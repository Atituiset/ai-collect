# SGLang推理优化


本文围绕 Scheduler 调度器里最核心的 Batch 构建逻辑展开：

**``` 1get_next_batch_to_run 2  -> get_new_batch_prefill 3  -> update_running_batch ```**

这条链路决定了下一次 forward 到底跑什么，是新的 prefill，还是已有请求的 decode。Batch 构建逻辑比较绕，建议先用下面的模拟请求把状态流转看懂，再回到代码里逐行阅读。

**一句话理解**

**Scheduler 每一轮调度，本质是在回答一个问题：**

**``` 1下一次 model forward 应该跑哪个 Batch？ ```**

这个问题不能只看 waiting\_queue，也不能只看 running\_batch，而是要同时处理 4 类状态：

**``` 1last_batch: 2上一轮刚 forward 完的 batch。 3它可能是 EXTEND，也可能是 DECODE。 4只有 EXTEND 才可能在本轮开头并入 running_batch。 5 6waiting_queue: 7等待第一次或下一段 prefill 的请求。 8 9running_batch: 10已经完成 prefill、后续可以 decode 的请求集合。 11 12chunked_req: 13  长 prompt 被切块后，尚未完成全部 prefill 的请求。 ```**

核心决策图如下：

**``` 1┌───────────────────────────────────────────────┐ 2│ get_next_batch_to_run                          │ 3│ 目标：返回下一次 forward 要跑的 ScheduleBatch  │ 4└───────────────────────┬───────────────────────┘ 5                        │ 6                        v 7┌───────────────────────────────────────────────┐ 8│ 1. 清理超时 / finished / 特殊 staging 状态     │ 9│    abort waiting timeout                       │ 10│    abort running timeout                       │ 11│    DLLM / HiSparse / chunked_req 特殊处理      │ 12└───────────────────────┬───────────────────────┘ 13                        │ 14                        v 15┌───────────────────────────────────────────────┐ 16│ 2. 如果 last_batch 是 EXTEND                  │ 17│    filter_batch 去掉 finished / chunked_req    │ 18│    再 merge 到 running_batch                   │ 19└───────────────────────┬───────────────────────┘ 20                        │ 21                        v 22┌───────────────────────────────────────────────┐ 23│ 3. 尝试构建新的 prefill batch                 │ 24│    get_new_batch_prefill                       │ 25└──────────────┬────────────────────────────────┘ 26               │ 27        有 new_batch? 28          │ 29    ┌─────┴─────┐ 30    │           │ 31   是           否 32    │           │ 33    v           v 34┌──────────┐   ┌────────────────────────────────┐ 35│ 返回     │   │ 4. running_batch 是否能 decode? │ 36│ EXTEND   │   └──────────────┬─────────────────┘ 37│ 或 MIXED │                  │ 38└──────────┘          ┌───────┴───────┐ 39                      │               │ 40                     能              不能 41                      │               │ 42                      v               v 43              ┌──────────────┐   ┌──────────┐ 44              │ update       │   │ 返回 None │ 45              │ running_batch│   └──────────┘ 46              └──────┬───────┘ 47                     │ 48                     v 49              ┌──────────────┐ 50              │ 返回 DECODE   │ 51              └──────────────┘ ```**

这里最重要的优先级是：

**``` 1last_batch 如果是 EXTEND，合入 running_batch 发生在本轮最开始。 2新 prefill batch 的优先级高于 running decode。 3只有 get_new_batch_prefill 返回 None，才会进入 update_running_batch。 ```**

换句话说：

**``` 1waiting_queue / chunked_req 能组出 EXTEND: 2  本轮跑 prefill。 3 4组不出 EXTEND，但 running_batch 里还有活请求: 5  本轮跑 decode。 6 7两者都没有: 8  本轮空转。 ```**

这个优先级解释了一个常见现象：即使 running\_batch 里已经有很多请求在等 decode，只要 Scheduler 还能安全接入新的 prefill，它仍然会先返回 EXTEND batch。

**关键代码位置**

**主要涉及以下文件：   - python/sglang/srt/managers/scheduler.py  - get\_next\_batch\_to\_run - get\_new\_batch\_prefill - \_get\_new\_batch\_prefill\_raw - update\_running\_batch  - python/sglang/srt/managers/schedule\_batch.py  - Req - ScheduleBatch - ScheduleBatch.init\_new - ScheduleBatch.prepare\_for\_extend - ScheduleBatch.prepare\_for\_decode - ScheduleBatch.filter\_batch - ScheduleBatch.merge\_batch - ScheduleBatch.mix\_with\_running  - python/sglang/srt/managers/schedule\_policy.py  - PrefillAdder - PrefillAdder.add\_one\_req - PrefillAdder.add\_chunked\_req - PrefillAdder.preempt\_to\_schedule**

**代码主干先拆开看**

**这部分代码最绕的原因是：一个函数里同时处理“调度决策”和“状态迁移”。建议先把三个函数的职责边界分清楚。**

**``` 1get_next_batch_to_run: 2调度总控。 3负责判断上轮 batch 是否需要合入 running_batch，并决定本轮返回 EXTEND / DECODE / None。 4 5get_new_batch_prefill: 6prefill admission。 7负责从 waiting_queue / chunked_req 中挑出可以 prefill 的请求，并构造成 EXTEND batch。 8 9update_running_batch: 10decode admission。 11  负责过滤 running_batch、检查 decode KV 空间、必要时 retract，然后构造成 DECODE batch。 ```**

**get\_next\_batch\_to\_run 的关键分段**

**可以把 get\_next\_batch\_to\_run 看成 5 个连续阶段。**

**``` 1阶段 1：处理不能继续等待或运行的请求 2 3_abort_on_waiting_timeout() 4_abort_on_running_timeout() 5DLLM / staging / chunked_req 特殊处理 ```**

这一段的重点不是构建 batch，而是保证后面的队列状态是干净的。已经超时或被 abort 的请求，不能继续参与 admission。

**``` 1阶段 2：处理上轮 batch 2 3if last_batch.forward_mode.is_extend(): 4    last_batch.filter_batch(chunked_req_to_exclude=...) 5    running_batch.merge_batch(last_batch) ```**

last\_batch 是 event loop 每轮末尾直接赋值的：

**``` 1batch = get_next_batch_to_run() 2run_batch(batch) 3process_batch_result(batch, result) 4self.last_batch = batch ```**

所以它可能是：

**``` 1EXTEND: 2  上轮跑的是 prefill。 3  本轮开头需要考虑是否把它合入 running_batch。 4 5DECODE: 6  上轮跑的是 decode。 7  它本来就是 running_batch 的推进结果，不需要再次 merge。 8 9None: 10  上轮没有 batch。 ```**

因此代码只处理：

**``` 1self.last_batch.forward_mode.is_extend() ```**

这是 prefill 到 decode 生命周期的入口。上一轮返回的 EXTEND batch 经过 forward 和采样后，会在本轮开头尝试合入 running\_batch。

这一段有两个关键细节：

**``` 1filter_batch: 2  去掉已经 generation finished 的请求。 3  通过 chunked_req_to_exclude 排除还没有完成全部 prefill 的 chunked_req。 4 5merge_batch: 6  合并 reqs / req_pool_indices / seq_lens / sampling_info 等 batch 级状态。 7  不是简单地 list extend。 ```**

这里“可以合入 running\_batch”的含义是：

**``` 1普通非 chunked prefill: 2  prefill forward 完成后，只要请求没有 generation finished，就可以合入 running_batch。 3 4chunked prefill: 5  中间 chunk 不能合入 running_batch。 6  只有最后一个 chunk 完成后，self.chunked_req 不再指向它，它才会通过 filter_batch 并被 merge。 7 8prefill-only / 已经生成结束的请求: 9  会被 filter_batch 过滤掉，不进入 decode。 ```**

**``` 1阶段 3：优先尝试新 prefill 2 3new_batch = get_new_batch_prefill() 4 5if new_batch is not None: 6    ret = new_batch ```**

这就是 prefill 优先级的来源。只要 get\_new\_batch\_prefill 能组出一个 batch，本轮就不会走普通 decode。

**``` 1阶段 4：没有 prefill 时，才推进 running decode 2 3if new_batch is None: 4    if not running_batch.is_empty() and not running_batch.is_prefill_only: 5        running_batch = update_running_batch(running_batch) 6        ret = running_batch ```**

这里的 update\_running\_batch 不只是“把 forward\_mode 改成 DECODE”，它会先过滤 finished 请求、检查内存、必要时 retract。

**``` 1阶段 5：返回前做 DP attention / ngram / schedule time 处理 2 3maybe_prepare_mlp_sync_batch(ret) 4_maybe_prepare_ngram_embedding(ret) 5set_schedule_time_batch(ret) ```**

所以主干决策是前 4 段，最后一段是返回 batch 前的补充处理。

**get\_new\_batch\_prefill 的关键分段**

**get\_new\_batch\_prefill 的核心是 admission：决定哪些请求可以进入本轮 EXTEND batch。  它不是按条数拿请求，而是按资源预算逐个尝试：**

**``` 1waiting_queue 2  -> policy.calc_priority 3  -> Req.init_next_round_input 4  -> PrefillAdder.add_one_req 5  -> can_run_list 6  -> ScheduleBatch.prepare_for_extend ```**

关键代码可以分成 6 段。

**``` 1阶段 1：快速失败 2 3if running_batch.batch_is_full and chunked_req is None: 4    return None 5 6if waiting_queue is empty and chunked_req is None: 7    return None 8 9if get_num_allocatable_reqs(running_bs) <= 0 and chunked_req is None: 10    running_batch.batch_is_full = True 11    return None ```**

这里检查的是“有没有必要尝试 prefill”。如果 running batch 已满、没有等待请求、请求池没有槽位，就直接返回 None，让外层去尝试 decode。

**``` 1阶段 2：给 waiting_queue 排优先级 2 3policy.calc_priority(waiting_queue, running_batch) ```**

这一步会改变 waiting\_queue 的调度顺序。后续 for req in waiting\_queue 不是原始到达顺序，而是策略处理后的顺序。

**``` 1阶段 3：创建 PrefillAdder 2 3adder = PrefillAdder( 4    page_size, 5    tree_cache, 6    token_to_kv_pool_allocator, 7    running_batch, 8    new_token_ratio, 9    max_prefill_tokens, 10    chunked_prefill_size, 11    ... 12) ```**

这里 PrefillAdder.\_\_init\_\_ 的参数名是 rem\_input\_tokens，但调用方传进去的是：

**``` 1self.max_prefill_tokens ```**

对应关系是：

**``` 1Scheduler._get_new_batch_prefill_raw: 2  PrefillAdder(..., self.max_prefill_tokens, chunked_prefill_size, ...) 3 4PrefillAdder.__init__: 5  def __init__(..., rem_input_tokens, rem_chunk_tokens, num_mixed_decode_tokens=0, ...): 6      self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens ```**

所以：

**``` 1rem_input_tokens 的初始值 2  = Scheduler.max_prefill_tokens 3  - mixed chunked prefill 中预留给 decode 的 token 数 ```**

Scheduler.max\_prefill\_tokens 又来自 model worker 初始化阶段：

**``` 1self.tp_worker.get_worker_info() 2  -> self.max_prefill_tokens ```**

从配置角度看，它最终受 ServerArgs.max\_prefill\_tokens、模型 worker 可用 KV/显存预算、后端实现等共同影响。对调度器这段代码而言，它就是“本轮 prefill 最多允许消耗多少 input token”的预算上限。

PrefillAdder 的作用是维护本轮 prefill 的动态预算：

**``` 1can_run_list: 2已经决定本轮可以跑的请求。 3 4rem_input_tokens: 5本轮 prefill input token 预算还剩多少。 6初始来自 self.max_prefill_tokens，后续每加入一个请求都会扣减。 7 8rem_total_tokens: 9KV cache 可用量扣掉 running_batch 未来 decode 预留后的余量。 10 11rem_chunk_tokens: 12chunked prefill 本轮还能吃多少 token。 13 14preempt_list: 15  priority preemption 下被撤回的 running 请求。 ```**

**``` 1阶段 4：优先处理未完成的 chunked_req 2 3if chunked_req is not None: 4    chunked_req.init_next_round_input() 5    chunked_req = adder.add_chunked_req(chunked_req) ```**

这个分支很关键。长 prompt 的中间 chunk 已经占用了部分状态，如果不优先继续调度，会造成资源和状态难以收敛。因此 chunked\_req 的优先级高于普通 waiting\_queue。

**``` 1阶段 5：逐个尝试 waiting_queue 里的请求 2 3for req in waiting_queue: 4    if running_batch.batch_is_full: 5        break or preempt_to_schedule(req) 6 7    req.init_next_round_input(tree_cache) 8    res = adder.add_one_req(req, ...) 9 10    if res != CONTINUE: 11        break ```**

这一段是最关键的地方。

req.init\_next\_round\_input(tree\_cache) 会先做：

**``` 1fill_ids = origin_input_ids + output_ids 2prefix_indices = tree_cache.match_prefix(...) 3extend_input_len = len(fill_ids) - len(prefix_indices) ```**

也就是说，进入 add\_one\_req 前，请求已经知道：

**``` 1完整序列多长 2命中多少 KV cache 3本轮真正还要计算多少 token ```**

adder.add\_one\_req 再判断：

**``` 1extend_input_len + max_new_tokens + page_size 是否超过 rem_total_tokens 2实际 input tokens 是否超过 rem_input_tokens 3chunked_prefill_size 是否需要截断 4prefix cache host hit 是否需要 load back 5是否需要 lock tree cache node ```**

如果允许加入：

**``` 1adder.can_run_list.append(req) 2adder._update_prefill_budget(...) ```**

\_update\_prefill\_budget 会扣减：

**``` 1rem_input_tokens -= ceil_paged_tokens(extend_input_len) 2rem_total_token_offset += extend_input_len + max_new_tokens + page_size 3rem_chunk_tokens -= extend_input_len       # 如果开启 chunked prefill ```**

如果不允许：

**``` 1返回 NO_TOKEN / OTHER 2外层 break，本轮 prefill batch 到此为止 ```**

**``` 1阶段 6：把 can_run_list 变成 EXTEND batch 2 3waiting_queue -= can_run_list 4 5new_batch = ScheduleBatch.init_new(can_run_list, ...) 6new_batch.prepare_for_extend() 7return new_batch ```**

ScheduleBatch.init\_new 只创建 batch 对象；prepare\_for\_extend 才真正把 Req 转成 forward 所需张量。

**prepare\_for\_extend 的关键产物**

**prepare\_for\_extend 的核心输入是每个请求的：**

**``` 1fill_ids 2prefix_indices 3extend_input_len ```**

它构造出的关键字段是：

**``` 1input_ids: 2每个请求未命中 prefix cache 的 token，全部拼平成一个 tensor。 3 4seq_lens: 5每个请求 forward 后的完整长度。 6 7prefix_lens: 8每个请求命中的 prefix 长度。 9 10extend_lens: 11每个请求本轮实际 extend 的 token 数。 12 13out_cache_loc: 14每个 extend token 对应的 KV cache 写入地址。 15 16req_pool_indices: 17每个请求在 req_to_token_pool 中的位置。 18 19sampling_info: 20  采样参数按 batch 聚合后的结构。 ```**

这一步还有一个容易忽略的副作用：它会更新 Req 自身的内存状态。

**``` 1req.req_pool_idx = req_pool_indices[i] 2req.kv_committed_len = seq_len 3req.kv_allocated_len = seq_len 4req.extend_batch_idx += 1 5req.cached_tokens / already_computed 更新 ```**

所以 prepare\_for\_extend 不是纯粹构造 tensor，它也把请求正式推进到了“已经为 prefill 分配资源”的状态。

**update\_running\_batch 的关键分段**

**update\_running\_batch 的任务是把 running\_batch 变成下一轮 decode batch。  主干可以拆成 4 段：**

**``` 1阶段 1：过滤已经完成的请求 2 3initial_bs = batch.batch_size() 4batch.filter_batch(v1_spec_info_filtered=True) 5 6if batch.is_empty(): 7    batch.batch_is_full = False 8    return batch ```**

这里会把已经 finished 的请求从 running\_batch 中移除，并同步过滤 req\_pool\_indices、seq\_lens、sampling\_info 等 batch 字段。

**``` 1阶段 2：检查下一步 decode 内存 2 3if not batch.check_decode_mem(): 4    retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(...) ```**

check\_decode\_mem 会估算下一次 decode 需要多少 KV cache：

**``` 1普通 decode: 2  每个请求通常需要 1 个新 token 的 KV 空间。 3 4page allocator: 5  如果跨 page 边界，需要按 page_size 申请。 6 7speculative decoding: 8  可能一次预留多个 draft / verify token 的空间。 ```**

如果空间不够，就进入 retract。

**``` 1阶段 3：retract 后把请求放回 waiting_queue 2 3for req in retracted_reqs: 4    self._add_request_to_queue(req, is_retracted=True) ```**

retract 会释放这些请求占用的 KV cache，但不会丢失请求进度。请求带着已有 output\_ids 回到 waiting\_queue，后续会再次通过 init\_next\_round\_input 做 prefix match。

**``` 1阶段 4：构建 DECODE batch 2 3batch.prepare_for_decode() 4return batch ```**

prepare\_for\_decode 做的事情和 prepare\_for\_extend 完全不同：

**``` 1input_ids: 2  不再是 prompt token，而是每个请求上一步采样出的最后一个 token。 3 4out_cache_loc: 5  为每个请求本轮 decode 的新 token 分配 KV 写入位置。 6 7seq_lens: 8  每个请求长度 +1。 9 10forward_mode: 11  改成 DECODE。 ```**

所以 update\_running\_batch 的本质是：

**``` 1running_batch 2  -> filter finished 3  -> ensure decode memory 4  -> retract if needed 5  -> prepare_for_decode 6  -> DECODE batch ```**

**先看三个重要对象**

**Req 是请求的状态机**

**Req 不是只保存输入 prompt，它会在调度和 forward 过程中持续变化。  一个生成请求可以简化成：**

**``` 1Req( 2  rid="R1", 3  origin_input_ids=[11, 12, 13, 14], 4  output_ids=[], 5  prefix_indices=[], 6  fill_ids=[], 7  extend_input_len=0, 8  req_pool_idx=None, 9) ```**

关键字段可以这样理解：

| 字段 | 含义 |
| --- | --- |
| origin\_input\_ids | 用户 prompt 的 token ids |
| output\_ids | 已经采样出来的输出 token ids |
| fill\_ids | 当前需要考虑的完整序列，通常是 origin\_input\_ids + output\_ids |
| prefix\_indices | prefix cache 命中的 KV 位置 |
| extend\_input\_len | 本轮 prefill 还需要实际 forward 的 token 数 |
| req\_pool\_idx | req-to-token pool 中给这个请求分配的槽位 |
| kv\_committed\_len | 已经确认写入 KV cache 的序列长度 |
| kv\_allocated\_len | 已经分配的 KV cache 长度 |

**ScheduleBatch 是 forward 前的批处理数据**

**ScheduleBatch 是 Scheduler 传给 model worker 的 batch 状态。  它既保存请求列表：**

**``` 1batch.reqs = [R1, R2, R3] ```**

也保存 forward 需要的张量：

**``` 1batch.input_ids 2batch.req_pool_indices 3batch.seq_lens 4batch.out_cache_loc 5batch.sampling_info ```**

最关键的是 forward\_mode：

**``` 1EXTEND: 2  prefill，新请求或 chunked prefill 的一段 prompt。 3 4DECODE: 5  decode，running_batch 中每个请求继续生成一个 token。 6 7MIXED: 8  mixed chunked prefill，把新 prefill 和已有 decode 混在同一次 forward。 9 10IDLE: 11  DP attention 等场景下用于同步的空 batch。 ```**

**PrefillAdder 是 admission controller**

**PrefillAdder 决定 waiting\_queue 里的请求哪些能进入新的 prefill batch。  它同时考虑：**

**``` 1请求池槽位是否足够 2KV cache token 预算是否足够 3max_prefill_tokens 是否足够 4chunked_prefill_size 是否足够 5prefix cache 命中后实际需要算多少 token 6running_batch 未来 decode 还要预留多少 token 7LoRA / priority / preemption / prefill delayer 等策略 ```**

因此 get\_new\_batch\_prefill 不是简单的 waiting\_queue[:batch\_size]。

**模拟实际的请求**

**下面用 4 个请求模拟主流程。假设：**

**``` 1page_size = 1 2max_prefill_tokens = 10 3chunked_prefill_size = None 4pp_max_micro_batch_size = 4 5req_to_token_pool 还有足够槽位 6KV cache 还有足够空间 7不开启 overlap / speculative / LoRA / disaggregation / DP attention ```**

初始请求：

| 请求 | prompt token | max\_new\_tokens | prefix cache 命中 | 初始 output\_ids |
| --- | --- | --- | --- | --- |
| R1 | [11,12,13,14] | 3 | 0 | [] |
| R2 | [21,22,23] | 2 | 1 个 token | [] |
| R3 | [31,32,33,34,35,36] | 2 | 3 个 token | [] |
| R4 | [41,42] | 2 | 0 | [] |

初始 Scheduler 状态：

**``` 1waiting_queue = [R1, R2, R3, R4] 2running_batch = [] 3last_batch = None 4chunked_req = None ```**

**模拟请求的主干路径图**

**下面这张图只对应后面 R1-R4 的简化场景：没有 DP attention、没有 overlap、没有 speculative decoding、没有 disaggregation，也没有 chunked prefill。它用来帮助先理解主干路径。**

**``` 1┌──────────────────────────────┐ 2│ get_next_batch_to_run         │ 3└───────────────┬──────────────┘ 4                │ 5                v 6┌──────────────────────────────┐ 7│ 1. 处理上轮 last_batch        │ 8│    EXTEND -> running_batch    │ 9└───────────────┬──────────────┘ 10                │ 11                v 12┌──────────────────────────────┐ 13│ 2. 尝试 get_new_batch_prefill │ 14└───────┬──────────────────────┘ 15        │ 16        │ 有新 prefill batch 17        v 18┌──────────────────────────────┐ 19│ 返回 EXTEND batch             │ 20└──────────────────────────────┘ 21 22        │ 没有新 prefill batch 23        v 24┌──────────────────────────────┐ 25│ 3. update_running_batch       │ 26│    running_batch -> DECODE    │ 27└───────┬──────────────────────┘ 28        │ 29        v 30┌──────────────────────────────┐ 31│ 返回 DECODE batch             │ 32└──────────────────────────────┘ ```**

注意第 1 步非常重要：如果上轮跑的是 prefill，last\_batch 会在下一轮调度开始时尝试并入 running\_batch。普通 prefill 完成且未 generation finished 的请求会进入 decode；未完成的 chunked prefill 会被排除，等最后一个 chunk 完成后再进入。

**第 1 轮：从 waiting\_queue 构建 prefill batch**

**第一次调用：**

**``` 1get_next_batch_to_run() ```**

此时：

**``` 1last_batch = None 2running_batch = [] 3waiting_queue = [R1, R2, R3, R4] ```**

**get\_next\_batch\_to\_run 前半段**

**因为没有 last\_batch，也没有已有 running\_batch，前半段基本不会改变状态。  随后进入：**

**``` 1new_batch = self.get_new_batch_prefill() ```**

**get\_new\_batch\_prefill 创建 PrefillAdder**

**get\_new\_batch\_prefill 内部会调用 \_get\_new\_batch\_prefill\_raw，先做几个快速判断：**

**``` 1running_batch.batch_is_full ?  否 2waiting_queue 是否为空 ?       否 3req pool 是否还有槽位 ?        是 ```**

然后创建 PrefillAdder。

此时 PrefillAdder 的关键预算可以简化理解为：

**``` 1rem_input_tokens = max_prefill_tokens = 10 2can_run_list = [] 3running_batch = [] ```**

这里的 10 来自本例前面假设的 max\_prefill\_tokens = 10。真实运行时，PrefillAdder 收到的是 Scheduler 的 self.max\_prefill\_tokens，然后赋给自己的 self.rem\_input\_tokens。

**逐个检查 waiting\_queue**

**R1 先调用：**

**``` 1req.init_next_round_input(self.tree_cache) ```**

R1 没有 prefix cache 命中：

**``` 1fill_ids = origin_input_ids + output_ids 2         = [11,12,13,14] + [] 3         = [11,12,13,14] 4 5prefix_indices = [] 6extend_input_len = len(fill_ids) - len(prefix_indices) 7                 = 4 - 0 8                 = 4 ```**

然后进入：

**``` 1adder.add_one_req(R1, ...) ```**

R1 占用 4 个 prefill token，允许加入：

**``` 1can_run_list = [R1] 2rem_input_tokens = 10 - 4 = 6 ```**

#### R2

R2 命中 1 个 prefix token：

**``` 1fill_ids = [21,22,23] 2prefix_indices = [kv_21] 3extend_input_len = 3 - 1 = 2 ```**

加入后：

**``` 1can_run_list = [R1, R2] 2rem_input_tokens = 6 - 2 = 4 ```**

#### R3

R3 命中 3 个 prefix token：

**``` 1fill_ids = [31,32,33,34,35,36] 2prefix_indices = [kv_31, kv_32, kv_33] 3extend_input_len = 6 - 3 = 3 ```**

加入后：

**``` 1can_run_list = [R1, R2, R3] 2rem_input_tokens = 4 - 3 = 1 ```**

#### R4

R4 需要 2 个 prefill token：

**``` 1fill_ids = [41,42] 2prefix_indices = [] 3extend_input_len = 2 ```**

但当前 rem\_input\_tokens = 1，不够放 R4。此时 add\_one\_req 会返回类似 AddReqResult.OTHER 或 NO\_TOKEN 的结果，调度停止继续扩张这个 prefill batch。

最终：

**``` 1can_run_list = [R1, R2, R3] 2waiting_queue = [R4] ```**

**构建 ScheduleBatch**

**get\_new\_batch\_prefill 接着执行：**

**``` 1new_batch = ScheduleBatch.init_new(...) 2new_batch.prepare_for_extend() ```**

init\_new 只是创建 batch 壳子，真正把请求变成 forward 张量的是 prepare\_for\_extend。

**prepare\_for\_extend 做了什么**

**对 R1、R2、R3 来说：**

**``` 1R1: 2fill_ids = [11,12,13,14] 3prefix_len = 0 4extend_len = 4 5本轮 input_ids = [11,12,13,14] 6seq_len = 4 7 8R2: 9fill_ids = [21,22,23] 10prefix_len = 1 11extend_len = 2 12本轮 input_ids = [22,23] 13seq_len = 3 14 15R3: 16fill_ids = [31,32,33,34,35,36] 17prefix_len = 3 18extend_len = 3 19本轮 input_ids = [34,35,36] 20  seq_len = 6 ```**

所以 Batch 张量大致是：

**``` 1batch.forward_mode = EXTEND 2batch.reqs = [R1, R2, R3] 3 4batch.input_ids = [ 511,12,13,14,   # R1 622,23,         # R2 734,35,36       # R3 8] 9 10batch.prefix_lens = [0, 1, 3] 11batch.extend_lens = [4, 2, 3] 12batch.extend_num_tokens = 9 13batch.seq_lens = [4, 3, 6] 14batch.seq_lens_sum = 13 ```**

同时 prepare\_for\_extend 会分配内存：

**``` 1req_pool_indices: 2  R1 -> req_pool_idx 0 3  R2 -> req_pool_idx 1 4  R3 -> req_pool_idx 2 5 6out_cache_loc: 7  给 9 个实际 extend token 分配 KV 写入位置 ```**

可视化：

**``` 1EXTEND batch 2 3R1: [11][12][13][14] 4     ^^^^^^^^^^^^^^^^ 5     本轮计算 4 token 6 7R2: [21][22][23] 8     ^^  ^^^^^^^ 9     命中  本轮计算 2 token 10 11R3: [31][32][33][34][35][36] 12     ^^^^^^^^^^^  ^^^^^^^^^^^ 13     命中 3 token   本轮计算 3 token ```**

get\_next\_batch\_to\_run 看到 new\_batch is not None，直接返回这个 EXTEND batch。

第 1 轮返回：

**``` 1return EXTEND([R1, R2, R3]) ```**

此时还没有立刻把它放进 running\_batch。它会先被 forward，forward 完后成为 last\_batch。下一轮调度开头，如果它是已完成 prefill 的请求，就会被合入 running\_batch。

**第 2 轮：EXTEND last\_batch 合入 running\_batch，再继续找新 prefill**

**假设第 1 轮 EXTEND forward 完成后，每个请求已经采样出第一个 token：**

**``` 1R1.output_ids = [101] 2R2.output_ids = [201] 3R3.output_ids = [301] ```**

Scheduler 会记录：

**``` 1last_batch = EXTEND([R1, R2, R3]) 2running_batch = [] 3waiting_queue = [R4] ```**

第二次调用：

**``` 1get_next_batch_to_run() ```**

**先判断 last\_batch 是否需要合并进 running\_batch**

**代码主干是：**

**``` 1if self.last_batch and self.last_batch.forward_mode.is_extend(): 2    self.last_batch.filter_batch(...) 3    if not self.last_batch.is_empty(): 4        if self.running_batch.is_empty(): 5            self.running_batch = self.last_batch 6        else: 7            self.running_batch.merge_batch(self.last_batch) ```**

这里先判断 last\_batch.forward\_mode.is\_extend()。如果上一轮跑的是 DECODE，就不会走这段 merge 逻辑。

filter\_batch 会把已经 generation finished 的请求过滤掉，也会按 chunked\_req\_to\_exclude 排除未完成的 chunked prefill。当前例子没有 chunked prefill，如果 R1/R2/R3 都还没结束：

**``` 1running_batch = [R1, R2, R3] 2last_batch 仍然是上轮 EXTEND batch 的对象引用 3waiting_queue = [R4] ```**

这里有个容易混淆的点：

**``` 1EXTEND batch forward 后，不会在同一轮立即变成 DECODE。 2它会在下一次 get_next_batch_to_run 的开头，根据是否 finished / 是否未完成 chunked prefill，决定能否合入 running_batch。 ```**

**再尝试新的 prefill**

**此时 waiting\_queue 还有 R4，所以 Scheduler 继续尝试：**

**``` 1new_batch = self.get_new_batch_prefill() ```**

R4 的状态：

**``` 1fill_ids = [41,42] 2prefix_indices = [] 3extend_input_len = 2 ```**

如果内存和请求槽位允许，R4 会组成新的 prefill batch：

**``` 1return EXTEND([R4]) ```**

第 2 轮返回：

**``` 1return EXTEND([R4]) ```**

注意这一轮虽然 running\_batch 里已经有 R1/R2/R3，但因为有新的 prefill 可以跑，所以优先跑 R4 的 prefill，而不是 decode R1/R2/R3。

状态图：

**``` 1第 2 轮开始 2 3last_batch:    EXTEND([R1,R2,R3]) 4running_batch: [] 5waiting_queue: [R4] 6 7合并 last_batch 后 8 9running_batch: [R1,R2,R3] 10waiting_queue: [R4] 11 12构建新 prefill 后 13 14return: EXTEND([R4]) ```**

**第 3 轮：没有新 prefill，开始 update\_running\_batch 做 decode**

**假设第 2 轮 R4 prefill 完成后：**

**``` 1R4.output_ids = [401] 2last_batch = EXTEND([R4]) 3running_batch = [R1, R2, R3] 4waiting_queue = [] ```**

第三次调用：

**``` 1get_next_batch_to_run() ```**

**先合入 R4**

**last\_batch 是 EXTEND，所以合入 running\_batch：**

**``` 1running_batch = [R1, R2, R3, R4] 2waiting_queue = [] ```**

**get\_new\_batch\_prefill 返回 None**

**因为 waiting\_queue 空了：**

**``` 1new_batch = None ```**

于是进入 decode 路径：

**``` 1self.running_batch = self.update_running_batch(self.running_batch) 2ret = self.running_batch if not self.running_batch.is_empty() else None ```**

**update\_running\_batch 的主流程**

**update\_running\_batch 可以拆成 4 步：**

**``` 11. filter_batch 2   过滤已经 finished 的请求。 3 42. check_decode_mem 5   判断下一步 decode 的 KV cache 是否够。 6 73. 必要时 retract_decode 8   如果 KV cache 不够，撤回部分请求，释放 KV，再把被撤回请求放回 waiting_queue。 9 104. prepare_for_decode 11   把 running_batch 改造成下一次 decode forward 需要的 batch。 ```**

如果 R1/R2/R3/R4 都还没结束，且 decode 内存足够，则进入：

**``` 1batch.prepare_for_decode() ```**

**prepare\_for\_decode 做了什么**

**此时每个请求的 output\_ids 至少有一个 token：**

**``` 1R1.output_ids = [101] 2R2.output_ids = [201] 3R3.output_ids = [301] 4R4.output_ids = [401] ```**

decode forward 的输入就是每个请求上一步采样出的最后一个 token：

**``` 1batch.forward_mode = DECODE 2batch.input_ids = [101, 201, 301, 401] ```**

同时为每个请求分配一个新的 KV 写入位置：

**``` 1batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1) ```**

并更新长度：

**``` 1prefill 后 seq_lens: 2R1 = 4 3R2 = 3 4R3 = 6 5R4 = 2 6 7prepare_for_decode 后 seq_lens: 8R1 = 5 9R2 = 4 10R3 = 7 11  R4 = 3 ```**

可视化：

**``` 1DECODE batch 2 3R1: [11][12][13][14] -> [101] -> 本轮输入 101，写第 5 个 KV 4R2: [21][22][23]     -> [201] -> 本轮输入 201，写第 4 个 KV 5R3: [31][32][33][34][35][36] -> [301] -> 本轮输入 301，写第 7 个 KV 6R4: [41][42]         -> [401] -> 本轮输入 401，写第 3 个 KV ```**

第 3 轮返回：

**``` 1return DECODE([R1, R2, R3, R4]) ```**

**一张表看完三轮状态变化**

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | 轮次 | 进入时 waiting\_queue | 进入时 running\_batch | 先处理 last\_batch | 返回 batch | | --- | --- | --- | --- | --- | | 第 1 轮 | [R1,R2,R3,R4] | [] | 无 | EXTEND([R1,R2,R3]) | | 第 2 轮 | [R4] | [] | [R1,R2,R3] -> running\_batch | EXTEND([R4]) | | 第 3 轮 | [] | [R1,R2,R3] | [R4] -> running\_batch | DECODE([R1,R2,R3,R4]) |   这就是最常见的主干逻辑： |

| ``` 1新请求 prefill 优先。 2prefill 结果下一轮合入 running_batch。 3没有 prefill 可跑时，running_batch 才进入 decode。 ``` |

**Batch 字段在 EXTEND 和 DECODE 中的区别**

**EXTEND batch**

**EXTEND 关注的是“哪些 prompt token 还没有 KV，需要补算”。**

**``` 1input_ids: 2所有请求未命中 prefix cache 的 token 拼平。 3 4prefix_lens: 5每个请求命中的 prefix 长度。 6 7extend_lens: 8每个请求本轮实际 prefill 的 token 数。 9 10extend_num_tokens: 11sum(extend_lens)。 12 13seq_lens: 14每个请求本轮 forward 后的完整序列长度。 15 16out_cache_loc: 17  每个 extend token 对应的 KV cache 写入位置。 ```**

**DECODE batch**

**DECODE 关注的是“每个请求继续生成一个 token”。**

**``` 1input_ids: 2  每个请求上一步采样出来的最后一个 token。 3 4prefix_lens / extend_lens: 5  decode 不需要这些 extend 专用字段。 6 7seq_lens: 8  每个请求加 1 后的完整长度。 9 10out_cache_loc: 11  每个请求新 token 的 KV cache 写入位置。 ```**

对比：

| 字段 | EXTEND | DECODE |
| --- | --- | --- |
| forward\_mode | EXTEND | DECODE |
| input\_ids | 未缓存的 prompt token 拼平 | 每个请求 1 个 last output token |
| seq\_lens | prompt/prefix 处理后的完整长度 | 原长度 + 1 |
| out\_cache\_loc | 每个 extend token 一个位置 | 每个请求一个位置 |
| prefix\_lens | 有意义 | 一般不使用 |
| extend\_lens | 有意义 | 一般不使用 |

**为什么 prefill 后要下一轮才合入 running\_batch**

**因为 get\_next\_batch\_to\_run 返回 batch 后，还要经过：**

**``` 1run_batch 2  -> model forward 3  -> sampling 4  -> process_batch_result 5  -> 更新 req.output_ids / finished 状态 ```**

只有 forward 结束后，Scheduler 才知道：

**``` 1这个请求是否 finished 2采样出了哪个 token 3output_ids 是否更新 4logprob / hidden states 等结果如何处理 ```**

所以 prefill batch 不能在构建出来时立刻进入 decode。它必须先作为 last\_batch 经过一次 forward 结果处理，然后在下一轮调度开头经过 filter\_batch。

合入规则可以简化成：

**``` 1last_batch 不是 EXTEND: 2不处理。 3例如上一轮是 DECODE，它已经属于 running_batch 的生命周期。 4 5last_batch 是 EXTEND，且请求已经 generation finished: 6filter_batch 过滤掉，不进入 running_batch。 7 8last_batch 是 EXTEND，但请求仍是未完成的 chunked_req: 9通过 chunked_req_to_exclude 排除，不进入 running_batch。 10 11last_batch 是 EXTEND，且请求 prefill 已完整完成、还需要继续生成: 12  合入 running_batch，后续参与 decode。 ```**

**update\_running\_batch 中的 retract 是什么**

**decode 阶段每轮都要给每个 running request 新增 KV cache。  如果当前 running\_batch 太大，下一步 decode 的 KV cache 不够，update\_running\_batch 会触发：**

**``` 1batch.retract_decode(...) ```**

可以把 retract 理解成：

**``` 1为了避免 decode OOM，临时撤回一部分 running 请求。 2撤回请求会释放 KV cache，并重新放回 waiting_queue。 3后续它们会以带 output_ids 的状态重新 prefill / prefix match。 ```**

简化例子：

**``` 1running_batch = [R1,R2,R3,R4] 2下一步 decode 需要 4 个 KV slot 3但当前只能再分配 2 个 KV slot ```**

可能发生：

**``` 1保留: [R1,R2] 2撤回: [R3,R4] 3 4running_batch = [R1,R2] 5waiting_queue = [R3,R4] ```**

撤回的请求不是丢掉，而是：

**``` 1R3.origin_input_ids 不变 2R3.output_ids 保留已经生成的 token 3R3.is_retracted = True 4R3 后续重新进入 waiting_queue ```**

下一次 init\_next\_round\_input 时：

**``` 1fill_ids = origin_input_ids + output_ids ```**

所以它会尝试从 prefix cache 找回已经算过的 KV，只补算缺失部分。

**chunked prefill 如何改变 Batch 构建**

**如果一个请求 prompt 很长，超过 chunked\_prefill\_size，它不会一次性 prefill 完，而是拆成多轮。  假设：**

**``` 1chunked_prefill_size = 4 2R5.origin_input_ids = [51,52,53,54,55,56,57,58,59] 3prefix 命中 0 ```**

第一段：

**``` 1fill_ids = [51,52,53,54] 2extend_input_len = 4 3new_chunked_req = R5 4chunked_req = R5 5return EXTEND([R5]) ```**

第二段调度时，Scheduler 会优先处理已有的 chunked\_req：

**``` 1R5.init_next_round_input() 2adder.add_chunked_req(R5) ```**

如果还没完成：

**``` 1fill_ids = [51,52,53,54,55,56,57,58] 2prefix_indices = 前 4 个 token 的 KV 3extend_input_len = 4 4chunked_req 仍然是 R5 ```**

最后一段：

**``` 1fill_ids = [51,52,53,54,55,56,57,58,59] 2prefix_indices = 前 8 个 token 的 KV 3extend_input_len = 1 4chunked_req = None ```**

可视化：

**``` 1R5 长 prompt: 2 3[51][52][53][54][55][56][57][58][59] 4 └──── chunk 1 ────┘ 5                 └──── chunk 2 ────┘ 6                                 └ chunk 3 ```**

get\_next\_batch\_to\_run 开头对 chunked\_req 有特殊处理，是为了避免未完成的 chunk 被过早合入 decode：

**``` 1chunked prefill 没结束: 2  不应该进入 running_batch decode。 3 4chunked prefill 全部结束: 5  才能作为完整请求进入 running_batch。 ```**

**mixed chunked prefill 是什么**

**普通情况下：**

**``` 1有新 prefill -> 跑 prefill 2没新 prefill -> 跑 decode ```**

开启 mixed chunked prefill 后，在满足条件时可以把两者混在一起：

**``` 1new_batch = 新请求的 EXTEND 2running_batch = 已有请求的 DECODE 3 4new_batch.mix_with_running(running_batch) 5 6return MIXED([prefill reqs + decode reqs]) ```**

简化图：

**``` 1MIXED batch input_ids 2 3prefill R5: [55][56][57][58] 4decode  R1: [102] 5decode  R2: [202] 6 7拼成: 8input_ids = [55,56,57,58,102,202] ```**

代码上在 \_get\_new\_batch\_prefill\_raw 的末尾：

**``` 1if self.is_mixed_chunk and not self.running_batch.is_empty(): 2    self.running_batch.filter_batch(...) 3    self.running_batch.prepare_for_decode() 4    new_batch.mix_with_running(self.running_batch) 5    self.running_batch = ScheduleBatch(reqs=[], ...) ```**

这样一次 forward 里既补长 prompt 的一段，也推进已有请求 decode。

**prefix cache 对 Batch 的影响**

**prefix cache 最直接影响的是：**

**``` 1prefix_indices 2extend_input_len 3input_ids 4out_cache_loc 数量 ```**

同样 6 个 token 的 prompt：

**``` 1R_no_cache: 2origin_input_ids = [1,2,3,4,5,6] 3prefix_indices = [] 4extend_input_len = 6 5input_ids = [1,2,3,4,5,6] 6 7R_cache: 8origin_input_ids = [1,2,3,4,5,6] 9prefix_indices = [kv1,kv2,kv3,kv4] 10extend_input_len = 2 11  input_ids = [5,6] ```**

所以 Scheduler admission 时不能只看 prompt 长度，而要先：

**``` 1req.init_next_round_input(self.tree_cache) ```**

让请求完成 prefix match，然后 PrefillAdder 再根据真实 extend\_input\_len 判断能不能加入 batch。

**三个函数的快速回顾**

**前面已经展开过关键代码分段，这里再用压缩伪代码把三者串起来，方便回看。**

**get\_next\_batch\_to\_run**

**去掉 DP attention、HiSparse、DLLM、ngram、overlap 等分支后，可以简化成：**

**``` 1def get_next_batch_to_run(): 2    # 1. 上轮 prefill 结果进入 running_batch 3    if last_batch and last_batch.forward_mode.is_extend(): 4        last_batch.filter_batch(chunked_req_to_exclude=...) 5        if not last_batch.is_empty(): 6            if running_batch.is_empty(): 7                running_batch = last_batch 8            else: 9                running_batch.merge_batch(last_batch) 10 11    # 2. 优先尝试新 prefill 12    new_batch = get_new_batch_prefill() 13    if new_batch is not None: 14        return new_batch 15 16    # 3. 没有 prefill，推进 decode 17    if not running_batch.is_empty() and not running_batch.is_prefill_only: 18        running_batch = update_running_batch(running_batch) 19        if not running_batch.is_empty(): 20            return running_batch 21 22    return None ```**

**get\_new\_batch\_prefill**

**简化后：**

**``` 1def get_new_batch_prefill(): 2    if running_batch.batch_is_full and chunked_req is None: 3        return None 4 5    if waiting_queue is empty and chunked_req is None: 6        return None 7 8    if req pool 没有槽位 and chunked_req is None: 9        running_batch.batch_is_full = True 10        return None 11 12    policy.calc_priority(waiting_queue, running_batch) 13 14    adder = PrefillAdder(...) 15 16    if chunked_req is not None: 17        chunked_req.init_next_round_input() 18        chunked_req = adder.add_chunked_req(chunked_req) 19 20    for req in waiting_queue: 21        if batch 已满: 22            break or preempt_to_schedule(req) 23 24        req.init_next_round_input(tree_cache) 25        result = adder.add_one_req(req, ...) 26 27        if result != CONTINUE: 28            break 29 30    if adder.can_run_list is empty: 31        return None 32 33    waiting_queue -= adder.can_run_list 34 35    new_batch = ScheduleBatch.init_new(adder.can_run_list, ...) 36    new_batch.prepare_for_extend() 37 38    if mixed chunked prefill: 39        new_batch.mix_with_running(running_batch) 40 41    return new_batch ```**

**update\_running\_batch**

**简化后：**

**``` 1def update_running_batch(batch): 2    initial_bs = batch.batch_size() 3 4    batch.filter_batch() 5    if batch.is_empty(): 6        batch.batch_is_full = False 7        return batch 8 9    if not batch.check_decode_mem(): 10        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(...) 11        self.new_token_ratio = new_token_ratio 12 13        for req in retracted_reqs: 14            self._add_request_to_queue(req, is_retracted=True) 15    else: 16        self.new_token_ratio = max( 17            self.new_token_ratio - self.new_token_ratio_decay, 18            self.min_new_token_ratio, 19        ) 20 21    if batch.batch_size() < initial_bs: 22        batch.batch_is_full = False 23 24    if batch.is_empty(): 25        return batch 26 27    batch.prepare_for_decode() 28    return batch ```**

**常见容易绕的点**

**1. running\_batch 不等于正在 forward 的 batch**

**running\_batch 指的是“已经进入 decode 生命周期的请求集合”。  当前返回给 forward 的 batch 可能是：**

**``` 1EXTEND new_batch 2DECODE running_batch 3MIXED new_batch + running_batch ```**

**2. EXTEND last\_batch 是 prefill 到 decode 的桥**

**last\_batch 不一定是 prefill batch。event loop 每轮都会执行 self.last\_batch = batch，所以上一轮返回 DECODE 时，last\_batch 也是 DECODE。  只有当 last\_batch.forward\_mode.is\_extend() 为 True 时，它才承担 prefill 到 decode 的桥接作用。下一轮调度开头会先 filter\_batch，排除已经 generation finished 的请求和未完成的 chunked prefill，再把剩下的请求合入 running\_batch。**

**3. input\_ids 在 EXTEND 和 DECODE 中含义完全不同**

**``` 1EXTEND input_ids: 2  prompt 中没有命中 prefix cache 的 token。 3 4DECODE input_ids: 5  每个请求上一步采样出的最后一个 token。 ```**

**4. prefix cache 命中会改变 admission 结果**

**两个 prompt 一样长的请求，命中多的请求 extend\_input\_len 更小，更容易被加入 prefill batch。**

**5. batch\_is\_full 是优化标记，不是绝对状态**

**running\_batch.batch\_is\_full 用来减少无意义的 prefill 尝试。某些场景会重置它，比如：**

**``` 1有请求完成并被 filter 掉 2开启 priority preemption 3hybrid SWA 4retract 后 running batch 变小 ```**

**6. 被 retract 的请求会回 waiting\_queue**

**retract 不是失败，也不是丢请求，而是为了释放 decode KV cache。请求保留 output\_ids，后面再通过 prefix cache 恢复。**

**最后用一句话串起来**

**ScheduleBatch 构建的核心可以这样记：**

**``` 1waiting_queue 里的 Req 2  -> init_next_round_input 做 prefix match 3  -> PrefillAdder 判断预算 4  -> ScheduleBatch.prepare_for_extend 变成 EXTEND forward 数据 5  -> forward 后作为 last_batch 6  -> 下一轮合入 running_batch 7  -> update_running_batch + prepare_for_decode 变成 DECODE forward 数据 ```**

理解了这个状态流转，再去看 DP attention、chunked prefill、overlap、speculative decoding、disaggregation 等分支，会容易很多。
