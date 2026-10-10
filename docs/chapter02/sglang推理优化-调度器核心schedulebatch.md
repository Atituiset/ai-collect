# SGLang推理优化


本文围绕 Scheduler 调度器里最核心的 Batch 构建逻辑展开：


```text
get_next_batch_to_run
  -> get_new_batch_prefill
  -> update_running_batch
```


这条链路决定了下一次 forward 到底跑什么，是新的 prefill，还是已有请求的 decode。Batch 构建逻辑比较绕，建议先用下面的模拟请求把状态流转看懂，再回到代码里逐行阅读。

**一句话理解**

**Scheduler 每一轮调度，本质是在回答一个问题：**


```text
下一次 model forward 应该跑哪个 Batch？
```


这个问题不能只看 waiting\_queue，也不能只看 running\_batch，而是要同时处理 4 类状态：


```text
last_batch:
上一轮刚 forward 完的 batch。
它可能是 EXTEND，也可能是 DECODE。
只有 EXTEND 才可能在本轮开头并入 running_batch。

waiting_queue:
等待第一次或下一段 prefill 的请求。

running_batch:
已经完成 prefill、后续可以 decode 的请求集合。

chunked_req:
  长 prompt 被切块后，尚未完成全部 prefill 的请求。
```


核心决策图如下：


```text
┌───────────────────────────────────────────────┐
│ get_next_batch_to_run                          │
│ 目标：返回下一次 forward 要跑的 ScheduleBatch  │
└───────────────────────┬───────────────────────┘
                        │
                        v
┌───────────────────────────────────────────────┐
│ 1. 清理超时 / finished / 特殊 staging 状态     │
│    abort waiting timeout                       │
│    abort running timeout                       │
│    DLLM / HiSparse / chunked_req 特殊处理      │
└───────────────────────┬───────────────────────┘
                        │
                        v
┌───────────────────────────────────────────────┐
│ 2. 如果 last_batch 是 EXTEND                  │
│    filter_batch 去掉 finished / chunked_req    │
│    再 merge 到 running_batch                   │
└───────────────────────┬───────────────────────┘
                        │
                        v
┌───────────────────────────────────────────────┐
│ 3. 尝试构建新的 prefill batch                 │
│    get_new_batch_prefill                       │
└──────────────┬────────────────────────────────┘
               │
        有 new_batch?
          │
    ┌─────┴─────┐
    │           │
   是           否
    │           │
    v           v
┌──────────┐   ┌────────────────────────────────┐
│ 返回     │   │ 4. running_batch 是否能 decode? │
│ EXTEND   │   └──────────────┬─────────────────┘
│ 或 MIXED │                  │
└──────────┘          ┌───────┴───────┐
                      │               │
                     能              不能
                      │               │
                      v               v
              ┌──────────────┐   ┌──────────┐
              │ update       │   │ 返回 None │
              │ running_batch│   └──────────┘
              └──────┬───────┘
                     │
                     v
              ┌──────────────┐
              │ 返回 DECODE   │
              └──────────────┘
```


这里最重要的优先级是：


```text
last_batch 如果是 EXTEND，合入 running_batch 发生在本轮最开始。
新 prefill batch 的优先级高于 running decode。
只有 get_new_batch_prefill 返回 None，才会进入 update_running_batch。
```


换句话说：


```text
waiting_queue / chunked_req 能组出 EXTEND:
  本轮跑 prefill。

组不出 EXTEND，但 running_batch 里还有活请求:
  本轮跑 decode。

两者都没有:
  本轮空转。
```


这个优先级解释了一个常见现象：即使 running\_batch 里已经有很多请求在等 decode，只要 Scheduler 还能安全接入新的 prefill，它仍然会先返回 EXTEND batch。

**关键代码位置**

**主要涉及以下文件：**

- python/sglang/srt/managers/scheduler.py
- get\_next\_batch\_to\_run
- get\_new\_batch\_prefill
- \_get\_new\_batch\_prefill\_raw
- update\_running\_batch
- python/sglang/srt/managers/schedule\_batch.py
- Req
- ScheduleBatch
- ScheduleBatch.init\_new
- ScheduleBatch.prepare\_for\_extend
- ScheduleBatch.prepare\_for\_decode
- ScheduleBatch.filter\_batch
- ScheduleBatch.merge\_batch
- ScheduleBatch.mix\_with\_running
- python/sglang/srt/managers/schedule\_policy.py
- PrefillAdder
- PrefillAdder.add\_one\_req
- PrefillAdder.add\_chunked\_req
- PrefillAdder.preempt\_to\_schedule

**代码主干先拆开看**

**这部分代码最绕的原因是：一个函数里同时处理“调度决策”和“状态迁移”。建议先把三个函数的职责边界分清楚。**


```text
get_next_batch_to_run:
调度总控。
负责判断上轮 batch 是否需要合入 running_batch，并决定本轮返回 EXTEND / DECODE / None。

get_new_batch_prefill:
prefill admission。
负责从 waiting_queue / chunked_req 中挑出可以 prefill 的请求，并构造成 EXTEND batch。

update_running_batch:
decode admission。
  负责过滤 running_batch、检查 decode KV 空间、必要时 retract，然后构造成 DECODE batch。
```


**get\_next\_batch\_to\_run 的关键分段**

**可以把 get\_next\_batch\_to\_run 看成 5 个连续阶段。**


```text
阶段 1：处理不能继续等待或运行的请求

_abort_on_waiting_timeout()
_abort_on_running_timeout()
DLLM / staging / chunked_req 特殊处理
```


这一段的重点不是构建 batch，而是保证后面的队列状态是干净的。已经超时或被 abort 的请求，不能继续参与 admission。


```text
阶段
：处理上轮 batch 2
if last_batch.forward_mode.is_extend():
    last_batch.filter_batch(chunked_req_to_exclude=...)
    running_batch.merge_batch(last_batch)
```


last\_batch 是 event loop 每轮末尾直接赋值的：


```python
batch = get_next_batch_to_run()
run_batch(batch)
process_batch_result(batch, result)
self.last_batch = batch
```


所以它可能是：


```text
EXTEND:
  上轮跑的是 prefill。
  本轮开头需要考虑是否把它合入 running_batch。

DECODE:
  上轮跑的是 decode。
  它本来就是 running_batch 的推进结果，不需要再次 merge。

None:
  上轮没有 batch。
```


因此代码只处理：


```python
self.last_batch.forward_mode.is_extend()
```


这是 prefill 到 decode 生命周期的入口。上一轮返回的 EXTEND batch 经过 forward 和采样后，会在本轮开头尝试合入 running\_batch。

这一段有两个关键细节：


```text
filter_batch:
  去掉已经 generation finished 的请求。
  通过 chunked_req_to_exclude 排除还没有完成全部 prefill 的 chunked_req。

merge_batch:
  合并 reqs / req_pool_indices / seq_lens / sampling_info 等 batch 级状态。
  不是简单地 list extend。
```


这里“可以合入 running\_batch”的含义是：


```python
普通非 chunked prefill:
  prefill forward 完成后，只要请求没有 generation finished，就可以合入 running_batch。

chunked prefill:
  中间 chunk 不能合入 running_batch。
  只有最后一个 chunk 完成后，self.chunked_req 不再指向它，它才会通过 filter_batch 并被 merge。

prefill-only / 已经生成结束的请求:
  会被 filter_batch 过滤掉，不进入 decode。
```



```text
阶段 3：优先尝试新 prefill

new_batch = get_new_batch_prefill()

if new_batch is not None:
    ret = new_batch
```


这就是 prefill 优先级的来源。只要 get\_new\_batch\_prefill 能组出一个 batch，本轮就不会走普通 decode。


```text
阶段 4：没有 prefill 时，才推进 running decode

if new_batch is None:
    if not running_batch.is_empty() and not running_batch.is_prefill_only:
        running_batch = update_running_batch(running_batch)
        ret = running_batch
```


这里的 update\_running\_batch 不只是“把 forward\_mode 改成 DECODE”，它会先过滤 finished 请求、检查内存、必要时 retract。


```text
阶段 5：返回前做 DP attention / ngram / schedule time 处理

maybe_prepare_mlp_sync_batch(ret)
_maybe_prepare_ngram_embedding(ret)
set_schedule_time_batch(ret)
```


所以主干决策是前 4 段，最后一段是返回 batch 前的补充处理。

**get\_new\_batch\_prefill 的关键分段**

**get\_new\_batch\_prefill 的核心是 admission：决定哪些请求可以进入本轮 EXTEND batch。  它不是按条数拿请求，而是按资源预算逐个尝试：**


```text
waiting_queue
  -> policy.calc_priority
  -> Req.init_next_round_input
  -> PrefillAdder.add_one_req
  -> can_run_list
  -> ScheduleBatch.prepare_for_extend
```


关键代码可以分成 6 段。


```python
阶段 1：快速失败

if running_batch.batch_is_full and chunked_req is None:
    return None

if waiting_queue is empty and chunked_req is None:
    return None

if get_num_allocatable_reqs(running_bs) <= 0 and chunked_req is None:
    running_batch.batch_is_full = True
    return None
```


这里检查的是“有没有必要尝试 prefill”。如果 running batch 已满、没有等待请求、请求池没有槽位，就直接返回 None，让外层去尝试 decode。


```text
阶段
：给 waiting_queue 排优先级 2
policy.calc_priority(waiting_queue, running_batch)
```


这一步会改变 waiting\_queue 的调度顺序。后续 for req in waiting\_queue 不是原始到达顺序，而是策略处理后的顺序。


```text
阶段 3：创建 PrefillAdder

adder = PrefillAdder(
    page_size,
    tree_cache,
    token_to_kv_pool_allocator,
    running_batch,
    new_token_ratio,
    max_prefill_tokens,
    chunked_prefill_size,
    ...
)
```


这里 PrefillAdder.\_\_init\_\_ 的参数名是 rem\_input\_tokens，但调用方传进去的是：


```python
self.max_prefill_tokens
```


对应关系是：


```python
Scheduler._get_new_batch_prefill_raw:
  PrefillAdder(..., self.max_prefill_tokens, chunked_prefill_size, ...)

PrefillAdder.__init__:
  def __init__(..., rem_input_tokens, rem_chunk_tokens, num_mixed_decode_tokens=0, ...):
      self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
```


所以：


```text
rem_input_tokens 的初始值
  = Scheduler.max_prefill_tokens
  - mixed chunked prefill 中预留给 decode 的 token 数
```


Scheduler.max\_prefill\_tokens 又来自 model worker 初始化阶段：


```python
self.tp_worker.get_worker_info()
  -> self.max_prefill_tokens
```


从配置角度看，它最终受 ServerArgs.max\_prefill\_tokens、模型 worker 可用 KV/显存预算、后端实现等共同影响。对调度器这段代码而言，它就是“本轮 prefill 最多允许消耗多少 input token”的预算上限。

PrefillAdder 的作用是维护本轮 prefill 的动态预算：


```python
can_run_list:
已经决定本轮可以跑的请求。

rem_input_tokens:
本轮 prefill input token 预算还剩多少。
初始来自 self.max_prefill_tokens，后续每加入一个请求都会扣减。

rem_total_tokens:
KV cache 可用量扣掉 running_batch 未来 decode 预留后的余量。

rem_chunk_tokens:
chunked prefill 本轮还能吃多少 token。

preempt_list:
  priority preemption 下被撤回的 running 请求。
```



```text
阶段 4：优先处理未完成的 chunked_req

if chunked_req is not None:
    chunked_req.init_next_round_input()
    chunked_req = adder.add_chunked_req(chunked_req)
```


这个分支很关键。长 prompt 的中间 chunk 已经占用了部分状态，如果不优先继续调度，会造成资源和状态难以收敛。因此 chunked\_req 的优先级高于普通 waiting\_queue。


```text
阶段 5：逐个尝试 waiting_queue 里的请求

for req in waiting_queue:
    if running_batch.batch_is_full:
        break or preempt_to_schedule(req)

    req.init_next_round_input(tree_cache)
    res = adder.add_one_req(req, ...)

    if res != CONTINUE:
        break
```


这一段是最关键的地方。

req.init\_next\_round\_input(tree\_cache) 会先做：


```text
fill_ids = origin_input_ids + output_ids
prefix_indices = tree_cache.match_prefix(...)
extend_input_len = len(fill_ids) - len(prefix_indices)
```


也就是说，进入 add\_one\_req 前，请求已经知道：


```text
完整序列多长
命中多少 KV cache
本轮真正还要计算多少 token
```


adder.add\_one\_req 再判断：


```text
extend_input_len + max_new_tokens + page_size 是否超过 rem_total_tokens
实际 input tokens 是否超过 rem_input_tokens
chunked_prefill_size 是否需要截断
prefix cache host hit 是否需要 load back
是否需要 lock tree cache node
```


如果允许加入：


```text
adder.can_run_list.append(req)
adder._update_prefill_budget(...)
```


\_update\_prefill\_budget 会扣减：


```text
rem_input_tokens -= ceil_paged_tokens(extend_input_len)
rem_total_token_offset += extend_input_len + max_new_tokens + page_size
rem_chunk_tokens -= extend_input_len       # 如果开启 chunked prefill
```


如果不允许：


```text
返回 NO_TOKEN / OTHER
外层 break，本轮 prefill batch 到此为止
```



```python
阶段 6：把 can_run_list 变成 EXTEND batch

waiting_queue -= can_run_list

new_batch = ScheduleBatch.init_new(can_run_list, ...)
new_batch.prepare_for_extend()
return new_batch
```


ScheduleBatch.init\_new 只创建 batch 对象；prepare\_for\_extend 才真正把 Req 转成 forward 所需张量。

**prepare\_for\_extend 的关键产物**

**prepare\_for\_extend 的核心输入是每个请求的：**


```text
fill_ids
prefix_indices
extend_input_len
```


它构造出的关键字段是：


```text
input_ids:
每个请求未命中 prefix cache 的 token，全部拼平成一个 tensor。

seq_lens:
每个请求 forward 后的完整长度。

prefix_lens:
每个请求命中的 prefix 长度。

extend_lens:
每个请求本轮实际 extend 的 token 数。

out_cache_loc:
每个 extend token 对应的 KV cache 写入地址。

req_pool_indices:
每个请求在 req_to_token_pool 中的位置。

sampling_info:
  采样参数按 batch 聚合后的结构。
```


这一步还有一个容易忽略的副作用：它会更新 Req 自身的内存状态。


```text
req.req_pool_idx = req_pool_indices[i]
req.kv_committed_len = seq_len
req.kv_allocated_len = seq_len
req.extend_batch_idx += 1
req.cached_tokens / already_computed 更新
```


所以 prepare\_for\_extend 不是纯粹构造 tensor，它也把请求正式推进到了“已经为 prefill 分配资源”的状态。

**update\_running\_batch 的关键分段**

**update\_running\_batch 的任务是把 running\_batch 变成下一轮 decode batch。  主干可以拆成 4 段：**


```python
阶段 1：过滤已经完成的请求

initial_bs = batch.batch_size()
batch.filter_batch(v1_spec_info_filtered=True)

if batch.is_empty():
    batch.batch_is_full = False
    return batch
```


这里会把已经 finished 的请求从 running\_batch 中移除，并同步过滤 req\_pool\_indices、seq\_lens、sampling\_info 等 batch 字段。


```text
阶段
：检查下一步 decode 内存 2
if not batch.check_decode_mem():
    retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(...)
```


check\_decode\_mem 会估算下一次 decode 需要多少 KV cache：


```text
普通 decode:
  每个请求通常需要 1 个新 token 的 KV 空间。

page allocator:
  如果跨 page 边界，需要按 page_size 申请。

speculative decoding:
  可能一次预留多个 draft / verify token 的空间。
```


如果空间不够，就进入 retract。


```python
阶段 3：retract 后把请求放回 waiting_queue

for req in retracted_reqs:
    self._add_request_to_queue(req, is_retracted=True)
```


retract 会释放这些请求占用的 KV cache，但不会丢失请求进度。请求带着已有 output\_ids 回到 waiting\_queue，后续会再次通过 init\_next\_round\_input 做 prefix match。


```python
阶段 4：构建 DECODE batch

batch.prepare_for_decode()
return batch
```


prepare\_for\_decode 做的事情和 prepare\_for\_extend 完全不同：


```text
input_ids:
  不再是 prompt token，而是每个请求上一步采样出的最后一个 token。

out_cache_loc:
  为每个请求本轮 decode 的新 token 分配 KV 写入位置。

seq_lens:
  每个请求长度 +1。

forward_mode:
  改成 DECODE。
```


所以 update\_running\_batch 的本质是：


```text
running_batch
  -> filter finished
  -> ensure decode memory
  -> retract if needed
  -> prepare_for_decode
  -> DECODE batch
```


**先看三个重要对象**

**Req 是请求的状态机**

**Req 不是只保存输入 prompt，它会在调度和 forward 过程中持续变化。  一个生成请求可以简化成：**


```text
Req(
  rid="R1",
  origin_input_ids=[11, 12, 13, 14],
  output_ids=[],
  prefix_indices=[],
  fill_ids=[],
  extend_input_len=0,
  req_pool_idx=None,
)
```


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


```text
batch.reqs = [R1, R2, R3]
```


也保存 forward 需要的张量：


```text
batch.input_ids
batch.req_pool_indices
batch.seq_lens
batch.out_cache_loc
batch.sampling_info
```


最关键的是 forward\_mode：


```text
EXTEND:
  prefill，新请求或 chunked prefill 的一段 prompt。

DECODE:
  decode，running_batch 中每个请求继续生成一个 token。

MIXED:
  mixed chunked prefill，把新 prefill 和已有 decode 混在同一次 forward。

IDLE:
  DP attention 等场景下用于同步的空 batch。
```


**PrefillAdder 是 admission controller**

**PrefillAdder 决定 waiting\_queue 里的请求哪些能进入新的 prefill batch。  它同时考虑：**


```text
请求池槽位是否足够
KV cache token 预算是否足够
max_prefill_tokens 是否足够
chunked_prefill_size 是否足够
prefix cache 命中后实际需要算多少 token
running_batch 未来 decode 还要预留多少 token
LoRA / priority / preemption / prefill delayer 等策略
```


因此 get\_new\_batch\_prefill 不是简单的 waiting\_queue[:batch\_size]。

**模拟实际的请求**

**下面用 4 个请求模拟主流程。假设：**


```text
page_size = 1
max_prefill_tokens = 10
chunked_prefill_size = None
pp_max_micro_batch_size = 4
req_to_token_pool 还有足够槽位
KV cache 还有足够空间
不开启 overlap / speculative / LoRA / disaggregation / DP attention
```


初始请求：

| 请求 | prompt token | max\_new\_tokens | prefix cache 命中 | 初始 output\_ids |
| --- | --- | --- | --- | --- |
| R1 | [11,12,13,14] | 3 | 0 | [] |
| R2 | [21,22,23] | 2 | 1 个 token | [] |
| R3 | [31,32,33,34,35,36] | 2 | 3 个 token | [] |
| R4 | [41,42] | 2 | 0 | [] |

初始 Scheduler 状态：


```text
waiting_queue = [R1, R2, R3, R4]
running_batch = []
last_batch = None
chunked_req = None
```


**模拟请求的主干路径图**

**下面这张图只对应后面 R1-R4 的简化场景：没有 DP attention、没有 overlap、没有 speculative decoding、没有 disaggregation，也没有 chunked prefill。它用来帮助先理解主干路径。**


```text
┌──────────────────────────────┐
│ get_next_batch_to_run         │
└───────────────┬──────────────┘
                │
                v
┌──────────────────────────────┐
│ 1. 处理上轮 last_batch        │
│    EXTEND -> running_batch    │
└───────────────┬──────────────┘
                │
                v
┌──────────────────────────────┐
│ 2. 尝试 get_new_batch_prefill │
└───────┬──────────────────────┘
        │
        │ 有新 prefill batch
        v
┌──────────────────────────────┐
│ 返回 EXTEND batch             │
└──────────────────────────────┘

        │ 没有新 prefill batch
        v
┌──────────────────────────────┐
│ 3. update_running_batch       │
│    running_batch -> DECODE    │
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐
│ 返回 DECODE batch             │
└──────────────────────────────┘
```


注意第 1 步非常重要：如果上轮跑的是 prefill，last\_batch 会在下一轮调度开始时尝试并入 running\_batch。普通 prefill 完成且未 generation finished 的请求会进入 decode；未完成的 chunked prefill 会被排除，等最后一个 chunk 完成后再进入。

**第 1 轮：从 waiting\_queue 构建 prefill batch**

**第一次调用：**


```text
get_next_batch_to_run()
```


此时：


```text
last_batch = None
running_batch = []
waiting_queue = [R1, R2, R3, R4]
```


**get\_next\_batch\_to\_run 前半段**

**因为没有 last\_batch，也没有已有 running\_batch，前半段基本不会改变状态。  随后进入：**


```python
new_batch = self.get_new_batch_prefill()
```


**get\_new\_batch\_prefill 创建 PrefillAdder**

**get\_new\_batch\_prefill 内部会调用 \_get\_new\_batch\_prefill\_raw，先做几个快速判断：**


```text
running_batch.batch_is_full ?  否
waiting_queue 是否为空 ?       否
req pool 是否还有槽位 ?        是
```


然后创建 PrefillAdder。

此时 PrefillAdder 的关键预算可以简化理解为：


```text
rem_input_tokens = max_prefill_tokens = 10
can_run_list = []
running_batch = []
```


这里的 10 来自本例前面假设的 max\_prefill\_tokens = 10。真实运行时，PrefillAdder 收到的是 Scheduler 的 self.max\_prefill\_tokens，然后赋给自己的 self.rem\_input\_tokens。

**逐个检查 waiting\_queue**

**R1 先调用：**


```python
req.init_next_round_input(self.tree_cache)
```


R1 没有 prefix cache 命中：


```text
fill_ids = origin_input_ids + output_ids
         = [11,12,13,14] + []
         = [11,12,13,14]

prefix_indices = []
extend_input_len = len(fill_ids) - len(prefix_indices)
                 = 4 - 0
                 = 4
```


然后进入：


```text
adder.add_one_req(R1, ...)
```


R1 占用 4 个 prefill token，允许加入：


```text
can_run_list = [R1]
rem_input_tokens = 10 - 4 = 6
```


#### R2

R2 命中 1 个 prefix token：


```text
fill_ids = [21,22,23]
prefix_indices = [kv_21]
extend_input_len = 3 - 1 = 2
```


加入后：


```text
can_run_list = [R1, R2]
rem_input_tokens = 6 - 2 = 4
```


#### R3

R3 命中 3 个 prefix token：


```text
fill_ids = [31,32,33,34,35,36]
prefix_indices = [kv_31, kv_32, kv_33]
extend_input_len = 6 - 3 = 3
```


加入后：


```text
can_run_list = [R1, R2, R3]
rem_input_tokens = 4 -
 = 1
```


#### R4

R4 需要 2 个 prefill token：


```text
fill_ids = [41,42]
prefix_indices = []
extend_input_len = 2
```


但当前 rem\_input\_tokens = 1，不够放 R4。此时 add\_one\_req 会返回类似 AddReqResult.OTHER 或 NO\_TOKEN 的结果，调度停止继续扩张这个 prefill batch。

最终：


```text
can_run_list = [R1, R2, R3]
waiting_queue = [R4]
```


**构建 ScheduleBatch**

**get\_new\_batch\_prefill 接着执行：**


```text
new_batch = ScheduleBatch.init_new(...)
new_batch.prepare_for_extend()
```


init\_new 只是创建 batch 壳子，真正把请求变成 forward 张量的是 prepare\_for\_extend。

**prepare\_for\_extend 做了什么**

**对 R1、R2、R3 来说：**


```text
R1:
fill_ids = [11,12,13,14]
prefix_len = 0
extend_len = 4
本轮 input_ids = [11,12,13,14]
seq_len = 4

R2:
fill_ids = [21,22,23]
prefix_len = 1
extend_len = 2
本轮 input_ids = [22,23]
seq_len = 3

R3:
fill_ids = [31,32,33,34,35,36]
prefix_len = 3
extend_len = 3
本轮 input_ids = [34,35,36]
  seq_len = 6
```


所以 Batch 张量大致是：


```text
batch.forward_mode = EXTEND
batch.reqs = [R1, R2, R3] 3 4batch.input_ids = [ 511,12,13,14,   # R1 622,23,         # R2 734,35,36       # R3 8] 9 10batch.prefix_lens = [0, 1,
] 11batch.extend_lens = [4, 2, 3] 12batch.extend_num_tokens = 9 13batch.seq_lens = [4, 3, 6] 14batch.seq_lens_sum = 13
```


同时 prepare\_for\_extend 会分配内存：


```text
req_pool_indices:
  R1 -> req_pool_idx 0
  R2 -> req_pool_idx 1
  R3 -> req_pool_idx 2

out_cache_loc:
  给 9 个实际 extend token 分配 KV 写入位置
```


可视化：


```text
EXTEND batch

R1: [11][12][13][14]
     ^^^^^^^^^^^^^^^^
     本轮计算 4 token

R2: [21][22][23]
     ^^  ^^^^^^^
     命中  本轮计算 2 token

R3: [31][32][33][34][35][36]
     ^^^^^^^^^^^  ^^^^^^^^^^^
     命中 3 token   本轮计算 3 token
```


get\_next\_batch\_to\_run 看到 new\_batch is not None，直接返回这个 EXTEND batch。

第 1 轮返回：


```python
return EXTEND([R1, R2, R3])
```


此时还没有立刻把它放进 running\_batch。它会先被 forward，forward 完后成为 last\_batch。下一轮调度开头，如果它是已完成 prefill 的请求，就会被合入 running\_batch。

**第 2 轮：EXTEND last\_batch 合入 running\_batch，再继续找新 prefill**

**假设第 1 轮 EXTEND forward 完成后，每个请求已经采样出第一个 token：**


```text
R1.output_ids = [101]
R2.output_ids = [201]
R3.output_ids = [301]
```


Scheduler 会记录：


```text
last_batch = EXTEND([R1, R2, R3])
running_batch = []
waiting_queue = [R4]
```


第二次调用：


```text
get_next_batch_to_run()
```


**先判断 last\_batch 是否需要合并进 running\_batch**

**代码主干是：**


```python
if self.last_batch and self.last_batch.forward_mode.is_extend():
    self.last_batch.filter_batch(...)
    if not self.last_batch.is_empty():
        if self.running_batch.is_empty():
            self.running_batch = self.last_batch
        else:
            self.running_batch.merge_batch(self.last_batch)
```


这里先判断 last\_batch.forward\_mode.is\_extend()。如果上一轮跑的是 DECODE，就不会走这段 merge 逻辑。

filter\_batch 会把已经 generation finished 的请求过滤掉，也会按 chunked\_req\_to\_exclude 排除未完成的 chunked prefill。当前例子没有 chunked prefill，如果 R1/R2/R3 都还没结束：


```text
running_batch = [R1, R2, R3]
last_batch 仍然是上轮 EXTEND batch 的对象引用
waiting_queue = [R4]
```


这里有个容易混淆的点：


```text
EXTEND batch forward 后，不会在同一轮立即变成 DECODE。
它会在下一次 get_next_batch_to_run 的开头，根据是否 finished / 是否未完成 chunked prefill，决定能否合入 running_batch。
```


**再尝试新的 prefill**

**此时 waiting\_queue 还有 R4，所以 Scheduler 继续尝试：**


```python
new_batch = self.get_new_batch_prefill()
```


R4 的状态：


```text
fill_ids = [41,42]
prefix_indices = []
extend_input_len = 2
```


如果内存和请求槽位允许，R4 会组成新的 prefill batch：


```python
return EXTEND([R4])
```


第 2 轮返回：


```python
return EXTEND([R4])
```


注意这一轮虽然 running\_batch 里已经有 R1/R2/R3，但因为有新的 prefill 可以跑，所以优先跑 R4 的 prefill，而不是 decode R1/R2/R3。

状态图：


```text
第
 轮开始 2
last_batch:    EXTEND([R1,R2,R3])
running_batch: []
waiting_queue: [R4]

合并 last_batch 后

running_batch: [R1,R2,R3]
waiting_queue: [R4]

构建新 prefill 后

return: EXTEND([R4])
```


**第 3 轮：没有新 prefill，开始 update\_running\_batch 做 decode**

**假设第 2 轮 R4 prefill 完成后：**


```text
R4.output_ids = [401]
last_batch = EXTEND([R4])
running_batch = [R1, R2, R3]
waiting_queue = []
```


第三次调用：


```text
get_next_batch_to_run()
```


**先合入 R4**

**last\_batch 是 EXTEND，所以合入 running\_batch：**


```text
running_batch = [R1, R2, R3, R4]
waiting_queue = []
```


**get\_new\_batch\_prefill 返回 None**

**因为 waiting\_queue 空了：**


```text
new_batch = None
```


于是进入 decode 路径：


```python
self.running_batch = self.update_running_batch(self.running_batch)
ret = self.running_batch if not self.running_batch.is_empty() else None
```


**update\_running\_batch 的主流程**

**update\_running\_batch 可以拆成 4 步：**


```text
1. filter_batch
   过滤已经 finished 的请求。

2. check_decode_mem
   判断下一步 decode 的 KV cache 是否够。

3. 必要时 retract_decode
   如果 KV cache 不够，撤回部分请求，释放 KV，再把被撤回请求放回 waiting_queue。

4. prepare_for_decode
   把 running_batch 改造成下一次 decode forward 需要的 batch。
```


如果 R1/R2/R3/R4 都还没结束，且 decode 内存足够，则进入：


```text
batch.prepare_for_decode()
```


**prepare\_for\_decode 做了什么**

**此时每个请求的 output\_ids 至少有一个 token：**


```text
R1.output_ids = [101]
R2.output_ids = [201]
R3.output_ids = [301]
R4.output_ids = [401]
```


decode forward 的输入就是每个请求上一步采样出的最后一个 token：


```text
batch.forward_mode = DECODE
batch.input_ids = [101, 201,
01,
01]
```


同时为每个请求分配一个新的 KV 写入位置：


```text
batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
```


并更新长度：


```text
prefill 后 seq_lens:
R1 = 4
R2 = 3
R3 = 6
R4 = 2

prepare_for_decode 后 seq_lens:
R1 = 5
R2 = 4
R3 = 7
  R4 = 3
```


可视化：


```text
DECODE batch 2 3R1: [11][12][13][14] -> [101] -> 本轮输入 101，写第 5 个 KV 4R2: [21][22][23]     -> [201] -> 本轮输入
01，写第 4 个 KV 5R3: [31][32][33][34][35][36] -> [301] -> 本轮输入
01，写第 7 个 KV 6R4: [41][42]         -> [401] -> 本轮输入
01，写第 3 个 KV
```


第 3 轮返回：


```python
return DECODE([R1, R2, R3, R4])
```


**一张表看完三轮状态变化**

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | 轮次 | 进入时 waiting\_queue | 进入时 running\_batch | 先处理 last\_batch | 返回 batch | | --- | --- | --- | --- | --- | | 第 1 轮 | [R1,R2,R3,R4] | [] | 无 | EXTEND([R1,R2,R3]) | | 第 2 轮 | [R4] | [] | [R1,R2,R3] -> running\_batch | EXTEND([R4]) | | 第 3 轮 | [] | [R1,R2,R3] | [R4] -> running\_batch | DECODE([R1,R2,R3,R4]) |   这就是最常见的主干逻辑： |

| ``` 1新请求 prefill 优先。 2prefill 结果下一轮合入 running_batch。 3没有 prefill 可跑时，running_batch 才进入 decode。 ``` |

**Batch 字段在 EXTEND 和 DECODE 中的区别**

**EXTEND batch**

**EXTEND 关注的是“哪些 prompt token 还没有 KV，需要补算”。**


```text
input_ids:
所有请求未命中 prefix cache 的 token 拼平。

prefix_lens:
每个请求命中的 prefix 长度。

extend_lens:
每个请求本轮实际 prefill 的 token 数。

extend_num_tokens:
sum(extend_lens)。

seq_lens:
每个请求本轮 forward 后的完整序列长度。

out_cache_loc:
  每个 extend token 对应的 KV cache 写入位置。
```


**DECODE batch**

**DECODE 关注的是“每个请求继续生成一个 token”。**


```text
input_ids:
  每个请求上一步采样出来的最后一个 token。

prefix_lens / extend_lens:
  decode 不需要这些 extend 专用字段。

seq_lens:
  每个请求加 1 后的完整长度。

out_cache_loc:
  每个请求新 token 的 KV cache 写入位置。
```


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


```text
run_batch
  -> model forward
  -> sampling
  -> process_batch_result
  -> 更新 req.output_ids / finished 状态
```


只有 forward 结束后，Scheduler 才知道：


```text
这个请求是否 finished
采样出了哪个 token
output_ids 是否更新
logprob / hidden states 等结果如何处理
```


所以 prefill batch 不能在构建出来时立刻进入 decode。它必须先作为 last\_batch 经过一次 forward 结果处理，然后在下一轮调度开头经过 filter\_batch。

合入规则可以简化成：


```text
last_batch 不是 EXTEND:
不处理。
例如上一轮是 DECODE，它已经属于 running_batch 的生命周期。

last_batch 是 EXTEND，且请求已经 generation finished:
filter_batch 过滤掉，不进入 running_batch。

last_batch 是 EXTEND，但请求仍是未完成的 chunked_req:
通过 chunked_req_to_exclude 排除，不进入 running_batch。

last_batch 是 EXTEND，且请求 prefill 已完整完成、还需要继续生成:
  合入 running_batch，后续参与 decode。
```


**update\_running\_batch 中的 retract 是什么**

**decode 阶段每轮都要给每个 running request 新增 KV cache。  如果当前 running\_batch 太大，下一步 decode 的 KV cache 不够，update\_running\_batch 会触发：**


```text
batch.retract_decode(...)
```


可以把 retract 理解成：


```text
为了避免 decode OOM，临时撤回一部分 running 请求。
撤回请求会释放 KV cache，并重新放回 waiting_queue。
后续它们会以带 output_ids 的状态重新 prefill / prefix match。
```


简化例子：


```text
running_batch = [R1,R2,R3,R4]
下一步 decode 需要 4 个 KV slot
但当前只能再分配 2 个 KV slot
```


可能发生：


```text
保留: [R1,R2]
撤回: [R3,R4]

running_batch = [R1,R2]
waiting_queue = [R3,R4]
```


撤回的请求不是丢掉，而是：


```text
R3.origin_input_ids 不变
R3.output_ids 保留已经生成的 token
R3.is_retracted = True
R3 后续重新进入 waiting_queue
```


下一次 init\_next\_round\_input 时：


```text
fill_ids = origin_input_ids + output_ids
```


所以它会尝试从 prefix cache 找回已经算过的 KV，只补算缺失部分。

**chunked prefill 如何改变 Batch 构建**

**如果一个请求 prompt 很长，超过 chunked\_prefill\_size，它不会一次性 prefill 完，而是拆成多轮。  假设：**


```text
chunked_prefill_size = 4
R5.origin_input_ids = [51,52,53,54,55,56,57,58,59]
prefix 命中 0
```


第一段：


```python
fill_ids = [51,52,53,54]
extend_input_len = 4
new_chunked_req = R5
chunked_req = R5
return EXTEND([R5])
```


第二段调度时，Scheduler 会优先处理已有的 chunked\_req：


```text
R5.init_next_round_input()
adder.add_chunked_req(R5)
```


如果还没完成：


```text
fill_ids = [51,52,53,54,55,56,57,58]
prefix_indices = 前 4 个 token 的 KV
extend_input_len = 4
chunked_req 仍然是 R5
```


最后一段：


```text
fill_ids = [51,52,53,54,55,56,57,58,59]
prefix_indices = 前 8 个 token 的 KV
extend_input_len = 1
chunked_req = None
```


可视化：


```text
R5 长 prompt:

[51][52][53][54][55][56][57][58][59]
 └──── chunk 1 ────┘
                 └──── chunk 2 ────┘
                                 └ chunk 3
```


get\_next\_batch\_to\_run 开头对 chunked\_req 有特殊处理，是为了避免未完成的 chunk 被过早合入 decode：


```text
chunked prefill 没结束:
  不应该进入 running_batch decode。

chunked prefill 全部结束:
  才能作为完整请求进入 running_batch。
```


**mixed chunked prefill 是什么**

**普通情况下：**


```text
有新 prefill -> 跑 prefill
没新 prefill -> 跑 decode
```


开启 mixed chunked prefill 后，在满足条件时可以把两者混在一起：


```python
new_batch = 新请求的 EXTEND
running_batch = 已有请求的 DECODE

new_batch.mix_with_running(running_batch)

return MIXED([prefill reqs + decode reqs])
```


简化图：


```text
MIXED batch input_ids

prefill R5: [55][56][57][58]
decode  R1: [102]
decode  R2: [202]

拼成:
input_ids = [55,56,57,58,102,202]
```


代码上在 \_get\_new\_batch\_prefill\_raw 的末尾：


```python
if self.is_mixed_chunk and not self.running_batch.is_empty():
    self.running_batch.filter_batch(...)
    self.running_batch.prepare_for_decode()
    new_batch.mix_with_running(self.running_batch)
    self.running_batch = ScheduleBatch(reqs=[], ...)
```


这样一次 forward 里既补长 prompt 的一段，也推进已有请求 decode。

**prefix cache 对 Batch 的影响**

**prefix cache 最直接影响的是：**


```text
prefix_indices
extend_input_len
input_ids
out_cache_loc 数量
```


同样 6 个 token 的 prompt：


```text
R_no_cache:
origin_input_ids = [1,2,3,4,5,6]
prefix_indices = []
extend_input_len = 6
input_ids = [1,2,3,4,5,6]

R_cache:
origin_input_ids = [1,2,3,4,5,6]
prefix_indices = [kv1,kv2,kv3,kv4]
extend_input_len = 2
  input_ids = [5,6]
```


所以 Scheduler admission 时不能只看 prompt 长度，而要先：


```python
req.init_next_round_input(self.tree_cache)
```


让请求完成 prefix match，然后 PrefillAdder 再根据真实 extend\_input\_len 判断能不能加入 batch。

**三个函数的快速回顾**

**前面已经展开过关键代码分段，这里再用压缩伪代码把三者串起来，方便回看。**

**get\_next\_batch\_to\_run**

**去掉 DP attention、HiSparse、DLLM、ngram、overlap 等分支后，可以简化成：**


```python
def get_next_batch_to_run():
    # 1. 上轮 prefill 结果进入 running_batch
    if last_batch and last_batch.forward_mode.is_extend():
        last_batch.filter_batch(chunked_req_to_exclude=...)
        if not last_batch.is_empty():
            if running_batch.is_empty():
                running_batch = last_batch
            else:
                running_batch.merge_batch(last_batch)

    # 2. 优先尝试新 prefill
    new_batch = get_new_batch_prefill()
    if new_batch is not None:
        return new_batch

    # 3. 没有 prefill，推进 decode
    if not running_batch.is_empty() and not running_batch.is_prefill_only:
        running_batch = update_running_batch(running_batch)
        if not running_batch.is_empty():
            return running_batch

    return None
```


**get\_new\_batch\_prefill**

**简化后：**


```python
def get_new_batch_prefill():
    if running_batch.batch_is_full and chunked_req is None:
        return None

    if waiting_queue is empty and chunked_req is None:
        return None

    if req pool 没有槽位 and chunked_req is None:
        running_batch.batch_is_full = True
        return None

    policy.calc_priority(waiting_queue, running_batch)

    adder = PrefillAdder(...)

    if chunked_req is not None:
        chunked_req.init_next_round_input()
        chunked_req = adder.add_chunked_req(chunked_req)

    for req in waiting_queue:
        if batch 已满:
            break or preempt_to_schedule(req)

        req.init_next_round_input(tree_cache)
        result = adder.add_one_req(req, ...)

        if result != CONTINUE:
            break

    if adder.can_run_list is empty:
        return None

    waiting_queue -= adder.can_run_list

    new_batch = ScheduleBatch.init_new(adder.can_run_list, ...)
    new_batch.prepare_for_extend()

    if mixed chunked prefill:
        new_batch.mix_with_running(running_batch)

    return new_batch
```


**update\_running\_batch**

**简化后：**


```python
def update_running_batch(batch):
    initial_bs = batch.batch_size()

    batch.filter_batch()
    if batch.is_empty():
        batch.batch_is_full = False
        return batch

    if not batch.check_decode_mem():
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(...)
        self.new_token_ratio = new_token_ratio

        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)
    else:
        self.new_token_ratio = max(
            self.new_token_ratio - self.new_token_ratio_decay,
            self.min_new_token_ratio,
        )

    if batch.batch_size() < initial_bs:
        batch.batch_is_full = False

    if batch.is_empty():
        return batch

    batch.prepare_for_decode()
    return batch
```


**常见容易绕的点**

**1. running\_batch 不等于正在 forward 的 batch**

**running\_batch 指的是“已经进入 decode 生命周期的请求集合”。  当前返回给 forward 的 batch 可能是：**


```text
EXTEND new_batch
DECODE running_batch
MIXED new_batch + running_batch
```


**2. EXTEND last\_batch 是 prefill 到 decode 的桥**

**last\_batch 不一定是 prefill batch。event loop 每轮都会执行 self.last\_batch = batch，所以上一轮返回 DECODE 时，last\_batch 也是 DECODE。  只有当 last\_batch.forward\_mode.is\_extend() 为 True 时，它才承担 prefill 到 decode 的桥接作用。下一轮调度开头会先 filter\_batch，排除已经 generation finished 的请求和未完成的 chunked prefill，再把剩下的请求合入 running\_batch。**

**3. input\_ids 在 EXTEND 和 DECODE 中含义完全不同**


```text
EXTEND input_ids:
  prompt 中没有命中 prefix cache 的 token。

DECODE input_ids:
  每个请求上一步采样出的最后一个 token。
```


**4. prefix cache 命中会改变 admission 结果**

**两个 prompt 一样长的请求，命中多的请求 extend\_input\_len 更小，更容易被加入 prefill batch。**

**5. batch\_is\_full 是优化标记，不是绝对状态**

**running\_batch.batch\_is\_full 用来减少无意义的 prefill 尝试。某些场景会重置它，比如：**


```text
有请求完成并被 filter 掉
开启 priority preemption
hybrid SWA
retract 后 running batch 变小
```


**6. 被 retract 的请求会回 waiting\_queue**

**retract 不是失败，也不是丢请求，而是为了释放 decode KV cache。请求保留 output\_ids，后面再通过 prefix cache 恢复。**

**最后用一句话串起来**

**ScheduleBatch 构建的核心可以这样记：**


```text
waiting_queue 里的 Req
  -> init_next_round_input 做 prefix match
  -> PrefillAdder 判断预算
  -> ScheduleBatch.prepare_for_extend 变成 EXTEND forward 数据
  -> forward 后作为 last_batch
  -> 下一轮合入 running_batch
  -> update_running_batch + prepare_for_decode 变成 DECODE forward 数据
```


理解了这个状态流转，再去看 DP attention、chunked prefill、overlap、speculative decoding、disaggregation 等分支，会容易很多。
