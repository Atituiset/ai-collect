# SGLang推理优化

本文围绕 Scheduler 调度器到模型 forward 之间最关键的三个 Batch 数据结构展开：

**``` 1ScheduleBatch 2  -> ModelWorkerBatch 3  -> ForwardBatch ```**

上一篇调度器ScheduleBatch重点解释 Scheduler 如何决定“下一轮跑 prefill 还是 decode”。本文继续往后走，解释这个决定出来的 ScheduleBatch 如何一步步变成模型真正执行时消费的 ForwardBatch然后模型开始使用input\_ids开始做module forward一系列工作。

**一句话理解**

**这三个对象不是重复封装，而是服务于三个不同层级：**

**``` 1ScheduleBatch: 2Scheduler 视角。 3关心请求生命周期、调度状态、req 列表、prefix cache、KV cache 分配、采样参数。 4 5ModelWorkerBatch: 6Scheduler -> worker 的传输视角。 7把 ScheduleBatch 中 forward 需要的字段打包出来，减少 worker 侧直接依赖调度器内部状态。 8 9ForwardBatch: 10ModelRunner / attention backend / model.forward 的执行视角。 11  在 ModelWorkerBatch 基础上补齐 positions、attention metadata、KV pool、DP padding、LoRA、MROPE 等运行时信息。 ```**

核心数据流如下：

**``` 1┌─────────────────────────────────────────────┐ 2│ Scheduler.get_next_batch_to_run              │ 3│ 产物：ScheduleBatch                          │ 4│                                             │ 5│ 包含：Req 列表、forward_mode、input_ids、     │ 6│ seq_lens、out_cache_loc、sampling_info       │ 7└──────────────────────┬──────────────────────┘ 8                       │ 9                       │ ScheduleBatch.get_model_worker_batch() 10                       v 11┌─────────────────────────────────────────────┐ 12│ Scheduler.run_batch / TPWorker               │ 13│ 产物：ModelWorkerBatch                       │ 14│                                             │ 15│ 包含：worker forward 必需字段，外加 logprob、 │ 16│ DP attention、multimodal、LoRA、spec 信息     │ 17└──────────────────────┬──────────────────────┘ 18                       │ 19                       │ ForwardBatch.init_new(...) 20                       v 21┌─────────────────────────────────────────────┐ 22│ TPWorker.forward_batch_generation            │ 23│ ModelRunner.forward                          │ 24│ 产物：ForwardBatch                           │ 25│                                             │ 26│ 包含：positions、extend_start_loc、KV pool、  │ 27│ attn_backend、MROPE、LoRA runtime metadata    │ 28└──────────────────────┬──────────────────────┘ 29                       │ 30                       v 31┌─────────────────────────────────────────────┐ 32│ model.forward(input_ids, positions, batch)   │ 33│                                             │ 34│ input_ids -> embed_tokens -> hidden_states   │ 35│ hidden_states + forward_batch -> layers      │ 36│ logits processor / sampler                   │ 37└─────────────────────────────────────────────┘ ```**

可以用一句话记：

**``` 1ScheduleBatch 管调度， 2ModelWorkerBatch 管跨层传递， 3ForwardBatch 管模型执行。 ```**

**关键代码位置**

**主要涉及以下文件：   - python/sglang/srt/managers/schedule\_batch.py  - ScheduleBatch - ScheduleBatch.prepare\_for\_extend - ScheduleBatch.prepare\_for\_decode - ScheduleBatch.get\_model\_worker\_batch - ModelWorkerBatch  - python/sglang/srt/managers/scheduler.py  - Scheduler.run\_batch - Scheduler.record\_batch\_in\_overlap  - python/sglang/srt/managers/tp\_worker.py  - TpModelWorker.forward\_batch\_generation  - python/sglang/srt/model\_executor/forward\_batch\_info.py  - ForwardBatch - ForwardBatch.init\_new - ForwardBatch.prepare\_mlp\_sync\_batch  - python/sglang/srt/model\_executor/model\_runner.py  - ModelRunner.forward - ModelRunner.forward\_decode - ModelRunner.forward\_extend - ModelRunner.sample  - python/sglang/srt/layers/attention/\*\_backend.py  - init\_forward\_metadata - attention forward  - python/sglang/srt/layers/logits\_processor.py  - LogitsProcessor  - python/sglang/srt/layers/sampler.py  - Sampler**

**三个对象先拆开看**

**ScheduleBatch：调度器手里的 Batch**

**ScheduleBatch 是 Scheduler 的核心状态对象。它知道“这一组请求为什么能跑、应该怎么跑、跑完之后怎么回到调度生命周期”。  它的典型字段可以分成 5 类：**

**``` 11. 请求和资源 2   reqs 3   req_to_token_pool 4   token_to_kv_pool_allocator 5   tree_cache 6 72. 调度模式 8   forward_mode 9   batch_is_full 10   chunked_req 11   is_prefill_only 12 133. forward 输入张量 14   input_ids 15   req_pool_indices 16   seq_lens 17   seq_lens_cpu 18   orig_seq_lens 19   out_cache_loc 20   output_ids 21 224. extend / decode 专用信息 23   prefix_lens 24   extend_lens 25   extend_num_tokens 26   extend_logprob_start_lens 27   decoding_reqs 28 295. 后处理和特殊能力 30   sampling_info 31   return_logprob 32   multimodal_inputs 33   spec_info 34   hicache_consumer_index ```**

这些字段可以用下面这张表来直观理解：

| 字段 | 通俗含义 | 在 batch 构建中的作用 |
| --- | --- | --- |
| reqs | 这一批真正包含哪些请求对象 | Scheduler 后续要把 next token、finish 状态、logprob、统计信息写回到这些 Req 上 |
| req\_to\_token\_pool | 请求行号到 KV slot 的映射表 | 记录每个请求每个 token 的 KV cache 位置，attention 读历史 KV 时会用到 |
| token\_to\_kv\_pool\_allocator | KV cache 空位分配器 | 给本轮新 token 分配 out\_cache\_loc，也负责释放或回收 KV cache |
| tree\_cache | prefix cache / radix cache | 判断 prompt 前缀哪些 token 已经算过，决定 prefix\_lens 和本轮还需要 prefill 的 token |
| forward\_mode | 这一轮 forward 类型 | 决定 batch 是 EXTEND、DECODE、MIXED 还是其他特殊模式 |
| batch\_is\_full | running batch 是否已经满 | 如果已满，Scheduler 可以跳过部分新 prefill admission 检查 |
| chunked\_req | 当前跨轮 chunked prefill 的请求 | 长 prompt 被切块时，用它记录那个还没 prefill 完的请求 |
| is\_prefill\_only | 这一批是否只做 prefill，不采样生成 | embedding、打分、max\_new\_tokens=0 等场景会走这类逻辑 |
| input\_ids | 本轮真的要送进模型的 token | prefill 时是未命中 cache 的 prompt token；decode 时通常是每个请求最新生成的一个 token |
| input\_embeds | 直接送进模型的 embedding | 多模态或 embedding override 场景可能绕过普通 token embedding |
| req\_pool\_indices | 每个请求在 req\_to\_token\_pool 里的行号 | attention 根据它找到每个请求对应的 KV slot 序列 |
| seq\_lens / seq\_lens\_cpu | 每个请求当前完整序列长度 | 包括 prompt 和已生成 token；decode position 通常由它推出 |
| orig\_seq\_lens | 原始序列长度 | 某些长上下文或特殊模型需要知道未 chunk 前的真实长度 |
| out\_cache\_loc | 本轮新 KV 要写入的 KV slot | 长度等于本轮实际 forward token 数；它是写地址，不是完整历史 KV 读取表 |
| output\_ids | 本轮 decode 输入 token 的张量形式 | decode batch 中通常来自每个请求上轮采样出的 token |
| prefix\_lens | 每个请求已经命中 cache 的前缀长度 | prefill 时决定哪些 token 不需要重复计算 |
| extend\_lens | 每个请求本轮要新算多少个 prompt token | 和 prefix\_lens 一起描述 EXTEND batch 的形状 |
| extend\_num\_tokens | 本轮所有 extend token 总数 | 等于拼平后的 extend input\_ids 长度 |
| extend\_logprob\_start\_lens | prompt logprob 从哪里开始算 | 返回 input logprob 时用于跳过不需要回传的前缀 |
| decoding\_reqs | MIXED batch 中正在 decode 的请求 | mixed chunked prefill 场景下，一个 batch 里可能同时有 decode 和 extend |
| sampling\_info | 采样参数的 batch 化结果 | 保存 temperature、top\_p、top\_k、grammar 等采样需要的信息 |
| return\_logprob | 是否需要返回 logprob | 决定 logits processor 是否额外保留 prompt/output logprob |
| top\_logprobs\_nums | 每个请求要返回多少个 top logprobs | OpenAI API 风格的 top logprobs 返回需要它 |
| token\_ids\_logprobs | 指定 token 的 logprob 查询列表 | 用户只关心某些 token 的概率时使用 |
| multimodal\_inputs | 图像、音频、视频等多模态输入 | 后续转换成 input\_embeds、MROPE 位置或 encoder 信息 |
| spec\_info | speculative decoding 的运行信息 | draft/verify 路径会用它改变输入、positions 和 logits 处理 |
| global\_num\_tokens | DP attention 下全局 token 分布 | 多 DP rank 需要对齐 token 数、padding 和 MLP 同步 |
| hicache\_consumer\_index | HiCache 数据加载同步索引 | 用于协调 CPU/GPU/远端 cache 的异步加载和消费 |

ScheduleBatch 的关键点是：它仍然保留 Req 对象列表。

**``` 1batch.reqs = [R1, R2, R3] ```**

这意味着它不仅能跑 forward，还能在 forward 后把结果写回每个请求：

**``` 1req.output_ids 2req.finished_reason 3req.input_token_logprobs 4req.hidden_states 5req.time_stats ```**

所以 ScheduleBatch 是“调度生命周期对象”，不是单纯的模型输入。

**ModelWorkerBatch：Scheduler 和 worker 之间的中间层**

**ModelWorkerBatch 在 schedule\_batch.py 中定义，它由：**

**``` 1ScheduleBatch.get_model_worker_batch() ```**

创建。

它的作用是把 ScheduleBatch 中 worker forward 需要的字段打包出来：

**``` 1forward_mode 2input_ids 3req_pool_indices 4seq_lens 5out_cache_loc 6sampling_info 7multimodal_inputs 8lora_ids 9spec_info 10extend_* 字段 11DP attention 字段 ```**

可以把 ModelWorkerBatch 字段理解成“worker 执行 forward 所需的一张快照”：

| 字段 | 通俗含义 | 和 ScheduleBatch 的关系 |
| --- | --- | --- |
| forward\_mode | worker 这轮要按什么模式跑 | 原样来自 ScheduleBatch.forward\_mode |
| input\_ids | 要送进模型的 token | 从 ScheduleBatch.input\_ids 拷贝过来 |
| req\_pool\_indices | 请求在 KV 映射表里的行号 | attention 需要它定位每个请求的上下文 |
| seq\_lens | 每个请求当前完整长度 | decode position、attention mask、logits 抽取都会用 |
| seq\_lens\_cpu | CPU 侧的序列长度副本 | 某些 backend 或 metadata 构建需要 CPU list/tensor |
| seq\_lens\_sum | batch 内所有请求长度之和 | 用于统计、DP attention 或 backend metadata |
| out\_cache\_loc | 本轮新 KV 写到哪里 | 从 Scheduler 已分配好的 KV slot 原样带到 worker |
| extend\_num\_tokens | EXTEND token 总数 | worker 用它知道 prefill 的拼平 token 数 |
| extend\_seq\_lens | 每个请求本轮 extend token 数 | 名字里有 seq\_lens，但含义是“本轮新算多少 token” |
| extend\_prefix\_lens | 每个请求已有 prefix cache 长度 | ForwardBatch 计算 positions 和 extend metadata 依赖它 |
| extend\_logprob\_start\_lens | input logprob 起算位置 | 只在需要返回 prompt logprob 时有意义 |
| extend\_input\_logprob\_token\_ids | 要计算 input logprob 的 token ids | logprob 后处理使用，普通生成通常不用重点关注 |
| sampling\_info | batch 化采样参数 | sampler 需要它决定每个请求怎么采样 |
| return\_logprob | 是否返回 logprob | logits processor 决定是否保留更多 logits 信息 |
| global\_num\_tokens / global\_num\_tokens\_for\_logprob | DP attention 的全局 token 视图 | worker 侧转成 GPU tensor 后用于 DP padding / sync |
| is\_extend\_in\_batch / all\_extend\_in\_batch | 当前全局 batch 里是否有 extend | DP attention、mixed batch、padding 决策会用 |
| can\_run\_dp\_cuda\_graph | DP 场景是否可走 cuda graph | 决定能不能使用更固定形状的执行路径 |
| multimodal\_inputs | 多模态输入信息 | worker/model runner 后续生成或合并多模态 embedding |
| encoder\_cached / encoder\_lens / encoder\_out\_cache\_loc | encoder-decoder 模型的 encoder 侧缓存信息 | 普通 decoder-only LLM 可以先忽略 |
| lora\_ids | 每个请求使用的 LoRA adapter | model runner 准备 LoRA batch 时使用 |
| input\_embeds / replace\_embeds / replace\_positions | 直接输入或替换部分 token embedding | 多模态、embedding override、特殊模型路径使用 |
| spec\_info / spec\_algorithm | speculative decoding 信息 | draft/target verify worker 需要它 |
| capture\_hidden\_mode | 是否要返回 hidden states | embedding、debug、MTP/PD 等路径可能需要 |
| reqs | 保留下来的请求对象列表 | 主要服务 grammar、request id、调试和少量后处理，不再负责 admission |

和 ScheduleBatch 相比，ModelWorkerBatch 有几个特点：

**``` 11. 它更接近 worker 输入，不负责调度 admission。 22. 它保留 reqs，主要是为了 grammar、rids、部分后处理和调试。 33. 它还没有 positions，也没有 attn_backend 的 runtime metadata。 44. 它不直接持有 token_to_kv_pool，KV pool 由 model_runner 注入到 ForwardBatch。 ```**

可以把它理解成：

**``` 1ScheduleBatch 的 forward 投影。 ```**

**ForwardBatch：模型真正消费的 Batch**

**ForwardBatch 在 model\_executor/forward\_batch\_info.py 中定义，由：**

**``` 1ForwardBatch.init_new(model_worker_batch, model_runner) ```**

创建。

它是 model runner、attention backend、model.forward、logits processor、sampler 真正使用的对象。

和 ModelWorkerBatch 相比，ForwardBatch 会补齐运行时信息：

**``` 1positions: 2decode 时通常是 seq_lens - 1。 3extend 时由 prefix_lens / seq_lens 计算每个 token 的位置。 4 5extend_start_loc: 6extend batch 中每个请求在拼平 input_ids 里的起始位置。 7 8req_to_token_pool / token_to_kv_pool: 9model_runner 注入的内存池对象，attention 写 KV 时需要。 10 11attn_backend: 12当前模型 runner 绑定的 attention backend。 13 14global_num_tokens_gpu: 15DP attention / MLP sync 场景下的全局 token 数。 16 17out_cache_loc_swa: 18hybrid SWA cache 场景下的 SWA KV 写入位置。 19 20mrope_positions: 21Qwen2-VL / MROPE 模型的位置编码。 22 23LoRA runtime metadata: 24  prepare_lora_batch 后供 LoRA 层使用。 ```**

ForwardBatch 字段可以理解成“模型执行现场的工具箱”：

| 字段 | 通俗含义 | 谁最关心它 |
| --- | --- | --- |
| forward\_mode | 当前是 prefill、decode、mixed 还是特殊 forward | model runner、attention backend、logits processor |
| batch\_size | 这一批有多少个请求 | model runner 和 backend metadata |
| input\_ids | 本轮模型要 embedding 的 token | model.forward 的第一入口 |
| positions | 每个 token 在原序列里的位置 | rotary embedding、MROPE、attention |
| req\_pool\_indices | 请求在 req\_to\_token\_pool 中的行号 | attention 读取完整上下文 KV |
| seq\_lens / seq\_lens\_cpu | 每个请求当前完整长度 | decode position、attention 范围、logits 抽取 |
| seq\_lens\_sum | 所有请求长度总和 | 部分 backend 统计和 metadata |
| out\_cache\_loc | 本轮新 KV 写入 token\_to\_kv\_pool 的位置 | attention layer 写 K/V |
| out\_cache\_loc\_swa | SWA cache 对应的新 KV 写入位置 | hybrid SWA attention |
| req\_to\_token\_pool | 请求 token 到 KV slot 的映射表 | attention 读历史和当前上下文 KV |
| token\_to\_kv\_pool | 真正存 K/V tensor 的 KV cache 池 | attention 读写 K/V |
| attn\_backend | 当前使用的 attention 实现 | 初始化 metadata、执行 attention kernel |
| extend\_prefix\_lens | 每个请求已经缓存的 prompt 前缀长度 | prefill attention 和 position 计算 |
| extend\_seq\_lens | 每个请求本轮新算的 prompt token 数 | prefill attention 需要知道每段 token 边界 |
| extend\_start\_loc | 每个请求在拼平 input\_ids 中的起点 | 从 flat token 张量切回每个请求时使用 |
| extend\_num\_tokens | 本轮 extend token 总数 | backend 初始化和 logprob 处理 |
| sampling\_info | 采样参数和 grammar 信息 | sampler |
| return\_logprob / top\_logprobs\_nums / token\_ids\_logprobs | logprob 返回配置 | logits processor |
| input\_embeds | 已经准备好的输入 embedding | 多模态或 embedding override 时替代 embed\_tokens(input\_ids) |
| mm\_inputs | 多模态输入运行时信息 | 多模态模型、MROPE 计算 |
| mrope\_positions | 多维 RoPE 位置 | Qwen2-VL 等 MROPE 模型 |
| lora\_ids | 本 batch 涉及的 LoRA adapter | LoRA manager 和 LoRA layer |
| global\_num\_tokens\_gpu | DP attention 下 GPU 侧全局 token 数 | DP padding、MLP sync、logprob 对齐 |
| num\_token\_non\_padded | 去掉 padding 后真实 token 数 | cuda graph / DP padding 场景 |
| spec\_info | speculative decoding 运行时信息 | draft/verify 模型路径 |
| capture\_hidden\_mode | 是否捕获 hidden states | 返回 hidden states、MTP、debug 路径 |
| rids | 请求 id 列表 | dump、trace、跨 step 追踪 |

因此 ForwardBatch 是真正的“执行期 Batch”。

**一张表对比三者**

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | 对象 | 所在层 | 谁创建 | 谁消费 | 主要职责 | | --- | --- | --- | --- | --- | | ScheduleBatch | Scheduler | get\_new\_batch\_prefill / update\_running\_batch | Scheduler、run\_batch、结果处理 | 调度状态、请求生命周期、KV 分配、采样状态 | | ModelWorkerBatch | Scheduler -> Worker | ScheduleBatch.get\_model\_worker\_batch | TPWorker.forward\_batch\_generation | 打包 worker forward 所需字段 | | ForwardBatch | Model Executor | ForwardBatch.init\_new | ModelRunner 、attention backend、model.forward、sampler | 补齐运行时 metadata，驱动模型执行 |   字段关系可以这样看： |

| ``` 1Req / Scheduler state 2  只在 ScheduleBatch 中完整存在 3 4forward tensor 5  ScheduleBatch 构造 6  ModelWorkerBatch 携带 7  ForwardBatch 执行 8 9runtime metadata 10  主要在 ForwardBatch 中补齐 ``` |

**普通生成请求的模拟数据**

**下面用 3 个请求模拟一次 prefill 和一次 decode。  假设：**

**``` 1page_size = 1 2不开启 DP attention 3不开启 speculative decoding 4不开启 LoRA 5不开启 multimodal 6不开启 overlap 7prefix cache 中 R2 命中 1 个 token，R3 命中 3 个 token 8token_to_kv_pool 中已经存在的 prefix cache 位置： 9R2 token 21 -> kv slot 80 10R3 token 31 -> kv slot 90 11R3 token 32 -> kv slot 91 12R3 token 33 -> kv slot 92 13 14本轮 EXTEND 新分配的 KV cache 位置从 100 开始 ```**

请求如下：

| 请求 | origin\_input\_ids | prefix cache 命中 | max\_new\_tokens |
| --- | --- | --- | --- |
| R1 | [11,12,13,14] | 0 | 3 |
| R2 | [21,22,23] | 1 | 2 |
| R3 | [31,32,33,34,35,36] | 3 | 2 |

进入 prefill 前：

**``` 1R1.output_ids = [] 2R2.output_ids = [] 3R3.output_ids = [] ```**

**第 1 步：Req 进入 ScheduleBatch**

**Scheduler 在 get\_new\_batch\_prefill 中先对每个请求调用：**

**``` 1req.init_next_round_input(tree_cache) ```**

得到：

**``` 1R1: 2fill_ids = [11,12,13,14] 3prefix_indices = [] 4extend_input_len = 4 5 6R2: 7fill_ids = [21,22,23] 8prefix_indices = [80] 9extend_input_len = 2 10 11R3: 12fill_ids = [31,32,33,34,35,36] 13prefix_indices = [90,91,92] 14  extend_input_len = 3 ```**

然后创建：

**``` 1new_batch = ScheduleBatch.init_new([R1, R2, R3], ...) 2new_batch.prepare_for_extend() ```**

prepare\_for\_extend 会把请求变成 EXTEND batch：

**``` 1ScheduleBatch.forward_mode = EXTEND 2ScheduleBatch.reqs = [R1, R2, R3] 3 4ScheduleBatch.input_ids = 5  [11,12,13,14, 22,23, 34,35,36] 6 7ScheduleBatch.prefix_lens = [0, 1, 3] 8ScheduleBatch.extend_lens = [4, 2, 3] 9ScheduleBatch.extend_num_tokens = 9 10ScheduleBatch.seq_lens = [4, 3, 6] 11ScheduleBatch.seq_lens_sum = 13 ```**

可视化：

**``` 1ScheduleBatch / EXTEND 2 3R1: [11][12][13][14] 4     └──── input_ids ────┘ 5 6R2: [21][22][23] 7     hit └ input ┘ 8 9R3: [31][32][33][34][35][36] 10     └── hit ──┘ └─ input ─┘ 11 12拼平后: 13input_ids = [11,12,13,14,22,23,34,35,36] ```**

这里 ScheduleBatch 还会完成内存分配：

**``` 1req_pool_indices: 2  R1 -> 0 3  R2 -> 1 4  R3 -> 2 5 6out_cache_loc: 7  为 9 个 extend token 分配 KV 写入位置： 8  [100,101,102,103,104,105,106,107,108] ```**

这里 out\_cache\_loc 的长度等于本轮实际 forward 的 token 数，也就是：

**``` 1R1 4 个 extend token 2R2 2 个 extend token 3R3 3 个 extend token 4总计 9 个 token ```**

它不包含已经命中 prefix cache 的 token。R2 的 21 已经在 kv slot 80，R3 的 31/32/33 已经在 kv slot 90/91/92，本轮只需要给新算的 11/12/13/14/22/23/34/35/36 分配新的 KV 写入位置。

**第 2 步：ScheduleBatch 转成 ModelWorkerBatch**

**Scheduler 在 run\_batch 中执行：**

**``` 1model_worker_batch = batch.get_model_worker_batch() ```**

对 EXTEND batch 来说，get\_model\_worker\_batch 会保留 extend 专用字段：

**``` 1ModelWorkerBatch.forward_mode = EXTEND 2ModelWorkerBatch.input_ids = [11,12,13,14,22,23,34,35,36] 3ModelWorkerBatch.req_pool_indices = [0,1,2] 4ModelWorkerBatch.seq_lens = [4,3,6] 5ModelWorkerBatch.out_cache_loc = [100,101,102,103,104,105,106,107,108] 6 7ModelWorkerBatch.extend_num_tokens = 9 8ModelWorkerBatch.extend_seq_lens = [4,2,3] 9ModelWorkerBatch.extend_prefix_lens = [0,1,3] 10ModelWorkerBatch.extend_logprob_start_lens = [...] ```**

这一步有一个重要分支：

**``` 1if self.forward_mode.is_decode_or_idle(): 2    extend_seq_lens = None 3    extend_prefix_lens = None 4else: 5    extend_seq_lens = self.extend_lens 6    extend_prefix_lens = self.prefix_lens ```**

也就是说：

**``` 1EXTEND / MIXED: 2  ModelWorkerBatch 携带 extend_* 信息。 3 4DECODE / IDLE: 5  extend_* 信息置 None。 ```**

这里有一个命名容易混淆的点：

**``` 1ModelWorkerBatch.extend_seq_lens 2  来自 ScheduleBatch.extend_lens。 3  表示每个请求本轮实际 extend 的 token 数。 4 5ModelWorkerBatch.seq_lens 6  来自 ScheduleBatch.seq_lens。 7  表示每个请求 forward 后的完整序列长度。 ```**

为什么需要 ModelWorkerBatch 这一层？

**``` 1ScheduleBatch 太重： 2  它包含调度器内部状态、tree cache、allocator、batch merge/filter 方法。 3 4worker forward 不需要全部调度状态： 5  它只需要 forward 张量、采样参数和特殊功能 metadata。 6 7overlap schedule 需要稳定快照： 8  overlap 模式会 record_batch_in_overlap(model_worker_batch)，避免 ScheduleBatch 后续被调度器修改时影响 forward stream。 ```**

**第 3 步：ModelWorkerBatch 转成 ForwardBatch**

**TP worker 收到 ModelWorkerBatch 后，在：**

**``` 1TPWorker.forward_batch_generation(...) ```**

中执行：

**``` 1forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner) ```**

这一步会先复制基础字段：

**``` 1ForwardBatch.forward_mode = EXTEND 2ForwardBatch.batch_size = len(seq_lens) = 3 3ForwardBatch.input_ids = [11,12,13,14,22,23,34,35,36] 4ForwardBatch.req_pool_indices = [0,1,2] 5ForwardBatch.seq_lens = [4,3,6] 6ForwardBatch.out_cache_loc = [100,101,102,103,104,105,106,107,108] 7ForwardBatch.sampling_info = ModelWorkerBatch.sampling_info ```**

然后补齐 model runner 运行时对象：

**``` 1ForwardBatch.req_to_token_pool = model_runner.req_to_token_pool 2ForwardBatch.token_to_kv_pool = model_runner.token_to_kv_pool 3ForwardBatch.attn_backend = model_runner.attn_backend ```**

接着计算 EXTEND 位置相关字段：

**``` 1extend_prefix_lens = [0,1,3] 2extend_seq_lens = [4,2,3] 3extend_num_tokens = 9 ```**

计算结果可以理解为：

**``` 1R1 positions: 0,1,2,3 2R2 positions: 1,2 3R3 positions: 3,4,5 ```**

拼平后：

**``` 1ForwardBatch.positions = 2  [0,1,2,3, 1,2, 3,4,5] 3 4ForwardBatch.extend_start_loc = 5  [0,4,6] ```**

可视化：

**``` 1ForwardBatch / EXTEND 2 3input_ids:  [11,12,13,14, 22,23, 34,35,36] 4positions:  [ 0, 1, 2, 3,  1, 2,  3, 4, 5] 5             └── R1 ────┘ └R2┘ └── R3 ───┘ 6 7extend_start_loc: 8  R1 starts at input_ids[0] 9  R2 starts at input_ids[4] 10  R3 starts at input_ids[6] ```**

注意：positions 不是在 ScheduleBatch 中生成的，而是在 ForwardBatch.init\_new 中根据 forward mode 和 extend 信息生成的。

**第 4 步：ForwardBatch 被 ModelRunner 使用**

**ForwardBatch 创建后进入：**

**``` 1ModelRunner.forward(forward_batch) ```**

普通生成模型会继续进入：

**``` 1ModelRunner._forward_raw 2  -> forward_extend 或 forward_decode ```**

EXTEND 路径中：

**``` 1self.attn_backend.init_forward_metadata(forward_batch) 2 3self.model.forward( 4    forward_batch.input_ids, 5    forward_batch.positions, 6    forward_batch, 7    ... 8) ```**

这说明 ForwardBatch 有三类消费者：

**``` 11. attention backend: 2   用 seq_lens / out_cache_loc / extend_* / positions 初始化 attention metadata。 3 42. model.forward: 5   直接消费 input_ids、positions、forward_batch。 6   各层 attention / MoE / Mamba / multimodal 逻辑会从 forward_batch 读取运行时信息。 7 83. logits processor / sampler: 9   用 forward_mode、seq_lens、positions、sampling_info、return_logprob 决定取哪些 logits 和如何采样。 ```**

模型 forward 后，ModelRunner.sample 会使用同一个 ForwardBatch：

**``` 1next_token_ids = self.sampler( 2    logits_output, 3    forward_batch.sampling_info, 4    forward_batch.return_logprob, 5    forward_batch.top_logprobs_nums, 6    forward_batch.token_ids_logprobs, 7    forward_batch.positions if decode else forward_batch.seq_lens - 1, 8) ```**

所以 ForwardBatch 不只是 attention 输入，它也贯穿 logits 和 sampling。

继续沿用 R1/R2/R3 的模拟数据，此时进入 ModelRunner.forward 的 EXTEND ForwardBatch 是：

**``` 1forward_mode = EXTEND 2batch_size = 3 3 4input_ids: 5[11,12,13,14, 22,23, 34,35,36] 6 7positions: 8[ 0, 1, 2, 3,  1, 2,  3, 4, 5] 9 10seq_lens: 11[4,3,6] 12 13extend_prefix_lens: 14[0,1,3] 15 16extend_seq_lens: 17[4,2,3] 18 19extend_start_loc: 20[0,4,6] 21 22req_pool_indices: 23  [0,1,2] ```**

这时 attention backend 会基于这些信息准备 metadata。可以粗略理解为：

**``` 1R1: 2本轮 token 范围 input_ids[0:4] 3写 KV 到 out_cache_loc[0:4] = [100,101,102,103] 4完整 seq_len = 4 5 6R2: 7prefix cache 已有 1 个 token 8本轮 token 范围 input_ids[4:6] 9写 KV 到 out_cache_loc[4:6] = [104,105] 10完整 seq_len = 3 11 12R3: 13prefix cache 已有 3 个 token 14本轮 token 范围 input_ids[6:9] 15写 KV 到 out_cache_loc[6:9] = [106,107,108] 16  完整 seq_len = 6 ```**

图示：

**``` 1ForwardBatch token layout 2 3flat idx:    0   1   2   3    4   5    6   7   8 4input_ids:  11  12  13  14   22  23   34  35  36 5positions:   0   1   2   3    1   2    3   4   5 6kv slot:    100 101 102 103  104 105  106 107 108 7request:    └──── R1 ────┘   └ R2 ┘   └── R3 ──┘ 8 9R1 last token flat idx = 3 10R2 last token flat idx = 5 11R3 last token flat idx = 8 ```**

把 prefix cache 和本轮新写入位置合起来看，请求的完整 KV 索引大致是：

**``` 1R1: 2tokens  [11, 12, 13, 14] 3kv slot [100,101,102,103] 4 5R2: 6tokens  [21, 22, 23] 7kv slot [ 80,104,105] 8 9R3: 10tokens  [31, 32, 33, 34, 35, 36] 11  kv slot [ 90, 91, 92,106,107,108] ```**

这里要特别区分两个概念：

**``` 1out_cache_loc: 2  本轮新 token 的 KV 写入位置。 3  它的长度等于本轮实际 forward token 数。 4 5req_to_token_pool.req_to_token: 6  每个请求完整上下文 token -> KV slot 的映射表。 7  attention 读取历史 KV 时主要依赖这张表。 ```**

也就是说，out\_cache\_loc 不是“读取历史 KV 的完整索引”。它更像是本轮新 KV 的写地址。历史 KV 的读取通常通过 req\_pool\_indices + seq\_lens 定位到 req\_to\_token\_pool.req\_to\_token，再从 token\_to\_kv\_pool 中读取对应 slot 的 K/V。

以 R2 为例：

**``` 1R2 完整上下文: 2  tokens  [21, 22, 23] 3  kv slot [80,104,105] 4 5其中： 6  21 是 prefix cache 命中的历史 token，KV 已经在 slot 80。 7  22/23 是本轮 EXTEND 新算的 token，KV 写到 out_cache_loc[4:6] = [104,105]。 ```**

attention 计算时可以这样理解：

**``` 11. 当前 hidden_states 计算出 22/23 的新 K/V。 22. 通过 out_cache_loc=[104,105] 把新 K/V 写入 token_to_kv_pool。 33. 当前 token 做 attention 时需要完整上下文 [21,22,23]。 44. 通过 req_to_token_pool.req_to_token 找到 [80,104,105]。 55. 从 token_to_kv_pool 读取这些 slot 的 K/V 参与 attention。 ```**

因此：

**``` 1写新 KV: 2  out_cache_loc -> token_to_kv_pool.set_kv_buffer(...) 3 4读上下文 KV: 5  req_to_token_pool.req_to_token + seq_lens -> token_to_kv_pool.get_*_buffer(...) ```**

**第 5 步：ForwardBatch 怎么进入模型文件**

**模型文件里并不会直接接收 ScheduleBatch 或 ModelWorkerBatch。它接收的是：**

**``` 1self.model.forward( 2    forward_batch.input_ids, 3    forward_batch.positions, 4    forward_batch, 5    ... 6) ```**

也就是说，模型文件入口通常是三个核心参数：

**``` 1input_ids: 2本轮要执行的 token ids。 3EXTEND 时是未缓存 prompt token 拼平。 4DECODE 时是每个请求上一步采样出的 token。 5 6positions: 7每个 input token 在原序列里的位置。 8由 ForwardBatch.init_new 生成。 9 10forward_batch: 11  携带 attention/KV/cache/logprob/sampling/LoRA/DP 等运行时信息。 ```**

以 python/sglang/srt/models/llama.py 为例，调用链可以简化成：

**``` 1ModelRunner.forward 2  -> ModelRunner.forward_extend / forward_decode 3  -> LlamaForCausalLM.forward(input_ids, positions, forward_batch) 4  -> LlamaModel.forward(input_ids, positions, forward_batch) 5  -> embed_tokens(input_ids) 得到 hidden_states 6  -> 每层 LlamaDecoderLayer.forward(positions, hidden_states, forward_batch, ...) 7  -> norm 8  -> logits_processor(input_ids, hidden_states, lm_head, forward_batch) ```**

对应代码结构是：

**``` 1LlamaForCausalLM.forward: 2hidden_states = self.model( 3      input_ids, 4      positions, 5      forward_batch, 6      input_embeds, 7      pp_proxy_tensors=pp_proxy_tensors, 8) 9 10return self.logits_processor( 11      input_ids, 12      hidden_states, 13      self.lm_head, 14      forward_batch, 15      aux_hidden_states, 16  ) ```**

再进入 LlamaModel.forward：

**``` 1if first PP rank: 2if input_embeds is None: 3      hidden_states = self.embed_tokens(input_ids) 4else: 5      hidden_states = input_embeds 6residual = None 7else: 8hidden_states = pp_proxy_tensors["hidden_states"] 9residual = pp_proxy_tensors["residual"] 10 11for layer in layers: 12hidden_states, residual = layer( 13      positions, 14      hidden_states, 15      forward_batch, 16      residual, 17) 18 19hidden_states = norm(hidden_states, residual) 20return hidden_states ```**

这就是 ForwardBatch 到模型文件，再到第一份 hidden\_states 的关键转换：

**``` 1ForwardBatch.input_ids 2  -> model.embed_tokens(input_ids) 3  -> hidden_states ```**

如果 ForwardBatch.input\_embeds 不为空，比如 multimodal、embedding override、某些 speculative / MTP 路径，则模型会跳过或替换普通 token embedding：

**``` 1forward_batch.input_embeds 2  -> kwargs["input_embeds"] 3  -> hidden_states = input_embeds ```**

可视化：

**``` 1ForwardBatch / EXTEND 2 3input_ids:  [11,12,13,14,22,23,34,35,36] 4positions:  [ 0, 1, 2, 3, 1, 2, 3, 4, 5] 5                 │ 6                 v 7model.embed_tokens(input_ids) 8                 │ 9                 v 10hidden_states: 11shape = [num_tokens, hidden_size] 12        = [9, hidden_size] 13                 │ 14                 v 15Transformer layers: 16  layer(positions, hidden_states, forward_batch, residual) ```**

把这 9 行 hidden\_states 展开看：

**``` 1hidden_states[0] = embed(11), position 0, belongs to R1 2hidden_states[1] = embed(12), position 1, belongs to R1 3hidden_states[2] = embed(13), position 2, belongs to R1 4hidden_states[3] = embed(14), position 3, belongs to R1 5 6hidden_states[4] = embed(22), position 1, belongs to R2 7hidden_states[5] = embed(23), position 2, belongs to R2 8 9hidden_states[6] = embed(34), position 3, belongs to R3 10hidden_states[7] = embed(35), position 4, belongs to R3 11hidden_states[8] = embed(36), position 5, belongs to R3 ```**

注意 R2/R3 的 position 不是从 0 开始，因为它们前面有 prefix cache 命中：

**``` 1R2 prefix_len = 1: 2  本轮只算 [22,23] 3  positions = [1,2] 4 5R3 prefix_len = 3: 6  本轮只算 [34,35,36] 7  positions = [3,4,5] ```**

也就是说，hidden\_states 的行数等于本轮实际 forward token 数，而不是原始 prompt token 数：

**``` 1R1 4 行 + R2 2 行 + R3 3 行 = 9 行 ```**

Decode 时同理：

**``` 1ForwardBatch / DECODE 2 3input_ids:  [101,201,301] 4positions:  [  4,  3,  6] 5                 │ 6                 v 7model.embed_tokens(input_ids) 8                 │ 9                 v 10hidden_states: 11  shape = [batch_size, hidden_size] 12        = [3, hidden_size] ```**

**第 6 步：hidden\_states 如何继续使用 ForwardBatch**

**得到 hidden\_states 后，模型层仍然会持续使用 forward\_batch。  典型 layer 调用是：**

**``` 1hidden_states, residual = layer( 2    positions, 3    hidden_states, 4    forward_batch, 5    residual, 6) ```**

在 layer 内部，attention 会依赖 forward\_batch 中的运行时信息：

**``` 1seq_lens: 2当前每个请求的序列长度。 3 4out_cache_loc: 5本轮新 KV 写入 token_to_kv_pool 的位置。 6它主要是写地址，不是完整历史 KV 的读取地址。 7 8req_pool_indices: 9请求在 req_to_token_pool 中的行号。 10attention 会结合 req_pool_indices、seq_lens 和 req_to_token_pool.req_to_token 找到完整上下文的 KV slot。 11 12extend_prefix_lens / extend_seq_lens / extend_start_loc: 13prefill attention 需要知道每个请求的 prefix 和本轮 token 区间。 14 15positions / mrope_positions: 16rotary embedding 或 MROPE 位置编码使用。 17 18attn_backend metadata: 19  init_forward_metadata(forward_batch) 已经根据 batch 形状准备好 backend 所需索引。 ```**

所以 hidden\_states 的计算不是只靠 input\_ids，而是：

**``` 1input_ids -> embedding -> hidden_states 2positions -> positional / rotary information 3forward_batch -> attention/KV/cache/runtime metadata ```**

最终最后一层输出的 hidden\_states 会进入 logits processor：

**``` 1logits_processor( 2  input_ids, 3  hidden_states, 4  lm_head, 5  forward_batch, 6) ```**

logits processor 会根据 forward\_batch.forward\_mode 决定取哪些位置的 logits：

**``` 1DECODE: 2  每个请求都有一个 decode token，通常取每行对应 logits。 3 4EXTEND: 5  通常只取每个请求最后一个 token 的 logits 用于下一 token 采样。 6  如果 return_logprob，还会额外处理 prompt token logprobs。 7 8PREFILL-ONLY: 9  可能只计算 logprob / embedding，不采样 next token。 ```**

然后 ModelRunner.sample 使用同一个 forward\_batch.sampling\_info 采样：

**``` 1LogitsProcessorOutput 2  -> Sampler(..., forward_batch.sampling_info, ...) 3  -> next_token_ids 4  -> GenerationBatchResult 5  -> Scheduler.process_batch_result 6  -> Req.output_ids 更新 ```**

继续使用 R1/R2/R3 的 EXTEND 示例，最后一层输出仍然是：

**``` 1final_hidden_states.shape = [9, hidden_size] ```**

但 prefill 生成下一个 token 时，不需要 9 个位置都采样。通常只需要每个请求最后一个 token 的 logits：

**``` 1R1: 2最后一个本轮 token = 14 3flat idx = 3 4用 final_hidden_states[3] 计算 next-token logits 5 6R2: 7最后一个本轮 token = 23 8flat idx = 5 9用 final_hidden_states[5] 计算 next-token logits 10 11R3: 12最后一个本轮 token = 36 13flat idx = 8 14  用 final_hidden_states[8] 计算 next-token logits ```**

可视化：

**``` 1final_hidden_states rows 2 3idx:        0    1    2    3     4    5     6    7    8 4token:     11   12   13   14    22   23    34   35   36 5request:   └──── R1 ─────┘     └ R2 ┘     └── R3 ───┘ 6sample:                   ^          ^              ^ 7                         R1         R2             R3 ```**

假设 sampler 得到：

**``` 1next_token_ids = [101, 201, 301] ```**

那么 GenerationBatchResult 会把它带回 Scheduler：

**``` 1GenerationBatchResult.next_token_ids = [101,201,301] ```**

随后 Scheduler.process\_batch\_result 会更新请求状态：

**``` 1R1.output_ids = [101] 2R2.output_ids = [201] 3R3.output_ids = [301] ```**

这三个请求下一轮合入 running\_batch 后，就进入 decode 生命周期。也就是后面 Decode 阶段使用的模拟数据来源。

**Decode 阶段的数据变化**

**假设上面 prefill 后采样出：**

**``` 1R1.output_ids = [101] 2R2.output_ids = [201] 3R3.output_ids = [301] ```**

下一轮没有新 prefill，Scheduler 会调用：

**``` 1running_batch = update_running_batch(running_batch) ```**

其中：

**``` 1batch.prepare_for_decode() ```**

会把 ScheduleBatch 改成 DECODE：

**``` 1ScheduleBatch.forward_mode = DECODE 2ScheduleBatch.input_ids = [101,201,301] 3ScheduleBatch.seq_lens = [5,4,7] 4ScheduleBatch.out_cache_loc = [109,110,111] ```**

这里假设 allocator 继续从 EXTEND 已经使用过的 100..108 后面分配，所以 3 个 decode token 分别写到：

**``` 1R1 decode token 101 -> kv slot 109 2R2 decode token 201 -> kv slot 110 3R3 decode token 301 -> kv slot 111 ```**

这里的 input\_ids 含义已经变了：

**``` 1EXTEND input_ids: 2  prompt 中没有命中 prefix cache 的 token，数量可以远大于 batch_size。 3 4DECODE input_ids: 5  每个请求一个 last output token，数量通常等于 batch_size。 ```**

Decode 的 ModelWorkerBatch：

**``` 1ModelWorkerBatch.forward_mode = DECODE 2ModelWorkerBatch.input_ids = [101,201,301] 3ModelWorkerBatch.req_pool_indices = [0,1,2] 4ModelWorkerBatch.seq_lens = [5,4,7] 5ModelWorkerBatch.out_cache_loc = [109,110,111] 6 7ModelWorkerBatch.extend_seq_lens = None 8ModelWorkerBatch.extend_prefix_lens = None 9ModelWorkerBatch.extend_num_tokens = 普通 decode 路径不使用 ```**

Decode 的 ForwardBatch.init\_new 会走另一条位置逻辑：

**``` 1if forward_mode.is_decode(): 2  positions = clamp_position(seq_lens) ```**

可以理解为：

**``` 1R1 seq_len = 5 -> position = 4 2R2 seq_len = 4 -> position = 3 3R3 seq_len = 7 -> position = 6 ```**

因此：

**``` 1ForwardBatch / DECODE 2 3input_ids: [101,201,301] 4positions: [4,3,6] 5seq_lens:  [5,4,7] 6out_cache_loc: [109,110,111] ```**

可视化：

**``` 1R1: [11][12][13][14][101] 2                         ^ 3                         decode position 4 4                         kv slot 109 5 6R2: [21][22][23][201] 7                    ^ 8                    decode position 3 9                    kv slot 110 10 11R3: [31][32][33][34][35][36][301] 12                                   ^ 13                                   decode position 6 14                                   kv slot 111 ```**

decode 完成这一轮 forward 后，完整 KV 索引会继续增长：

**``` 1R1: 2tokens  [11, 12, 13, 14,101] 3kv slot [100,101,102,103,109] 4 5R2: 6tokens  [21, 22, 23,201] 7kv slot [ 80,104,105,110] 8 9R3: 10tokens  [31, 32, 33, 34, 35, 36,301] 11  kv slot [ 90, 91, 92,106,107,108,111] ```**

decode attention 的读写关系也是同一套逻辑。以 R2 为例：

**``` 1本轮 decode input token: 2  201 3 4本轮写新 KV: 5  out_cache_loc = [110] 6 7attention 读取完整上下文 KV: 8  tokens  [21, 22, 23,201] 9  kv slot [80,104,105,110] ```**

所以 decode 阶段不是只读 out\_cache\_loc=110。110 是新 token 的写入位置；attention 读取的是 req\_to\_token\_pool.req\_to\_token 中记录的完整上下文 KV slot，包括历史的 80/104/105 和本轮新写入的 110。

**三层 Batch 的字段流转表**

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 以 EXTEND 为例：   | 信息 | ScheduleBatch | ModelWorkerBatch | ForwardBatch | | --- | --- | --- | --- | | 请求列表 | reqs=[R1,R2,R3] | reqs 保留用于 grammar/rids 等 | rids ，通常不再依赖完整 Req | | forward mode | EXTEND | EXTEND | EXTEND | | input ids | 已拼平 | 原样携带 | 原样执行 | | req pool idx | alloc\_for\_extend 得到 | 原样携带 | attention/KV 使用 | | seq lens | prepare\_for\_extend 得到 | 原样携带 | attention/logits 使用 | | out cache loc | KV allocator 分配，例如 [100..108] | 原样携带 | attention 写 KV 使用 | | prefix/extend lens | list 形式 | list 形式 | tensor + CPU list | | positions | 无 | 无 | compute\_position 得到 | | hidden\_states | 无 | 无 | 模型文件中由 embed\_tokens(input\_ids) 或 input\_embeds 生成 | | KV pool 对象 | allocator 引用 | 无 | model\_runner.token\_to\_kv\_pool | | attention backend | 无 | 无 | model\_runner.attn\_backend | | sampling info | 创建并维护 | 原样携带 | sampler 使用 |   以 DECODE 为例：   | 信息 | ScheduleBatch | ModelWorkerBatch | ForwardBatch | | --- | --- | --- | --- | | input ids | last output token | 原样携带 | 原样执行 | | seq lens | prepare\_for\_decode 中 +1 | 原样携带 | 生成 decode positions | | extend fields | 不再使用 | None | None | | positions | 无 | 无 | seq\_lens - 1 | | hidden\_states | 无 | 无 | 模型文件中由 decode token embedding 得到 | | out cache loc | 每个请求一个新 KV 位置，例如 [109,110,111] | 原样携带 | attention 写 KV 使用 | |

**三个对象分别在哪些地方使用**

**ScheduleBatch 的使用位置**

**ScheduleBatch 主要在 Scheduler 和调度相关 worker 中使用：**

**``` 1Scheduler.get_next_batch_to_run: 2返回下一轮要跑的 ScheduleBatch。 3 4get_new_batch_prefill: 5创建 EXTEND ScheduleBatch。 6 7update_running_batch: 8把 running_batch 更新成 DECODE ScheduleBatch。 9 10ScheduleBatch.prepare_for_extend: 11构建 prefill tensor 并分配 KV。 12 13ScheduleBatch.prepare_for_decode: 14构建 decode tensor 并分配 KV。 15 16ScheduleBatch.filter_batch / merge_batch: 17维护 running_batch 生命周期。 18 19Scheduler.process_batch_result: 20根据 forward 结果更新 req.output_ids / finished / logprob。 21 22speculative workers: 23  部分 spec v1 路径仍直接接收 ScheduleBatch。 ```**

**ModelWorkerBatch 的使用位置**

**ModelWorkerBatch 主要出现在 Scheduler 到 worker 的边界：**

**``` 1ScheduleBatch.get_model_worker_batch: 2创建 ModelWorkerBatch。 3 4Scheduler.run_batch: 5普通 generation 大多数路径都会先得到 ModelWorkerBatch。 6 7overlap schedule: 8record_batch_in_overlap(model_worker_batch) 保存 forward stream 需要的快照。 9 10TPWorker.forward_batch_generation: 11接收 ModelWorkerBatch 并转换为 ForwardBatch。 12 13speculative workers: 14target worker / draft worker 之间也会传 ModelWorkerBatch。 15 16embedding path: 17  forward_batch_embedding 也使用 ModelWorkerBatch。 ```**

**ForwardBatch 的使用位置**

**ForwardBatch 是模型执行层通用对象：**

**``` 1TPWorker.forward_batch_generation: 2ForwardBatch.init_new(model_worker_batch, model_runner)。 3 4ModelRunner.forward: 5接收 ForwardBatch。 6 7ModelRunner.forward_extend / forward_decode: 8根据 forward_mode 分发执行。 9 10attention backend: 11init_forward_metadata(forward_batch)。 12 13model.forward: 14model.forward(input_ids, positions, forward_batch)。 15 16logits processor: 17根据 forward_mode / seq_lens / extend 信息抽取 logits。 18 19sampler: 20使用 sampling_info、positions、logprob 字段采样 next_token_ids。 21 22cuda graph / piecewise cuda graph: 23capture/replay 时也构造或复用 ForwardBatch。 24 25DP attention: 26  ForwardBatch.prepare_mlp_sync_batch 处理 global_num_tokens 和 padding。 ```**

**为什么不直接从 ScheduleBatch 到 ForwardBatch**

**看起来可以省掉 ModelWorkerBatch，但实际有几个原因需要中间层：**

**``` 11. 隔离 Scheduler 内部状态 2   ScheduleBatch 带有 reqs、tree_cache、allocator、filter/merge 等调度逻辑。 3   model worker 不应该依赖这些调度细节。 4 52. 支持 overlap schedule 6   overlap 模式下 Scheduler 线程和 forward stream 并行。 7   ModelWorkerBatch 是一个更稳定的 forward 快照。 8 93. 支持不同 worker 类型 10   TP worker、speculative worker、embedding worker、MLX worker 都可以围绕 ModelWorkerBatch 建接口。 11 124. 运行时对象必须由 ModelRunner 注入 13   token_to_kv_pool、attn_backend、device-specific positions 不属于 Scheduler。 14   它们在 ForwardBatch.init_new 中由 model_runner 注入更合理。 ```**

**常见容易绕的点**

**1. input\_ids 在三层中是同一批 token，但含义取决于 forward\_mode**

**``` 1EXTEND: 2  input_ids 是未命中 prefix cache 的 prompt token 拼平。 3 4DECODE: 5  input_ids 是每个请求上一轮采样出的最后一个 token。 ```**

**2. positions 不在 ScheduleBatch 中**

**Scheduler 不直接生成 positions。它只提供：**

**``` 1seq_lens 2prefix_lens 3extend_lens 4extend_num_tokens ```**

真正的 positions 在 ForwardBatch.init\_new 中生成，因为它依赖 model runner 的后端、device 和特殊模型逻辑。

**3. out\_cache\_loc 由 Scheduler 分配，但由 attention 执行使用**

**``` 1ScheduleBatch.prepare_for_extend / prepare_for_decode: 2  分配 out_cache_loc。 3 4ForwardBatch: 5  携带 out_cache_loc。 6 7attention backend / attention layer: 8  根据 out_cache_loc 把新 KV 写入 token_to_kv_pool。 ```**

**4. sampling\_info 从 ScheduleBatch 一直传到 ForwardBatch**

**sampling\_info 在 ScheduleBatch.prepare\_for\_extend 或 decode 生命周期中创建/维护，然后一路传到 ForwardBatch。采样时 ModelRunner.sample 会使用它：**

**``` 1temperature 2top_p 3top_k 4min_p 5presence/frequency penalty 6grammar mask 7logprob 参数 ```**

**5. ForwardBatch 是局部执行对象**

**ForwardBatch 通常只在一次 forward 中使用。forward 结束后，结果会通过 GenerationBatchResult 回到 Scheduler，再由 ScheduleBatch 和 Req 更新长期状态。**

**``` 1ForwardBatch: 2  执行期临时对象。 3 4ScheduleBatch / Req: 5  调度和请求生命周期状态。 ```**

**最后串起来**

**一次普通 generation 的主路径可以总结成：**

**``` 1Req 2-> Scheduler admission 3-> ScheduleBatch.prepare_for_extend / prepare_for_decode 4-> ScheduleBatch.get_model_worker_batch 5-> TPWorker.forward_batch_generation 6-> ForwardBatch.init_new 7-> attention_backend.init_forward_metadata 8-> model.forward(input_ids, positions, forward_batch) 9-> embed_tokens(input_ids) / input_embeds 10-> hidden_states 11-> transformer layers(hidden_states, positions, forward_batch) 12-> logits processor 13-> sampler 14-> GenerationBatchResult 15-> Scheduler.process_batch_result 16  -> Req.output_ids / finished 状态更新 ```**

理解这条链路后，再看 DP attention、overlap schedule、speculative decoding、cuda graph、multimodal、LoRA 等分支，本质上都是在这三个 Batch 对象上增加字段、补 runtime metadata，或者改变它们的转换时机。
