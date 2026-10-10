# SGLang推理优化

本文围绕 Scheduler 调度器到模型 forward 之间最关键的三个 Batch 数据结构展开：


```text
ScheduleBatch
  -> ModelWorkerBatch
  -> ForwardBatch
```


上一篇调度器ScheduleBatch重点解释 Scheduler 如何决定“下一轮跑 prefill 还是 decode”。本文继续往后走，解释这个决定出来的 ScheduleBatch 如何一步步变成模型真正执行时消费的 ForwardBatch然后模型开始使用input\_ids开始做module forward一系列工作。

**一句话理解**

**这三个对象不是重复封装，而是服务于三个不同层级：**


```text
ScheduleBatch:
Scheduler 视角。
关心请求生命周期、调度状态、req 列表、prefix cache、KV cache 分配、采样参数。

ModelWorkerBatch:
Scheduler -> worker 的传输视角。
把 ScheduleBatch 中 forward 需要的字段打包出来，减少 worker 侧直接依赖调度器内部状态。

ForwardBatch:
ModelRunner / attention backend / model.forward 的执行视角。
  在 ModelWorkerBatch 基础上补齐 positions、attention metadata、KV pool、DP padding、LoRA、MROPE 等运行时信息。
```


核心数据流如下：


```text
┌─────────────────────────────────────────────┐
│ Scheduler.get_next_batch_to_run              │
│ 产物：ScheduleBatch                          │
│                                             │
│ 包含：Req 列表、forward_mode、input_ids、     │
│ seq_lens、out_cache_loc、sampling_info       │
└──────────────────────┬──────────────────────┘
                       │
                       │ ScheduleBatch.get_model_worker_batch()
                       v
┌─────────────────────────────────────────────┐
│ Scheduler.run_batch / TPWorker               │
│ 产物：ModelWorkerBatch                       │
│                                             │
│ 包含：worker forward 必需字段，外加 logprob、 │
│ DP attention、multimodal、LoRA、spec 信息     │
└──────────────────────┬──────────────────────┘
                       │
                       │ ForwardBatch.init_new(...)
                       v
┌─────────────────────────────────────────────┐
│ TPWorker.forward_batch_generation            │
│ ModelRunner.forward                          │
│ 产物：ForwardBatch                           │
│                                             │
│ 包含：positions、extend_start_loc、KV pool、  │
│ attn_backend、MROPE、LoRA runtime metadata    │
└──────────────────────┬──────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────┐
│ model.forward(input_ids, positions, batch)   │
│                                             │
│ input_ids -> embed_tokens -> hidden_states   │
│ hidden_states + forward_batch -> layers      │
│ logits processor / sampler                   │
└─────────────────────────────────────────────┘
```


可以用一句话记：


```text
ScheduleBatch 管调度，
ModelWorkerBatch 管跨层传递，
ForwardBatch 管模型执行。
```


**关键代码位置**

**主要涉及以下文件：**

- python/sglang/srt/managers/schedule\_batch.py
- ScheduleBatch
- ScheduleBatch.prepare\_for\_extend
- ScheduleBatch.prepare\_for\_decode
- ScheduleBatch.get\_model\_worker\_batch
- ModelWorkerBatch
- python/sglang/srt/managers/scheduler.py
- Scheduler.run\_batch
- Scheduler.record\_batch\_in\_overlap
- python/sglang/srt/managers/tp\_worker.py
- TpModelWorker.forward\_batch\_generation
- python/sglang/srt/model\_executor/forward\_batch\_info.py
- ForwardBatch
- ForwardBatch.init\_new
- ForwardBatch.prepare\_mlp\_sync\_batch
- python/sglang/srt/model\_executor/model\_runner.py
- ModelRunner.forward
- ModelRunner.forward\_decode
- ModelRunner.forward\_extend
- ModelRunner.sample
- python/sglang/srt/layers/attention/\*\_backend.py
- init\_forward\_metadata
- attention forward
- python/sglang/srt/layers/logits\_processor.py
- LogitsProcessor
- python/sglang/srt/layers/sampler.py
- Sampler

**三个对象先拆开看**

**ScheduleBatch：调度器手里的 Batch**

**ScheduleBatch 是 Scheduler 的核心状态对象。它知道“这一组请求为什么能跑、应该怎么跑、跑完之后怎么回到调度生命周期”。  它的典型字段可以分成 5 类：**


```text
1. 请求和资源
   reqs
   req_to_token_pool
   token_to_kv_pool_allocator
   tree_cache

2. 调度模式
   forward_mode
   batch_is_full
   chunked_req
   is_prefill_only

3. forward 输入张量
   input_ids
   req_pool_indices
   seq_lens
   seq_lens_cpu
   orig_seq_lens
   out_cache_loc
   output_ids

4. extend / decode 专用信息
   prefix_lens
   extend_lens
   extend_num_tokens
   extend_logprob_start_lens
   decoding_reqs

5. 后处理和特殊能力
   sampling_info
   return_logprob
   multimodal_inputs
   spec_info
   hicache_consumer_index
```


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


```text
batch.reqs = [R1, R2, R3]
```


这意味着它不仅能跑 forward，还能在 forward 后把结果写回每个请求：


```text
req.output_ids
req.finished_reason
req.input_token_logprobs
req.hidden_states
req.time_stats
```


所以 ScheduleBatch 是“调度生命周期对象”，不是单纯的模型输入。

**ModelWorkerBatch：Scheduler 和 worker 之间的中间层**

**ModelWorkerBatch 在 schedule\_batch.py 中定义，它由：**


```text
ScheduleBatch.get_model_worker_batch()
```


创建。

它的作用是把 ScheduleBatch 中 worker forward 需要的字段打包出来：


```text
forward_mode
input_ids
req_pool_indices
seq_lens
out_cache_loc
sampling_info
multimodal_inputs
lora_ids
spec_info
extend_* 字段
DP attention 字段
```


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


```text
1. 它更接近 worker 输入，不负责调度 admission。
2. 它保留 reqs，主要是为了 grammar、rids、部分后处理和调试。
3. 它还没有 positions，也没有 attn_backend 的 runtime metadata。
4. 它不直接持有 token_to_kv_pool，KV pool 由 model_runner 注入到 ForwardBatch。
```


可以把它理解成：


```text
ScheduleBatch 的 forward 投影。
```


**ForwardBatch：模型真正消费的 Batch**

**ForwardBatch 在 model\_executor/forward\_batch\_info.py 中定义，由：**


```text
ForwardBatch.init_new(model_worker_batch, model_runner)
```


创建。

它是 model runner、attention backend、model.forward、logits processor、sampler 真正使用的对象。

和 ModelWorkerBatch 相比，ForwardBatch 会补齐运行时信息：


```text
positions:
decode 时通常是 seq_lens - 1。
extend 时由 prefix_lens / seq_lens 计算每个 token 的位置。

extend_start_loc:
extend batch 中每个请求在拼平 input_ids 里的起始位置。

req_to_token_pool / token_to_kv_pool:
model_runner 注入的内存池对象，attention 写 KV 时需要。

attn_backend:
当前模型 runner 绑定的 attention backend。

global_num_tokens_gpu:
DP attention / MLP sync 场景下的全局 token 数。

out_cache_loc_swa:
hybrid SWA cache 场景下的 SWA KV 写入位置。

mrope_positions:
Qwen2-VL / MROPE 模型的位置编码。

LoRA runtime metadata:
  prepare_lora_batch 后供 LoRA 层使用。
```


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


```text
page_size = 1
不开启 DP attention
不开启 speculative decoding
不开启 LoRA
不开启 multimodal
不开启 overlap
prefix cache 中 R2 命中 1 个 token，R3 命中 3 个 token
token_to_kv_pool 中已经存在的 prefix cache 位置：
R2 token 21 -> kv slot 80
R3 token 31 -> kv slot 90
R3 token 32 -> kv slot 91
R3 token 33 -> kv slot 92

本轮 EXTEND 新分配的 KV cache 位置从 100 开始
```


请求如下：

| 请求 | origin\_input\_ids | prefix cache 命中 | max\_new\_tokens |
| --- | --- | --- | --- |
| R1 | [11,12,13,14] | 0 | 3 |
| R2 | [21,22,23] | 1 | 2 |
| R3 | [31,32,33,34,35,36] | 3 | 2 |

进入 prefill 前：


```text
R1.output_ids = []
R2.output_ids = []
R3.output_ids = []
```


**第 1 步：Req 进入 ScheduleBatch**

**Scheduler 在 get\_new\_batch\_prefill 中先对每个请求调用：**


```text
req.init_next_round_input(tree_cache)
```


得到：


```text
R1:
fill_ids = [11,12,13,14]
prefix_indices = []
extend_input_len = 4

R2:
fill_ids = [21,22,23]
prefix_indices = [80]
extend_input_len = 2

R3:
fill_ids = [31,32,33,34,35,36]
prefix_indices = [90,91,92]
  extend_input_len = 3
```


然后创建：


```text
new_batch = ScheduleBatch.init_new([R1, R2, R3], ...)
new_batch.prepare_for_extend()
```


prepare\_for\_extend 会把请求变成 EXTEND batch：


```text
ScheduleBatch.forward_mode = EXTEND
ScheduleBatch.reqs = [R1, R2, R3] 3 4ScheduleBatch.input_ids = 5  [11,12,13,14, 22,23,
4,35,36] 6 7ScheduleBatch.prefix_lens = [0, 1, 3] 8ScheduleBatch.extend_lens = [4, 2, 3] 9ScheduleBatch.extend_num_tokens = 9 10ScheduleBatch.seq_lens = [4, 3, 6] 11ScheduleBatch.seq_lens_sum = 13
```


可视化：


```text
ScheduleBatch / EXTEND

R1: [11][12][13][14]
     └──── input_ids ────┘

R2: [21][22][23]
     hit └ input ┘

R3: [31][32][33][34][35][36]
     └── hit ──┘ └─ input ─┘

拼平后:
input_ids = [11,12,13,14,22,23,34,35,36]
```


这里 ScheduleBatch 还会完成内存分配：


```text
req_pool_indices:
  R1 -> 0
  R2 -> 1
  R3 -> 2

out_cache_loc:
  为 9 个 extend token 分配 KV 写入位置：
  [100,101,102,103,104,105,106,107,108]
```


这里 out\_cache\_loc 的长度等于本轮实际 forward 的 token 数，也就是：


```text
R1 4 个 extend token
R2 2 个 extend token
R3 3 个 extend token
总计 9 个 token
```


它不包含已经命中 prefix cache 的 token。R2 的 21 已经在 kv slot 80，R3 的 31/32/33 已经在 kv slot 90/91/92，本轮只需要给新算的 11/12/13/14/22/23/34/35/36 分配新的 KV 写入位置。

**第 2 步：ScheduleBatch 转成 ModelWorkerBatch**

**Scheduler 在 run\_batch 中执行：**


```text
model_worker_batch = batch.get_model_worker_batch()
```


对 EXTEND batch 来说，get\_model\_worker\_batch 会保留 extend 专用字段：


```text
ModelWorkerBatch.forward_mode = EXTEND
ModelWorkerBatch.input_ids = [11,12,13,14,22,23,34,35,36]
ModelWorkerBatch.req_pool_indices = [0,1,2]
ModelWorkerBatch.seq_lens = [4,3,6]
ModelWorkerBatch.out_cache_loc = [100,101,102,103,104,105,106,107,108]

ModelWorkerBatch.extend_num_tokens = 9
ModelWorkerBatch.extend_seq_lens = [4,2,3]
ModelWorkerBatch.extend_prefix_lens = [0,1,3]
ModelWorkerBatch.extend_logprob_start_lens = [...]
```


这一步有一个重要分支：


```python
if self.forward_mode.is_decode_or_idle():
    extend_seq_lens = None
    extend_prefix_lens = None
else:
    extend_seq_lens = self.extend_lens
    extend_prefix_lens = self.prefix_lens
```


也就是说：


```text
EXTEND / MIXED:
  ModelWorkerBatch 携带 extend_* 信息。

DECODE / IDLE:
  extend_* 信息置 None。
```


这里有一个命名容易混淆的点：


```text
ModelWorkerBatch.extend_seq_lens
  来自 ScheduleBatch.extend_lens。
  表示每个请求本轮实际 extend 的 token 数。

ModelWorkerBatch.seq_lens
  来自 ScheduleBatch.seq_lens。
  表示每个请求 forward 后的完整序列长度。
```


为什么需要 ModelWorkerBatch 这一层？


```text
ScheduleBatch 太重：
  它包含调度器内部状态、tree cache、allocator、batch merge/filter 方法。

worker forward 不需要全部调度状态：
  它只需要 forward 张量、采样参数和特殊功能 metadata。

overlap schedule 需要稳定快照：
  overlap 模式会 record_batch_in_overlap(model_worker_batch)，避免 ScheduleBatch 后续被调度器修改时影响 forward stream。
```


**第 3 步：ModelWorkerBatch 转成 ForwardBatch**

**TP worker 收到 ModelWorkerBatch 后，在：**


```text
TPWorker.forward_batch_generation(...)
```


中执行：


```python
forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
```


这一步会先复制基础字段：


```text
ForwardBatch.forward_mode = EXTEND
ForwardBatch.batch_size = len(seq_lens) = 3
ForwardBatch.input_ids = [11,12,13,14,22,23,34,35,36]
ForwardBatch.req_pool_indices = [0,1,2]
ForwardBatch.seq_lens = [4,3,6]
ForwardBatch.out_cache_loc = [100,101,102,103,104,105,106,107,108]
ForwardBatch.sampling_info = ModelWorkerBatch.sampling_info
```


然后补齐 model runner 运行时对象：


```text
ForwardBatch.req_to_token_pool = model_runner.req_to_token_pool
ForwardBatch.token_to_kv_pool = model_runner.token_to_kv_pool
ForwardBatch.attn_backend = model_runner.attn_backend
```


接着计算 EXTEND 位置相关字段：


```text
extend_prefix_lens = [0,1,3]
extend_seq_lens = [4,2,3]
extend_num_tokens = 9
```


计算结果可以理解为：


```text
R1 positions: 0,1,2,3
R2 positions: 1,2
R3 positions: 3,4,5
```


拼平后：


```text
ForwardBatch.positions =
  [0,1,2,3, 1,2,
,4,5] 3
ForwardBatch.extend_start_loc =
  [0,4,6]
```


可视化：


```text
ForwardBatch / EXTEND 2 3input_ids:  [11,12,13,14,
2,23,
4,35,36]
positions:  [ 0, 1, 2, 3,  1, 2,  3, 4,
] 5             └── R1 ────┘ └R2┘ └── R3 ───┘

extend_start_loc:
  R1 starts at input_ids[0]
  R2 starts at input_ids[4]
  R3 starts at input_ids[6]
```


注意：positions 不是在 ScheduleBatch 中生成的，而是在 ForwardBatch.init\_new 中根据 forward mode 和 extend 信息生成的。

**第 4 步：ForwardBatch 被 ModelRunner 使用**

**ForwardBatch 创建后进入：**


```text
ModelRunner.forward(forward_batch)
```


普通生成模型会继续进入：


```text
ModelRunner._forward_raw
  -> forward_extend 或 forward_decode
```


EXTEND 路径中：


```python
self.attn_backend.init_forward_metadata(forward_batch)

self.model.forward(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
    ...
)
```


这说明 ForwardBatch 有三类消费者：


```text
1. attention backend:
   用 seq_lens / out_cache_loc / extend_* / positions 初始化 attention metadata。

2. model.forward:
   直接消费 input_ids、positions、forward_batch。
   各层 attention / MoE / Mamba / multimodal 逻辑会从 forward_batch 读取运行时信息。

3. logits processor / sampler:
   用 forward_mode、seq_lens、positions、sampling_info、return_logprob 决定取哪些 logits 和如何采样。
```


模型 forward 后，ModelRunner.sample 会使用同一个 ForwardBatch：


```python
next_token_ids = self.sampler(
    logits_output,
    forward_batch.sampling_info,
    forward_batch.return_logprob,
    forward_batch.top_logprobs_nums,
    forward_batch.token_ids_logprobs,
    forward_batch.positions if decode else forward_batch.seq_lens - 1,
)
```


所以 ForwardBatch 不只是 attention 输入，它也贯穿 logits 和 sampling。

继续沿用 R1/R2/R3 的模拟数据，此时进入 ModelRunner.forward 的 EXTEND ForwardBatch 是：


```text
forward_mode = EXTEND
batch_size = 3 3 4input_ids: 5[11,12,13,14, 22,23,
4,35,36] 6 7positions: 8[ 0, 1, 2, 3,  1, 2,  3,
,
] 9 10seq_lens: 11[4,3,6] 12 13extend_prefix_lens: 14[0,1,3] 15 16extend_seq_lens: 17[4,2,3] 18 19extend_start_loc: 20[0,4,6] 21 22req_pool_indices: 23  [0,1,2]
```


这时 attention backend 会基于这些信息准备 metadata。可以粗略理解为：


```text
R1:
本轮 token 范围 input_ids[0:4]
写 KV 到 out_cache_loc[0:4] = [100,101,102,103]
完整 seq_len = 4

R2:
prefix cache 已有 1 个 token
本轮 token 范围 input_ids[4:6]
写 KV 到 out_cache_loc[4:6] = [104,105]
完整 seq_len = 3

R3:
prefix cache 已有 3 个 token
本轮 token 范围 input_ids[6:9]
写 KV 到 out_cache_loc[6:9] = [106,107,108]
  完整 seq_len = 6
```


图示：


```text
ForwardBatch token layout 2 3flat idx:    0   1   2   3    4   5    6   7   8 4input_ids:  11  12  13  14
2  23
4  35  36 5positions:   0   1   2   3    1   2    3


kv slot:    100 101 102 103  104 105  106 107 108
request:    └──── R1 ────┘   └ R2 ┘   └── R3 ──┘

R1 last token flat idx = 3
R2 last token flat idx = 5
R3 last token flat idx = 8
```


把 prefix cache 和本轮新写入位置合起来看，请求的完整 KV 索引大致是：


```text
R1:
tokens  [11, 12, 13, 14]
kv slot [100,101,102,103]

R2:
tokens  [21, 22, 23]
kv slot [
0,104,105] 8
R3:
tokens  [31, 32, 33, 34, 35, 36]
  kv slot [ 90, 91, 92,106,107,108]
```


这里要特别区分两个概念：


```text
out_cache_loc:
  本轮新 token 的 KV 写入位置。
  它的长度等于本轮实际 forward token 数。

req_to_token_pool.req_to_token:
  每个请求完整上下文 token -> KV slot 的映射表。
  attention 读取历史 KV 时主要依赖这张表。
```


也就是说，out\_cache\_loc 不是“读取历史 KV 的完整索引”。它更像是本轮新 KV 的写地址。历史 KV 的读取通常通过 req\_pool\_indices + seq\_lens 定位到 req\_to\_token\_pool.req\_to\_token，再从 token\_to\_kv\_pool 中读取对应 slot 的 K/V。

以 R2 为例：


```text
R2 完整上下文:
  tokens  [21, 22, 23]
  kv slot [80,104,105]

其中：
  21 是 prefix cache 命中的历史 token，KV 已经在 slot 80。
  22/23 是本轮 EXTEND 新算的 token，KV 写到 out_cache_loc[4:6] = [104,105]。
```


attention 计算时可以这样理解：


```text
1. 当前 hidden_states 计算出
2/23 的新 K/V。 22. 通过 out_cache_loc=[104,105] 把新 K/V 写入 token_to_kv_pool。
3. 当前 token 做 attention 时需要完整上下文 [21,22,23]。
4. 通过 req_to_token_pool.req_to_token 找到 [80,104,105]。
5. 从 token_to_kv_pool 读取这些 slot 的 K/V 参与 attention。
```


因此：


```text
写新 KV:
  out_cache_loc -> token_to_kv_pool.set_kv_buffer(...)

读上下文 KV:
  req_to_token_pool.req_to_token + seq_lens -> token_to_kv_pool.get_*_buffer(...)
```


**第 5 步：ForwardBatch 怎么进入模型文件**

**模型文件里并不会直接接收 ScheduleBatch 或 ModelWorkerBatch。它接收的是：**


```python
self.model.forward(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
    ...
)
```


也就是说，模型文件入口通常是三个核心参数：


```text
input_ids:
本轮要执行的 token ids。
EXTEND 时是未缓存 prompt token 拼平。
DECODE 时是每个请求上一步采样出的 token。

positions:
每个 input token 在原序列里的位置。
由 ForwardBatch.init_new 生成。

forward_batch:
  携带 attention/KV/cache/logprob/sampling/LoRA/DP 等运行时信息。
```


以 python/sglang/srt/models/llama.py 为例，调用链可以简化成：


```text
ModelRunner.forward
  -> ModelRunner.forward_extend / forward_decode
  -> LlamaForCausalLM.forward(input_ids, positions, forward_batch)
  -> LlamaModel.forward(input_ids, positions, forward_batch)
  -> embed_tokens(input_ids) 得到 hidden_states
  -> 每层 LlamaDecoderLayer.forward(positions, hidden_states, forward_batch, ...)
  -> norm
  -> logits_processor(input_ids, hidden_states, lm_head, forward_batch)
```


对应代码结构是：


```python
LlamaForCausalLM.forward:
hidden_states = self.model(
      input_ids,
      positions,
      forward_batch,
      input_embeds,
      pp_proxy_tensors=pp_proxy_tensors,
)

return self.logits_processor(
      input_ids,
      hidden_states,
      self.lm_head,
      forward_batch,
      aux_hidden_states,
  )
```


再进入 LlamaModel.forward：


```python
if first PP rank:
if input_embeds is None:
      hidden_states = self.embed_tokens(input_ids)
else:
      hidden_states = input_embeds
residual = None
else:
hidden_states = pp_proxy_tensors["hidden_states"]
residual = pp_proxy_tensors["residual"]

for layer in layers:
hidden_states, residual = layer(
      positions,
      hidden_states,
      forward_batch,
      residual,
)

hidden_states = norm(hidden_states, residual)
return hidden_states
```


这就是 ForwardBatch 到模型文件，再到第一份 hidden\_states 的关键转换：


```text
ForwardBatch.input_ids
  -> model.embed_tokens(input_ids)
  -> hidden_states
```


如果 ForwardBatch.input\_embeds 不为空，比如 multimodal、embedding override、某些 speculative / MTP 路径，则模型会跳过或替换普通 token embedding：


```text
forward_batch.input_embeds
  -> kwargs["input_embeds"]
  -> hidden_states = input_embeds
```


可视化：


```text
ForwardBatch / EXTEND 2 3input_ids:  [11,12,13,14,22,23,34,35,36] 4positions:  [ 0, 1,
,
, 1, 2, 3,
,
] 5                 │
                 v
model.embed_tokens(input_ids)
                 │
                 v
hidden_states:
shape = [num_tokens, hidden_size]
        = [9, hidden_size]
                 │
                 v
Transformer layers:
  layer(positions, hidden_states, forward_batch, residual)
```


把这 9 行 hidden\_states 展开看：


```text
hidden_states[0] = embed(11), position 0, belongs to R1
hidden_states[1] = embed(12), position 1, belongs to R1
hidden_states[2] = embed(13), position 2, belongs to R1
hidden_states[3] = embed(14), position 3, belongs to R1

hidden_states[4] = embed(22), position 1, belongs to R2
hidden_states[5] = embed(23), position 2, belongs to R2

hidden_states[6] = embed(34), position 3, belongs to R3
hidden_states[7] = embed(35), position 4, belongs to R3
hidden_states[8] = embed(36), position 5, belongs to R3
```


注意 R2/R3 的 position 不是从 0 开始，因为它们前面有 prefix cache 命中：


```text
R2 prefix_len = 1:
  本轮只算 [22,23]
  positions = [1,2]

R3 prefix_len = 3:
  本轮只算 [34,35,36]
  positions = [3,4,5]
```


也就是说，hidden\_states 的行数等于本轮实际 forward token 数，而不是原始 prompt token 数：


```text
R1 4 行 + R2
 行 + R3
 行 = 9 行
```


Decode 时同理：


```text
ForwardBatch / DECODE

input_ids:  [101,201,301]
positions:  [  4,  3,  6]
                 │
                 v
model.embed_tokens(input_ids)
                 │
                 v
hidden_states:
  shape = [batch_size, hidden_size]
        = [3, hidden_size]
```


**第 6 步：hidden\_states 如何继续使用 ForwardBatch**

**得到 hidden\_states 后，模型层仍然会持续使用 forward\_batch。  典型 layer 调用是：**


```text
hidden_states, residual = layer(
    positions,
    hidden_states,
    forward_batch,
    residual,
)
```


在 layer 内部，attention 会依赖 forward\_batch 中的运行时信息：


```text
seq_lens:
当前每个请求的序列长度。

out_cache_loc:
本轮新 KV 写入 token_to_kv_pool 的位置。
它主要是写地址，不是完整历史 KV 的读取地址。

req_pool_indices:
请求在 req_to_token_pool 中的行号。
attention 会结合 req_pool_indices、seq_lens 和 req_to_token_pool.req_to_token 找到完整上下文的 KV slot。

extend_prefix_lens / extend_seq_lens / extend_start_loc:
prefill attention 需要知道每个请求的 prefix 和本轮 token 区间。

positions / mrope_positions:
rotary embedding 或 MROPE 位置编码使用。

attn_backend metadata:
  init_forward_metadata(forward_batch) 已经根据 batch 形状准备好 backend 所需索引。
```


所以 hidden\_states 的计算不是只靠 input\_ids，而是：


```text
input_ids -> embedding -> hidden_states
positions -> positional / rotary information
forward_batch -> attention/KV/cache/runtime metadata
```


最终最后一层输出的 hidden\_states 会进入 logits processor：


```text
logits_processor(
  input_ids,
  hidden_states,
  lm_head,
  forward_batch,
)
```


logits processor 会根据 forward\_batch.forward\_mode 决定取哪些位置的 logits：


```text
DECODE:
  每个请求都有一个 decode token，通常取每行对应 logits。

EXTEND:
  通常只取每个请求最后一个 token 的 logits 用于下一 token 采样。
  如果 return_logprob，还会额外处理 prompt token logprobs。

PREFILL-ONLY:
  可能只计算 logprob / embedding，不采样 next token。
```


然后 ModelRunner.sample 使用同一个 forward\_batch.sampling\_info 采样：


```text
LogitsProcessorOutput
  -> Sampler(..., forward_batch.sampling_info, ...)
  -> next_token_ids
  -> GenerationBatchResult
  -> Scheduler.process_batch_result
  -> Req.output_ids 更新
```


继续使用 R1/R2/R3 的 EXTEND 示例，最后一层输出仍然是：


```text
final_hidden_states.shape = [9, hidden_size]
```


但 prefill 生成下一个 token 时，不需要 9 个位置都采样。通常只需要每个请求最后一个 token 的 logits：


```text
R1:
最后一个本轮 token = 14
flat idx = 3
用 final_hidden_states[3] 计算 next-token logits

R2:
最后一个本轮 token = 23
flat idx = 5
用 final_hidden_states[5] 计算 next-token logits

R3:
最后一个本轮 token = 36
flat idx = 8
  用 final_hidden_states[8] 计算 next-token logits
```


可视化：


```text
final_hidden_states rows 2 3idx:        0    1    2    3     4    5     6    7    8 4token:     11   12   13   14    22
3    34
5   36 5request:   └──── R1 ─────┘     └ R2 ┘     └── R3 ───┘ 6sample:                   ^          ^              ^ 7                         R1         R2             R3
```


假设 sampler 得到：


```text
next_token_ids = [101,
01,
01]
```


那么 GenerationBatchResult 会把它带回 Scheduler：


```text
GenerationBatchResult.next_token_ids = [101,201,301]
```


随后 Scheduler.process\_batch\_result 会更新请求状态：


```text
R1.output_ids = [101]
R2.output_ids = [201]
R3.output_ids = [301]
```


这三个请求下一轮合入 running\_batch 后，就进入 decode 生命周期。也就是后面 Decode 阶段使用的模拟数据来源。

**Decode 阶段的数据变化**

**假设上面 prefill 后采样出：**


```text
R1.output_ids = [101]
R2.output_ids = [201]
R3.output_ids = [301]
```


下一轮没有新 prefill，Scheduler 会调用：


```text
running_batch = update_running_batch(running_batch)
```


其中：


```text
batch.prepare_for_decode()
```


会把 ScheduleBatch 改成 DECODE：


```text
ScheduleBatch.forward_mode = DECODE
ScheduleBatch.input_ids = [101,201,301]
ScheduleBatch.seq_lens = [5,4,7]
ScheduleBatch.out_cache_loc = [109,110,111]
```


这里假设 allocator 继续从 EXTEND 已经使用过的 100..108 后面分配，所以 3 个 decode token 分别写到：


```text
R1 decode token 101 -> kv slot 109
R2 decode token 201 -> kv slot 110
R3 decode token 301 -> kv slot 111
```


这里的 input\_ids 含义已经变了：


```text
EXTEND input_ids:
  prompt 中没有命中 prefix cache 的 token，数量可以远大于 batch_size。

DECODE input_ids:
  每个请求一个 last output token，数量通常等于 batch_size。
```


Decode 的 ModelWorkerBatch：


```text
ModelWorkerBatch.forward_mode = DECODE
ModelWorkerBatch.input_ids = [101,201,301]
ModelWorkerBatch.req_pool_indices = [0,1,2]
ModelWorkerBatch.seq_lens = [5,4,7]
ModelWorkerBatch.out_cache_loc = [109,110,111]

ModelWorkerBatch.extend_seq_lens = None
ModelWorkerBatch.extend_prefix_lens = None
ModelWorkerBatch.extend_num_tokens = 普通 decode 路径不使用
```


Decode 的 ForwardBatch.init\_new 会走另一条位置逻辑：


```text
if forward_mode.is_decode():
  positions = clamp_position(seq_lens)
```


可以理解为：


```text
R1 seq_len = 5 -> position = 4
R2 seq_len = 4 -> position = 3
R3 seq_len = 7 -> position = 6
```


因此：


```text
ForwardBatch / DECODE

input_ids: [101,201,301]
positions: [4,3,6]
seq_lens:  [5,4,7]
out_cache_loc: [109,110,111]
```


可视化：


```text
R1: [11][12][13][14][101]
                         ^
                         decode position 4
                         kv slot 109

R2: [21][22][23][201]
                    ^
                    decode position 3
                    kv slot 110

R3: [31][32][33][34][35][36][301]
                                   ^
                                   decode position 6
                                   kv slot 111
```


decode 完成这一轮 forward 后，完整 KV 索引会继续增长：


```text
R1:
tokens  [11, 12, 13, 14,101]
kv slot [100,101,102,103,109]

R2:
tokens  [21, 22, 23,201]
kv slot [
0,104,105,110] 8
R3:
tokens  [31, 32, 33, 34, 35, 36,301]
  kv slot [ 90, 91, 92,106,107,108,111]
```


decode attention 的读写关系也是同一套逻辑。以 R2 为例：


```text
本轮 decode input token:
  201

本轮写新 KV:
  out_cache_loc = [110]

attention 读取完整上下文 KV:
  tokens  [21, 22, 23,201]
  kv slot [80,104,105,110]
```


所以 decode 阶段不是只读 out\_cache\_loc=110。110 是新 token 的写入位置；attention 读取的是 req\_to\_token\_pool.req\_to\_token 中记录的完整上下文 KV slot，包括历史的 80/104/105 和本轮新写入的 110。

**三层 Batch 的字段流转表**

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 以 EXTEND 为例：   | 信息 | ScheduleBatch | ModelWorkerBatch | ForwardBatch | | --- | --- | --- | --- | | 请求列表 | reqs=[R1,R2,R3] | reqs 保留用于 grammar/rids 等 | rids ，通常不再依赖完整 Req | | forward mode | EXTEND | EXTEND | EXTEND | | input ids | 已拼平 | 原样携带 | 原样执行 | | req pool idx | alloc\_for\_extend 得到 | 原样携带 | attention/KV 使用 | | seq lens | prepare\_for\_extend 得到 | 原样携带 | attention/logits 使用 | | out cache loc | KV allocator 分配，例如 [100..108] | 原样携带 | attention 写 KV 使用 | | prefix/extend lens | list 形式 | list 形式 | tensor + CPU list | | positions | 无 | 无 | compute\_position 得到 | | hidden\_states | 无 | 无 | 模型文件中由 embed\_tokens(input\_ids) 或 input\_embeds 生成 | | KV pool 对象 | allocator 引用 | 无 | model\_runner.token\_to\_kv\_pool | | attention backend | 无 | 无 | model\_runner.attn\_backend | | sampling info | 创建并维护 | 原样携带 | sampler 使用 |   以 DECODE 为例：   | 信息 | ScheduleBatch | ModelWorkerBatch | ForwardBatch | | --- | --- | --- | --- | | input ids | last output token | 原样携带 | 原样执行 | | seq lens | prepare\_for\_decode 中 +1 | 原样携带 | 生成 decode positions | | extend fields | 不再使用 | None | None | | positions | 无 | 无 | seq\_lens - 1 | | hidden\_states | 无 | 无 | 模型文件中由 decode token embedding 得到 | | out cache loc | 每个请求一个新 KV 位置，例如 [109,110,111] | 原样携带 | attention 写 KV 使用 | |

**三个对象分别在哪些地方使用**

**ScheduleBatch 的使用位置**

**ScheduleBatch 主要在 Scheduler 和调度相关 worker 中使用：**


```text
Scheduler.get_next_batch_to_run:
返回下一轮要跑的 ScheduleBatch。

get_new_batch_prefill:
创建 EXTEND ScheduleBatch。

update_running_batch:
把 running_batch 更新成 DECODE ScheduleBatch。

ScheduleBatch.prepare_for_extend:
构建 prefill tensor 并分配 KV。

ScheduleBatch.prepare_for_decode:
构建 decode tensor 并分配 KV。

ScheduleBatch.filter_batch / merge_batch:
维护 running_batch 生命周期。

Scheduler.process_batch_result:
根据 forward 结果更新 req.output_ids / finished / logprob。

speculative workers:
  部分 spec v1 路径仍直接接收 ScheduleBatch。
```


**ModelWorkerBatch 的使用位置**

**ModelWorkerBatch 主要出现在 Scheduler 到 worker 的边界：**


```text
ScheduleBatch.get_model_worker_batch:
创建 ModelWorkerBatch。

Scheduler.run_batch:
普通 generation 大多数路径都会先得到 ModelWorkerBatch。

overlap schedule:
record_batch_in_overlap(model_worker_batch) 保存 forward stream 需要的快照。

TPWorker.forward_batch_generation:
接收 ModelWorkerBatch 并转换为 ForwardBatch。

speculative workers:
target worker / draft worker 之间也会传 ModelWorkerBatch。

embedding path:
  forward_batch_embedding 也使用 ModelWorkerBatch。
```


**ForwardBatch 的使用位置**

**ForwardBatch 是模型执行层通用对象：**


```text
TPWorker.forward_batch_generation:
ForwardBatch.init_new(model_worker_batch, model_runner)。

ModelRunner.forward:
接收 ForwardBatch。

ModelRunner.forward_extend / forward_decode:
根据 forward_mode 分发执行。

attention backend:
init_forward_metadata(forward_batch)。

model.forward:
model.forward(input_ids, positions, forward_batch)。

logits processor:
根据 forward_mode / seq_lens / extend 信息抽取 logits。

sampler:
使用 sampling_info、positions、logprob 字段采样 next_token_ids。

cuda graph / piecewise cuda graph:
capture/replay 时也构造或复用 ForwardBatch。

DP attention:
  ForwardBatch.prepare_mlp_sync_batch 处理 global_num_tokens 和 padding。
```


**为什么不直接从 ScheduleBatch 到 ForwardBatch**

**看起来可以省掉 ModelWorkerBatch，但实际有几个原因需要中间层：**


```text
1. 隔离 Scheduler 内部状态
   ScheduleBatch 带有 reqs、tree_cache、allocator、filter/merge 等调度逻辑。
   model worker 不应该依赖这些调度细节。

2. 支持 overlap schedule
   overlap 模式下 Scheduler 线程和 forward stream 并行。
   ModelWorkerBatch 是一个更稳定的 forward 快照。

3. 支持不同 worker 类型
   TP worker、speculative worker、embedding worker、MLX worker 都可以围绕 ModelWorkerBatch 建接口。

4. 运行时对象必须由 ModelRunner 注入
   token_to_kv_pool、attn_backend、device-specific positions 不属于 Scheduler。
   它们在 ForwardBatch.init_new 中由 model_runner 注入更合理。
```


**常见容易绕的点**

**1. input\_ids 在三层中是同一批 token，但含义取决于 forward\_mode**


```text
EXTEND:
  input_ids 是未命中 prefix cache 的 prompt token 拼平。

DECODE:
  input_ids 是每个请求上一轮采样出的最后一个 token。
```


**2. positions 不在 ScheduleBatch 中**

**Scheduler 不直接生成 positions。它只提供：**


```text
seq_lens
prefix_lens
extend_lens
extend_num_tokens
```


真正的 positions 在 ForwardBatch.init\_new 中生成，因为它依赖 model runner 的后端、device 和特殊模型逻辑。

**3. out\_cache\_loc 由 Scheduler 分配，但由 attention 执行使用**


```text
ScheduleBatch.prepare_for_extend / prepare_for_decode:
  分配 out_cache_loc。

ForwardBatch:
  携带 out_cache_loc。

attention backend / attention layer:
  根据 out_cache_loc 把新 KV 写入 token_to_kv_pool。
```


**4. sampling\_info 从 ScheduleBatch 一直传到 ForwardBatch**

**sampling\_info 在 ScheduleBatch.prepare\_for\_extend 或 decode 生命周期中创建/维护，然后一路传到 ForwardBatch。采样时 ModelRunner.sample 会使用它：**


```text
temperature
top_p
top_k
min_p
presence/frequency penalty
grammar mask
logprob 参数
```


**5. ForwardBatch 是局部执行对象**

**ForwardBatch 通常只在一次 forward 中使用。forward 结束后，结果会通过 GenerationBatchResult 回到 Scheduler，再由 ScheduleBatch 和 Req 更新长期状态。**


```text
ForwardBatch:
  执行期临时对象。

ScheduleBatch / Req:
  调度和请求生命周期状态。
```


**最后串起来**

**一次普通 generation 的主路径可以总结成：**


```text
Req
-> Scheduler admission
-> ScheduleBatch.prepare_for_extend / prepare_for_decode
-> ScheduleBatch.get_model_worker_batch
-> TPWorker.forward_batch_generation
-> ForwardBatch.init_new
-> attention_backend.init_forward_metadata
-> model.forward(input_ids, positions, forward_batch)
-> embed_tokens(input_ids) / input_embeds
-> hidden_states
-> transformer layers(hidden_states, positions, forward_batch)
-> logits processor
-> sampler
-> GenerationBatchResult
-> Scheduler.process_batch_result
  -> Req.output_ids / finished 状态更新
```


理解这条链路后，再看 DP attention、overlap schedule、speculative decoding、cuda graph、multimodal、LoRA 等分支，本质上都是在这三个 Batch 对象上增加字段、补 runtime metadata，或者改变它们的转换时机。
