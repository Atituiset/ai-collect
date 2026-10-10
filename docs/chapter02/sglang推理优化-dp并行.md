# SGLang推理优化


本文梳理 SGLang 内置 Native DP 在单机上的启动、rank 映射、ZMQ 请求分发和 NCCL/TP 集合通信逻辑。

这里讨论的是 python -m sglang.launch\_server 的内置 DP，也就是 DataParallelController + launch\_dp\_schedulers 这条路径，不是 SGLang Model Gateway，也不是 DP attention。

示例配置：


```bash
--nnodes 1
--dp 2
--tp 4
```


假设单机 8 张 GPU，总共启动 2 个 DP replica，每个 replica 使用 4 张 GPU 做 TP。

**一句话理解**

**普通 Native DP 是“多个完整模型副本”。  在单机 8 卡 dp2 + tp4 下：**


```text
DP0 replica:
  使用 GPU 0,1,2,3
  内部是一个 TP4 group
  tp_rank = 0,1,2,3

DP1 replica:
  使用 GPU 4,5,6,7
  内部是另一个 TP4 group
  tp_rank = 0,1,2,3
```


注意：两个 DP replica 里的 tp\_rank 都是 0..3。它们不是同一个 TP8 world，而是两个互相独立的 TP4 world。

这和 DP attention 很不一样：


```text
普通 Native DP:
  dp2 + tp4 = 两套独立 TP4 模型副本，总共 8 卡。

DP attention:
  dp4 + tp16 = 一个 TP16 world 内切出 4 个 attention DP group。
```


**关键代码位置**

**主要涉及以下文件：**

- python/sglang/srt/managers/data\_parallel\_controller.py
- launch\_dp\_schedulers
- launch\_tensor\_parallel\_group\_thread
- launch\_tensor\_parallel\_group
- round\_robin\_scheduler
- total\_requests\_scheduler
- total\_tokens\_scheduler
- python/sglang/srt/managers/scheduler.py
- init\_ipc\_channels
- recv\_requests
- python/sglang/srt/server\_args.py
- PortArgs.init\_new
- \_handle\_data\_parallelism
- python/sglang/srt/distributed/parallel\_state.py
- initialize\_model\_parallel

**总 GPU 数怎么理解**

**普通 Native DP 下，总 GPU 数大致是：**


```text
total_gpus = dp_size * tp_size * pp_size
```


本例没有 PP：


```text
dp_size = 2
tp_size = 4
pp_size = 1

total_gpus = 2 * 4 * 1 = 8
```


如果只写 --dp 2 且 --tp 仍是默认 1，那么只会是 2 个单卡副本，不会自动用满 8 卡。单机 8 卡 DP2 的典型配置应该显式设置 --tp 4。

**launch\_dp\_schedulers 做了什么**

**普通 DP 入口在 DataParallelController.\_\_init\_\_：**


```python
if server_args.enable_dp_attention:
    self.launch_dp_attention_schedulers(server_args, port_args)
else:
    self.launch_dp_schedulers(server_args, port_args)
    self.control_message_step = 1
```


不开启 DP attention 时，会进入：


```python
def launch_dp_schedulers(self, server_args, port_args):
    base_gpu_id = 0

    for dp_rank in range(server_args.dp_size):
        tmp_port_args = PortArgs.init_new(server_args)
        tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
        tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

        sockets.append(bind_port(tmp_port_args.nccl_port))

        thread = threading.Thread(
            target=self.launch_tensor_parallel_group_thread,
            args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
        )

        base_gpu_id += (
            server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
        )

        self.workers[dp_rank] = get_zmq_socket(
            self.context,
            zmq.PUSH,
            tmp_port_args.scheduler_input_ipc_name,
            True,
        )
```


这段可以拆成 5 步。

**第 1 步：按 dp\_rank 循环**

**本例 dp\_size=2：**


```text
第一次循环:
  dp_rank = 0

第二次循环:
  dp_rank = 1
```


每个 dp\_rank 都会启动一套完整 TP group。

**第 2 步：给每个 DP replica 创建独立 PortArgs**

**普通 DP 下：**


```text
tmp_port_args = PortArgs.init_new(server_args)
```


PortArgs.init\_new 在非 DP attention 下会生成一组本机 IPC / port：


```text
return PortArgs(
    tokenizer_ipc_name="ipc://...",
    scheduler_input_ipc_name="ipc://...",
    detokenizer_ipc_name="ipc://...",
    nccl_port=nccl_port,
    rpc_ipc_name="ipc://...",
    metrics_ipc_name="ipc://...",
)
```


随后代码把 tokenizer/detokenizer 的 IPC 改回共享的主通道：


```text
tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
```


所以：


```text
scheduler_input_ipc_name:
每个 DP replica 独立。
DataParallelController 用它把请求发给某个 DP。

nccl_port:
每个 DP replica 独立。
每个 DP 内部建立自己的 TP process group。

tokenizer_ipc_name / detokenizer_ipc_name:
多个 DP replica 共享。
  输出统一回到同一个 tokenizer/detokenizer 路径。
```


**第 3 步：预占每个 DP 的 NCCL port**

**代码：**


```text
sockets.append(bind_port(tmp_port_args.nccl_port))
```


这一步是为了避免多个 DP replica 在启动前拿到冲突端口。

随后在真正启动线程前释放：


```text
for sock in sockets:
    sock.close()
```


含义是：


```text
DP0 有自己的 nccl_port，比如 31001。
DP1 有自己的 nccl_port，比如
1002。 3
两个 DP replica 的 NCCL/Torch distributed world 是隔离的。
```


**第 4 步：给每个 DP replica 创建 controller -> scheduler 的 ZMQ PUSH socket**

**代码：**


```python
self.workers[dp_rank] = get_zmq_socket(
    self.context,
    zmq.PUSH,
    tmp_port_args.scheduler_input_ipc_name,
    True,
)
```


self.workers 是 DataParallelController 手里的 worker socket 列表：


```python
self.workers[0] -> DP0 scheduler input
self.workers[1] -> DP1 scheduler input
```


后续负载均衡就是选择 self.workers[i]，然后：


```python
self.workers[target_worker].send_pyobj(req)
```


**第 5 步：base\_gpu\_id 递增，保证不同 DP replica 用不同 GPU**

**代码：**


```text
base_gpu_id += (
    server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
)
```


本例：


```text
tp_size = 4
pp_size = 1
gpu_id_step = 1

每个 DP replica 占 4 张 GPU。
```


所以：


```text
DP0:
  base_gpu_id = 0
  使用 GPU 0,1,2,3 4 5DP1: 6  base_gpu_id = 4 7  使用 GPU
,5,6,7
```


**launch\_tensor\_parallel\_group 做了什么**

**普通 DP 下，launch\_tensor\_parallel\_group 是被每个 DP replica 分别调用的：**


```text
DP0 thread -> launch_tensor_parallel_group(..., base_gpu_id=0, dp_rank=0)
DP1 thread -> launch_tensor_parallel_group(..., base_gpu_id=4, dp_rank=1)
```


**第 1 步：算当前 node 的 TP rank 范围**

**单机 nnodes=1、tp\_size=4：**


```text
nnodes_per_tp_group = 1
tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
tp_rank_range = range(0, 4)
```


所以每个 DP replica 都会启动：


```text
tp_rank = 0,1,2,3
```


注意：DP0 和 DP1 的 tp\_rank 都是 0..3。它们属于不同进程组，所以 rank 编号可以重复。

**第 2 步：普通 DP 不会从 tp\_rank 反推 dp\_rank**

**DP attention 模式下，dp\_rank 是由 tp\_rank 反推的。  普通 DP 模式下不是这样。dp\_rank 来自外层 launch\_dp\_schedulers：**


```text
DP0 thread 传入 dp_rank=0
DP1 thread 传入 dp_rank=1
```


所以这段 DP attention 专用逻辑不会执行：


```text
if server_args.enable_dp_attention:
    _, _, dp_rank = compute_dp_attention_world_info(...)
    rank_port_args = PortArgs.init_new(server_args, dp_rank, worker_ports)
    rank_port_args.nccl_port = port_args.nccl_port
```


普通 DP 下：


```text
rank_port_args = port_args
```


也就是说，一个 DP replica 内的所有 TP rank 共用同一个 rank\_port\_args，它们使用同一个 scheduler input IPC 和同一个 NCCL port。

**第 3 步：计算 GPU id**

**代码：**


```text
gpu_id = (
    server_args.base_gpu_id
    + base_gpu_id
    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
)
```


本例没有 PP，gpu\_id\_step=1：


```text
DP0, base_gpu_id=0:
tp_rank 0 -> GPU 0
tp_rank 1 -> GPU 1
tp_rank 2 -> GPU 2
tp_rank 3 -> GPU 3

DP1, base_gpu_id=4:
tp_rank 0 -> GPU 4
tp_rank 1 -> GPU 5
tp_rank 2 -> GPU 6
  tp_rank 3 -> GPU 7
```


**第 4 步：计算 attention / MoE rank**

**普通 DP 下：**


```text
attn_dp_size = 1
attn_tp_size = server_args.tp_size // 1 // server_args.attn_cp_size
```


如果 attn\_cp\_size=1：


```text
attn_tp_size = tp_size = 4
attn_tp_rank = tp_rank
attn_dp_rank = 0
```


也就是说，在普通 DP replica 内：


```text
attention TP group == TP group
attention DP size == 1
```


普通 DP 的“DP”是在多个 replica 之间做请求分流，不是在 attention 层内部切 DP。

**第 5 步：启动 scheduler 进程**

**每个 TP rank 都会启动一个 scheduler 进程：**


```python
proc = mp.Process(
    target=self.run_scheduler_process_func,
    args=(
        server_args,
        rank_port_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        writer,
    ),
)
```


本例最终进程表可以理解成：


```text
DP replica  local GPU  tp_rank  dp_rank  nccl world  scheduler input
DP0         GPU 0      0        0        DP0 TP4     DP0 port
DP0         GPU 1      1        0        DP0 TP4     DP0 port
DP0         GPU 2      2        0        DP0 TP4     DP0 port
DP0         GPU 3      3        0        DP0 TP4     DP0 port

DP1         GPU 4      0        1        DP1 TP4     DP1 port
DP1         GPU 5      1        1        DP1 TP4     DP1 port
DP1         GPU 6      2        1        DP1 TP4     DP1 port
DP1         GPU 7      3        1        DP1 TP4     DP1 port
```


**ZMQ 请求路径**

**请求从 tokenizer/http frontend 到 DP replica 的路径是：**

**``` 1Tokenizer / HTTP frontend 2**

DataParallelController 支持几种分发策略：


```python
dispatch_lookup = {
    LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
    LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
    LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
    LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
}
```


普通 round-robin 逻辑：


```python
if self.status[self.round_robin_counter]:
    self.workers[self.round_robin_counter].send_pyobj(req)
    self.round_robin_counter = (
        self.round_robin_counter + 1
    ) % len(self.workers)
```


本例就是：


```text
request 0 -> DP0
request 1 -> DP1
request 2 -> DP0
request 3 -> DP1
...
```


如果请求显式带了 routed\_dp\_rank：


```python
if req.routed_dp_rank is not None:
    self.workers[req.routed_dp_rank].send_pyobj(req)
```


则可以直接路由到指定 DP replica。

**Scheduler 内部如何收请求**

**每个 DP replica 内部有 4 个 TP rank，但不是每个 rank 都直接从 ZMQ 收请求。  Scheduler.init\_ipc\_channels：**


```python
if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
    self.recv_from_tokenizer = get_zmq_socket(
        context, zmq.PULL, port_args.scheduler_input_ipc_name, False
    )
    self.recv_from_rpc = get_zmq_socket(
        context, zmq.DEALER, port_args.rpc_ipc_name, False
    )
else:
    self.recv_from_tokenizer = None
    self.recv_from_rpc = None
```


普通 DP 下 attn\_tp\_rank == tp\_rank，所以只有每个 DP replica 的 tp\_rank=0 会收请求：


```text
DP0:
GPU0 / tp_rank0 直接收 ZMQ 请求
GPU1 / tp_rank1 不直接收
GPU2 / tp_rank2 不直接收
GPU3 / tp_rank3 不直接收

DP1:
GPU4 / tp_rank0 直接收 ZMQ 请求
GPU5 / tp_rank1 不直接收
GPU6 / tp_rank2 不直接收
  GPU7 / tp_rank3 不直接收
```


随后 Scheduler.recv\_requests 会把 rank0 收到的请求 broadcast 给同一个 TP group 内的其他 rank：


```python
elif self.tp_size != 1:
    recv_reqs = broadcast_pyobj(
        recv_reqs,
        self.tp_group.rank,
        self.tp_cpu_group,
        src=self.tp_group.ranks[0],
    )
```


所以 DP0 内部是：

**``` 1DP0 rank0 收 ZMQ 请求 2**

DP1 内部同理。

**NCCL / TP 集合通信逻辑**

**普通 Native DP 下，两个 DP replica 没有跨 DP 的 NCCL collective。  每个 DP replica 内部单独初始化 TP group：**


```text
DP0 NCCL world:
  GPU 0,1,2,3
  tp_rank 0,1,2,3 4 5DP1 NCCL world: 6  GPU
,5,6,7 7  tp_rank 0,1,2,3
```


每个 replica 内部的模型计算和集合通信就是普通 TP：


```text
tensor parallel all-reduce
all-gather
reduce-scatter
attention TP broadcast
logits / sampler 相关同步
```


但这些通信只发生在当前 DP replica 的 TP4 world 内。

对比 DP attention：


```text
普通 DP:
  DP0 和 DP1 是两个独立模型副本。
  DP0 的 TP collective 不需要 DP1 参与。
  DP1 没请求时，DP1 可以 idle，不会影响 DP0 forward。

DP attention:
  所有 rank 在同一个 TP world 内。
  attention 可分 DP group，但 MLP/MoE 前后需要同步 token 元信息。
  idle DP 也可能要进入 collective，避免其他 rank hang。
```


因此普通 DP 没有 prepare\_mlp\_sync\_batch\_raw 这套跨 DP token 元信息同步，也没有 DP attention 里的 global\_num\_tokens、DP padding mode、DP hidden states gather/scatter 逻辑。

**KV cache 和模型权重的含义**

**普通 Native DP 是完整模型副本。  在 dp2 + tp4 下：**


```text
DP0:
  拥有一套模型权重分片。
  TP4 内每张 GPU 持有当前 TP rank 对应的权重 shard。
  DP0 管自己的请求和 KV cache。

DP1:
  拥有另一套模型权重分片。
  TP4 内每张 GPU 持有当前 TP rank 对应的权重 shard。
  DP1 管自己的请求和 KV cache。
```


DP0 和 DP1 之间不共享请求状态，也不共享 KV cache。

这也是普通 DP 的核心取舍：


```text
优点:
  简单，隔离性强。
  一个 DP replica 忙，不要求其他 DP replica 同步参与。
  适合吞吐扩展。

代价:
  模型权重和 KV cache 按 DP replica 复制。
  同样 8 张卡，dp2+tp4 每个请求最多只用其中一个 TP4 replica。
```


**框架图**


```text
                         ┌────────────────────────────┐
                         │ Tokenizer / HTTP frontend   │
                         └──────────────┬─────────────┘
                                        │ ZMQ
                                        v
                         ┌────────────────────────────┐
                         │ DataParallelController     │
                         │ workers[0] -> DP0 port     │
                         │ workers[1] -> DP1 port     │
                         └──────┬──────────────┬──────┘
                                │              │
                                │ ZMQ          │ ZMQ
                                v              v
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ DP0 replica                           │  │ DP1 replica                           │
│ NCCL world A, TP4                     │  │ NCCL world B, TP4                     │
│ scheduler_input = DP0 port            │  │ scheduler_input = DP1 port            │
│                                       │  │                                       │
│ GPU0 tp_rank0  ZMQ receiver / leader  │  │ GPU4 tp_rank0  ZMQ receiver / leader  │
│ GPU1 tp_rank1                         │  │ GPU5 tp_rank1                         │
│ GPU2 tp_rank2                         │  │ GPU6 tp_rank2                         │
│ GPU3 tp_rank3                         │  │ GPU7 tp_rank3                         │
│                                       │  │                                       │
│ rank0 broadcasts requests             │  │ rank0 broadcasts requests             │
│ within DP0 TP group                   │  │ within DP1 TP group                   │
└──────────────────────────────────────┘  └──────────────────────────────────────┘
```


**请求的完整链路**


```text
1. Tokenizer / HTTP frontend 产生请求。

2. DataParallelController 通过 ZMQ 收到请求。

3. DataParallelController 根据负载均衡策略选择 DP0 或 DP1。

4. 请求通过 ZMQ 发到对应 DP replica 的 scheduler input IPC。

5. 该 DP replica 内部只有 tp_rank0 直接收请求。

6. tp_rank0 把请求 broadcast 给同 replica 内其他 TP rank。

7. 这个 DP replica 的 TP4 group 独立执行 model forward。

8. forward 内部的 NCCL collective 只发生在当前 DP replica 的 4 张 GPU 内。

9. 输出通过共享 tokenizer/detokenizer IPC 回到上层。
```


**SGLang Model Gateway 如何实现类似 DP 扩展**

**SGLang Model Gateway，简称 SMG，提供的是另一种更现代的 DP 扩展方式。  Native DP 是：**


```text
一个 sglang.launch_server 进程树
  -> 内部 DataParallelController
  -> 内部启动多个 DP replica
  -> Controller 用 ZMQ 把请求分给 replica
```


SMG 是：


```text
一个 Rust gateway / router
  -> 管理多个独立 SGLang worker server
  -> 通过 HTTP 或 gRPC 把请求路由到某个 worker
```


也就是说，SMG 不把多个 DP replica 塞进同一个 SGLang server 的内部进程结构里，而是把它们提升为多个独立 worker 实例，再由 gateway 统一做路由。

**Co-launch 模式**

**SMG 的 Python launcher 支持一条命令同时启动 router 和多个 worker：**


```bash
python -m sglang_router.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 2 \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```


入口在：


```text
sgl-model-gateway/bindings/python/src/sglang_router/launch_server.py
```


核心逻辑是：


```text
worker_ports = find_available_ports(
    args.router_dp_worker_base_port, server_args.dp_size
)

for i, worker_port in enumerate(worker_ports):
    proc = launch_server_process(server_args, worker_port, i)
    server_processes.append(proc)

protocol = "grpc" if server_args.grpc_mode else "http"
router_args.worker_urls = [
    f"{protocol}://{server_args.host}:{port}" for port in worker_ports
]

launch_router(router_args)
```


每个 worker 启动时会改写参数：


```text
server_args.port = worker_port
server_args.base_gpu_id = dp_id * server_args.tp_size
server_args.dp_size = 1
```


所以单机 8 卡 dp2 + tp4 在 SMG co-launch 下可以理解为：


```text
Router:
对外监听 port
0000 3
Worker 0:
独立 sglang server
port = 31000 附近的某个可用端口
base_gpu_id = 0
dp_size = 1
tp_size = 4
使用 GPU 0,1,2,3

Worker 1:
独立 sglang server
port = 31000 附近的另一个可用端口
base_gpu_id = 4
dp_size = 1
tp_size = 4
  使用 GPU 4,5,6,7
```


注意这里 worker 内部已经不是 Native DP 了：


```text
每个 worker 自己是普通 tp4 server；
DP 这个概念由 router 层实现；
worker 之间通过 HTTP/gRPC 隔离。
```


**Separate launch 模式**

**也可以先独立启动 worker，再让 router 连接它们：**


```bash
# Worker 0
python -m sglang.launch_server \
--model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--tp-size 4 \
--base-gpu-id 0 \
--port 8000

# Worker 1
python -m sglang.launch_server \
--model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--tp-size 4 \
--base-gpu-id 4 \
--port 8001

# Router
python -m sglang_router.launch_router \
--worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
--policy cache_aware \
--host 0.0.0.0 \
--port 30000
```


这时请求路径是：

**``` 1Client 2**

这和 Native DP 的形态很像，都是“两个 TP4 副本做吞吐扩展”。区别是 Native DP 的 router 是 SGLang server 内部的 DataParallelController，SMG 的 router 是独立 Rust gateway。

**SMG 的 worker registry 和 policy**

**SMG 会把 worker 注册到 worker registry：**


```text
sgl-model-gateway/src/core/worker_registry.rs
```


这个 registry 按 model 维护 worker 索引，并支持：


```text
按 model 查 worker
按健康状态过滤 worker
按 worker type 区分 regular / prefill / decode
为 consistent hashing 维护 hash ring
```


路由策略由 policy factory 创建：


```text
sgl-model-gateway/src/policies/factory.rs
```


支持的策略包括：


```text
random
round_robin
power_of_two
cache_aware
bucket
manual
consistent_hashing
prefix_hash
```


其中 cache\_aware 是 SMG 相比 Native DP 的一个重要能力。它维护近似 prefix tree，根据请求前缀和已有 cache affinity 选择 worker：


```text
如果系统负载均衡:
  尽量把相似 prefix 的请求路由到可能已经有 prefix cache 的 worker。

如果系统负载不均衡:
  切到负载均衡逻辑，把请求发给更空闲的 worker。
```


这比 Native DP 的 round\_robin / total\_requests / total\_tokens 更适合真实在线流量，因为 LLM serving 的性能经常受 prefix cache 命中率影响。

**DP-aware worker discovery**

**SMG 也有 DP-aware discovery：**


```text
sgl-model-gateway/src/core/steps/worker/local/discover_dp.rs
```


核心逻辑：


```text
let info = get_server_info(url, api_key).await?;
let dp_size = info.dp_size.ok_or_else(...)?;
let model_id = info.model_id
    .or(info.served_model_name)
    .or_else(...)
```


它会从 worker 的 server info 里发现：


```text
dp_size
model_id
served_model_name
model_path
```


这让 gateway 可以在更复杂部署中理解 worker 的 DP 信息，而不是只把 worker 当成一个普通 URL。

**为什么 SMG 更好**

**Native DP 的优点是实现简单：一个 SGLang server 内部起多个 replica，内部 ZMQ 分发请求。但它的问题也来自这个设计。  SMG 更适合生产的原因主要有这些。  第一，路由策略更强。  Native DP 主要是：**


```text
round_robin
total_requests
total_tokens
follow_bootstrap_room
```


SMG 支持：


```text
cache_aware
power_of_two
prefix_hash
manual
consistent_hashing
bucket
round_robin
random
```


其中 cache-aware / prefix-hash 对 prefix cache 友好，可以减少重复 prefill，提高缓存命中。

第二，worker 是独立 server，隔离性更好。

Native DP：


```text
一个父进程树内部管理多个 DP replica。
某个 replica 的异常更容易影响整个 server 生命周期。
升级、替换、下线粒度较粗。
```


SMG：


```text
每个 worker 是独立 SGLang server。
router 只通过 HTTP/gRPC 和 worker 通信。
worker 可以独立重启、扩缩容、替换。
```


第三，故障处理更完善。

SMG 有 circuit breaker 和 retry：


```text
sgl-model-gateway/src/core/circuit_breaker.rs
sgl-model-gateway/src/core/retry.rs
```


它可以对失败 worker 做熔断、半开探测、重试和 backoff。Native DP 的 DataParallelController 更偏本地进程内分发，不是完整的生产 gateway 故障治理系统。

第四，观测和运维能力更强。

SMG 是独立 gateway，天然适合做：


```text
worker selection metrics
router latency metrics
error metrics
worker health check
multi-worker registry
multi-model routing
Kubernetes service discovery
mesh / multi-router 扩展
```


Native DP 更像 SGLang server 内部的一个简单 DP controller。

第五，部署形态更灵活。

Native DP 受限于单个 SGLang launch 进程树，并且本地代码里明确限制：


```python
assert not (
    self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention
), "multi-node data parallel is not supported unless dp attention!"
```


也就是说，普通 Native DP 不支持多机 DP。

SMG 可以连接任意多个 worker URL：


```text
同机多个 worker
多机多个 worker
Kubernetes 多 pod
HTTP worker
gRPC worker
PD disaggregation worker
```


因此它更适合作为真正的生产 DP routing 层。

**Native DP 和 SMG 的对比**


```text
实现位置:
Native DP:
    SGLang server 内部，Python DataParallelController。
SMG:
    独立 Rust gateway，外部管理多个 worker server。

worker 形态:
Native DP:
    一个 launch_server 内部多个 replica。
SMG:
    多个独立 launch_server worker。

通信方式:
Native DP:
    Controller 到 scheduler 使用 ZMQ IPC。
SMG:
    Router 到 worker 使用 HTTP/gRPC。

NCCL:
Native DP:
    每个 replica 内部一个 TP NCCL world。
SMG:
    每个 worker 内部自己建立 TP NCCL world。
    Router 不参与 NCCL。

负载均衡:
Native DP:
    简单 in-process 方法。
SMG:
    cache-aware、power-of-two、prefix-hash、consistent hashing 等。

故障隔离:
Native DP:
    replica 在同一 server 进程树内。
SMG:
    worker 独立，gateway 可跳过不健康 worker。

生产推荐:
Native DP:
    更适合理解机制或少量旧框架兼容。
SMG:
    推荐用于生产 DP routing。
```


**和 DP attention 的核心区别**


```text
维度:
普通 DP:
    DP 是外层多个完整 replica。

DP attention:
    DP 是折叠在全局 TP world 里的 attention 维度。

进程组:
普通 DP:
    dp2 + tp4 = 两个独立 TP4 NCCL world。

DP attention:
    dp4 + tp16 = 一个 TP16 NCCL world。

请求分发:
普通 DP:
    controller 把请求发给某个完整 replica。

DP attention:
    controller 把请求发给某个 attention DP group leader。

跨 DP 同步:
普通 DP:
    没有跨 DP 的 MLP/MoE token 同步。

DP attention:
    MLP/MoE 前后需要同步 global_num_tokens，并按 DP slice gather/scatter hidden_states。

KV cache:
普通 DP:
    每个 DP replica 独立持有自己的 KV cache。

DP attention:
    attention DP group 维度参与 KV / hidden states 的特殊组织。
```


**总结**


```text
ZMQ 管请求:
DataParallelController 把请求发给 DP0 或 DP1。

rank 逻辑管副本:
DP0 是一套 TP4，DP1 是另一套 TP4。
两套 TP rank 都是 0..3，但属于不同 NCCL world。

NCCL 管模型并行:
每个 DP replica 内部做自己的 TP collective。
DP0 和 DP1 之间不做模型 forward collective。

普通 DP 的本质:
  用更多完整副本提升吞吐，用显存复制换隔离和简单调度。
```

