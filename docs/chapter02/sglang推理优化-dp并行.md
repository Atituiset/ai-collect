# SGLang推理优化


本文梳理 SGLang 内置 Native DP 在单机上的启动、rank 映射、ZMQ 请求分发和 NCCL/TP 集合通信逻辑。

这里讨论的是 python -m sglang.launch\_server 的内置 DP，也就是 DataParallelController + launch\_dp\_schedulers 这条路径，不是 SGLang Model Gateway，也不是 DP attention。

示例配置：

**``` 1--nnodes 1 2--dp 2 3--tp 4 ```**

假设单机 8 张 GPU，总共启动 2 个 DP replica，每个 replica 使用 4 张 GPU 做 TP。

**一句话理解**

**普通 Native DP 是“多个完整模型副本”。  在单机 8 卡 dp2 + tp4 下：**

**``` 1DP0 replica: 2  使用 GPU 0,1,2,3 3  内部是一个 TP4 group 4  tp_rank = 0,1,2,3 5 6DP1 replica: 7  使用 GPU 4,5,6,7 8  内部是另一个 TP4 group 9  tp_rank = 0,1,2,3 ```**

注意：两个 DP replica 里的 tp\_rank 都是 0..3。它们不是同一个 TP8 world，而是两个互相独立的 TP4 world。

这和 DP attention 很不一样：

**``` 1普通 Native DP: 2  dp2 + tp4 = 两套独立 TP4 模型副本，总共 8 卡。 3 4DP attention: 5  dp4 + tp16 = 一个 TP16 world 内切出 4 个 attention DP group。 ```**

**关键代码位置**

**主要涉及以下文件：   - python/sglang/srt/managers/data\_parallel\_controller.py  - launch\_dp\_schedulers - launch\_tensor\_parallel\_group\_thread - launch\_tensor\_parallel\_group - round\_robin\_scheduler - total\_requests\_scheduler - total\_tokens\_scheduler  - python/sglang/srt/managers/scheduler.py  - init\_ipc\_channels - recv\_requests  - python/sglang/srt/server\_args.py  - PortArgs.init\_new - \_handle\_data\_parallelism  - python/sglang/srt/distributed/parallel\_state.py  - initialize\_model\_parallel**

**总 GPU 数怎么理解**

**普通 Native DP 下，总 GPU 数大致是：**

**``` 1total_gpus = dp_size * tp_size * pp_size ```**

本例没有 PP：

**``` 1dp_size = 2 2tp_size = 4 3pp_size = 1 4 5total_gpus = 2 * 4 * 1 = 8 ```**

如果只写 --dp 2 且 --tp 仍是默认 1，那么只会是 2 个单卡副本，不会自动用满 8 卡。单机 8 卡 DP2 的典型配置应该显式设置 --tp 4。

**launch\_dp\_schedulers 做了什么**

**普通 DP 入口在 DataParallelController.\_\_init\_\_：**

**``` 1if server_args.enable_dp_attention: 2    self.launch_dp_attention_schedulers(server_args, port_args) 3else: 4    self.launch_dp_schedulers(server_args, port_args) 5    self.control_message_step = 1 ```**

不开启 DP attention 时，会进入：

**``` 1def launch_dp_schedulers(self, server_args, port_args): 2    base_gpu_id = 0 3 4    for dp_rank in range(server_args.dp_size): 5        tmp_port_args = PortArgs.init_new(server_args) 6        tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name 7        tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name 8 9        sockets.append(bind_port(tmp_port_args.nccl_port)) 10 11        thread = threading.Thread( 12            target=self.launch_tensor_parallel_group_thread, 13            args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event), 14        ) 15 16        base_gpu_id += ( 17            server_args.tp_size * server_args.pp_size * server_args.gpu_id_step 18        ) 19 20        self.workers[dp_rank] = get_zmq_socket( 21            self.context, 22            zmq.PUSH, 23            tmp_port_args.scheduler_input_ipc_name, 24            True, 25        ) ```**

这段可以拆成 5 步。

**第 1 步：按 dp\_rank 循环**

**本例 dp\_size=2：**

**``` 1第一次循环: 2  dp_rank = 0 3 4第二次循环: 5  dp_rank = 1 ```**

每个 dp\_rank 都会启动一套完整 TP group。

**第 2 步：给每个 DP replica 创建独立 PortArgs**

**普通 DP 下：**

**``` 1tmp_port_args = PortArgs.init_new(server_args) ```**

PortArgs.init\_new 在非 DP attention 下会生成一组本机 IPC / port：

**``` 1return PortArgs( 2    tokenizer_ipc_name="ipc://...", 3    scheduler_input_ipc_name="ipc://...", 4    detokenizer_ipc_name="ipc://...", 5    nccl_port=nccl_port, 6    rpc_ipc_name="ipc://...", 7    metrics_ipc_name="ipc://...", 8) ```**

随后代码把 tokenizer/detokenizer 的 IPC 改回共享的主通道：

**``` 1tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name 2tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name ```**

所以：

**``` 1scheduler_input_ipc_name: 2每个 DP replica 独立。 3DataParallelController 用它把请求发给某个 DP。 4 5nccl_port: 6每个 DP replica 独立。 7每个 DP 内部建立自己的 TP process group。 8 9tokenizer_ipc_name / detokenizer_ipc_name: 10多个 DP replica 共享。 11  输出统一回到同一个 tokenizer/detokenizer 路径。 ```**

**第 3 步：预占每个 DP 的 NCCL port**

**代码：**

**``` 1sockets.append(bind_port(tmp_port_args.nccl_port)) ```**

这一步是为了避免多个 DP replica 在启动前拿到冲突端口。

随后在真正启动线程前释放：

**``` 1for sock in sockets: 2    sock.close() ```**

含义是：

**``` 1DP0 有自己的 nccl_port，比如 31001。 2DP1 有自己的 nccl_port，比如 31002。 3 4两个 DP replica 的 NCCL/Torch distributed world 是隔离的。 ```**

**第 4 步：给每个 DP replica 创建 controller -> scheduler 的 ZMQ PUSH socket**

**代码：**

**``` 1self.workers[dp_rank] = get_zmq_socket( 2    self.context, 3    zmq.PUSH, 4    tmp_port_args.scheduler_input_ipc_name, 5    True, 6) ```**

self.workers 是 DataParallelController 手里的 worker socket 列表：

**``` 1self.workers[0] -> DP0 scheduler input 2self.workers[1] -> DP1 scheduler input ```**

后续负载均衡就是选择 self.workers[i]，然后：

**``` 1self.workers[target_worker].send_pyobj(req) ```**

**第 5 步：base\_gpu\_id 递增，保证不同 DP replica 用不同 GPU**

**代码：**

**``` 1base_gpu_id += ( 2    server_args.tp_size * server_args.pp_size * server_args.gpu_id_step 3) ```**

本例：

**``` 1tp_size = 4 2pp_size = 1 3gpu_id_step = 1 4 5每个 DP replica 占 4 张 GPU。 ```**

所以：

**``` 1DP0: 2  base_gpu_id = 0 3  使用 GPU 0,1,2,3 4 5DP1: 6  base_gpu_id = 4 7  使用 GPU 4,5,6,7 ```**

**launch\_tensor\_parallel\_group 做了什么**

**普通 DP 下，launch\_tensor\_parallel\_group 是被每个 DP replica 分别调用的：**

**``` 1DP0 thread -> launch_tensor_parallel_group(..., base_gpu_id=0, dp_rank=0) 2DP1 thread -> launch_tensor_parallel_group(..., base_gpu_id=4, dp_rank=1) ```**

**第 1 步：算当前 node 的 TP rank 范围**

**单机 nnodes=1、tp\_size=4：**

**``` 1nnodes_per_tp_group = 1 2tp_size_per_node = server_args.tp_size // nnodes_per_tp_group 3tp_rank_range = range(0, 4) ```**

所以每个 DP replica 都会启动：

**``` 1tp_rank = 0,1,2,3 ```**

注意：DP0 和 DP1 的 tp\_rank 都是 0..3。它们属于不同进程组，所以 rank 编号可以重复。

**第 2 步：普通 DP 不会从 tp\_rank 反推 dp\_rank**

**DP attention 模式下，dp\_rank 是由 tp\_rank 反推的。  普通 DP 模式下不是这样。dp\_rank 来自外层 launch\_dp\_schedulers：**

**``` 1DP0 thread 传入 dp_rank=0 2DP1 thread 传入 dp_rank=1 ```**

所以这段 DP attention 专用逻辑不会执行：

**``` 1if server_args.enable_dp_attention: 2    _, _, dp_rank = compute_dp_attention_world_info(...) 3    rank_port_args = PortArgs.init_new(server_args, dp_rank, worker_ports) 4    rank_port_args.nccl_port = port_args.nccl_port ```**

普通 DP 下：

**``` 1rank_port_args = port_args ```**

也就是说，一个 DP replica 内的所有 TP rank 共用同一个 rank\_port\_args，它们使用同一个 scheduler input IPC 和同一个 NCCL port。

**第 3 步：计算 GPU id**

**代码：**

**``` 1gpu_id = ( 2    server_args.base_gpu_id 3    + base_gpu_id 4    + ((pp_rank % pp_size_per_node) * tp_size_per_node) 5    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step 6) ```**

本例没有 PP，gpu\_id\_step=1：

**``` 1DP0, base_gpu_id=0: 2tp_rank 0 -> GPU 0 3tp_rank 1 -> GPU 1 4tp_rank 2 -> GPU 2 5tp_rank 3 -> GPU 3 6 7DP1, base_gpu_id=4: 8tp_rank 0 -> GPU 4 9tp_rank 1 -> GPU 5 10tp_rank 2 -> GPU 6 11  tp_rank 3 -> GPU 7 ```**

**第 4 步：计算 attention / MoE rank**

**普通 DP 下：**

**``` 1attn_dp_size = 1 2attn_tp_size = server_args.tp_size // 1 // server_args.attn_cp_size ```**

如果 attn\_cp\_size=1：

**``` 1attn_tp_size = tp_size = 4 2attn_tp_rank = tp_rank 3attn_dp_rank = 0 ```**

也就是说，在普通 DP replica 内：

**``` 1attention TP group == TP group 2attention DP size == 1 ```**

普通 DP 的“DP”是在多个 replica 之间做请求分流，不是在 attention 层内部切 DP。

**第 5 步：启动 scheduler 进程**

**每个 TP rank 都会启动一个 scheduler 进程：**

**``` 1proc = mp.Process( 2    target=self.run_scheduler_process_func, 3    args=( 4        server_args, 5        rank_port_args, 6        gpu_id, 7        tp_rank, 8        attn_cp_rank, 9        moe_dp_rank, 10        moe_ep_rank, 11        pp_rank, 12        dp_rank, 13        writer, 14    ), 15) ```**

本例最终进程表可以理解成：

**``` 1DP replica  local GPU  tp_rank  dp_rank  nccl world  scheduler input 2DP0         GPU 0      0        0        DP0 TP4     DP0 port 3DP0         GPU 1      1        0        DP0 TP4     DP0 port 4DP0         GPU 2      2        0        DP0 TP4     DP0 port 5DP0         GPU 3      3        0        DP0 TP4     DP0 port 6 7DP1         GPU 4      0        1        DP1 TP4     DP1 port 8DP1         GPU 5      1        1        DP1 TP4     DP1 port 9DP1         GPU 6      2        1        DP1 TP4     DP1 port 10DP1         GPU 7      3        1        DP1 TP4     DP1 port ```**

**ZMQ 请求路径**

**请求从 tokenizer/http frontend 到 DP replica 的路径是：**

**``` 1Tokenizer / HTTP frontend 2**

DataParallelController 支持几种分发策略：

**``` 1dispatch_lookup = { 2    LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler, 3    LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler, 4    LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler, 5    LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler, 6} ```**

普通 round-robin 逻辑：

**``` 1if self.status[self.round_robin_counter]: 2    self.workers[self.round_robin_counter].send_pyobj(req) 3    self.round_robin_counter = ( 4        self.round_robin_counter + 1 5    ) % len(self.workers) ```**

本例就是：

**``` 1request 0 -> DP0 2request 1 -> DP1 3request 2 -> DP0 4request 3 -> DP1 5... ```**

如果请求显式带了 routed\_dp\_rank：

**``` 1if req.routed_dp_rank is not None: 2    self.workers[req.routed_dp_rank].send_pyobj(req) ```**

则可以直接路由到指定 DP replica。

**Scheduler 内部如何收请求**

**每个 DP replica 内部有 4 个 TP rank，但不是每个 rank 都直接从 ZMQ 收请求。  Scheduler.init\_ipc\_channels：**

**``` 1if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0: 2    self.recv_from_tokenizer = get_zmq_socket( 3        context, zmq.PULL, port_args.scheduler_input_ipc_name, False 4    ) 5    self.recv_from_rpc = get_zmq_socket( 6        context, zmq.DEALER, port_args.rpc_ipc_name, False 7    ) 8else: 9    self.recv_from_tokenizer = None 10    self.recv_from_rpc = None ```**

普通 DP 下 attn\_tp\_rank == tp\_rank，所以只有每个 DP replica 的 tp\_rank=0 会收请求：

**``` 1DP0: 2GPU0 / tp_rank0 直接收 ZMQ 请求 3GPU1 / tp_rank1 不直接收 4GPU2 / tp_rank2 不直接收 5GPU3 / tp_rank3 不直接收 6 7DP1: 8GPU4 / tp_rank0 直接收 ZMQ 请求 9GPU5 / tp_rank1 不直接收 10GPU6 / tp_rank2 不直接收 11  GPU7 / tp_rank3 不直接收 ```**

随后 Scheduler.recv\_requests 会把 rank0 收到的请求 broadcast 给同一个 TP group 内的其他 rank：

**``` 1elif self.tp_size != 1: 2    recv_reqs = broadcast_pyobj( 3        recv_reqs, 4        self.tp_group.rank, 5        self.tp_cpu_group, 6        src=self.tp_group.ranks[0], 7    ) ```**

所以 DP0 内部是：

**``` 1DP0 rank0 收 ZMQ 请求 2**

DP1 内部同理。

**NCCL / TP 集合通信逻辑**

**普通 Native DP 下，两个 DP replica 没有跨 DP 的 NCCL collective。  每个 DP replica 内部单独初始化 TP group：**

**``` 1DP0 NCCL world: 2  GPU 0,1,2,3 3  tp_rank 0,1,2,3 4 5DP1 NCCL world: 6  GPU 4,5,6,7 7  tp_rank 0,1,2,3 ```**

每个 replica 内部的模型计算和集合通信就是普通 TP：

**``` 1tensor parallel all-reduce 2all-gather 3reduce-scatter 4attention TP broadcast 5logits / sampler 相关同步 ```**

但这些通信只发生在当前 DP replica 的 TP4 world 内。

对比 DP attention：

**``` 1普通 DP: 2  DP0 和 DP1 是两个独立模型副本。 3  DP0 的 TP collective 不需要 DP1 参与。 4  DP1 没请求时，DP1 可以 idle，不会影响 DP0 forward。 5 6DP attention: 7  所有 rank 在同一个 TP world 内。 8  attention 可分 DP group，但 MLP/MoE 前后需要同步 token 元信息。 9  idle DP 也可能要进入 collective，避免其他 rank hang。 ```**

因此普通 DP 没有 prepare\_mlp\_sync\_batch\_raw 这套跨 DP token 元信息同步，也没有 DP attention 里的 global\_num\_tokens、DP padding mode、DP hidden states gather/scatter 逻辑。

**KV cache 和模型权重的含义**

**普通 Native DP 是完整模型副本。  在 dp2 + tp4 下：**

**``` 1DP0: 2  拥有一套模型权重分片。 3  TP4 内每张 GPU 持有当前 TP rank 对应的权重 shard。 4  DP0 管自己的请求和 KV cache。 5 6DP1: 7  拥有另一套模型权重分片。 8  TP4 内每张 GPU 持有当前 TP rank 对应的权重 shard。 9  DP1 管自己的请求和 KV cache。 ```**

DP0 和 DP1 之间不共享请求状态，也不共享 KV cache。

这也是普通 DP 的核心取舍：

**``` 1优点: 2  简单，隔离性强。 3  一个 DP replica 忙，不要求其他 DP replica 同步参与。 4  适合吞吐扩展。 5 6代价: 7  模型权重和 KV cache 按 DP replica 复制。 8  同样 8 张卡，dp2+tp4 每个请求最多只用其中一个 TP4 replica。 ```**

**框架图**

**``` 1                         ┌────────────────────────────┐ 2                         │ Tokenizer / HTTP frontend   │ 3                         └──────────────┬─────────────┘ 4                                        │ ZMQ 5                                        v 6                         ┌────────────────────────────┐ 7                         │ DataParallelController     │ 8                         │ workers[0] -> DP0 port     │ 9                         │ workers[1] -> DP1 port     │ 10                         └──────┬──────────────┬──────┘ 11                                │              │ 12                                │ ZMQ          │ ZMQ 13                                v              v 14┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐ 15│ DP0 replica                           │  │ DP1 replica                           │ 16│ NCCL world A, TP4                     │  │ NCCL world B, TP4                     │ 17│ scheduler_input = DP0 port            │  │ scheduler_input = DP1 port            │ 18│                                       │  │                                       │ 19│ GPU0 tp_rank0  ZMQ receiver / leader  │  │ GPU4 tp_rank0  ZMQ receiver / leader  │ 20│ GPU1 tp_rank1                         │  │ GPU5 tp_rank1                         │ 21│ GPU2 tp_rank2                         │  │ GPU6 tp_rank2                         │ 22│ GPU3 tp_rank3                         │  │ GPU7 tp_rank3                         │ 23│                                       │  │                                       │ 24│ rank0 broadcasts requests             │  │ rank0 broadcasts requests             │ 25│ within DP0 TP group                   │  │ within DP1 TP group                   │ 26└──────────────────────────────────────┘  └──────────────────────────────────────┘ ```**

**请求的完整链路**

**``` 11. Tokenizer / HTTP frontend 产生请求。 2 32. DataParallelController 通过 ZMQ 收到请求。 4 53. DataParallelController 根据负载均衡策略选择 DP0 或 DP1。 6 74. 请求通过 ZMQ 发到对应 DP replica 的 scheduler input IPC。 8 95. 该 DP replica 内部只有 tp_rank0 直接收请求。 10 116. tp_rank0 把请求 broadcast 给同 replica 内其他 TP rank。 12 137. 这个 DP replica 的 TP4 group 独立执行 model forward。 14 158. forward 内部的 NCCL collective 只发生在当前 DP replica 的 4 张 GPU 内。 16 179. 输出通过共享 tokenizer/detokenizer IPC 回到上层。 ```**

**SGLang Model Gateway 如何实现类似 DP 扩展**

**SGLang Model Gateway，简称 SMG，提供的是另一种更现代的 DP 扩展方式。  Native DP 是：**

**``` 1一个 sglang.launch_server 进程树 2  -> 内部 DataParallelController 3  -> 内部启动多个 DP replica 4  -> Controller 用 ZMQ 把请求分给 replica ```**

SMG 是：

**``` 1一个 Rust gateway / router 2  -> 管理多个独立 SGLang worker server 3  -> 通过 HTTP 或 gRPC 把请求路由到某个 worker ```**

也就是说，SMG 不把多个 DP replica 塞进同一个 SGLang server 的内部进程结构里，而是把它们提升为多个独立 worker 实例，再由 gateway 统一做路由。

**Co-launch 模式**

**SMG 的 Python launcher 支持一条命令同时启动 router 和多个 worker：**

**``` 1python -m sglang_router.launch_server \ 2  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \ 3  --dp-size 2 \ 4  --tp-size 4 \ 5  --host 0.0.0.0 \ 6  --port 30000 ```**

入口在：

**``` 1sgl-model-gateway/bindings/python/src/sglang_router/launch_server.py ```**

核心逻辑是：

**``` 1worker_ports = find_available_ports( 2    args.router_dp_worker_base_port, server_args.dp_size 3) 4 5for i, worker_port in enumerate(worker_ports): 6    proc = launch_server_process(server_args, worker_port, i) 7    server_processes.append(proc) 8 9protocol = "grpc" if server_args.grpc_mode else "http" 10router_args.worker_urls = [ 11    f"{protocol}://{server_args.host}:{port}" for port in worker_ports 12] 13 14launch_router(router_args) ```**

每个 worker 启动时会改写参数：

**``` 1server_args.port = worker_port 2server_args.base_gpu_id = dp_id * server_args.tp_size 3server_args.dp_size = 1 ```**

所以单机 8 卡 dp2 + tp4 在 SMG co-launch 下可以理解为：

**``` 1Router: 2对外监听 port 30000 3 4Worker 0: 5独立 sglang server 6port = 31000 附近的某个可用端口 7base_gpu_id = 0 8dp_size = 1 9tp_size = 4 10使用 GPU 0,1,2,3 11 12Worker 1: 13独立 sglang server 14port = 31000 附近的另一个可用端口 15base_gpu_id = 4 16dp_size = 1 17tp_size = 4 18  使用 GPU 4,5,6,7 ```**

注意这里 worker 内部已经不是 Native DP 了：

**``` 1每个 worker 自己是普通 tp4 server； 2DP 这个概念由 router 层实现； 3worker 之间通过 HTTP/gRPC 隔离。 ```**

**Separate launch 模式**

**也可以先独立启动 worker，再让 router 连接它们：**

**``` 1# Worker 0 2python -m sglang.launch_server \ 3--model-path meta-llama/Meta-Llama-3.1-8B-Instruct \ 4--tp-size 4 \ 5--base-gpu-id 0 \ 6--port 8000 7 8# Worker 1 9python -m sglang.launch_server \ 10--model-path meta-llama/Meta-Llama-3.1-8B-Instruct \ 11--tp-size 4 \ 12--base-gpu-id 4 \ 13--port 8001 14 15# Router 16python -m sglang_router.launch_router \ 17--worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \ 18--policy cache_aware \ 19--host 0.0.0.0 \ 20--port 30000 ```**

这时请求路径是：

**``` 1Client 2**

这和 Native DP 的形态很像，都是“两个 TP4 副本做吞吐扩展”。区别是 Native DP 的 router 是 SGLang server 内部的 DataParallelController，SMG 的 router 是独立 Rust gateway。

**SMG 的 worker registry 和 policy**

**SMG 会把 worker 注册到 worker registry：**

**``` 1sgl-model-gateway/src/core/worker_registry.rs ```**

这个 registry 按 model 维护 worker 索引，并支持：

**``` 1按 model 查 worker 2按健康状态过滤 worker 3按 worker type 区分 regular / prefill / decode 4为 consistent hashing 维护 hash ring ```**

路由策略由 policy factory 创建：

**``` 1sgl-model-gateway/src/policies/factory.rs ```**

支持的策略包括：

**``` 1random 2round_robin 3power_of_two 4cache_aware 5bucket 6manual 7consistent_hashing 8prefix_hash ```**

其中 cache\_aware 是 SMG 相比 Native DP 的一个重要能力。它维护近似 prefix tree，根据请求前缀和已有 cache affinity 选择 worker：

**``` 1如果系统负载均衡: 2  尽量把相似 prefix 的请求路由到可能已经有 prefix cache 的 worker。 3 4如果系统负载不均衡: 5  切到负载均衡逻辑，把请求发给更空闲的 worker。 ```**

这比 Native DP 的 round\_robin / total\_requests / total\_tokens 更适合真实在线流量，因为 LLM serving 的性能经常受 prefix cache 命中率影响。

**DP-aware worker discovery**

**SMG 也有 DP-aware discovery：**

**``` 1sgl-model-gateway/src/core/steps/worker/local/discover_dp.rs ```**

核心逻辑：

**``` 1let info = get_server_info(url, api_key).await?; 2let dp_size = info.dp_size.ok_or_else(...)?; 3let model_id = info.model_id 4    .or(info.served_model_name) 5    .or_else(...) ```**

它会从 worker 的 server info 里发现：

**``` 1dp_size 2model_id 3served_model_name 4model_path ```**

这让 gateway 可以在更复杂部署中理解 worker 的 DP 信息，而不是只把 worker 当成一个普通 URL。

**为什么 SMG 更好**

**Native DP 的优点是实现简单：一个 SGLang server 内部起多个 replica，内部 ZMQ 分发请求。但它的问题也来自这个设计。  SMG 更适合生产的原因主要有这些。  第一，路由策略更强。  Native DP 主要是：**

**``` 1round_robin 2total_requests 3total_tokens 4follow_bootstrap_room ```**

SMG 支持：

**``` 1cache_aware 2power_of_two 3prefix_hash 4manual 5consistent_hashing 6bucket 7round_robin 8random ```**

其中 cache-aware / prefix-hash 对 prefix cache 友好，可以减少重复 prefill，提高缓存命中。

第二，worker 是独立 server，隔离性更好。

Native DP：

**``` 1一个父进程树内部管理多个 DP replica。 2某个 replica 的异常更容易影响整个 server 生命周期。 3升级、替换、下线粒度较粗。 ```**

SMG：

**``` 1每个 worker 是独立 SGLang server。 2router 只通过 HTTP/gRPC 和 worker 通信。 3worker 可以独立重启、扩缩容、替换。 ```**

第三，故障处理更完善。

SMG 有 circuit breaker 和 retry：

**``` 1sgl-model-gateway/src/core/circuit_breaker.rs 2sgl-model-gateway/src/core/retry.rs ```**

它可以对失败 worker 做熔断、半开探测、重试和 backoff。Native DP 的 DataParallelController 更偏本地进程内分发，不是完整的生产 gateway 故障治理系统。

第四，观测和运维能力更强。

SMG 是独立 gateway，天然适合做：

**``` 1worker selection metrics 2router latency metrics 3error metrics 4worker health check 5multi-worker registry 6multi-model routing 7Kubernetes service discovery 8mesh / multi-router 扩展 ```**

Native DP 更像 SGLang server 内部的一个简单 DP controller。

第五，部署形态更灵活。

Native DP 受限于单个 SGLang launch 进程树，并且本地代码里明确限制：

**``` 1assert not ( 2    self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention 3), "multi-node data parallel is not supported unless dp attention!" ```**

也就是说，普通 Native DP 不支持多机 DP。

SMG 可以连接任意多个 worker URL：

**``` 1同机多个 worker 2多机多个 worker 3Kubernetes 多 pod 4HTTP worker 5gRPC worker 6PD disaggregation worker ```**

因此它更适合作为真正的生产 DP routing 层。

**Native DP 和 SMG 的对比**

**``` 1实现位置: 2Native DP: 3    SGLang server 内部，Python DataParallelController。 4 5SMG: 6    独立 Rust gateway，外部管理多个 worker server。 7 8worker 形态: 9Native DP: 10    一个 launch_server 内部多个 replica。 11 12SMG: 13    多个独立 launch_server worker。 14 15通信方式: 16Native DP: 17    Controller 到 scheduler 使用 ZMQ IPC。 18 19SMG: 20    Router 到 worker 使用 HTTP/gRPC。 21 22NCCL: 23Native DP: 24    每个 replica 内部一个 TP NCCL world。 25 26SMG: 27    每个 worker 内部自己建立 TP NCCL world。 28    Router 不参与 NCCL。 29 30负载均衡: 31Native DP: 32    简单 in-process 方法。 33 34SMG: 35    cache-aware、power-of-two、prefix-hash、consistent hashing 等。 36 37故障隔离: 38Native DP: 39    replica 在同一 server 进程树内。 40 41SMG: 42    worker 独立，gateway 可跳过不健康 worker。 43 44生产推荐: 45Native DP: 46    更适合理解机制或少量旧框架兼容。 47 48SMG: 49    推荐用于生产 DP routing。 ```**

**和 DP attention 的核心区别**

**``` 1维度: 2普通 DP: 3    DP 是外层多个完整 replica。 4 5DP attention: 6    DP 是折叠在全局 TP world 里的 attention 维度。 7 8进程组: 9普通 DP: 10    dp2 + tp4 = 两个独立 TP4 NCCL world。 11 12DP attention: 13    dp4 + tp16 = 一个 TP16 NCCL world。 14 15请求分发: 16普通 DP: 17    controller 把请求发给某个完整 replica。 18 19DP attention: 20    controller 把请求发给某个 attention DP group leader。 21 22跨 DP 同步: 23普通 DP: 24    没有跨 DP 的 MLP/MoE token 同步。 25 26DP attention: 27    MLP/MoE 前后需要同步 global_num_tokens，并按 DP slice gather/scatter hidden_states。 28 29KV cache: 30普通 DP: 31    每个 DP replica 独立持有自己的 KV cache。 32 33DP attention: 34    attention DP group 维度参与 KV / hidden states 的特殊组织。 ```**

**总结**

**``` 1ZMQ 管请求: 2DataParallelController 把请求发给 DP0 或 DP1。 3 4rank 逻辑管副本: 5DP0 是一套 TP4，DP1 是另一套 TP4。 6两套 TP rank 都是 0..3，但属于不同 NCCL world。 7 8NCCL 管模型并行: 9每个 DP replica 内部做自己的 TP collective。 10DP0 和 DP1 之间不做模型 forward collective。 11 12普通 DP 的本质: 13  用更多完整副本提升吞吐，用显存复制换隔离和简单调度。 ```**
