# LLM大模型是如何运行起来的：从Hello World到vLLM的运行时深度解析

## 第一部分：从一行代码说起

打开你的终端，敲下这行命令：

```
curl http://localhost:8000/v1/chat/completions \  
  -H "Content-Type: application/json" \  
  -d '{  
    "model": "Qwen3-8B",  
    "messages": [{"role":"user","content":"你好"}]  
  }'
```

回车。等待。1秒、2秒、3秒……终于，屏幕上缓缓吐出一个"你好！有什么我可以帮你的吗？"。

**但就在这几秒钟里，你的GPU正在经历一场风暴。**

打开`nvidia-smi`，你会看到：显存占用瞬间从0暴涨到16GB，利用率从0%飙到95%以上，风扇开始咆哮。如果你同时开着`htop`，你会发现CPU几乎没怎么动——它像个悠闲的监工，而GPU里那几万个小学生正在疯狂做算术。

**为什么这行代码会吃掉几十GB显存？为什么它不能像传统程序那样瞬间返回？**

这是两个根本性的问题，也是本文要回答的核心。

传统程序是**逻辑执行**。CPU取指令、做判断、跳分支，本质上是在"走步骤"。你写个`if-else`，CPU就像个老教授，一步步推演逻辑，快则纳秒级，慢也毫秒级。

但大模型不是"走步骤"，它是**算概率**。你输入"你好"，模型要计算这15万个词中，下一个词是"！"的概率是多少，是"吗"的概率是多少。这个计算不是查表，而是让输入向量穿过一个数百层的神经网络，每一层都是**数十亿参数的矩阵乘法**。

**一句话总结：传统程序算"步骤"，大模型算"概率"。**

而这个"算概率"的过程，需要把几百亿个数字同时加载到GPU里，用几万个核心并行计算。这就是为什么显存会暴涨，为什么响应有延迟——**你不是在调用一个API，你是在启动一座微型工厂。**

## 第二部分：拆开黑盒——模型文件里到底有什么？

下载一个Qwen3-8B模型，解压后你会看到一文件夹的文件。很多人看到这一堆东西就懵了：不是说好的"模型"吗？怎么像个杂货铺？

让我带你解剖这个"大脑"。

```
Qwen3-8B/  
├── config.json                    ← 大脑的"布线图"  
├── model-00001-of-00005.safetensors  ← 大脑的"神经连接权重"  
├── model-00002-of-00005.safetensors  
├── ...  
├── model-00005-of-00005.safetensors  
├── model.safetensors.index.json     ← 权重"目录表"  
├── tokenizer.json                   ← "语言翻译器"  
├── tokenizer_config.json  
├── vocab.json + merges.txt          ← 词表和分词规则  
└── generation_config.json           ← 生成策略默认值
```

### 2.1 config.json = 大脑的布线图

打开`config.json`，里面没有神秘代码，就是一堆数字：

```
{  
  "hidden_size": 4096,           // 每层神经元的宽度  
  "num_hidden_layers": 32,         // 神经网络的层数  
  "num_attention_heads": 32,       // 注意力头的数量  
  "max_position_embeddings": 32768, // 最大能读多少字  
  "torch_dtype": "bfloat16"       // 每个数字占2字节  
}
```

**没有这份布线图，那堆权重文件就是一堆无意义的数字。** 框架读到它，才知道"哦，这是一个32层、每层4096维的Transformer，我要按这个结构去组装"。

### 2.2 safetensors = 大脑的神经连接权重

这是模型真正的"肉身"。Qwen3-8B有80亿个参数，每个参数是一个数字（通常是2字节的bfloat16）。简单乘法：80亿 × 2字节 = **16GB**。

这些数字是什么？你可以把它们想象成大脑里每个神经元连接的"强度"。训练阶段，模型看了海量的互联网文本，通过反向传播不断调整这些数字，最终让它们编码了语言的知识和规律。

**推理时，这些数字是冻结的——它们不会被修改。** 模型只是用你输入的token，去查询这些已经训练好的权重，算出一个概率分布。

为什么分成5个文件？因为单个16GB的文件在加载时容易内存溢出，分片后可以实现**按需加载**——比如你只推理文本，可能不需要加载视觉编码器的部分。

### 2.3 tokenizer系列 = 语言翻译器

模型看不懂"你好"这两个字。它只认识数字。

tokenizer的作用就是把`"你好"` → `[243, 56, 129]`这样的token ID序列。这个过程叫**编码（Encoding）**。反过来，模型输出的数字序列也要通过tokenizer转回文字，叫**解码（Decoding）**。

Qwen3的词表有15万多个token，覆盖中英文和常见符号。`vocab.json`是"字典"，`merges.txt`是BPE分词的合并规则——告诉系统如何把单词拆成子词（比如"unhappiness" → "un" + "happiness"）。

### 2.4 格式之争：为什么safetensors淘汰了.bin？

早年的PyTorch模型用`.bin`格式，它本质上是Python的pickle序列化。**危险在于：pickle可以执行任意代码。** 你下载一个来路不明的.bin文件，加载时它可能就在你的服务器上运行了恶意脚本。

**safetensors是HuggingFace推出的安全格式**，它只存储纯张量数据，不执行代码，加载速度更快，还支持内存映射（mmap）——不需要把整个文件读进内存，用多少加载多少。

另外还有**GGUF格式**（llama.cpp专用），专为本地CPU推理设计，支持量化压缩；以及**HF格式**（就是我们上面看到的目录结构），是生态最通用的标准。

**标准化的力量在于：无论模型是Meta做的Llama、阿里做的Qwen，还是DeepSeek做的R1，只要遵循HF格式，你就能用同一套代码加载和推理。** 这是大模型生态繁荣的基石。

## 第三部分：动力心脏——为什么必须是GPU？

### 3.1 CPU vs GPU：大学教授 vs 小学生

假设你要算一道复杂的数学题：计算100万个向量的点积。

**CPU像几个大学教授。** 逻辑极强，能做微积分、能写论文，但只有8个核心（甚至64个）。每个核心一个周期只能算几个数。

**GPU像几万个小学生。** 单个小学生只会做加减乘除，逻辑能力弱，但胜在人多。NVIDIA A100有6912个CUDA核心，H100更是上万。你把100万个向量拆成小块，分给几万个小学生同时算，每人算几个，瞬间出结果。

大模型的推理，本质上是**海量矩阵乘法**。一个8B模型的单次前向传播，涉及数百个`[batch, seq_len, hidden_size]`的张量运算。这种计算的特点是：**逻辑简单（都是乘加），但数据量巨大，且高度并行。**

这正是GPU的甜点场景。

### 3.2 显存（VRAM）：比算力更稀缺的资源

很多人买GPU只看"算力"（TFLOPS），但跑大模型时，**显存才是瓶颈。**

为什么？因为模型权重本身就占满了显存。Qwen3-8B用bfloat16要16GB，加上KV Cache（后面会讲）、激活值、临时缓冲区，实际占用轻松超过20GB。如果你只有一张24GB的RTX 3090，几乎塞不下更大的模型。

**买大显存才是王道。** 这也是为啥推理卡（如A100 80GB）比游戏卡（RTX 4090 24GB）贵那么多——不是算力差距，是显存差距。

### 3.3 CUDA：GPU的"Windows操作系统"

有了硬件，还需要软件来指挥。CUDA就是NVIDIA给GPU写的操作系统。

你可以把CUDA理解为**GPU的指令集+运行时**。没有CUDA，你的Python代码根本不知道如何让那几万个"小学生"协同工作。PyTorch、TensorFlow这些框架，底层最终都翻译成CUDA kernel（GPU上的小程序）来执行。

CUDA的核心抽象是**kernel函数**：你写一段并行逻辑，指定"用几万个线程来执行"，CUDA负责把它们调度到GPU的物理核心上。像矩阵乘法这种操作，NVIDIA还提供了高度优化的cuBLAS库——你调个API，底层已经是工程师们调了几年的极致优化。

## 第四部分：如何让大脑跑起来——部署实战

### 4.1 原始阶段：用transformers库直接加载

最简单的方式，几行代码就能跑：

```
from transformers import AutoModelForCausalLM, AutoTokenizer  
  
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto", device_map="auto")  
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")  
  
inputs = tokenizer("你好", return_tensors="pt").to(model.device)  
outputs = model.generate(**inputs, max_new_tokens=100)  
print(tokenizer.decode(outputs[0]))
```

**但这是最慢、最占显存的方式。**

痛点在哪？

**第一，显存碎片。** transformers库默认给每个请求预分配连续的KV Cache空间。如果请求长度不确定，要么浪费（预分配多了），要么OOM（预分配少了）。更糟的是，不同长度的请求之间会产生大量无法利用的"碎片"显存。

**第二，KV Cache浪费。** 这是大模型推理的隐形杀手。KV Cache是Transformer解码阶段的核心数据结构——为了生成第N个token，模型需要"记住"前面所有token的Key和Value。一个长对话的KV Cache可以轻松占掉几GB甚至几十GB显存。传统实现里，这些缓存被连续存储，无法灵活共享和复用。

**第三，吞吐量低。** transformers一次只能处理一个请求（或一个batch），GPU算力大量闲置。

### 4.2 进阶方案：vLLM的PagedAttention革命

2023年，UC Berkeley的Sky Computing Lab（没错，就是那个造出Spark和Ray的神仙实验室）推出了vLLM。它的核心创新只有一个：**PagedAttention**。

PagedAttention的思想直接抄了操作系统最经典的设计——**虚拟内存分页**。

让我用类比解释：

想象GPU显存是一个仓库。传统做法是：每个请求来了，你给它划一片**连续的区域**放KV Cache。就像仓库里给每个客户预留一整排货架。客户A要100个token，你划100个连续货位；客户B要50个，你划50个连续货位。

问题是：客户走了，留下的空位大小不一，新客户的需求又各不相同，仓库里很快充满了无法利用的"缝隙"。这就是**显存碎片**。

**PagedAttention说：为什么非要连续？**

它把KV Cache切成固定大小的**块（Block）**，默认每个块存16个token的K/V。然后给每个请求维护一个**块表（Block Table）**，记录"逻辑上第几个块，对应物理仓库的哪几个分散货位"。

```
操作系统概念        vLLM对应概念  
─────────────────────────────────  
页帧(Page Frame)    →  Block（物理存储单元）  
页表(Page Table)    →  Block Table（映射表）  
页内偏移(Offset)    →  Slot偏移  
页大小              →  block_size（默认16个token）
```

**核心价值：物理块不需要连续。** 仓库里任何空闲的小块都能被利用，请求结束后释放的块立即被其他请求复用。显存利用率从传统的60%提升到**96%以上**。

### 4.3 vLLM部署实战：几行代码启动高性能服务

安装：

```
pip install vllm
```

启动OpenAI兼容的API服务：

```
vllm serve Qwen/Qwen3-8B \  
  --tensor-parallel-size 1 \  
  --max-model-len 32768 \  
  --gpu-memory-utilization 0.9
```

就这么一行，你得到了一个兼容OpenAI接口的高性能推理服务。测试一下：

```
curl http://localhost:8000/v1/chat/completions \  
  -H "Content-Type: application/json" \  
  -d '{  
    "model": "Qwen/Qwen3-8B",  
    "messages": [{"role":"user","content":"写一首关于夏天的诗"}],  
    "max_tokens": 500  
  }'
```

**效果对比：**

| 指标 | transformers | vLLM |
| --- | --- | --- |
| 单卡吞吐量 | ~2 req/s | ~20 req/s |
| 显存利用率 | ~60% | ~96% |
| 长上下文支持 | 易OOM | 稳定支持100k+ |
| 并发能力 | 弱 | 强 |

vLLM还内置了**Continuous Batching**（连续批处理）：不像传统batching要等一个batch全部完成才处理下一批，vLLM的调度器会动态把新请求塞进正在运行的batch里，只要GPU还有算力余量。这进一步榨干了GPU的利用率。

## 第五部分：高阶视角——如果你来写一个推理框架

假设你要从零写一个LLM推理引擎，你需要解决哪些问题？

### 5.1 功能清单：推理引擎的四大支柱

**1. 模型加载**

- 读取safetensors分片，按config.json重建网络结构
- 支持多卡并行（Tensor Parallelism）：把一层网络的权重切到多张卡上
- 支持流水线并行（Pipeline Parallelism）：把不同层放到不同卡上

**2. Batch调度**

- 接收请求队列，按优先级和长度组合成最优batch
- 实现Continuous Batching或Inflight Batching
- 处理抢占（Preemption）：显存不够时，把低优先级请求的KV Cache swap到CPU内存

**3. 显存管理**

- 实现PagedAttention或类似的块管理机制
- 监控KV Cache生命周期，及时回收
- 支持Prefix Caching：相同系统提示词（如"你是一个 helpful assistant"）的KV Cache只算一次，后续请求复用

**4. 并行策略**

- **Tensor Parallelism（TP）**：把矩阵乘法的维度拆开。比如隐藏层4096维，两张卡各算2048维，最后合并。适合单节点多卡。
- **Pipeline Parallelism（PP）**：把32层网络分给4张卡，每张卡管8层。数据像流水线一样流过各卡。适合跨节点。
- **Sequence Parallelism（SP）**：把长序列的token维度拆开，不同卡处理不同位置的token。

### 5.2 架构图：数据如何流过推理引擎

```
用户请求  
   ↓  
[HTTP API层]  ← 兼容OpenAI接口格式  
   ↓  
[调度器 Scheduler]  ← 决定哪些请求进batch、何时抢占  
   ↓  
[显存管理器 Block Manager]  ← 分配/回收物理block  
   ↓  
[推理引擎 Engine]  ← 执行模型前向传播（GPU kernel）  
   ↓  
[输出流 Output Stream]  ← 逐token返回，支持SSE流式输出  
   ↓  
用户看到文字逐字出现
```

### 5.3 算力演进：从GPU到异构计算

今天，NVIDIA GPU是LLM推理的绝对主流。但未来正在变化：

- **TPU（Google）**：专为Transformer设计的矩阵运算单元，在特定场景下性价比极高
- **NPU（华为昇腾、苹果Neural Engine）**：端侧推理的专用芯片，功耗极低
- **Groq LPU**：确定性架构，延迟极低，适合对实时性要求极高的场景

**未来的推理框架必须解决异构计算问题**——同一份模型，要能无缝跑在NVIDIA GPU、AMD GPU、TPU、NPU上。这要求框架层（如PyTorch、vLLM）和硬件层之间有良好的抽象层（如Triton、OpenAI Triton编译器）。

## 第六部分：总结与未来——软件工程的范式转移

### 6.1 我们从"写代码"变成了"炼模型"

十年前，软件工程师的核心产出是`.py`、`.cpp`文件，是逻辑和算法。

今天，核心产出变成了**模型权重文件**和**Prompt工程**。你不再写`if user_intent == "query"`，而是写"你是一个智能助手，请判断用户意图是查询、购买还是投诉……"

**软件正在从"确定性逻辑"走向"概率性智能"。**

### 6.2 自然语言正在成为新的"高级编程语言"

Prompt Engineering不是"调参数"那么简单。它在本质上是一种**声明式编程**：

- 传统编程：你告诉计算机**怎么做**（步骤、逻辑、控制流）
- Prompt编程：你告诉计算机**做什么**（目标、约束、边界条件）

未来的程序员可能不需要精通C++内存管理或Python装饰器，但需要精通：

- **如何定义问题**：把模糊的业务需求转化为清晰的AI任务
- **如何约束AI**：通过System Prompt、RAG上下文、工具调用来限定AI的行为边界
- **如何评估质量**：建立评测体系，知道模型什么时候靠谱、什么时候在胡说

### 6.3 Agentic时代：软件是活的

最激进的展望是：**软件不再是静态的二进制文件，而是动态的、具备自主决策能力的Agent。**

想象一个客服系统：它不是按预设流程回答，而是理解你的问题、查询知识库、调用工具、生成回复，甚至主动追问以澄清需求。它每天都在"运行"，每天都在根据新数据微调自己的行为。

**这意味着运维模式的根本转变。** 以前你部署一个服务，跑一年可能都不变。未来的AI服务需要持续监控、持续评估、持续微调——它是个"活物"，需要"饲养"。

## 写在最后

回到开头那行curl命令。

现在你知道了，当你按下回车时，背后发生的是：16GB的权重被加载到GPU显存，tokenizer把你的文字切成token，调度器把请求排进batch，PagedAttention管理着分散的KV Cache块，几万个CUDA核心在并行计算矩阵乘法，softmax输出概率分布，采样器从中选出一个token，然后这个过程重复几十次，最终组合成你看到的回复。

**这不是魔法，这是工程。**

从Hello World到vLLM，从transformers到PagedAttention，我们走过的是一条"把不可能变成可能"的路。而这条路才刚刚开始——Agentic AI、异构计算、实时推理，下一个十年，我们会看到更多奇迹。

如果你读到这里，希望你已经从"只会调API"跨过了那道鸿沟，真正理解了这颗"电子大脑"是如何跳动起来的。

> **推荐阅读与参考**：
>
> - • vLLM官方论文：Efficient Memory Management for Large Language Model Serving with PagedAttention
> - • Qwen3技术报告（HuggingFace/ModelScope）
> - • FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

[#vLLM](javascript:;)[#大模型推理原理](javascript:;)[#模型文件结构](javascript:;)
