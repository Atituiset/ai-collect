# llama.cpp 源码解析（三）：114 个 Builder——一套代码如何驾驭 130+ 种架构

> 好久不见，这是 **llama.cpp 源码系列的第三篇**。第一篇我们拆解了 GGML（推理引擎），知道了"数据怎么算"。第二篇我们拆解了 GGUF（模型文件格式），知道了"模型怎么存"。这篇回答中间缺失的一环：**给定一个叫做 "qwen3" 的 GGUF 文件，谁来决定这个模型的每一层应该执行哪些 GGML 操作？**

## 引言： 先锚定问题

读完前两篇，你手上已经有了两样东西：

**第一样：GGML 操作集合。** 你知道 `ggml_mul_mat` 可以做矩阵乘法，`ggml_rms_norm` 可以做归一化，`ggml_rope` 可以做位置编码，`ggml_flash_attn_ext` 可以做融合注意力。这些操作是"乐高积木块"——它们不依赖任何特定模型，对所有人一视同仁。

**第二样：GGUF 模型文件。** 你知道模型文件里有超参数（"我是 qwen3 架构、8 层、768 维"）、有张量索引（"blk.0.attn\_q.weight 在第 X 字节，形状是 768×768"）、有权重数据。

**缺失的"图纸"在哪？** LLaMA 用积木的方式是 QKV→RoPE→FlashAttn→FFN→残差。Mamba 用积木的方式是 1D卷积→状态空间更新。两者的"图纸"完全不一样。谁负责把 GGML 积木块按正确的顺序拼成计算图？

这就是架构系统做的事。整个系统的核心代码分布在四个文件里，职责分明：

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `llama-arch.h` | 636 | 定义"架构族"（LLaMA、Qwen、Mamba...） |
| `llama-arch.cpp` | 906 | 定义每种架构的"特征表"（有没有 Attention？有没有 RoPE？...） |
| `llama-model.h` | 643 | 定义"型号"（7B、13B、8x7B...）和层结构体 |
| `src/models/` | 114 文件 | 真正的 Builder——把架构翻译成 GGML 计算图 |

## 第一章：你会怎么设计？三个方案，两个失败

### 方案一：一个巨型函数，130 个 if-else

最直觉的做法：一个 `build_graph(string arch)` 函数，用 `if-else` 分派：

```
if (arch == "llama") {
    for (每层) { rms_norm(...); matmul(q_proj, ...); rope(...); attn(...); ffn(...); }
} else if (arch == "qwen3") {
    for (每层) { ...; rms_norm(q, ...); rms_norm(k, ...); ... }  // 多两行
} else if (arch == "mamba") {
    for (每层) { ssm_conv(...); ... }  // 完全不同
}
// ... 130 个分支
```

这个函数会膨胀到 26,000 行。每次新增一个架构——比如刚发布的 DeepSeek V4——就要在这个巨型函数里再塞一个 200 行的分支。没有人愿意维护。而且 llama.cpp 的 PR 规范要求代码必须由人类贡献者亲自编写和维护（不能用 AI 生成后直接提交）——每个新架构都需要理解它的人持续维护。

### 方案二：JSON/YAML 配置文件，通用引擎解析

看上去更优雅：

```
layers:  - type: rms_norm  - type: matmul    weight: attn_q  - type: rope  - type: flash_attn  - type: matmul    weight: attn_out  - type: add_residual  - type: rms_norm  - type: swiglu_ffn
```

**问题在哪？** 架构之间的差异不是"参数不同"，而是**计算图的结构不同**。

Qwen3 在 QKV 投影和 RoPE 之间**插入了**两个额外的归一化操作——图中多了两个 `ggml_rms_norm` 节点。Mamba 的整个层**不是**"RMS Norm → QKV → Attention → FFN"的结构——它是 SSM（状态空间模型），内部是 1D 卷积和状态更新。DeepSeek V4 的 decode 阶段需要对 KV Cache 做**动态压缩**——标准 Transformer 根本没有这个操作。

要描述这些差异，你的配置文件语言需要支持条件分支、循环、动态节点插入。你需要的不是 JSON 文件，而是一个图灵完备的 DSL。到这个地步，不如直接用 C++。

### 方案三（llama.cpp 的实际选择）：每个架构一个 C++ 类

```
src/models/
├── llama.cpp      (248行)  — LLaMA 1/2/3 + Mixtral MoE
├── qwen3.cpp      (156行)  — Qwen3
├── mamba.cpp      (137行)  — Mamba（无 Attention）
├── gemma.cpp      (144行)  — Gemma
├── deepseek2.cpp  (439行)  — DeepSeek V2/V3/R1（含 MLA 和 MoE）
├── rwkv7.cpp      (196行)  — RWKV v7（类 RNN）
└── ... 114 个文件，平均 ~200 行/个
```

**这看起来暴力——114 个类，每个手动描述计算图。但读懂源码之后你会发现，每个 Builder 其实只是一份"组合清单"——告诉 GGML 基类"用 helper X 接 helper Y 接 helper Z"。真正的重活在基类里。** 把这 114 个 Builder 想象成 IKEA 的 114 套家具图纸——每套图纸只告诉你"木板 A 接木板 B 接螺丝 C"，但螺丝和木板本身（GGML 操作）是所有图纸通用的。变了的是组装顺序，而不是零件。核心矛盾在这里被化解：架构变体的数量（130+）和核心操作的数量（~20 个 helper 覆盖 90% 场景）之间差了近一个数量级。

> **[图1：Builder 模式的四层架构]** — 顶层是 114 个 Builder，中间是 19 个 helper 组成的 Transformer 设计模式库，底层是 GGML 操作集合。箭头标注"Builder 只表达差异，helper 封装共享逻辑"。

## 第二章：三个关键设计——为什么 114 个 Builder 不失控

### 设计一：19 个 helper 函数——Builder 不需要"造积木"，只需要"拼积木"

`llm_graph_context` 基类（`src/llama-graph.h:710`）提供了 19 个 helper 函数。每个 helper 封装了一个常见的 Transformer 操作模式——在内部创建若干个 GGML 操作节点，对外暴露为一个函数调用。

| Helper | 封装了什么 GGML 操作组合 | 多常用？ |
| --- | --- | --- |
| `build_inp_embd` | `ggml_get_rows(tok_embd, tokens)` — 把 token ID 查表成向量 | 100% 的架构 |
| `build_norm` | `ggml_rms_norm` — 归一化，支持 RMS/Layer/Group 三种 | 95% 的架构 |
| `build_lora_mm` | `ggml_mul_mat` + 可选的 LoRA 矩阵乘法和加法 | 95% 的架构 |
| `build_attn` | 写 KV Cache + Flash Attention + 输出投影 — 约 10 个 GGML 节点 | 85% 的架构 |
| `build_ffn` | SwiGLU：gate×up→SiLU→×down — 约 5 个 GGML 节点 | 80% 的架构 |
| `build_moe_ffn` | MoE 路由器 + top-k 选择 + 专家 SwiGLU + 加权求和 | 15% 的架构 |
| `build_mamba_layer` | Mamba 的一整层：投影→卷积→离散化→状态更新→输出—约 20 个节点 | 5% 的架构 |
| ... | ... | ... |

**这些 helper 是架构系统真正的"引擎"。** 如果没有它们，每个 Builder 都需要手动生成数百行 GGML 操作代码——这会直接拖垮可维护性。有了它们，Builder 的代码从"造积木"变成"拼积木"——指定顺序即可。

举例：LLaMA 的 FFN 部分在 Builder 里就是两行：

```
cur = build_norm(cur, ffn_norm_weight, ...);     // 归一化cur = build_ffn(cur, up_w, gate_w, down_w, ...); // SwiGLU FFN
```

而 `build_ffn` 内部展开后是 5 个 GGML 操作节点（`ggml_mul_mat`×3 + `ggml_silu` + `ggml_mul`）。Builder 不需要知道这些细节——它只需要说"我要一个 SwiGLU FFN，这是权重，你帮我搞定"。

### 设计二：超集结构体——一个 `llama_layer` 管所有架构

不同架构的层结构天差地别。llama.cpp 的做法不是为每种架构定义不同的层结构体，然后搞一个复杂的继承体系——而是定义了一个**超集**结构体，包含 40+ 个字段，覆盖所有 130+ 种架构可能需要的全部权重。

```
llama_layer 字段（按类别）:

注意力相关:                     FFN 相关:                   SSM 相关 (Mamba用):
wq, wk, wv, wo                  ffn_gate                    ssm_in
bq, bk, bv, bo                  ffn_up                      ssm_dt
wqkv (融合QKV投影)               ffn_down                    ssm_a
attn_norm (pre-attn)            ffn_norm (pre-FFN)          ssm_d
attn_norm_2 (Falcon双norm)                                  ssm_conv1d
attn_q_norm (Qwen3的Q-norm)     MoE 相关:                   ssm_norm
attn_k_norm (Qwen3的K-norm)     ffn_gate_inp (路由器)
attn_out_norm                   ffn_gate_exps (多专家gate)   DeepSeek MLA:
                                ffn_up_exps (多专家up)      wq_a
                                ffn_down_exps (多专家down)  wkv_a
```

**各架构的实际使用情况：**

| 架构 | 用到的字段 | 其余字段 |
| --- | --- | --- |
| LLaMA | `attn_norm` , `wq/wk/wv/wo`, `ffn_norm`, `ffn_gate/up/down`（9个） | 全是 nullptr |
| Qwen3 | LLaMA 的 9 个 + `attn_q_norm`, `attn_k_norm`（11个） | 全是 nullptr |
| Mixtral MoE | LLaMA 的 5 个 attn 字段 + MoE 字段（~15个） | 全是 nullptr |
| Mamba | `ssm_in/dt/a/d/conv1d` , `ssm_norm`（8个） | 全部 attn/ffn 字段为 nullptr |
| DeepSeek V2 | LLaMA + MLA + MoE（20+个） | 其余 |

**核心原则：`nullptr` 就是多态。** 不需要虚函数，不需要抽象接口，不需要 `dynamic_cast`。Builder 里自然地写成条件检查——MoE 模型的 `ffn_gate_inp` 非空，稠密模型为 nullptr。同一个 Builder 内部的 `if (field != nullptr)` 就是"不同架构的差异处"。

**为什么不用继承？** 假设用继承，你需要的继承树可能是这样：

```
        BaseLayer       /    |    \  AttnLayer MoELayer SSMLayer     /  \      |QwenLayer DeepSeekLayer  ← 到这里已经炸了：DeepSeek需要Attn+MoE+MLA
```

DeepSeek 同时需要 Attention（`wq/wk/wv/wo`）、MoE（`ffn_gate_inp/exps`）和 MLA 潜注意力压缩（`wq_a/wkv_a`）。在单继承体系里找不到合适的父类——从 `AttnLayer` 继承会丢失 MoE 字段，从 `MoELayer` 继承会丢失 MLA 字段，从两边多继承会触发 diamond problem。而现实中还有更交叉的组合：Gemma 4 同时有 Attention + MoE + 滑动窗口 + per-layer 输入投影——130+ 种架构的差异根本不是线性的分类树，而是一张完全图。

**超集结构体用一个朴素的想法绕过了所有这些问题：把可能需要的字段全部列出来，用 nullptr 表示"不需要"。** C++ 里没有继承开销（都是直接成员访问），没有虚函数开销（不需要 vtable），没有 `dynamic_cast`（不需要运行时类型检查）。简单、直接、够用。这是 llama.cpp "实用主义哲学"的又一个体现。

> **[图2：超集结构体 vs 继承体系]** — 左侧是 llama.cpp 的实际做法（一个大 struct，用 nullptr 区分），右侧是继承体系的反例（DeepSeek 在继承树里找不到合适的单一父类）。

### 设计三：运行时多态 + 编译期多态

**运行时多态**：同一个 LLaMA Builder（`llm_build_llama`）同时覆盖稠密模型和 MoE 模型。不是两个类——是一个类里的 if-else：

```
if (model.layers[il].ffn_gate_inp == nullptr) → 普通 FFN 分支else → MoE FFN 分支
```

模型加载时，稠密模型的 `ffn_gate_inp` 保持 nullptr（GGUF 中没有 MoE 路由器权重），MoE 模型的为非空。**Builder 不需要知道加载的是什么模型——它只检查"这个字段有没有值"。**

**编译期多态**：LLaMA Builder 的模板参数 `<bool embed>` 将一份源码编译成两个版本：

- `embed=false`

  → decode 模式：有 KV Cache，用 `build_attn_inp_kv` 做输入，有 lm\_head 输出 logits
- `embed=true`

  → prefill/embedding 模式：无 KV Cache，用 `build_attn_inp_no_cache` 做输入，不加 lm\_head（输出 embedding）

`if constexpr (embed)` 在编译期求值——死代码分支不出现在最终二进制里。不是运行时判断，是编译时就决定好的。

## 第三章：LLaMA Builder 的结构——所有 Transformer Builder 的模板

`src/models/llama.cpp`（248 行）是最经典的 Builder。几乎所有 Transformer 架构的 Builder 都以它为蓝本。

**它的本质是什么？** 是一个围绕 `for (int il = 0; il < n_layer; ++il)` 循环展开的、用 GGML 操作编写的"数学公式到计算图的翻译器"。每一步对应 Transformer 论文里的一行公式。

我用 MiniMind 的实际维度（`n_embd=768, n_head=8, n_head_kv=4, n_ff=2432, n_vocab=6400`）来标注每一步。

### 3.1 输入准备

Builder 首先要创建计算图的"输入叶子节点"——这些节点的数据不是计算的，而是从外部传入的。

第一步是 token embedding：创建一个 `ggml_get_rows` 节点——从 `token_embd` 矩阵中取出当前 token 对应的行。`token_embd` 的形状是 `[768, 6400]`——768 维 × 6400 个 token。对于输入 token 1968（"你好"），取出第 1968 行，得到一个 768 维的向量。这个操作在 GGML 里就是一个 `ggml_get_rows` 节点。

第二步是位置编码：创建一个位置占位节点。对于"你好"这个 token，它在序列中的位置是 0，RoPE 需要这个信息。

第三步是 Attention 输入类型选择：这里用到了模板参数。如果是 decode 模式（`embed=false`），K 和 V 从 KV Cache 中读取，只计算新 token 的 Q。如果是 prefill 模式（`embed=true`），Q、K、V 全部从当前 batch 计算。这个决定影响后续 `build_attn` 的行为——第一篇第九章详细讲过这两种模式的区别。

### 3.2 层循环：N 次重复的核心

```
for(int il =0; il < n_layer;++il){
```

`n_layer` 的值来自 GGUF 的 KV 元数据——MiniMind 是 8，LLaMA-7B 是 32，LLaMA-70B 是 80。Builder 不写死层数，从超参数动态读取。**这就是为什么同一个 Builder 可以处理 7B 和 70B——它们的层数不同，但每层的结构完全一样。**

### 3.3 每层的计算——先 Attention，后 FFN

每层执行七步，每一步在 graph\_ctx 里创建 1-3 个 GGML 节点：

**第 1 步：RMS 归一化。** 输入是原始 token 向量（维度 768），过 RMS Norm 后还是 768 维。`build_norm` 内部创建一个 `ggml_rms_norm` 节点——`op=RMS_NORM, src=[cur, attn_norm_weight], data=NULL`。attn\_norm\_weight 是 768 维的可学习缩放向量 γ。

**第 2 步：Q、K、V 投影。** 三个矩阵乘法：  
 - `Q = Wq × cur`：`Wq` 形状 `[768, 768]` × `cur` 形状 `[768, 1]` → `Q` 形状 `[768, 1]`  
 - `K = Wk × cur`：`Wk` 形状 `[384, 768]` × `cur` 形状 `[768, 1]` → `K` 形状 `[384, 1]`  
 - `V = Wv × cur`：`Wv` 形状 `[384, 768]` × `cur` 形状 `[768, 1]` → `V` 形状 `[384, 1]`

注意 K 和 V 的维度是 384，不是 768——这是 **GQA（分组查询注意力）** 的效果。Q 有 8 个头（768÷96维/头=8），K 和 V 只有 4 个头（384÷96维/头=4）。每 2 个 Q 头共享 1 组 KV。这直接缩小了 KV Cache 到一半（第一篇第六章讲过）。

`build_lora_mm` 不只是 `ggml_mul_mat`。如果这个权重被附着了 LoRA adapter（微调修正矩阵），它还会插入额外的乘法和加法节点。Builder 不需要知道 LoRA 是否存在——helper 自动处理。

**第 3 步：可选 bias。** Qwen 系列模型的 Q、K、V 投影有 bias——LLaMA 没有。Builder 里的 `if (model.layers[il].bq)` 检查：如果 bias 权重存在（非 NULL），加一个 `ggml_add` 节点。不存在就跳过。这就是 `llama_layer` 超集设计的实际使用——一个 nullptr 检查就处理了架构差异。

**第 4 步：Reshape 成多头形状。** Q 从 `[768, 1]` reshape 成 `[96, 8, 1]`（96 维/头 × 8 头 × 1 token）。这是"空操作"——只改 ne/nb，不拷贝数据（第一篇 1.4 节的视图系统）。K 和 V reshape 成 `[96, 4, 1]`（头数只有 4）。

**第 5 步：RoPE 位置编码。** 对 Q 和 K 的前 96 维（`n_rot=96`）做旋转变换。每对相邻维度 `(x, y)` 旋转一个和 token 位置 m 相关的角度 `m×θ`。旋转后，模型能从 `(x, y)` 的几何关系反推"这是第 m 个 token"。RoPE 的全部参数（`freq_base`, `freq_scale`, `ext_factor` 等）都是从 GGUF KV 元数据读取的——Builder 不写死任何数值。

**第 6 步：注意力计算（融合操作）。**`build_attn` 内部创建 Flash Attention 节点——把 Q×K^T、Softmax、×V 三个操作融合成一个 GGML 节点。避免了两个中间矩阵的读写（第一篇第七章和第八章讨论过融合操作的优势）。之后再加上输出投影 `Wo × attn_out`（Wo 形状 `[768, 768]`）。整个步骤输出一个 768 维向量。

**第 7 步：残差连接。** 把 Attention 的输出和进入 Attention 之前的原始输入相加——`cur + inpSA`。这就是"残差"的含义：Attention 只需要学习输出和输入之间的**差值**，原始输入直接传过去。

### 3.4 FFN 部分——一个 if-else 决定命运

Attention 之后是 FFN。先做 RMS Norm，然后根据 `ffn_gate_inp` 是否为空分叉：

**如果是稠密模型（LLaMA 家族）：** 走 SwiGLU 路径。`build_ffn` 内部创建大约 5 个 GGML 节点：  
 - `gate = SiLU(W_gate × cur)` — W\_gate 形状 `[2432, 768]`，过 SiLU 激活  
 - `up = W_up × cur` — W\_up 形状 `[2432, 768]`  
 - `ffn = W_down × (gate ⊙ up)` — W\_down 形状 `[768, 2432]`，把 2432 维压回 768 维

2432 是"FFN 中间维度"（`n_ff`）——比 768 大很多，给模型提供了更大的表示空间。`⊙` 是逐元素乘法——这就是"SwiGLU"的"门控"机制：gate 值大的维度被保留，gate 值小的被压制。

**如果是 MoE 模型（Mixtral 家族）：** 走 MoE 路径。`build_moe_ffn` 内部更复杂：  
 - 路由器计算 8 个专家的得分：`softmax(W_gate_inp × cur)`  
 - 选分数最高的 2 个专家  
 - 只对这 2 个专家的 up/gate/down 权重做 SwiGLU  
 - 两个专家的输出按 softmax 分数加权求和

MoE 的精髓：虽然有 8 套完整 FFN 权重（每个约 3.7MB），但每次只激活 2 套。计算量相当于稠密模型的 2/8 = 25%，但参数量是 8 倍。存储成本高，推理成本低。

最后做第二层残差连接——`cur + ffn_inp`。本层输出变成下一层的输入。

### 3.5 输出与拓扑排序

层循环结束后，最后的 RMS Norm + lm\_head 投影：`output × cur`。`output` 形状 `[6400, 768]`，cur 形状 `[768, 1]`，输出 `[6400, 1]`——这就是 6400 个 token 的原始得分（logits）。

最后调用 `ggml_build_forward_expand(gf, cur)`——从 cur 出发，沿着 `src[]` 指针向上回溯，收集所有 287 个 GGML 操作节点，做拓扑排序（第一篇第四章讲过的 DFS 算法）。排序结果存入 `cgraph->nodes[]`。

**Builder 的构造函数到这里结束。计算图描述完毕。287 个 ggml\_tensor 全部 `data=NULL`。真正的矩阵乘法要等 `ggml_graph_compute` 被调用时才发生。**

### 3.6 Builder 之间的差异只出现在"不同之处"

对比 `llama.cpp`（LLaMA）和 `qwen3.cpp`（Qwen3），差异只有两行：

```
// LLaMA: QKV投影后直接RoPE
Qcur = build_lora_mm(wq, cur, ...);
Kcur = build_lora_mm(wk, cur, ...);
// → RoPE

// Qwen3: Q和K在RoPE前各多过一次归一化
Qcur = build_lora_mm(wq, cur, ...);
Kcur = build_lora_mm(wk, cur, ...);
Qcur = build_norm(Qcur, attn_q_norm, NULL, LLM_NORM_RMS, il);  // ← 多这行
Kcur = build_norm(Kcur, attn_k_norm, NULL, LLM_NORM_RMS, il);  // ← 多这行
// → RoPE
```

**Qwen3 的 Builder 几乎就是 LLaMA 的 Builder 加两行。** 这不是巧合——130+ 种架构里，绝大多数 Transformer 变体都只是"标准 LLaMA 模板"的微调。基类提供共享逻辑，Builder 只表达差异。

但怎么让程序知道 "qwen3" 这个字符串应该对应到 Qwen3 的 Builder，而不是 LLaMA 的？这就是下一章的主题。

## 第四章：四层匹配——"qwen3" 字符串如何找到它的 Builder

GGUF 文件里有一行键值对：`"general.architecture" = "qwen3"`（第二篇第一章展示过 MiniMind 的完整 KV 元数据）。这一个字符串，触发四层转换。

### 第一层：string → `llm_arch` 枚举

`llama-arch.cpp` 有一个约 100 个条目的静态映射表。每个条目是一对字符串和枚举值：

```
{"llama",   LLM_ARCH_LLAMA}{"qwen3",   LLM_ARCH_QWEN3}{"mamba",   LLM_ARCH_MAMBA}{"rwkv7",   LLM_ARCH_RWKV7}{"deepseek2", LLM_ARCH_DEEPSEEK2}...
```

这一层只做一件事：把人类可读的字符串翻译成机器用的枚举。没有歧义，没有模糊匹配——找不到就直接报错退出。

### 第二层：`llm_arch` → 架构特征表

同一个 `llm_arch` 下的模型并非完全一样。LLaMA-3 和 LLaMA-2 就有差异——LLaMA-3 引入了 rope\_factors（不同层用不同的 RoPE 频率缩放）。所以 llama.cpp 不只在"架构族"层面区分模型，还有一个**架构特征表**，记录每种架构的"能力清单"。

这个表（也在 `llama-arch.cpp` 中）是 Builder 在运行时的参考——Builder 在执行时查询"我当前这个架构有没有 QK 归一化？""有没有 MoE？""用的是什么类型的 RoPE？"然后走对应的分支。这不是一个配置字典，而是一个"能力矩阵"，Builder 用它来做 `if/else` 决策。

### 第三层：`llm_arch` + 超参数 → `llm_type` 枚举

同一架构族还有不同大小的变体。LLaMA 有 7B（32层，4096维）、13B（40层，5120维）、70B（80层，8192维）。它们都是 `LLM_ARCH_LLAMA`，但层数和维度不同——这意味着计算图的节点总数不同（32 层是 ~290 个节点，80 层是 ~700 个节点）。

`llm_type`（`llama-model.h:22-141`）同时编码架构和参数量。130+ 个条目，命名直接告诉你参数量：

- `LLM_TYPE_7B`

  — LLaMA-7B
- `LLM_TYPE_13B`

  — LLaMA-13B
- `LLM_TYPE_8x7B`

  — Mixtral MoE，8 个专家，每专家 7B
- `LLM_TYPE_8x22B`

  — Mixtral Large MoE
- `LLM_TYPE_35B_A3B`

  — Qwen3.5 MoE，总参数 35B，激活 3B
- `LLM_TYPE_397B_A17B`

  — Qwen3.5 最大 MoE，总 397B，激活 17B

`_A_` 后缀是 MoE 的激活参数，`x` 格式是专家数 × 每专家参数量。

### 第四层：`llm_type` → Builder 实例创建

最后一个 switch（在 `llama-model.cpp` 内）根据 `llm_type` 创建对应的 Builder。但一个 Builder 可以对应多个 `llm_type`：

```
LLM_TYPE_7B  ─┐LLM_TYPE_13B ─┼→ llm_build_llama<false>LLM_TYPE_70B ─┘LLM_TYPE_8x7B ───→ llm_build_llama<false>  （同一个！MoE分叉在Builder内部）LLM_TYPE_Qwen3_?B → llm_build_qwen3LLM_TYPE_671B   ───→ llm_build_deepseek2
```

**关键**：`llama-model.cpp` 中的这个 switch 不需要知道 Builder 内部细节。它只需要知道"这个 llm\_type 对应哪个 Builder 类"。Builder 自己负责读取特征表、检查 nullable 字段、决定走哪个分支。这是职责分离——调度器和 Builder 各管各的。

> **[图3：四层匹配链]** — 纵向展示 "qwen3" → LLM\_ARCH\_QWEN3 → 特征表 → LLM\_TYPE\_?B → llm\_build\_qwen3 的完整流程。每层标注对应的源文件和数据结构。

## 第五章：cb 回调——Builder 和 GPU 调度器之间的"秘密通道"

每个 Builder 在创建 GGML 操作节点后，会立即调用一个 `cb()` 函数：

```
cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);cb(cur, "attn_norm", il);  // ← 每个节点后都有这一行
```

**`cb` 做什么？** 它是 `llm_graph_context` 构造时从外部传入的一个回调函数指针（`llm_graph_params.cb`，定义在 `src/llama-graph.h:757`）。它的唯一任务是通知后端调度器（`ggml_backend_sched`，系列第七篇的主题）："我刚创建了一个节点，它是第 `il` 层的 `attn_norm` 操作。"

调度器根据三个因素决定这个节点在 CPU 还是 GPU 执行：

- **操作类型**

  ：`attn_norm`（归一化）计算量小，放 CPU 即可；`Qcur`（矩阵乘法）计算量大，放 GPU
- **数据位置**

  ：权重张量的 `buffer` 在哪个设备上？如果能和权重在同一设备，避免跨设备数据传输
- **层深度**

  ：前几层的节点可能和 embedding 共享 CPU，后几层可能全部在 GPU

**没有 cb 会怎样？** 图可以正常执行——所有操作在默认设备上跑。cb 不是必须的，但它是"CPU+GPU 混合推理"性能的开关。没有它，调度器就不知道每个操作的计算特征，只能靠启发式规则猜测。

## 第六章：四种架构差异，四种处理方式

不是所有架构差异都需要全新 Builder。llama.cpp 按差异程度用了四种策略，从"多写几行"到"独立项目"。

### 级别一：多几个节点（~5 行代码）

**例子：Qwen3 的 QK 归一化。** LLaMA 的 Q 和 K 投影后直接做 RoPE。Qwen3 在中间插入了额外的 RMS Norm。

在代码层面，就是 Builder 里多两行 `build_norm` 调用——因为 `attn_q_norm` 和 `attn_k_norm` 字段在 LLaMA 模型加载时保持 nullptr（跳过），在 Qwen3 加载时为非空（执行）。同一个 Builder 不需要关心"当前是什么模型"——它只检查字段是否为空。为空就跳过，非空就多加一个归一化节点。

这体现了超集结构体的威力：一个架构的"额外特性"对其他架构完全透明——nullptr 让它们自然跳过。

### 级别二：整个操作模式不同（~150 行新 Builder）

**例子：Mamba（没有 Attention）。** LLaMA Builder 的层循环是"norm→QKV→attn→残差→norm→SwiGLU→残差"。Mamba 的层循环是"SSM 状态空间更新→残差"。

基类提供了 `build_mamba_layer` helper（封装了投影→1D 卷积→离散化→状态更新→输出投影共约 20 个 GGML 操作）。Mamba Builder 的层循环里没有 `build_attn`、没有 `build_ffn`、没有 `ggml_rope`——只是一个 `build_mamba_layer` 调用。计算图的结构完全变了。

但 Builder 本身只需要 137 行——因为大部分模板代码（输入准备、循环骨架、输出层、拓扑排序）可以复用。只有"层的具体实现"需要替换。

### 级别三：一个字段决定分叉（~30 行，已在 Builder 内）

**例子：Mixtral MoE。** LLaMA Builder 不需要改——它本来就有一个 `if (ffn_gate_inp == nullptr)` 分叉。稠密模型走普通 FFN，MoE 模型走 MoE FFN。同一个 Builder 类、同一套代码，用一个字段是否为 nullptr 区分。

这种设计让 `llm_build_llama` 同时覆盖了 LLaMA 家族（稠密）和 Mixtral 家族（MoE）。不是两个 Builder——是一个 Builder，两个分支。开发者不需要为 MoE 单独建一个新文件。

### 级别四：全新架构概念（1392 行或独立项目）

**例子：DeepSeek V4（2026 年 4 月发布，百万上下文）。** V4 的核心理念是"动态 KV Cache 压缩"——在 decode 阶段实时压缩 KV 状态以减少百万上下文的内存需求。这是标准 Transformer 完全没有的概念。llama.cpp 基类的任何 helper 都不能直接描述它——需要全新的数据结构（`dsv4_decode_compressor`）、全新的注意力掩码（三种不同策略，不同阶段切换）、全新的通道混合方式（`dsv4_hc_mix`）。

**四种策略的选择标准：**

| 差异级别 | 工作量 | 什么时候用 |
| --- | --- | --- |
| 多几个节点 | ~5 行 | 现有 Builder 内加几行，架构变体 |
| 新子图模式 | ~150 行新 Builder | 需要不同的 GGML 操作组合，但仍用基类 helper |
| nullable 分叉 | ~30 行 | 两种模式共存于同一架构家族（稠密 vs MoE） |
| 全新概念 | 1392 行+ | 需要新 helper、新数据结构、甚至 fork 项目 |

> **[图4：四种架构差异的对比]** — 四个横向色带，从绿色（多几行）到蓝色（新Builder）到橙色（nullable分叉）到红色（独立项目）。每级配有具体架构示例和代码差异示意。

## 第七章：Builder 模式的边界——antirez 的 DeepSeek V4 案例

2026 年 4 月 DeepSeek V4 发布——1.6 万亿参数、百万上下文。llama.cpp 官方至今（5 月 26 日）未适配。Redis 作者 antirez fork 了项目，写了 1392 行的 `deepseek4.cpp`。

**为什么是 1392 行？** V4 的核心特性——动态 KV 压缩、三阶段注意力掩码、混合通道——不是任何现有 helper 能直接支持的。antirez 在 Builder 里引入了一整套新机制：`dsv4_decode_compressor`（decode 时压缩 KV Cache）、`dsv4_state_pair`（压缩后的状态管理）、三阶段掩码（RAW\_WINDOW / COMPRESS\_CAUSAL / ATTN\_STATIC）。

**这暴露了 Builder 模式的边界。** 当新架构的核心理念和现有 helper 库完全不兼容时，写新 Builder 和写专用引擎的成本开始接近。事实上 antirez 后来又写了 `ds4.c`——一个完全不依赖 llama.cpp 的纯 C 推理引擎，专为 V4 Flash 优化。

当然这里纯粹是想看看新的推理框架是怎么适配一些新的机制的，deepseek v4后续有时间单独出文章学习吧，这又是个复杂的大工程

好了，架构系统讲完了。下一篇系列应该会推出第四篇：**推理管线**

最近工作生活有点忙，断更了太久，之后一个月争取持续更新，下次见啦！
