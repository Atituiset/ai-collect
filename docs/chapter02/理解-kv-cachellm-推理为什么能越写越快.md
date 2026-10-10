# 理解 KV Cache：LLM 推理为什么能越写越快

只要你和现在的 AI 工具交流，无论是 Codex、Claude，还是 ChatGPT、DeepSeek、豆包，你应该都注意过一件事：它并不是一次性把完整答案吐出来，而是一个字接着一个字，慢慢形成一行字，再逐渐生成一整段话，直到所有结果都返回完毕。

而你看到的第一个 token，往往要等一会儿才出现，比后面出现的字慢得多。可一旦第一个 token 出现，AI 开始输出之后，后面的内容就会几乎连续地流式生成。

为什么 LLM 的生成结果一开始比较慢，后面却能连续输出？这就是本文要科普的小知识。

在这个现象背后，有一个有意思的工程机制：**KV Cache**（KV 缓存）。它的目的很直接：让 LLM 推理更快。

**KV Cache 的效果**

在进入技术细节之前，先来看一个直观对比：

可以看出，有 KV Cache 和没有 KV Cache 的 LLM 推理，速度差异非常明显。

接下来，我们从最基础的原理开始理解它。

LLM 是如何生成 token 的

现在大多数 LLM 底层使用的都是 Transformer 架构。

你可以先简单理解成：Transformer 是一种专门处理序列数据的模型结构，它擅长根据上下文关系，判断每个 token 应该如何理解。

当一段文本被输入模型后，Transformer 会处理所有输入 token，并为每个 token 生成一个 **hidden state**。你可以把 hidden state 理解成：模型在结合上下文之后，对这个 token 形成的一组内部表示。

随后，这些 hidden state 会被投影到词表空间，得到 **logits**，也就是词表中每个词对应的分数。

但真正重要的是：**只有最后一个 token 的 logits 会被用来预测下一个 token**。

模型会从这些 logits 中采样，得到下一个 token，然后把这个 token 追加到输入序列后面，再重复同样的过程。

这里有一个关键点：为了生成下一个 token，我们只需要最近那个 token 的 hidden state。其他 token 的 hidden state，更多是中间计算结果。

**Attention 实际上在计算什么**

Attention 你可以理解为是一种「信息选择机制」：它会判断当前 token 应该关注上下文里的哪些 token，并从这些位置取回信息。

在 Transformer 的每一层里，每个 token 都会生成三个向量：

- Query，简称 Q，表示「我现在想找什么信息」；
- Key，简称 K，表示「我这里有什么信息可被匹配」；
- Value，简称 V，表示「如果你关注我，真正拿走的内容是什么」。

Attention 会用 Query 和 Key 做乘法，得到注意力分数，再用这些分数对 Value 做加权求和。

现在我们只关注最后一个 token。

在计算 QK^T 时，T 表示转置。简单来说，就是让每个 Query 都能和所有 Key 做一次乘法，算出注意力分数。

对最后一个 token 来说，QK^T 的最后一行会用到：

- 最后一个 token 的 Query 向量
- 序列中所有 token 的 Key 向量

而这一行最终的 Attention 输出会用到：

- 同一个 Query 向量
- 所有 token 的 Key 和 Value 向量

所以，为了计算我们真正需要的那个 hidden state，每一层 Attention 都需要：最新 token 的 Q，以及整个序列中所有 token 的 K 和 V。

**存在的冗余**

假设生成第 50 个 token 时，需要 token 1 到 token 50 的 K 和 V。生成第 51 个 token 时，又需要 token 1 到 token 51 的 K 和 V。

但问题来了，token 1 到 token 49 的 K 和 V，其实之前已经计算过了。

它们不会改变。

同样的输入，同样的模型参数，输出也会相同。

如果每一步都重新从头计算这些 K 和 V，模型就在重复做大量已经做过的工作。

这就是每一步里的 O(n) 冗余计算。放到整个生成过程里，就会变成 O(n²) 级别的浪费。

**KV Cache 如何解决冗余**

KV Cache 的做法很简单：**不要在每一步重新计算所有 K 和 V，而是把它们存起来**。

对于每一个新 token，模型只需要：

1. 只为最新 token 计算 Q、K 和 V
2. 把新的 K 和 V 追加到缓存中
3. 从缓存中读取所有历史 K 和 V
4. 用新的 Q 去和完整的 K/V 缓存做 Attention

这就是 KV Cache。

每一层、每一步，只新增一个 K 和一个 V。其他历史 token 的 K 和 V，都直接从缓存里读取。

需要注意的是，Attention 计算本身仍然会随着序列长度增长。因为最新 token 的 Query 仍然要和所有历史 Key 计算注意力分数，并根据这些分数从对应的 Value 中取回信息。

但昂贵的 K/V 投影计算，只需要对每个 token 做一次，而不是每一步都重复做一次。

**Time-to-First-Token**

现在就能理解，为什么第一个 token 通常会比较慢。

当你发送一个 prompt 时，模型需要先处理完整输入。它会对整个 prompt 做一次前向计算，并为每个 token 计算和缓存 K/V。这个阶段叫做**prefill**。prefill 通常是整个请求中计算最重的阶段。因为模型要一次性处理完整的输入上下文。

一旦 cache 建好，后续每生成一个 token，就只需要对最新 token 做一次前向计算。

这也解释了为什么：第一个 token 前的等待时间更长，而后续 token 会连续流式输出。

这个初始延迟通常被称为 Time-to-First-Token，也就是 TTFT。**prompt 越长，prefill 越重，TTFT 也就越长**。优化 TTFT 本身也是一个很大的话题，比如 chunked prefill、speculative decoding、prompt caching 等。

但核心动态始终是一样的：构建 cache 很贵，读取 cache 很便宜。

**KV Cache 的代价**

KV Cache 本质上是在用内存换计算。

每一层都要为每个 token 存储 K 和 V 向量。

以 Qwen 2.5 72B 为例，如果是 80 层、32K context、hidden dim 8192，那么单个请求的 KV Cache 就可能消耗数 GB 的 GPU 显存。

当并发请求达到几百个时，KV Cache 的显存占用甚至可能超过模型权重本身。

这也是为什么 GQA 和 MQA 会出现。在标准 Multi-Head Attention 里，每个 Query head 通常都有对应的 Key / Value head。head 越多，需要缓存的 K 和 V 也越多。

Multi-Query Attention，简称：MQA，做法更激进：多个 Query head 共享同一组 Key / Value，因此 KV Cache 会明显变小。

Grouped-Query Attention，简称：GQA，则是折中方案：把多个 Query head 分成若干组，每组共享一组 Key / Value。它牺牲少量结构自由度，换来更低的 KV Cache 显存占用，同时尽量保持模型质量。

它们的核心思路是：让多个 Query head 共享更少的 Key / Value head，从而减少 KV Cache 的显存占用，同时尽量保持模型质量不受明显影响。

这也是为什么加长上下文窗口并不简单。如果 context length 翻倍，那么单个请求需要的 KV Cache 也会翻倍。

这意味着同样的 GPU 显存下，可以同时服务的用户数量会减少。

还有一个相关思路叫 PagedAttention，它主要解决的是大规模推理服务中 KV Cache 的内存管理问题。这里就不展开讲了，后面我们再来科普。

**小结**

KV Cache 消除了自回归生成过程中的大量重复计算。

历史 token 产生的 K 和 V 向量不会变化，所以只需要计算一次，然后存起来。之后每生成一个新 token，只需要计算这个新 token 自己的 Q、K 和 V，再让新的 Q 基于完整的 K/V 缓存完成 Attention 计算。

这就是为什么 LLM 的第一个 token 往往更慢：因为模型要先完成 prefill，建立完整的 KV Cache。而一旦 cache 建好，后续 token 就可以逐步、连续地生成。

KV Cache 在实践中能显著提升推理速度。代价是 GPU 显存占用增加，而在大规模 LLM 服务中，显存往往会成为比计算更关键的瓶颈。

今天几乎所有主流 LLM serving stack，比如 vLLM、TGI、TensorRT-LLM，都会围绕 KV Cache 做进一步优化。

**参考资料：**

Avi Chawla：KV Caching in LLMs, Explained Visually.
