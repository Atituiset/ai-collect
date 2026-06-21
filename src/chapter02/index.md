# 第2章 LLM 推理机制与优化

Prefill/Decode、KV Cache、vLLM、SGLang、RL对齐 —— 推理与训练优化全栈

## 本章文章

- [LLM 推理为什么先慢后快？从 Prefill、Decode 到 KV Cache 讲清楚](./llm-推理为什么先慢后快从-prefilldecode-到-kv-cache-讲清楚.md) — Prefill/Decode 两阶段与自回归生成原理
- [大模型推理为什么慢？从 Prefill、Decode 到 KV Cache，一文讲透推理加速全流程](./大模型推理为什么慢从-prefilldecode-到-kv-cache一文讲透推理加速全流程.md) — 推理加速全流程：从 Prefill 到 KV Cache 优化
- [大模型推理到底在做什么？从 Prefill、Decode 到 vLLM，一次讲透](./大模型推理到底在做什么从-prefilldecode-到-vllm一次讲透.md) — 推理本质：预测下一个 token 的完整流程
- [彻底搞懂 LLM 推理全过程](./彻底搞懂-llm-推理全过程.md) — LLM 推理从输入到输出的每个环节
- [LLM推理全流程解析：从输入到输出的底层逻辑](./llm推理全流程解析从输入到输出的底层逻辑.md) — 9 步推理流水线：分词、嵌入、采样、解码
- [理解 KV Cache：LLM 推理为什么能越写越快](./理解-kv-cachellm-推理为什么能越写越快.md) — KV Cache 原理、GQA/MQA 优化与内存管理
- [LLM推理优化核心技术](./llm推理优化核心技术.md) — 量化、蒸馏、投机解码等推理优化技术全景
- [大模型推理系统入门：从模型优化到调度优化](./大模型推理系统入门从模型优化到调度优化.md) — vLLM 技术路线图：模型优化与调度优化
- [vLLM 如何管理 KV Cache：从 vLLM 本地 Block 管理到跨实例缓存池](./vllm-如何管理-kv-cache从-vllm-本地-block-管理到跨实例缓存池.md) — Block 状态机、PagedAttention、Prefix Caching
- [LLM大模型是如何运行起来的：从Hello World到vLLM的运行时深度解析](./llm大模型是如何运行起来的从hello-world到vllm的运行时深度解析.md) — vLLM 运行时架构与请求处理全流程
- [SGLang推理优化-调度器核心ScheduleBatch](./sglang推理优化-调度器核心schedulebatch.md) — SGLang 调度器核心 ScheduleBatch 实现
- [SGLang推理优化-Scheduler 到 Forward 推理的 Batch 数据流](./sglang推理优化-scheduler-到-forward-推理的-batch-数据流.md) — 从 Scheduler 到 Forward 的 Batch 数据流
- [SGLang推理优化-DP并行](./sglang推理优化-dp并行.md) — SGLang DP 并行推理优化
- [2026 年面向 LLM 的 RL方法总结：从 PPO 到 DPO 到 GRPO，再到多智能体 RL](./2026-年面向-llm-的-rl方法总结从-ppo-到-dpo-到-grpo再到多智能体-rl.md) — PPO→DPO→GRPO→多智能体 RL 对齐方法演进
