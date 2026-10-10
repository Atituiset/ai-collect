# 推理 → 行动 → 观察：用 LangChain + Python 实现一个智能体循环

如果你用过 ChatGPT 或 Claude的话对标准聊天机器人的工作方式应该不陌生：提问然后得到一个回答。但如果交给它一个多步骤任务呢？比如：“帮我找到最便宜航班，查询我的常旅客积分，并预订最佳选项”。

这时标准聊天机器人就会卡住，因为它没有在单次响应之后继续推进的机制。

智能体循环架构（agentic loop architecture）就是为了解决这一局限而出现的。下面这篇介绍会说明它是什么、如何工作，以及如何把它实现出来。

## 简单聊天机器人与 AI 智能体的区别

聊天机器人以单次通过（single pass）的方式运行他的交互流程是：用户发送消息，LLM 生成响应，流程结束。一次输入对应一次输出，没有任何记忆或状态被带入下一步。

智能体则是被设计用来行动（act），而不只是响应。两者之间的差异，可以归结为一个非常基础的编程概念`while` 循环。

智能体循环是一种迭代循环：LLM 在其中使用工具、根据反馈做出调整，并反复执行这一过程，直到任务彻底完成。

## 智能体循环的 5 个阶段

智能体循环一般跨越五个阶段。

1. 感知（Perceive）：智能体接收输入，可能是用户的 Prompt、某个 API 的响应，或是一条错误消息。
2. 推理（Reason）：LLM 处理这段上下文，决定下一步该做什么。
3. 规划（Plan）：面对复杂目标时，智能体会把目标拆解为更小的子任务。
4. 行动（Act）：智能体执行具体动作——运行代码、查询数据库或调用 API。
5. 观察（Observe）：智能体查看动作的结果，判断它是否生效，或者计划是否需要调整。

在多数任务中，核心执行循环可以进一步简化为持续迭代的三步：推理 → 行动 → 观察。

智能体循环的核心迭代

## 架构 / 流程图

下面是这一架构的简单可视化示意：

详细的智能体循环架构

## 循环实际是怎么跑起来的：一个简单示例

设想给一个智能体下达这样的指令：“找出 2026 年发表的、关于智能体记忆的被引用次数最多的论文，并对其进行总结。”

循环的执行过程大致如下：

- 第 1 次迭代（推理 → 行动 → 观察）：智能体推理出需要搜索 2026 年的论文；通过调用搜索 API 进行行动；观察到一份包含 15 篇论文的列表。
- 第 2 次迭代：智能体推理出需要拿到被引用次数最多那篇结果的正文；通过文档检索工具进行行动；观察到完整的摘要。
- 第 3 次迭代：智能体推理出信息已经足够；生成总结，结束循环。

在底层，整个架构归结为几行逻辑的Python 伪代码：

```
 while not done:    
    response = call_llm(messages)    
        
    if response has tool_calls:    
        results = execute_tools(response.tool_calls)    
        messages.append(results)    
    else:    
        done = True    
         return response
```

## 用 Python 实现一个基础版本（LangChain）

下面是用 Python、LangChain 和一个 Oracle 数据库连接来实现智能体循环的基础示例，其中 Oracle 数据库连接被用于工具执行。

```
 from langchain.agents import create_agent    
from langchain_core.tools import tool    
from langchain_core.messages import AIMessage, ToolMessage    
  
## 1. 定义智能体可以执行（ACT）的工具    
@tool    
def calculate(expression: str) -> str:    
    """对一个数学表达式求值。"""    
    pass # 此处实现    
  
@tool    
def timezone_convert(time_str: str, from_city: str, to_city: str) -> str:    
    """在城市之间转换本地时间。"""    
    pass # 此处实现    
  
## 2. 创建带工具的智能体（这会编译出 StateGraph 循环）    
agent = create_agent(    
    model=llm,    
    tools=[calculate, timezone_convert],    
    system_prompt="You are a precise assistant. Use tools to find answers."    
)    
  
## 3. 运行迭代循环    
QUESTION = "If I fly from London at 14:00 for 6 hours, what time do I land in New York?"    
  
for chunk in agent.stream({"messages": [("human", QUESTION)]}, stream_mode="values"):    
    last_msg = chunk["messages"][-1]    
        
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:    
        for call in last_msg.tool_calls:    
            print(f"[ACT] → Executing {call['name']}")    
    elif isinstance(last_msg, ToolMessage):    
        print(f"[OBSERVE] ← Result received")    
    elif isinstance(last_msg, AIMessage) and last_msg.content:    
         print(f"\\nFinal Answer: {last_msg.content}")
```

## 什么时候该用智能体循环，什么时候不该用

适合使用的场景：

- 完成一个任务所需的步骤数无法事先确定。
- 系统需要根据中间步骤的结果调整自身策略——例如某次搜索失败时，要换不同的关键词再试。
- 任务完成度比原始速度更重要。

不适合使用的场景：

- 固定序列的工作流：如果整个流程都是高度可预测、步骤固定的，更合适的做法是写一个确定性的代码 Pipeline。
- 简单任务：只需要一次 LLM 调用加一次工具调用就能解决的事情，没必要套上迭代循环的开销。
- 严格的延迟约束：循环的每一次迭代都要做一次 LLM 调用，处理时间和 Token 成本都会累加。如果速度是首要优先级，智能体循环可能会偏慢。
