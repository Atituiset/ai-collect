# AutoGen 多智能体实战：Coder、Reviewer、Executor 自动跑完一次代码评审

多智能体系统（Multi-Agent Systems，MAS）代表了智能软件设计的一次转向：不再让一个庞大的 LLM 包揽所有问题，而是把复杂任务拆分到若干彼此协作的智能体上，每个智能体都有明确的角色、记忆和工具集。Microsoft AutoGen 这类框架把这种架构带进了纯 Python 环境，可以直接搭建 Planner（规划者）、Executor（执行者）和 Critic（评审者）等角色，让它们彼此自主对话。

## 什么是多智能体系统

多智能体系统是一种计算架构：多个自主智能体相互交互、协商并协作，去解决对单个智能体来说过于复杂或低效的问题。每个智能体都会感知环境、维护内部状态并采取面向目标的行动。

这意味着多个语言模型实例（或模型加工具的混合体）通过结构化消息进行通信。我们可以把它类比成一支虚拟团队：一个智能体编写代码，另一个负责审查，第三个负责检索网络以获取背景信息，再由一个管理者智能体协调整个流程。

## 为什么选择 AutoGen

AutoGen 由 Microsoft Research 开发是目前用于构建多智能体对话系统的最成熟、最具生产可用性的 Python 框架之一。它抽象了智能体之间的通信、工具调用以及对话历史管理的复杂度，同时保留了较高的可定制空间。

主要特性：

- **ConversableAgent**：所有智能体的基类，支持 LLM 后端、人类代理（human proxy）和工具调用。
- **AssistantAgent**：预配置的 LLM 智能体，用于遵循指令并编写代码。
- **UserProxyAgent**：作为 human-in-the-loop 或自主执行器，可在本地运行代码。
- **GroupChat 与 GroupChatManager**：协调多智能体之间的圆桌讨论，发言者选择逻辑可配置。

架构概览

这种模式被称为 **Planner-Executor-Critic（规划-执行-评审）** 循环，是大多数生产级多智能体部署的骨架。

## 相关工具与框架

本文聚焦 AutoGen但多智能体生态本身相当丰富：

- **LangChain / LangGraph**：基于图（graph）的智能体编排，自带状态机，适合循环工作流。
- **CrewAI**：在 AutoGen 和 LangChain 之上的更高层抽象，适合基于角色的智能体团队。
- **Semantic Kernel**：Microsoft 推出的 SDK，用于在 C# 和 Python 中整合 LLM 与记忆、技能和插件。
- **OpenAI Swarm**：一个轻量级实验框架，用于探索智能体之间的任务交接（handoff）。

AutoGen 的优势在于原生支持代码执行、对话历史管理较为完善，以及 GroupChat 的动态机制，使它特别适合工程导向的工作流。

## 完整代码示例与可视化

下面的示例演示一个双智能体的 AutoGen 配置：AssistantAgent 负责编写一段 Python 数据分析脚本，UserProxyAgent 负责执行并回报结果。这套模式可以直接迁移到自动化报表流水线、数据工程任务和研究助理类应用上。

安装

```
 pip install pyautogen matplotlib pandas
```

完整示例

```
 import autogen  
import os  
  
## -------------------------------------------------------  
## 第 1 步：配置 LLM 后端  
## AutoGen 支持 OpenAI、Azure、Anthropic 以及本地模型。  
## 这里使用环境变量，避免把密钥写入代码。  
## -------------------------------------------------------  
config_list = [  
    {  
        "model": "gpt-4o",  
        "api_key": os.environ.get("OPENAI_API_KEY"),  
    }  
]  
  
llm_config = {  
    "config_list": config_list,  
    "temperature": 0.2,       # 较低的 temperature = 更确定、更精确的代码  
    "cache_seed": 42,         # 可复现的结果，便于调试  
}  
  
## -------------------------------------------------------  
## 第 2 步：定义 AssistantAgent（编码者）  
## 该智能体接收任务指令并编写代码。  
## 它本身不执行代码，执行工作交给另一个智能体。  
## -------------------------------------------------------  
assistant = autogen.AssistantAgent(  
    name="DataAnalystAgent",  
    llm_config=llm_config,  
    system_message="""  
    You are an expert Python data analyst.  
    When given a data task, write clean, well-commented Python code.  
    Always save any generated charts as PNG files.  
    Respond with ONLY the code block. Do not explain unless asked.  
    When the task is complete, reply with TERMINATE.  
    """,  
)  
  
## -------------------------------------------------------  
## 第 3 步：定义 UserProxyAgent（执行者）  
## 该智能体代理人类角色。在 NEVER 模式下，  
## 它完全自主运行，不需要人类输入。  
## 代码执行通过本地 Docker 容器或当前 Python 环境进行沙箱化。  
## -------------------------------------------------------  
user_proxy = autogen.UserProxyAgent(  
    name="ExecutorAgent",  
    human_input_mode="NEVER",           # 完全自主执行  
    max_consecutive_auto_reply=10,       # 自动回复次数的安全上限  
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),  
    code_execution_config={  
        "work_dir": "agent_workspace",   # 代码在该目录下运行  
        "use_docker": False,             # 生产环境建议设为 True 以实现隔离  
    },  
    llm_config=False,                   # 执行者不使用 LLM  
)  
  
## -------------------------------------------------------  
## 第 4 步：用一个具体任务发起对话  
## UserProxy 把任务发送给 AssistantAgent。  
## AutoGen 会自动处理来回的对话过程。  
## -------------------------------------------------------  
task = """  
Generate a synthetic sales dataset with 100 rows containing:  
- 'month' (Jan to Dec, repeated)  
- 'product' (randomly from ['Widget A', 'Widget B', 'Widget C'])  
- 'revenue' (random float between 1000 and 50000)  
- 'units_sold' (random int between 10 and 500)  
  
Then:  
1. Compute total revenue and average units sold per product.  
2. Plot a bar chart of total revenue by product.  
3. Save the chart as 'revenue_by_product.png'.  
4. Print the summary statistics to the console.  
"""  
  
user_proxy.initiate_chat(  
    recipient=assistant,  
    message=task,  
    clear_history=True,  
)  
  
 print("Agent task completed. Check agent_workspace/ for output files.")
```

当 `initiate_chat` 被调用后，会按以下顺序展开：

1. `ExecutorAgent` 把任务字符串发送给 `DataAnalystAgent`。
2. `DataAnalystAgent` 把任务和它的 system prompt 一起发送给 LLM，返回一段 Python 代码块。
3. `ExecutorAgent` 检测到代码块，将其提取出来，并在 `agent_workspace` 目录下运行。
4. 标准输出和任何报错都会被捕获，并作为下一条消息回传给 `DataAnalystAgent`。
5. 如果代码出错，`DataAnalystAgent` 会自我修正并重新提交；该循环持续，直到执行成功或达到 `max_consecutive_auto_reply` 设定的上限。
6. 一旦成功，`DataAnalystAgent` 回复 `TERMINATE`，结束本轮对话。

扩展到 GroupChat（三智能体流水线）

```
 # 增加一个 ReviewerAgent，在执行前对代码进行评审  
reviewer = autogen.AssistantAgent(  
    name="ReviewerAgent",  
    llm_config=llm_config,  
    system_message="""  
    You are a senior Python code reviewer.  
    Review code for correctness, efficiency, and security.  
    If the code is acceptable, reply: 'APPROVED: proceed.'  
    If not, explain what to fix clearly and concisely.  
    """,  
)  
  
## GroupChat 让三个智能体以圆桌方式相互交流  
group_chat = autogen.GroupChat(  
    agents=[user_proxy, assistant, reviewer],  
    messages=[],  
    max_round=15,  
    speaker_selection_method="auto",   # AutoGen 根据上下文选择下一位发言者  
)  
  
manager = autogen.GroupChatManager(  
    groupchat=group_chat,  
    llm_config=llm_config,  
)  
  
## 启动小组工作流  
user_proxy.initiate_chat(  
    recipient=manager,  
    message=task,  
    clear_history=True,  
 )
```

在这套三智能体配置下，流程变成了Coder 编写代码，Reviewer 评审代码，Executor 执行通过评审的版本。这模拟的是真实工程团队中的代码评审流程，并且完全自动化。

## 基于 AutoGen 的多智能体系统：优势

- **任务拆解与专业化**：复杂工作流被拆分到职责聚焦的智能体上，输出质量相比单智能体提示有明显提升。
- **自我纠错循环**：智能体迭代地评审并修订彼此的输出，在不需要人工介入的情况下减少错误。
- **可扩展性**：可以向小组中加入新的智能体，而不必重构已有结构，Manager 会动态完成消息路由。
- **后端可选**：AutoGen 可与 OpenAI、Azure OpenAI、Anthropic、Mistral 以及本地模型（通过 LiteLLM）协同工作，避免厂商锁定。
- **代码执行集成**：在智能体循环中原生支持运行、测试和调试代码，是工程任务上的关键差异点。
- **支持 human-in-the-loop**：可以把任意智能体切换为交互模式，在任意决策节点引入专家判断，而不必重新设计架构。
- **活跃的开源社区**：AutoGen 的 GitHub 社区规模庞大、更新频繁，扩展生态成熟，被 Microsoft 以及金融、医疗等行业的公司采用。
- **可审计性**：智能体之间的完整对话历史都会被记录并可检视，相比黑盒式的单模型输出，决策过程更容易追溯。

## 总结

多智能体系统代表了 AI 解决复杂问题方式上的一次真正的架构演进。把认知能力分布到一组专精、协作的智能体上，而不是让单个模型包揽全部事情，可以在同一个架构动作中得到模块化、自我纠错、可扩展性和可解释性。

本文展示的模式，从双智能体的代码执行循环，到三智能体的 GroupChat 评审都不是研究原型，而是已经在金融、医疗、零售等行业规模化运行的生产可部署架构。

本文给出的代码示例只是一个可运行的起点，真正的深度在于：如何设计智能体角色、如何处理失败模式、如何保护执行环境，以及如何把智能体与组织自身的数据层连接起来。这才是工程判断真正发挥作用的地方。

by PySquad

喜欢就关注一下吧：
