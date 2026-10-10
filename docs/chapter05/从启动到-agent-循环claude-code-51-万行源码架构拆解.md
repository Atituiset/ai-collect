# 从启动到 Agent 循环：Claude Code 51 万行源码架构拆解

> 「Claude Code 源码拆解」系列 · 第二篇

Claude Code 的架构比你想象的复杂得多。这篇文章我从启动入口开始，一路拆到 Agent 核心循环，顺带挖出几个让我意外的设计决策。看完你会明白：为什么一个"命令行工具"里有 439 个 React 组件，为什么安全代码比执行代码多好几倍。

直接进。

一、整体架构长啥样

整个系统大致分五层：

cli.tsx ─── 命令行入口，处理 --version 这种快速路径  
   │  
   ▼  
main.tsx─── 785KB 的"万物初始化"（是的，单文件 785KB）  
   │  
   ▼  
QueryEngine / query.ts─── Agent 循环的核心  
   │                          一直转圈：调 API → 要工具？执行 → 再调 API  
   │  
   ├──→ tools/─── 40+ 个工具（Bash、文件读写、搜索、Agent...）  
   ├──→ services/─── API 客户端、MCP 协议、上下文压缩、遥测  
   ├──→ components/─── React/Ink 终端 UI（439 个组件文件，没看错）  
   └──→ bridge/ ─── 和 Desktop 应用 / 远程环境的桥接

接下来展开讲几个我觉得值得说的部分。

二、启动过程：一场毫秒级的优化战争

> --version 的零加载快速路径

打开 cli.tsx，第一段逻辑就很有意思：

async function main():Promise&lt;void&gt; {  
constargs = process.argv.slice(2);  
  
// --version: 不加载任何模块，直接打印退出  
if(args.length ===1&& (args[0] ==="--version"|| args[0] ==="-v")) {  
    console.log(`${"2.1.88"} (Claude Code)`);  
    return;  
  }  
  
// 其他路径才开始加载模块  
const{ profileCheckpoint } =awaitimport("../utils/startupProfiler.js");  
profileCheckpoint("cli\_entry");  
// ...  
}

这个项目依赖 192 个 npm 包。正常启动要加载一大堆模块，但 --version 这种命令根本不需要——所以它被放在所有 import 之前，直接 return。

看起来不起眼，但对一个重量级 CLI 工具来说，这种"能不加载就不加载"的思路贯穿了整个入口层。后面还有 --claude-in-chrome-mcp、--daemon-worker 等好几个快速路径，都是"判断 args → 动态 import 最少量的代码 → 干完活就走"。

> import 间隙里的"偷跑"

main.tsx 的开头更有意思：

// 这些副作用必须在所有其他 import 之前运行：  
import{ profileCheckpoint }from"./utils/startupProfiler.js";  
profileCheckpoint("main\_tsx\_entry");  
  
// 启动 MDM 配置子进程，让它和后面的 import 并行跑  
import{ startMdmRawRead }from"./utils/settings/mdm/rawRead.js";  
startMdmRawRead();  
  
// 启动 macOS 钥匙串预读取（OAuth + API Key），也并行  
import{ startKeychainPrefetch }from"./utils/secureStorage/keychainPrefetch.js";  
startKeychainPrefetch();  
  
// 然后才是正常的 100 多行 import...  
import{ CommandasCommanderCommand }from"@commander-js/extra-typings";  
importchalkfrom"chalk";  
importReactfrom"react";

Node.js 的 import 是同步串行的，光是 evaluate 这 100 多个 import 大约要 135ms。在这段时间里，钥匙串读取、MDM 配置查询这些 I/O 操作已经在后台跑了。等 import 完了，它们也差不多完了。

这种在 module evaluation 阶段"偷跑"异步任务的做法，不能说优雅，但确实有效。注释里也很实诚地解释了为什么要这么做——因为 isRemoteManagedSettingsEligible() 在 macOS 上会同步 spawn 两次 keychain 读取，加起来要 65ms。

三、Agent 循环：看起来简单，做起来要命

> 核心循环本身并不复杂

所有 AI Agent 的内核都是同一个模式：

你说话 → 调 API → API 回答  
                    │  
              回答里有工具调用？  
             /              \  
            是                 否  
            │                  │  
        执行工具           结束，返回  
        把结果塞回去  
        重新调 API ──────→ API 回答

这个循环实现在 query.ts（67KB）里。核心参数长这样：

export type QueryParams= {  
  messages:Message[];        // 对话历史  
  systemPrompt:SystemPrompt;// 系统提示词  
  canUseTool:CanUseToolFn;  // "这个工具能不能用？"的判断函数  
  toolUseContext:ToolUseContext;  
  maxTurns?:number;         // 别让它无限循环  
  taskBudget?: { total:number};// token 预算  
};

循环本身不难理解。难的是围绕这个循环长出来的一堆东西。

> 并发执行工具：不是所有工具都能一起跑

当 AI 一次回复里要求调 5 个工具时，怎么执行？全串行太慢，全并行又有风险——如果两个 Bash 命令同时改同一个文件呢？

StreamingToolExecutor 的解法是按工具类型分区：

// 工具编排的核心逻辑  
for(const{ isConcurrencySafe, blocks }of  
     partitionToolCalls(toolUseMessages, currentContext)) {  
  
if(isConcurrencySafe) {  
    // 只读工具（读文件、grep、glob）→ 并发跑，最多 10 路  
    for await(constupdateofrunToolsConcurrently(...)) {  
      yieldupdate  
    }  
  }else{  
    // 有副作用的工具（bash、写文件）→ 一个一个来  
    for await(constupdateofrunToolsSerially(...)) {  
      yieldupdate  
    }  
  }  
}

每个工具在定义时就声明了自己是不是 isConcurrencySafe。FileRead、Grep、Glob 这种只读工具可以放心并发；Bash、FileEdit 这种有副作用的必须排队。

并发上限默认 10，可以通过环境变量 CLAUDE\_CODE\_MAX\_TOOL\_USE\_CONCURRENCY 调。这个设计不复杂，但很实用。

四、40 多个工具里，Bash 最离谱

工具分几大类：文件操作（读/写/编辑/搜索）、Shell 执行、Web 访问、子 Agent、MCP 协议对接、规划模式等。

其中 BashTool 是绝对的重量级。主文件 157KB 已经够夸张了，但它的配套文件更吓人：

BashTool/  
├── BashTool.tsx              # 157KB - 主逻辑  
├── bashSecurity.ts           # 100KB - 安全分析  
├── bashPermissions.ts        #  96KB - 权限判定  
├── readOnlyValidation.ts     #  67KB - 只读校验  
├── pathValidation.ts         #  43KB - 路径校验  
├── sedValidation.ts          #  21KB - sed 命令校验  
├── prompt.ts                 #  21KB - 给 AI 看的工具说明  
└── ... 还有 8 个文件

安全相关的代码加起来超过 **300KB**。也就是说，"检查这条命令安不安全"的代码量，是"执行这条命令"的好几倍。

为什么要这么大？因为他们自己写了一个 Bash 解析器（utils/bash/bashParser.ts，128KB），能把 shell 命令解析成 AST。只有理解了命令的语义结构，才能准确判断 rm -rf / 是危险的，而 echo "rm -rf /" 是安全的。

> 项目里的 Bash 解析器全家桶加起来有 237KB。一个 AI 编程工具，为了安全，自己写了个 shell 解析器。这个投入说明了一件事——让 AI 执行命令这件事，安全不是锦上添花，是安身立命的根基。

五、108 个被抹掉的模块

读代码时到处能看到这种模式：

const feature = (flag) =>false;// 永远返回 false  
  
constcoordinatorModeModule =feature("COORDINATOR\_MODE")  
  ?require("./coordinator/coordinatorMode.js")  
  :null;  
  
constassistantModule =feature("KAIROS")  
  ?require("./assistant/index.js")  
  :null;

feature() 是 Bun 运行时的编译时函数。Anthropic 内部构建时某些 flag 返回 true，代码被保留；发到 npm 的版本全是 false，代码被死代码消除（DCE）干掉了。

通过搜集所有 feature() 调用，可以发现有 **108 个模块**在发布版中不存在。其中一些挺有意思的：

**COORDINATOR\_MODE** — 多 Agent 协调器，一个指挥多个

**DAEMON** — 后台守护进程，不退出一直跑

**WEB\_BROWSER\_TOOL** — 浏览器自动化工具

**VOICE\_MODE** — 语音输入

**CONTEXT\_COLLAPSE** — 上下文智能折叠（比压缩更激进）

**BUDDY** — 就是上篇说的电子宠物系统

这些模块画出了一条很清晰的路线：从"编程助手"走向"自主代理"。KAIROS 尤其值得关注——它代表了一种不需要人一直盯着的、后台自己干活的 Agent 形态。

六、三个出乎意料的工程选择

> 终端 UI 用 React 写的，439 个组件

439 个组件文件。用的是 React + Ink（一个把 React 渲染到终端的框架）。

PromptInput.tsx 有 347KB。一个命令行输入框，347KB。

里面有虚拟滚动（VirtualMessageList.tsx，145KB）、模糊搜索（FuzzyPicker.tsx）、完整的设计系统（Dialog、Tabs、ProgressBar、ThemedBox……）。

一个命令行工具的 UI 复杂度，赶上一个中型 Web 应用了。

> 系统提示词里的信息量

constants/prompts.ts（53KB）是系统提示词的核心。几个发现：

// 最前沿的模型标识  
const FRONTIER\_MODEL\_NAME = "Claude Opus 4.6";  
  
const CLAUDE\_4\_5\_OR\_4\_6\_MODEL\_IDS = {  
  opus: "claude-opus-4-6",  
  sonnet: "claude-sonnet-4-6",  
  haiku: "claude-haiku-4-5-20251001",  
};

还有一个 Prompt Cache 的优化：系统提示词被分成"全局可缓存"和"用户特定"两段，中间用一个 boundary 标记隔开。boundary 之前的（工具描述、通用规则）可以跨用户复用缓存，之后的（项目记忆、个人设置）不能。这对降低 API 开销很关键。

> 每一分钱都在追踪

cost-tracker.ts 追踪了 API 调用的所有细节：总费用（精确到美元）、输入/输出 token 数、缓存命中 token 数、API 耗时、新增/删除代码行数。每次对话结束都能告诉你花了多少钱，改了多少行代码。

这个设计我挺欣赏——AI 工具对用户最大的不信任感之一就是"不知道在花多少钱"，把账单透明化，是尊重用户的基本动作。

写在最后

拆完这一层之后，我对 Claude Code 的印象从"AI 接了个 API"彻底升级了。

启动阶段的毫秒级抠优化，Agent 循环里按类型分区的并发调度，BashTool 里三百多 KB 的安全代码——这些东西加在一起，说明这是一个被真实用户打磨了很久的工具，不是一个 demo。

当然也有让我皱眉的地方：main.tsx 单文件 785KB，PromptInput.tsx 单文件 347KB，这种文件规模放哪都是维护噩梦。但也许 Anthropic 的工程师觉得：有 AI 帮我看代码，大文件也没那么可怕了？

> **一个问题留给大家：**
>
> main.tsx
>
> 单文件 785KB、终端 UI 439 个组件——  
> 你觉得这是技术债，还是 AI 辅助开发时代"文件大小不再重要"的新信号？  
>   
> **评论区聊聊 👇**

**下一篇，我们钻进 query.ts——那个 67KB 的 Agent 核心，看它怎么处理流式响应、怎么做错误恢复，还有那段"巫师注释"背后到底踩了什么坑。**

本文基于 npm 包 @anthropic-ai/claude-code v2.1.88 的反编译源码进行分析  
仅用于技术学习 · 源码版权归 Anthropic 所有
