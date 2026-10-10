import { defineConfig } from 'vitepress'

// GitHub Pages 项目站点 https://atituiset.github.io/ai-collect/
const base = '/ai-collect/'
const repo = 'https://github.com/Atituiset/ai-collect'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  lang: 'zh-CN',
  title: 'AI 技术全景',
  titleTemplate: '从 Transformer 到 Agent 工程',
  description:
    '一本关于 AI 基础架构、推理优化与 Agent 工程的开源电子书：Transformer 与模型架构、LLM 推理优化、AI Infra 与硬件加速、Agent Harness 工程、Claude Code 源码解读与行业全景。',

  base,
  lastUpdated: true,
  cleanUrls: true,
  srcExclude: ['README.md'],

  markdown: {
    // shiki 3.x 不再内置 cuda 语法, 用 cpp 高亮兜底
    languageAlias: { cuda: 'cpp' },
    // 章节内含 KaTeX 数学公式
    math: true,
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#2e7d5b', media: '(prefers-color-scheme: light)' }],
    ['meta', { name: 'theme-color', content: '#18191c', media: '(prefers-color-scheme: dark)' }],
    ['meta', { property: 'og:title', content: 'AI 技术全景：从 Transformer 到 Agent 工程' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          '一本关于 AI 基础架构、推理优化与 Agent 工程的开源电子书：Transformer、LLM 推理优化、AI Infra、Agent Harness 与 Claude Code 源码解读。',
      },
    ],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
    // 学习笔记悬浮组件（localStorage），VitePress SPA 路由切换由 theme/index.ts 通知其重挂载
    ['script', { src: `${base}notes.js` }],
  ],

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/favicon.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: 'Transformer 基础', link: '/chapter01/' },
      { text: 'LLM 推理优化', link: '/chapter02/' },
      { text: 'Agent 工程', link: '/chapter04/' },
      { text: 'Claude Code 源码', link: '/chapter05/' },
    ],

    sidebar: [
      {
        text: '第1章 Transformer 与模型架构基础',
        link: '/chapter01/',
        collapsed: false,
        items: [
          { text: '终于把 Transformer 算法搞懂了！！', link: '/chapter01/终于把-transformer-算法搞懂了' },
          { text: 'Transformer 推理全流程简单拆解', link: '/chapter01/transformer-推理全流程简单拆解' },
          { text: 'Transformer 全貌——一个 token 从输入到输出的完整旅程', link: '/chapter01/transformer-全貌一个-token-从输入到输出的完整旅程' },
          { text: 'AI算法：MoE混合专家架构', link: '/chapter01/ai算法moe混合专家架构' },
          { text: 'Attention 机制与 LLM 的数学能力', link: '/chapter01/attention-机制与-llm-的数学能力' },
          { text: '大模型上下文窗口为什么有限？这3个问题必须搞清楚', link: '/chapter01/大模型上下文窗口为什么有限这3个问题必须搞清楚' },
        ],
      },
      {
        text: '第2章 LLM 推理机制与优化',
        link: '/chapter02/',
        collapsed: false,
        items: [
          { text: 'LLM 推理为什么先慢后快？从 Prefill、Decode 到 KV Cache 讲清楚', link: '/chapter02/llm-推理为什么先慢后快从-prefilldecode-到-kv-cache-讲清楚' },
          { text: '大模型推理为什么慢？从 Prefill、Decode 到 KV Cache，一文讲透推理加速全流程', link: '/chapter02/大模型推理为什么慢从-prefilldecode-到-kv-cache一文讲透推理加速全流程' },
          { text: '大模型推理到底在做什么？从 Prefill、Decode 到 vLLM，一次讲透', link: '/chapter02/大模型推理到底在做什么从-prefilldecode-到-vllm一次讲透' },
          { text: '彻底搞懂 LLM 推理全过程', link: '/chapter02/彻底搞懂-llm-推理全过程' },
          { text: 'LLM推理全流程解析：从输入到输出的底层逻辑', link: '/chapter02/llm推理全流程解析从输入到输出的底层逻辑' },
          { text: '理解 KV Cache：LLM 推理为什么能越写越快', link: '/chapter02/理解-kv-cachellm-推理为什么能越写越快' },
          { text: 'LLM推理优化核心技术', link: '/chapter02/llm推理优化核心技术' },
          { text: '大模型推理系统入门：从模型优化到调度优化', link: '/chapter02/大模型推理系统入门从模型优化到调度优化' },
          { text: 'vLLM 如何管理 KV Cache：从 vLLM 本地 Block 管理到跨实例缓存池', link: '/chapter02/vllm-如何管理-kv-cache从-vllm-本地-block-管理到跨实例缓存池' },
          { text: 'LLM大模型是如何运行起来的：从Hello World到vLLM的运行时深度解析', link: '/chapter02/llm大模型是如何运行起来的从hello-world到vllm的运行时深度解析' },
          { text: 'SGLang推理优化-调度器核心ScheduleBatch', link: '/chapter02/sglang推理优化-调度器核心schedulebatch' },
          { text: 'SGLang推理优化-Scheduler 到 Forward 推理的 Batch 数据流', link: '/chapter02/sglang推理优化-scheduler-到-forward-推理的-batch-数据流' },
          { text: 'SGLang推理优化-DP并行', link: '/chapter02/sglang推理优化-dp并行' },
          { text: '2026 年面向 LLM 的 RL方法总结：从 PPO 到 DPO 到 GRPO，再到多智能体 RL', link: '/chapter02/2026-年面向-llm-的-rl方法总结从-ppo-到-dpo-到-grpo再到多智能体-rl' },
        ],
      },
      {
        text: '第3章 AI Infra 与硬件加速',
        link: '/chapter03/',
        collapsed: false,
        items: [
          { text: '万字长文｜彻底理解 GPU 架构与深度学习加速原理', link: '/chapter03/万字长文彻底理解-gpu-架构与深度学习加速原理' },
          { text: '万字入门AI Infra：深入理解大模型中的数学与Infra优化', link: '/chapter03/万字入门ai-infra深入理解大模型中的数学与infra优化' },
          { text: 'AI Infra入门干货总结：大模型是如何高效推理的', link: '/chapter03/ai-infra入门干货总结大模型是如何高效推理的' },
          { text: 'llama.cpp 源码解析（三）：114 个 Builder——一套代码如何驾驭 130+ 种架构', link: '/chapter03/llamacpp-源码解析三114-个-builder一套代码如何驾驭-130-种架构' },
        ],
      },
      {
        text: '第4章 Agent 架构与 Harness 工程',
        link: '/chapter04/',
        collapsed: false,
        items: [
          { text: 'OpenClaw 的前世今生：一张图讲明白 从“大模型会说”到“工程化会做”的演进', link: '/chapter04/openclaw-的前世今生一张图讲明白-从大模型会说到工程化会做的演进' },
          { text: 'OpenClaw vs Hermes 架构深度拆解', link: '/chapter04/openclaw-vs-hermes-架构深度拆解' },
          { text: '(99+ 封私信 _ 80 条消息) AI agent到底有多大创新？ - 知乎', link: '/chapter04/99-封私信-80-条消息-ai-agent到底有多大创新---知乎' },
          { text: '推理 → 行动 → 观察：用 LangChain + Python 实现一个智能体循环', link: '/chapter04/推理-行动-观察用-langchain-python-实现一个智能体循环' },
          { text: 'AutoGen 多智能体实战：Coder、Reviewer、Executor 自动跑完一次代码评审', link: '/chapter04/autogen-多智能体实战coderreviewerexecutor-自动跑完一次代码评审' },
          { text: '一个爆火的 AI Agent 工作流项目', link: '/chapter04/一个爆火的-ai-agent-工作流项目' },
          { text: 'Agent Harness 解析：智能体架构深度拆解', link: '/chapter04/agent-harness-解析智能体架构深度拆解' },
          { text: 'Agent Harness Engineering全景综述，决定Agent系统的上限', link: '/chapter04/agent-harness-engineering全景综述决定agent系统的上限' },
          { text: '大白话讲透驾驭工程(Harness Engineering)六大组件', link: '/chapter04/大白话讲透驾驭工程harness-engineering六大组件' },
          { text: '从Prompt、Context到Harness，工程的三次进化与终局之战', link: '/chapter04/从promptcontext到harness工程的三次进化与终局之战' },
          { text: '从零搭建 Harness Engineering 架构：Rule、Skill、Sub-Agent 完整落地路径', link: '/chapter04/从零搭建-harness-engineering-架构ruleskillsub-agent-完整落地路径' },
          { text: '9大Harness的记忆能力对比', link: '/chapter04/9大harness的记忆能力对比' },
          { text: 'Rust 正在接管 AI Agent 的基础设施层', link: '/chapter04/rust-正在接管-ai-agent-的基础设施层' },
        ],
      },
      {
        text: '第5章 Claude Code 源码深度解读',
        link: '/chapter05/',
        collapsed: false,
        items: [
          { text: 'Claude Code 源码解读：包含Agent Loop、Context 管理、Tool 调用、Memory、Muti Agent等模块', link: '/chapter05/claude-code-源码解读包含agent-loopcontext-管理tool-调用memorymuti-agent等模块' },
          { text: 'Claude Code 架构详解', link: '/chapter05/claude-code-架构详解' },
          { text: 'Claude Code 泄漏源码分析', link: '/chapter05/claude-code-泄漏源码分析' },
          { text: '从Claude-Code源码学习系统提示词的工程系统', link: '/chapter05/从claude-code源码学习系统提示词的工程系统' },
          { text: '从启动到 Agent 循环：Claude Code 51 万行源码架构拆解', link: '/chapter05/从启动到-agent-循环claude-code-51-万行源码架构拆解' },
          { text: 'Claude Code 的 12 个可复用的 Agentic Harness 设计模式', link: '/chapter05/claude-code-的-12-个可复用的-agentic-harness-设计模式' },
          { text: '又一个神级 Claude Code 插件，开源了！', link: '/chapter05/又一个神级-claude-code-插件开源了' },
          { text: '基于顶级 Agent（Claude Code）的 Harness 工程搭建式业务 Agent 评测方案', link: '/chapter05/基于顶级-agentclaude-code的-harness-工程搭建式业务-agent-评测方案' },
        ],
      },
      {
        text: '第6章 行业全景与职业发展',
        link: '/chapter06/',
        collapsed: false,
        items: [
          { text: '大模型领域核心技术全景图（2026年版）', link: '/chapter06/大模型领域核心技术全景图2026年版' },
          { text: '(99+ 封私信 _ 80 条消息) Skill不就是prompt吗？为啥被吹成这样？ - 知乎', link: '/chapter06/99-封私信-80-条消息-skill不就是prompt吗为啥被吹成这样---知乎' },
          { text: '250 行 Python 写一个 CLI AI Agent', link: '/chapter06/250-行-python-写一个-cli-ai-agent' },
          { text: 'Hello Agent CLI 开篇', link: '/chapter06/hello-agent-cli-开篇' },
          { text: '分享一些 AI 面试经验：简历与能力框架', link: '/chapter06/分享一些-ai-面试经验简历与能力框架' },
        ],
      },
    ],

    outline: {
      level: [2, 3],
      label: '本页大纲',
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    editLink: {
      pattern: `${repo}/edit/main/:path`,
      text: '在 GitHub 上编辑此页',
    },

    socialLinks: [{ icon: 'github', link: repo }],

    docFooter: { prev: '上一页', next: '下一页' },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: { dateStyle: 'short', timeStyle: 'short' },
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    skipToContentLabel: '跳转到内容',

    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2026 AI Collect',
    },
  },
})
