// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="index.html"><strong aria-hidden="true">1.</strong> 简介</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter01/index.html"><strong aria-hidden="true">2.</strong> 第1章 Transformer 与模型架构基础</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter01/终于把-transformer-算法搞懂了.html"><strong aria-hidden="true">2.1.</strong> 终于把 Transformer 算法搞懂了！！</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter01/transformer-推理全流程简单拆解.html"><strong aria-hidden="true">2.2.</strong> Transformer 推理全流程简单拆解</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter01/transformer-全貌一个-token-从输入到输出的完整旅程.html"><strong aria-hidden="true">2.3.</strong> Transformer 全貌——一个 token 从输入到输出的完整旅程</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter01/ai算法moe混合专家架构.html"><strong aria-hidden="true">2.4.</strong> AI算法：MoE混合专家架构</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter01/attention-机制与-llm-的数学能力.html"><strong aria-hidden="true">2.5.</strong> Attention 机制与 LLM 的数学能力</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter01/大模型上下文窗口为什么有限这3个问题必须搞清楚.html"><strong aria-hidden="true">2.6.</strong> 大模型上下文窗口为什么有限？这3个问题必须搞清楚</a></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/index.html"><strong aria-hidden="true">3.</strong> 第2章 LLM 推理机制与优化</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/llm-推理为什么先慢后快从-prefilldecode-到-kv-cache-讲清楚.html"><strong aria-hidden="true">3.1.</strong> LLM 推理为什么先慢后快？从 Prefill、Decode 到 KV Cache 讲清楚</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/大模型推理为什么慢从-prefilldecode-到-kv-cache一文讲透推理加速全流程.html"><strong aria-hidden="true">3.2.</strong> 大模型推理为什么慢？从 Prefill、Decode 到 KV Cache，一文讲透推理加速全流程</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/大模型推理到底在做什么从-prefilldecode-到-vllm一次讲透.html"><strong aria-hidden="true">3.3.</strong> 大模型推理到底在做什么？从 Prefill、Decode 到 vLLM，一次讲透</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/彻底搞懂-llm-推理全过程.html"><strong aria-hidden="true">3.4.</strong> 彻底搞懂 LLM 推理全过程</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/llm推理全流程解析从输入到输出的底层逻辑.html"><strong aria-hidden="true">3.5.</strong> LLM推理全流程解析：从输入到输出的底层逻辑</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/理解-kv-cachellm-推理为什么能越写越快.html"><strong aria-hidden="true">3.6.</strong> 理解 KV Cache：LLM 推理为什么能越写越快</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/llm推理优化核心技术.html"><strong aria-hidden="true">3.7.</strong> LLM推理优化核心技术</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/大模型推理系统入门从模型优化到调度优化.html"><strong aria-hidden="true">3.8.</strong> 大模型推理系统入门：从模型优化到调度优化</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/vllm-如何管理-kv-cache从-vllm-本地-block-管理到跨实例缓存池.html"><strong aria-hidden="true">3.9.</strong> vLLM 如何管理 KV Cache：从 vLLM 本地 Block 管理到跨实例缓存池</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/llm大模型是如何运行起来的从hello-world到vllm的运行时深度解析.html"><strong aria-hidden="true">3.10.</strong> LLM大模型是如何运行起来的：从Hello World到vLLM的运行时深度解析</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/sglang推理优化-调度器核心schedulebatch.html"><strong aria-hidden="true">3.11.</strong> SGLang推理优化-调度器核心ScheduleBatch</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/sglang推理优化-scheduler-到-forward-推理的-batch-数据流.html"><strong aria-hidden="true">3.12.</strong> SGLang推理优化-Scheduler 到 Forward 推理的 Batch 数据流</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/sglang推理优化-dp并行.html"><strong aria-hidden="true">3.13.</strong> SGLang推理优化-DP并行</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter02/2026-年面向-llm-的-rl方法总结从-ppo-到-dpo-到-grpo再到多智能体-rl.html"><strong aria-hidden="true">3.14.</strong> 2026 年面向 LLM 的 RL方法总结：从 PPO 到 DPO 到 GRPO，再到多智能体 RL</a></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter03/index.html"><strong aria-hidden="true">4.</strong> 第3章 AI Infra 与硬件加速</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter03/万字长文彻底理解-gpu-架构与深度学习加速原理.html"><strong aria-hidden="true">4.1.</strong> 万字长文｜彻底理解 GPU 架构与深度学习加速原理</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter03/万字入门ai-infra深入理解大模型中的数学与infra优化.html"><strong aria-hidden="true">4.2.</strong> 万字入门AI Infra：深入理解大模型中的数学与Infra优化</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter03/ai-infra入门干货总结大模型是如何高效推理的.html"><strong aria-hidden="true">4.3.</strong> AI Infra入门干货总结：大模型是如何高效推理的</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter03/llamacpp-源码解析三114-个-builder一套代码如何驾驭-130-种架构.html"><strong aria-hidden="true">4.4.</strong> llama.cpp 源码解析（三）：114 个 Builder——一套代码如何驾驭 130+ 种架构</a></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/index.html"><strong aria-hidden="true">5.</strong> 第4章 Agent 架构与 Harness 工程</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/openclaw-的前世今生一张图讲明白-从大模型会说到工程化会做的演进.html"><strong aria-hidden="true">5.1.</strong> OpenClaw 的前世今生：一张图讲明白 从“大模型会说”到“工程化会做”的演进</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/openclaw-vs-hermes-架构深度拆解.html"><strong aria-hidden="true">5.2.</strong> OpenClaw vs Hermes 架构深度拆解</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/99-封私信-80-条消息-ai-agent到底有多大创新---知乎.html"><strong aria-hidden="true">5.3.</strong> (99+ 封私信 _ 80 条消息) AI agent到底有多大创新？ - 知乎</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/推理-行动-观察用-langchain-python-实现一个智能体循环.html"><strong aria-hidden="true">5.4.</strong> 推理 → 行动 → 观察：用 LangChain + Python 实现一个智能体循环</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/autogen-多智能体实战coderreviewerexecutor-自动跑完一次代码评审.html"><strong aria-hidden="true">5.5.</strong> AutoGen 多智能体实战：Coder、Reviewer、Executor 自动跑完一次代码评审</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/一个爆火的-ai-agent-工作流项目.html"><strong aria-hidden="true">5.6.</strong> 一个爆火的 AI Agent 工作流项目</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/agent-harness-解析智能体架构深度拆解.html"><strong aria-hidden="true">5.7.</strong> Agent Harness 解析：智能体架构深度拆解</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/agent-harness-engineering全景综述决定agent系统的上限.html"><strong aria-hidden="true">5.8.</strong> Agent Harness Engineering全景综述，决定Agent系统的上限</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/大白话讲透驾驭工程harness-engineering六大组件.html"><strong aria-hidden="true">5.9.</strong> 大白话讲透驾驭工程(Harness Engineering)六大组件</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/从promptcontext到harness工程的三次进化与终局之战.html"><strong aria-hidden="true">5.10.</strong> 从Prompt、Context到Harness，工程的三次进化与终局之战</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/从零搭建-harness-engineering-架构ruleskillsub-agent-完整落地路径.html"><strong aria-hidden="true">5.11.</strong> 从零搭建 Harness Engineering 架构：Rule、Skill、Sub-Agent 完整落地路径</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/9大harness的记忆能力对比.html"><strong aria-hidden="true">5.12.</strong> 9大Harness的记忆能力对比</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter04/rust-正在接管-ai-agent-的基础设施层.html"><strong aria-hidden="true">5.13.</strong> Rust 正在接管 AI Agent 的基础设施层</a></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/index.html"><strong aria-hidden="true">6.</strong> 第5章 Claude Code 源码深度解读</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/claude-code-源码解读包含agent-loopcontext-管理tool-调用memorymuti-agent等模块.html"><strong aria-hidden="true">6.1.</strong> Claude Code 源码解读：包含Agent Loop、Context 管理、Tool 调用、Memory、Muti Agent等模块</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/claude-code-架构详解.html"><strong aria-hidden="true">6.2.</strong> Claude Code 架构详解</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/claude-code-泄漏源码分析.html"><strong aria-hidden="true">6.3.</strong> Claude Code 泄漏源码分析</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/从claude-code源码学习系统提示词的工程系统.html"><strong aria-hidden="true">6.4.</strong> 从Claude-Code源码学习系统提示词的工程系统</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/从启动到-agent-循环claude-code-51-万行源码架构拆解.html"><strong aria-hidden="true">6.5.</strong> 从启动到 Agent 循环：Claude Code 51 万行源码架构拆解</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/claude-code-的-12-个可复用的-agentic-harness-设计模式.html"><strong aria-hidden="true">6.6.</strong> Claude Code 的 12 个可复用的 Agentic Harness 设计模式</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/又一个神级-claude-code-插件开源了.html"><strong aria-hidden="true">6.7.</strong> 又一个神级 Claude Code 插件，开源了！</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter05/基于顶级-agentclaude-code的-harness-工程搭建式业务-agent-评测方案.html"><strong aria-hidden="true">6.8.</strong> 基于顶级 Agent（Claude Code）的 Harness 工程搭建式业务 Agent 评测方案</a></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter06/index.html"><strong aria-hidden="true">7.</strong> 第6章 行业全景与职业发展</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter06/大模型领域核心技术全景图2026年版.html"><strong aria-hidden="true">7.1.</strong> 大模型领域核心技术全景图（2026年版）</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter06/99-封私信-80-条消息-skill不就是prompt吗为啥被吹成这样---知乎.html"><strong aria-hidden="true">7.2.</strong> (99+ 封私信 _ 80 条消息) Skill不就是prompt吗？为啥被吹成这样？ - 知乎</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter06/250-行-python-写一个-cli-ai-agent.html"><strong aria-hidden="true">7.3.</strong> 250 行 Python 写一个 CLI AI Agent</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter06/hello-agent-cli-开篇.html"><strong aria-hidden="true">7.4.</strong> Hello Agent CLI 开篇</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="chapter06/分享一些-ai-面试经验简历与能力框架.html"><strong aria-hidden="true">7.5.</strong> 分享一些 AI 面试经验：简历与能力框架</a></span></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split('#')[0].split('?')[0];
        if (current_page.endsWith('/')) {
            current_page += 'index.html';
        }
        const links = Array.prototype.slice.call(this.querySelectorAll('a'));
        const l = links.length;
        for (let i = 0; i < l; ++i) {
            const link = links[i];
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The 'index' page is supposed to alias the first chapter in the book.
            if (link.href === current_page
                || i === 0
                && path_to_root === ''
                && current_page.endsWith('/index.html')) {
                link.classList.add('active');
                let parent = link.parentElement;
                while (parent) {
                    if (parent.tagName === 'LI' && parent.classList.contains('chapter-item')) {
                        parent.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', e => {
            if (e.target.tagName === 'A') {
                const clientRect = e.target.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                sessionStorage.setItem('sidebar-scroll-offset', clientRect.top - sidebarRect.top);
            }
        }, { passive: true });
        const sidebarScrollOffset = sessionStorage.getItem('sidebar-scroll-offset');
        sessionStorage.removeItem('sidebar-scroll-offset');
        if (sidebarScrollOffset !== null) {
            // preserve sidebar scroll position when navigating via links within sidebar
            const activeSection = this.querySelector('.active');
            if (activeSection) {
                const clientRect = activeSection.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                const currentOffset = clientRect.top - sidebarRect.top;
                this.scrollTop += currentOffset - parseFloat(sidebarScrollOffset);
            }
        } else {
            // scroll sidebar to current active section when navigating via
            // 'next/previous chapter' buttons
            const activeSection = document.querySelector('#mdbook-sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        const sidebarAnchorToggles = document.querySelectorAll('.chapter-fold-toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(el => {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define('mdbook-sidebar-scrollbox', MDBookSidebarScrollbox);


// ---------------------------------------------------------------------------
// Support for dynamically adding headers to the sidebar.

(function() {
    // This is used to detect which direction the page has scrolled since the
    // last scroll event.
    let lastKnownScrollPosition = 0;
    // This is the threshold in px from the top of the screen where it will
    // consider a header the "current" header when scrolling down.
    const defaultDownThreshold = 150;
    // Same as defaultDownThreshold, except when scrolling up.
    const defaultUpThreshold = 300;
    // The threshold is a virtual horizontal line on the screen where it
    // considers the "current" header to be above the line. The threshold is
    // modified dynamically to handle headers that are near the bottom of the
    // screen, and to slightly offset the behavior when scrolling up vs down.
    let threshold = defaultDownThreshold;
    // This is used to disable updates while scrolling. This is needed when
    // clicking the header in the sidebar, which triggers a scroll event. It
    // is somewhat finicky to detect when the scroll has finished, so this
    // uses a relatively dumb system of disabling scroll updates for a short
    // time after the click.
    let disableScroll = false;
    // Array of header elements on the page.
    let headers;
    // Array of li elements that are initially collapsed headers in the sidebar.
    // I'm not sure why eslint seems to have a false positive here.
    // eslint-disable-next-line prefer-const
    let headerToggles = [];
    // This is a debugging tool for the threshold which you can enable in the console.
    let thresholdDebug = false;

    // Updates the threshold based on the scroll position.
    function updateThreshold() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;

        // The number of pixels below the viewport, at most documentHeight.
        // This is used to push the threshold down to the bottom of the page
        // as the user scrolls towards the bottom.
        const pixelsBelow = Math.max(0, documentHeight - (scrollTop + windowHeight));
        // The number of pixels above the viewport, at least defaultDownThreshold.
        // Similar to pixelsBelow, this is used to push the threshold back towards
        // the top when reaching the top of the page.
        const pixelsAbove = Math.max(0, defaultDownThreshold - scrollTop);
        // How much the threshold should be offset once it gets close to the
        // bottom of the page.
        const bottomAdd = Math.max(0, windowHeight - pixelsBelow - defaultDownThreshold);
        let adjustedBottomAdd = bottomAdd;

        // Adjusts bottomAdd for a small document. The calculation above
        // assumes the document is at least twice the windowheight in size. If
        // it is less than that, then bottomAdd needs to be shrunk
        // proportional to the difference in size.
        if (documentHeight < windowHeight * 2) {
            const maxPixelsBelow = documentHeight - windowHeight;
            const t = 1 - pixelsBelow / Math.max(1, maxPixelsBelow);
            const clamp = Math.max(0, Math.min(1, t));
            adjustedBottomAdd *= clamp;
        }

        let scrollingDown = true;
        if (scrollTop < lastKnownScrollPosition) {
            scrollingDown = false;
        }

        if (scrollingDown) {
            // When scrolling down, move the threshold up towards the default
            // downwards threshold position. If near the bottom of the page,
            // adjustedBottomAdd will offset the threshold towards the bottom
            // of the page.
            const amountScrolledDown = scrollTop - lastKnownScrollPosition;
            const adjustedDefault = defaultDownThreshold + adjustedBottomAdd;
            threshold = Math.max(adjustedDefault, threshold - amountScrolledDown);
        } else {
            // When scrolling up, move the threshold down towards the default
            // upwards threshold position. If near the bottom of the page,
            // quickly transition the threshold back up where it normally
            // belongs.
            const amountScrolledUp = lastKnownScrollPosition - scrollTop;
            const adjustedDefault = defaultUpThreshold - pixelsAbove
                + Math.max(0, adjustedBottomAdd - defaultDownThreshold);
            threshold = Math.min(adjustedDefault, threshold + amountScrolledUp);
        }

        if (documentHeight <= windowHeight) {
            threshold = 0;
        }

        if (thresholdDebug) {
            const id = 'mdbook-threshold-debug-data';
            let data = document.getElementById(id);
            if (data === null) {
                data = document.createElement('div');
                data.id = id;
                data.style.cssText = `
                    position: fixed;
                    top: 50px;
                    right: 10px;
                    background-color: 0xeeeeee;
                    z-index: 9999;
                    pointer-events: none;
                `;
                document.body.appendChild(data);
            }
            data.innerHTML = `
                <table>
                  <tr><td>documentHeight</td><td>${documentHeight.toFixed(1)}</td></tr>
                  <tr><td>windowHeight</td><td>${windowHeight.toFixed(1)}</td></tr>
                  <tr><td>scrollTop</td><td>${scrollTop.toFixed(1)}</td></tr>
                  <tr><td>pixelsAbove</td><td>${pixelsAbove.toFixed(1)}</td></tr>
                  <tr><td>pixelsBelow</td><td>${pixelsBelow.toFixed(1)}</td></tr>
                  <tr><td>bottomAdd</td><td>${bottomAdd.toFixed(1)}</td></tr>
                  <tr><td>adjustedBottomAdd</td><td>${adjustedBottomAdd.toFixed(1)}</td></tr>
                  <tr><td>scrollingDown</td><td>${scrollingDown}</td></tr>
                  <tr><td>threshold</td><td>${threshold.toFixed(1)}</td></tr>
                </table>
            `;
            drawDebugLine();
        }

        lastKnownScrollPosition = scrollTop;
    }

    function drawDebugLine() {
        if (!document.body) {
            return;
        }
        const id = 'mdbook-threshold-debug-line';
        const existingLine = document.getElementById(id);
        if (existingLine) {
            existingLine.remove();
        }
        const line = document.createElement('div');
        line.id = id;
        line.style.cssText = `
            position: fixed;
            top: ${threshold}px;
            left: 0;
            width: 100vw;
            height: 2px;
            background-color: red;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(line);
    }

    function mdbookEnableThresholdDebug() {
        thresholdDebug = true;
        updateThreshold();
        drawDebugLine();
    }

    window.mdbookEnableThresholdDebug = mdbookEnableThresholdDebug;

    // Updates which headers in the sidebar should be expanded. If the current
    // header is inside a collapsed group, then it, and all its parents should
    // be expanded.
    function updateHeaderExpanded(currentA) {
        // Add expanded to all header-item li ancestors.
        let current = currentA.parentElement;
        while (current) {
            if (current.tagName === 'LI' && current.classList.contains('header-item')) {
                current.classList.add('expanded');
            }
            current = current.parentElement;
        }
    }

    // Updates which header is marked as the "current" header in the sidebar.
    // This is done with a virtual Y threshold, where headers at or below
    // that line will be considered the current one.
    function updateCurrentHeader() {
        if (!headers || !headers.length) {
            return;
        }

        // Reset the classes, which will be rebuilt below.
        const els = document.getElementsByClassName('current-header');
        for (const el of els) {
            el.classList.remove('current-header');
        }
        for (const toggle of headerToggles) {
            toggle.classList.remove('expanded');
        }

        // Find the last header that is above the threshold.
        let lastHeader = null;
        for (const header of headers) {
            const rect = header.getBoundingClientRect();
            if (rect.top <= threshold) {
                lastHeader = header;
            } else {
                break;
            }
        }
        if (lastHeader === null) {
            lastHeader = headers[0];
            const rect = lastHeader.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            if (rect.top >= windowHeight) {
                return;
            }
        }

        // Get the anchor in the summary.
        const href = '#' + lastHeader.id;
        const a = [...document.querySelectorAll('.header-in-summary')]
            .find(element => element.getAttribute('href') === href);
        if (!a) {
            return;
        }

        a.classList.add('current-header');

        updateHeaderExpanded(a);
    }

    // Updates which header is "current" based on the threshold line.
    function reloadCurrentHeader() {
        if (disableScroll) {
            return;
        }
        updateThreshold();
        updateCurrentHeader();
    }


    // When clicking on a header in the sidebar, this adjusts the threshold so
    // that it is located next to the header. This is so that header becomes
    // "current".
    function headerThresholdClick(event) {
        // See disableScroll description why this is done.
        disableScroll = true;
        setTimeout(() => {
            disableScroll = false;
        }, 100);
        // requestAnimationFrame is used to delay the update of the "current"
        // header until after the scroll is done, and the header is in the new
        // position.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                // Closest is needed because if it has child elements like <code>.
                const a = event.target.closest('a');
                const href = a.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    threshold = targetElement.getBoundingClientRect().bottom;
                    updateCurrentHeader();
                }
            });
        });
    }

    // Takes the nodes from the given head and copies them over to the
    // destination, along with some filtering.
    function filterHeader(source, dest) {
        const clone = source.cloneNode(true);
        clone.querySelectorAll('mark').forEach(mark => {
            mark.replaceWith(...mark.childNodes);
        });
        dest.append(...clone.childNodes);
    }

    // Scans page for headers and adds them to the sidebar.
    document.addEventListener('DOMContentLoaded', function() {
        const activeSection = document.querySelector('#mdbook-sidebar .active');
        if (activeSection === null) {
            return;
        }

        const main = document.getElementsByTagName('main')[0];
        headers = Array.from(main.querySelectorAll('h2, h3, h4, h5, h6'))
            .filter(h => h.id !== '' && h.children.length && h.children[0].tagName === 'A');

        if (headers.length === 0) {
            return;
        }

        // Build a tree of headers in the sidebar.

        const stack = [];

        const firstLevel = parseInt(headers[0].tagName.charAt(1));
        for (let i = 1; i < firstLevel; i++) {
            const ol = document.createElement('ol');
            ol.classList.add('section');
            if (stack.length > 0) {
                stack[stack.length - 1].ol.appendChild(ol);
            }
            stack.push({level: i + 1, ol: ol});
        }

        // The level where it will start folding deeply nested headers.
        const foldLevel = 3;

        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            const level = parseInt(header.tagName.charAt(1));

            const currentLevel = stack[stack.length - 1].level;
            if (level > currentLevel) {
                // Begin nesting to this level.
                for (let nextLevel = currentLevel + 1; nextLevel <= level; nextLevel++) {
                    const ol = document.createElement('ol');
                    ol.classList.add('section');
                    const last = stack[stack.length - 1];
                    const lastChild = last.ol.lastChild;
                    // Handle the case where jumping more than one nesting
                    // level, which doesn't have a list item to place this new
                    // list inside of.
                    if (lastChild) {
                        lastChild.appendChild(ol);
                    } else {
                        last.ol.appendChild(ol);
                    }
                    stack.push({level: nextLevel, ol: ol});
                }
            } else if (level < currentLevel) {
                while (stack.length > 1 && stack[stack.length - 1].level > level) {
                    stack.pop();
                }
            }

            const li = document.createElement('li');
            li.classList.add('header-item');
            li.classList.add('expanded');
            if (level < foldLevel) {
                li.classList.add('expanded');
            }
            const span = document.createElement('span');
            span.classList.add('chapter-link-wrapper');
            const a = document.createElement('a');
            span.appendChild(a);
            a.href = '#' + header.id;
            a.classList.add('header-in-summary');
            filterHeader(header.children[0], a);
            a.addEventListener('click', headerThresholdClick);
            const nextHeader = headers[i + 1];
            if (nextHeader !== undefined) {
                const nextLevel = parseInt(nextHeader.tagName.charAt(1));
                if (nextLevel > level && level >= foldLevel) {
                    const toggle = document.createElement('a');
                    toggle.classList.add('chapter-fold-toggle');
                    toggle.classList.add('header-toggle');
                    toggle.addEventListener('click', () => {
                        li.classList.toggle('expanded');
                    });
                    const toggleDiv = document.createElement('div');
                    toggleDiv.textContent = '❱';
                    toggle.appendChild(toggleDiv);
                    span.appendChild(toggle);
                    headerToggles.push(li);
                }
            }
            li.appendChild(span);

            const currentParent = stack[stack.length - 1];
            currentParent.ol.appendChild(li);
        }

        const onThisPage = document.createElement('div');
        onThisPage.classList.add('on-this-page');
        onThisPage.append(stack[0].ol);
        const activeItemSpan = activeSection.parentElement;
        activeItemSpan.after(onThisPage);
    });

    document.addEventListener('DOMContentLoaded', reloadCurrentHeader);
    document.addEventListener('scroll', reloadCurrentHeader, { passive: true });
})();

