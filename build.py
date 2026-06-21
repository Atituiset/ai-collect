#!/usr/bin/env python3
"""
AI Book Builder - 从保存的网页构建静态书籍站点
用法: python3 build.py
输出: book/ 目录，可直接部署到 GitHub Pages
"""

import os
import re
import glob
import shutil
import hashlib
import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

SRC_DIR = Path(__file__).parent / "collects" / "收集"
OUT_DIR = Path(__file__).parent / "book"
ASSETS_DIR = OUT_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
ARTICLES_DIR = OUT_DIR / "articles"

SITE_TITLE = "AI 技术全景：从 Transformer 到 Agent 工程"
SITE_DESC = "一本关于 AI 基础架构、推理优化与 Agent 工程的开源电子书"


@dataclass
class Article:
    slug: str
    title: str
    source_file: str
    chapter: int
    order: int
    summary: str = ""
    img_local: int = 0
    img_online: int = 0


CHAPTERS = [
    {
        "id": 1,
        "title": "Transformer 与模型架构基础",
        "slug": "ch1-transformer",
        "desc": "从 Transformer 原理到 MoE 架构，理解大模型的基石",
    },
    {
        "id": 2,
        "title": "LLM 推理机制与优化",
        "slug": "ch2-inference",
        "desc": "Prefill/Decode、KV Cache、vLLM、SGLang、RL对齐 —— 推理与训练优化全栈",
    },
    {
        "id": 3,
        "title": "AI Infra 与硬件加速",
        "slug": "ch3-infra",
        "desc": "GPU 架构、算子融合、llama.cpp —— 基础设施层深度解析",
    },
    {
        "id": 4,
        "title": "Agent 架构与 Harness 工程",
        "slug": "ch4-agent",
        "desc": "OpenClaw、Harness 六大组件、多智能体协作 —— Agent 系统设计",
    },
    {
        "id": 5,
        "title": "Claude Code 源码深度解读",
        "slug": "ch5-claude-code",
        "desc": "架构拆解、提示词工程、设计模式 —— 从源码理解顶级 Agent",
    },
    {
        "id": 6,
        "title": "行业全景与职业发展",
        "slug": "ch6-career",
        "desc": "技术全景图、Agent CLI 实战、面试经验 —— 从技术到职业",
    },
]


ARTICLE_MAP = {
    1: [
        ("终于把 Transformer 算法搞懂了！！.html", "Transformer 算法入门：自注意力、编码器-解码器架构"),
        ("Transformer 推理全流程简单拆解.html", "Transformer 推理原理：从文本到生成的全流程解析"),
        ("Transformer 全貌——一个 token 从输入到输出的完整旅程.html", "Encoder-Decoder-GPT 三族谱，完整数据流旅程"),
        ("AI算法：MoE混合专家架构.html", "MoE 稀疏激活架构：路由机制、负载均衡与实际应用"),
        ("Attention 机制与 LLM 的数学能力.html", "Attention 数学原理与 LLM 数学推理的局限性"),
        ("大模型上下文窗口为什么有限？这3个问题必须搞清楚.html", "注意力平方复杂度、KV Cache 内存、位置编码限制"),
    ],
    2: [
        ("LLM 推理为什么先慢后快？从 Prefill、Decode 到 KV Cache 讲清楚.html", "Prefill/Decode 两阶段与自回归生成原理"),
        ("大模型推理为什么慢？从 Prefill、Decode 到 KV Cache，一文讲透推理加速全流程.html", "推理加速全流程：从 Prefill 到 KV Cache 优化"),
        ("大模型推理到底在做什么？从 Prefill、Decode 到 vLLM，一次讲透.html", "推理本质：预测下一个 token 的完整流程"),
        ("彻底搞懂 LLM 推理全过程.html", "LLM 推理从输入到输出的每个环节"),
        ("LLM推理全流程解析：从输入到输出的底层逻辑.html", "9 步推理流水线：分词、嵌入、采样、解码"),
        ("理解 KV Cache：LLM 推理为什么能越写越快.html", "KV Cache 原理、GQA/MQA 优化与内存管理"),
        ("LLM推理优化核心技术.html", "量化、蒸馏、投机解码等推理优化技术全景"),
        ("大模型推理系统入门：从模型优化到调度优化.html", "vLLM 技术路线图：模型优化与调度优化"),
        ("vLLM 如何管理 KV Cache：从 vLLM 本地 Block 管理到跨实例缓存池.html", "Block 状态机、PagedAttention、Prefix Caching"),
        ("LLM大模型是如何运行起来的：从Hello World到vLLM的运行时深度解析.html", "vLLM 运行时架构与请求处理全流程"),
        ("SGLang推理优化-调度器核心ScheduleBatch.html", "SGLang 调度器核心 ScheduleBatch 实现"),
        ("SGLang推理优化-Scheduler 到 Forward 推理的 Batch 数据流.html", "从 Scheduler 到 Forward 的 Batch 数据流"),
        ("SGLang推理优化-DP并行.html", "SGLang DP 并行推理优化"),
        ("2026 年面向 LLM 的 RL方法总结：从 PPO 到 DPO 到 GRPO，再到多智能体 RL.html", "PPO→DPO→GRPO→多智能体 RL 对齐方法演进"),
    ],
    3: [
        ("万字长文｜彻底理解 GPU 架构与深度学习加速原理.html", "GPU vs CPU、CUDA 编程模型、Tensor Core 加速"),
        ("\u4e07\u5b57\u5165\u95e8AI\u00a0Infra\uff1a\u6df1\u5165\u7406\u89e3\u5927\u6a21\u578b\u4e2d\u7684\u6570\u5b66\u4e0eInfra\u4f18\u5316.html", "RMSNorm\u3001\u7b97\u5b50\u878d\u5408\u3001\u5185\u5b58\u4f18\u5316\u7b49\u6570\u5b66\u4e0e Infra \u4f18\u5316"),
        ("AI Infra入门干货总结：大模型是如何高效推理的.html", "推理框架对比、批处理策略、KV Cache 管理"),
        ("llama.cpp 源码解析（三）：114 个 Builder——一套代码如何驾驭 130+ 种架构.html", "GGML 操作集、GGUF 模型格式、Builder 架构模式"),
    ],
    4: [
        ("OpenClaw 的前世今生：一张图讲明白 从\u201c大模型会说\u201d到\u201c工程化会做\u201d的演进.html", "LLM\u2192Function Call\u2192Agent\u2192OpenClaw \u6f14\u8fdb\u8def\u7ebf"),
        ("OpenClaw vs Hermes 架构深度拆解.html", "两大 Agent 框架架构对比与设计取舍"),
        ("(99+ 封私信 _ 80 条消息) AI agent到底有多大创新？ - 知乎.html", "Agent vs 传统 Task Bot 的本质讨论"),
        ("推理 → 行动 → 观察：用 LangChain + Python 实现一个智能体循环.html", "ReAct 模式与 Agent Loop 完整实现"),
        ("AutoGen 多智能体实战：Coder、Reviewer、Executor 自动跑完一次代码评审.html", "多智能体协作：Coder/Reviewer/Executor 角色分工"),
        ("一个爆火的 AI Agent 工作流项目.html", "Sim 工作流平台：画布式 Agent 编排"),
        ("Agent Harness 解析：智能体架构深度拆解.html", "Harness 六大模块拆解与架构设计"),
        ("Agent Harness Engineering全景综述，决定Agent系统的上限.html", "全栈基础设施层七大模块综述"),
        ("大白话讲透驾驭工程(Harness Engineering)六大组件.html", "Harness 六组件通俗解读"),
        ("从Prompt、Context到Harness，工程的三次进化与终局之战.html", "工程三次进化：Prompt→Context→Harness"),
        ("从零搭建 Harness Engineering 架构：Rule、Skill、Sub-Agent 完整落地路径.html", "Rule/Skill/Sub-Agent 完整落地路径"),
        ("9大Harness的记忆能力对比.html", "9 大 Agent 框架的记忆机制对比"),
        ("Rust 正在接管 AI Agent 的基础设施层.html", "Rust 在 Agent 基础设施中的性能优势"),
    ],
    5: [
        ("Claude Code 源码解读：包含Agent Loop、Context 管理、Tool 调用、Memory、Muti Agent等模块.html", "Agent Loop/Context/Tool/Memory/Muti Agent 模块拆解"),
        ("Claude Code 架构详解.html", "多层架构协同：从用户输入到自主执行"),
        ("Claude Code 泄漏源码分析.html", "从启动到 Agent Loop 的完整源码分析"),
        ("从Claude-Code源码学习系统提示词的工程系统.html", "提示词装配系统：多 Section 动态注入"),
        ("从启动到 Agent 循环：Claude Code 51 万行源码架构拆解.html", "完整架构拆解：入口→Agent 循环"),
        ("Claude Code 的 12 个可复用的 Agentic Harness 设计模式.html", "12 个 Agent 设计模式：记忆、工具、编排"),
        ("又一个神级 Claude Code 插件，开源了！.html", "Claude Code 插件生态与开源工具"),
        ("基于顶级 Agent（Claude Code）的 Harness 工程搭建式业务 Agent 评测方案.html", "Agent 搭建评测 Harness 的完整方案"),
    ],
    6: [
        ("大模型领域核心技术全景图（2026年版）.html", "2026 全景图：训练/推理/Agent/应用全栈"),
        ("(99+ 封私信 _ 80 条消息) Skill不就是prompt吗？为啥被吹成这样？ - 知乎.html", "Skill \u6982\u5ff5\u8fa8\u6790\uff1a\u4ece prompt \u5230\u5de5\u7a0b\u5316\u80fd\u529b"),
        ("250 行 Python 写一个 CLI AI Agent.html", "从零用 Python 实现 CLI Agent"),
        ("Hello Agent CLI 开篇.html", "Agent CLI 实战教程框架"),
        ("\u5206\u4eab\u4e00\u4e9b\u00a0AI\u00a0\u9762\u8bd5\u7ecf\u9a8c\uff1a\u7b80\u5386\u4e0e\u80fd\u529b\u6846\u67b6.html", "AI \u9762\u8bd5\u51c6\u5907\uff1a\u7b80\u5386\u4f18\u5316\u4e0e\u80fd\u529b\u6846\u67b6"),
    ],
}


def slugify(text):
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text).strip('-')
    return text[:80].lower() or 'untitled'


def extract_article(src_file):
    with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    title = _extract_title(raw)
    content = _extract_content(raw)
    content = _fix_image_paths(content, src_file)
    content = _clean_content(content)

    return title, content


def _extract_title(raw):
    m = re.search(r'id="activity-name"[^>]*>(.*?)</', raw, re.DOTALL)
    if m:
        return re.sub(r'<[^>]*>', '', m.group(1)).strip()

    m = re.search(r'<title[^>]*>([^<]+)</title>', raw)
    if m:
        t = m.group(1).strip()
        t = re.sub(r'\s*[-_|].*$', '', t).strip()
        return t
    return ""


def _extract_content(raw):
    # WeChat: extract #js_content
    m = re.search(r'id="js_content"[^>]*>(.*?)(?:id="js_content_end"|class="rich_media_tool_area"|class="rich_media_area_extra"|id="content_bottom_area")', raw, re.DOTALL)
    if m:
        return m.group(1)

    # WeChat fallback: js_content to next major section
    m = re.search(r'id="js_content"[^>]*>(.*?)</div>\s*</div>\s*<div\s+class="rich_media_area_extra"', raw, re.DOTALL)
    if m:
        return m.group(1)

    # WeChat fallback 2: js_content with depth tracking via simple heuristic
    m = re.search(r'id="js_content"[^>]*>(.*)', raw, re.DOTALL)
    if m:
        tail = m.group(1)
        content = _truncate_at_boundary(tail)
        if len(content) > 200:
            return content

    # Zhihu: RichText - content spans multiple divs, match more broadly
    m = re.search(r'class="RichText\s+ztext[^"]*"[^>]*>(.*)', raw, re.DOTALL)
    if m:
        tail = m.group(1)
        # Truncate at the next AnswerItem or ContentActions or AnswerActions
        cutoff_markers = ['class="ContentItem-actions"', 'class="AnswerItem"', 
                          'class="RichContent-actions"', 'class="Reward"',
                          'class="CornerButtons"', 'class="ContentItem-collapse"',
                          'class="RichTextCopyright"', 'itemprop="commentCount"']
        best = len(tail)
        for marker in cutoff_markers:
            pos = tail.find(marker)
            if pos > 0 and pos < best:
                best = pos
        zhihu_content = tail[:best]
        if len(zhihu_content) > 200:
            return zhihu_content

    return "<p>内容提取失败，请查看原文。</p>"


def _truncate_at_boundary(html):
    markers = [
        'id="js_content_end"',
        'class="rich_media_tool',
        'class="rich_media_area_extra"',
        'id="content_bottom_area"',
        'class="author_profile',
        'class="wx_bottom_modal',
        'class="reward_dialog',
        'class="album_con"',
        'class="adv_keyword',
        'class="font-pannel',
        'class="sougou"',
        'class="js_pc_qr_code"',
        '阅读原文',
        '微信扫一扫赞赏',
    ]
    best_pos = len(html)
    for marker in markers:
        pos = html.find(marker)
        if pos > 0 and pos < best_pos:
            best_pos = pos

    content = html[:best_pos]

    # Try to not break in the middle of a tag
    last_close = content.rfind('</')
    last_open = content.rfind('<')
    if last_open > last_close:
        content = content[:last_open]

    return content


def _clean_content(html):
    html = re.sub(r'<section[^>]*>', '<div>', html)
    html = re.sub(r'</section>', '</div>', html)
    html = re.sub(r'\s+style="[^"]*"', '', html)
    html = re.sub(r'<span\s*>\s*</span>', '', html)
    html = re.sub(r'<a[^>]*></a>', '', html)
    html = re.sub(r'<mp-style-type[^>]*>.*?</mp-style-type>', '', html, flags=re.DOTALL)
    html = re.sub(r'<span\s+leaf="">\s*</span>', '', html)
    html = re.sub(r'(<(?:div|p|blockquote)>\s*){3,}', '<div>', html)
    html = re.sub(r'(</(?:div|p|blockquote)>\s*){3,}', '</div>', html)
    html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL)
    html = re.sub(r'class="list-paddingleft-\d+"', '', html)
    html = re.sub(r'<div>\s*</div>', '', html)
    html = re.sub(r'\n{3,}', '\n\n', html)
    return html


def _fix_image_paths(content, src_file):
    files_dir = str(src_file).rsplit('.html', 1)[0] + '_files'
    if not os.path.isdir(files_dir):
        base = str(src_file).rsplit('.html', 1)[0]
        for suffix in ['_files', ' _files']:
            candidate = base + suffix
            if os.path.isdir(candidate):
                files_dir = candidate
                break

    def replace_img_src(m):
        full_tag = m.group(0)
        local_src = re.search(r'src="(\./[^"]+)"', full_tag)
        data_src = re.search(r'data-src="(https://mmbiz[^"]+)"', full_tag)

        if local_src:
            local_path = local_src.group(1)
            basename = os.path.basename(local_path)
            ext = _detect_image_ext(os.path.join(files_dir, basename))
            new_name = f"{basename}{ext}" if ext and not basename.endswith(ext) else basename
            new_src = f"../assets/images/{_safe_filename(src_file)}_{new_name}"
            full_tag = re.sub(r'src="\./[^"]+"', f'src="{new_src}"', full_tag)

        elif data_src:
            online_url = data_src.group(1)
            full_tag = re.sub(r'data-src="[^"]+"', f'src="{online_url}"', full_tag)

        zhihu_local = re.search(r'src="(v2-[^"]+)"', full_tag)
        if zhihu_local:
            basename = zhihu_local.group(1)
            new_src = f"../assets/images/{_safe_filename(src_file)}_{basename}"
            full_tag = re.sub(r'src="v2-[^"]+"', f'src="{new_src}"', full_tag)

        zhihu_noscript = re.search(r'data-original="(https://picx?\.zhimg\.com/[^"]+)"', full_tag)
        if zhihu_noscript:
            online_url = zhihu_noscript.group(1)
            if 'src="' not in full_tag or 'src="data:' in full_tag:
                full_tag = re.sub(r'src="[^"]*"', f'src="{online_url}"', full_tag, count=1)

        full_tag = re.sub(r'\s*data-src="[^"]*"', '', full_tag)
        full_tag = re.sub(r'\s*data-actualsrc="[^"]*"', '', full_tag)
        full_tag = re.sub(r'\s*data-original="[^"]*"', '', full_tag)

        return full_tag

    content = re.sub(r'<img[^>]+/?>', replace_img_src, content)
    return content


def _detect_image_ext(filepath):
    if not os.path.exists(filepath):
        parent_dir = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        if os.path.isdir(parent_dir):
            for f in os.listdir(parent_dir):
                if f.startswith(basename):
                    _, ext = os.path.splitext(f)
                    return ext
        return '.webp'
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
        if header[:4] == b'\x89PNG':
            return '.png'
        elif header[:3] == b'\xff\xd8\xff':
            return '.jpg'
        elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return '.webp'
        elif header[:6] in (b'GIF87a', b'GIF89a'):
            return '.gif'
        elif header[:4] == b'%PDF':
            return '.pdf'
    except Exception:
        pass
    return '.webp'


def _safe_filename(src_file):
    name = os.path.basename(str(src_file))
    name = re.sub(r'[^\w\u4e00-\u9fff]', '_', name)
    h = hashlib.md5(name.encode()).hexdigest()[:8]
    return h


def copy_local_images(src_file, article_hash):
    files_dir = str(src_file).rsplit('.html', 1)[0] + '_files'
    if not os.path.isdir(files_dir):
        base = str(src_file).rsplit('.html', 1)[0]
        for suffix in ['_files', ' _files']:
            candidate = base + suffix
            if os.path.isdir(candidate):
                files_dir = candidate
                break
    if not os.path.isdir(files_dir):
        return

    for f in os.listdir(files_dir):
        fpath = os.path.join(files_dir, f)
        if not os.path.isfile(fpath):
            continue
        if f.endswith(('.css', '.js', '.下载', '.html')):
            continue
        if f in ('frame.html',):
            continue

        ext = _detect_image_ext(fpath)
        if not f.endswith(ext):
            new_name = f"{f}{ext}"
        else:
            new_name = f

        dest_name = f"{article_hash}_{new_name}"
        dest = IMAGES_DIR / dest_name
        try:
            shutil.copy2(fpath, dest)
        except Exception:
            pass


def generate_article_html(article, content, chapter_info):
    chapter_nav = ""
    for ch in CHAPTERS:
        active = "active" if ch["id"] == chapter_info["id"] else ""
        chapter_nav += f'<a href="../chapters/{ch["slug"]}.html" class="nav-chapter {active}">{ch["title"]}</a>\n'

    prev_link = ""
    next_link = ""
    ch_articles = ARTICLE_MAP.get(chapter_info["id"], [])
    for i, (fname, _) in enumerate(ch_articles):
        slug = slugify(fname.replace('.html', ''))
        if fname == article.source_file:
            if i > 0:
                p_slug = slugify(ch_articles[i-1][0].replace('.html', ''))
                prev_link = f'<a href="{p_slug}.html" class="nav-prev">&larr; 上一篇</a>'
            if i < len(ch_articles) - 1:
                n_slug = slugify(ch_articles[i+1][0].replace('.html', ''))
                next_link = f'<a href="{n_slug}.html" class="nav-next">下一篇 &rarr;</a>'

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{article.title} - {SITE_TITLE}</title>
<link rel="stylesheet" href="../assets/style.css">
</head>
<body>
<aside class="sidebar">
<div class="sidebar-header">
<a href="../index.html" class="site-title">{SITE_TITLE}</a>
</div>
<nav class="sidebar-nav">
{chapter_nav}
</nav>
</aside>
<main class="main-content">
<article class="article">
<header class="article-header">
<div class="article-meta">第{chapter_info["id"]}章 · {chapter_info["title"]}</div>
<h1 class="article-title">{article.title}</h1>
<p class="article-summary">{article.summary}</p>
</header>
<div class="article-body">
{content}
</div>
<footer class="article-footer">
<div class="article-nav">
{prev_link}
<a href="../chapters/{chapter_info["slug"]}.html" class="nav-toc">目录</a>
{next_link}
</div>
</footer>
</article>
</main>
<button class="sidebar-toggle" onclick="document.body.classList.toggle('sidebar-open')">&#9776;</button>
<script src="../assets/theme.js"></script>
</body>
</html>'''


def generate_chapter_html(chapter, articles):
    chapter_nav = ""
    for ch in CHAPTERS:
        active = "active" if ch["id"] == chapter["id"] else ""
        chapter_nav += f'<a href="{ch["slug"]}.html" class="nav-chapter {active}">{ch["title"]}</a>\n'

    article_items = ""
    for a in articles:
        article_items += f'''
<a href="../articles/{a.slug}.html" class="article-card">
<span class="article-order">{a.order:02d}</span>
<div class="article-card-body">
<h3>{a.title}</h3>
<p>{a.summary}</p>
</div>
<span class="arrow">&rarr;</span>
</a>'''

    prev_ch = next_ch = None
    for i, ch in enumerate(CHAPTERS):
        if ch["id"] == chapter["id"]:
            if i > 0:
                prev_ch = CHAPTERS[i - 1]
            if i < len(CHAPTERS) - 1:
                next_ch = CHAPTERS[i + 1]

    nav_links = ""
    if prev_ch:
        nav_links += f'<a href="{prev_ch["slug"]}.html" class="nav-prev">&larr; {prev_ch["title"]}</a>'
    nav_links += '<a href="../index.html" class="nav-toc">首页</a>'
    if next_ch:
        nav_links += f'<a href="{next_ch["slug"]}.html" class="nav-next">{next_ch["title"]} &rarr;</a>'

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>第{chapter["id"]}章 {chapter["title"]} - {SITE_TITLE}</title>
<link rel="stylesheet" href="../assets/style.css">
</head>
<body>
<aside class="sidebar">
<div class="sidebar-header">
<a href="../index.html" class="site-title">{SITE_TITLE}</a>
</div>
<nav class="sidebar-nav">
{chapter_nav}
</nav>
</aside>
<main class="main-content">
<div class="chapter-page">
<header class="chapter-header">
<span class="chapter-number">第 {chapter["id"]} 章</span>
<h1>{chapter["title"]}</h1>
<p class="chapter-desc">{chapter["desc"]}</p>
</header>
<div class="article-list">
{article_items}
</div>
<nav class="chapter-nav">{nav_links}</nav>
</div>
</main>
<button class="sidebar-toggle" onclick="document.body.classList.toggle('sidebar-open')">&#9776;</button>
<script src="../assets/theme.js"></script>
</body>
</html>'''


def generate_index_html(all_articles):
    toc_html = ""
    for ch in CHAPTERS:
        ch_articles = [a for a in all_articles if a.chapter == ch["id"]]
        article_count = len(ch_articles)
        toc_html += f'''
<div class="toc-chapter">
<a href="chapters/{ch["slug"]}.html" class="toc-chapter-link">
<span class="toc-chapter-num">第 {ch["id"]} 章</span>
<h2>{ch["title"]}</h2>
<p>{ch["desc"]}</p>
<span class="toc-count">{article_count} 篇</span>
</a>
</div>'''

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{SITE_TITLE}</title>
<meta name="description" content="{SITE_DESC}">
<link rel="stylesheet" href="assets/style.css">
</head>
<body class="home-page">
<main class="home-content">
<header class="home-header">
<h1>{SITE_TITLE}</h1>
<p class="home-subtitle">{SITE_DESC}</p>
<div class="home-stats">
<span>{len(all_articles)} 篇文章</span>
<span>{len(CHAPTERS)} 个章节</span>
</div>
</header>
<div class="toc-grid">
{toc_html}
</div>
<footer class="home-footer">
<p>本站内容收集整理自互联网公开技术博客，仅供学习交流使用</p>
</footer>
</main>
<script src="assets/theme.js"></script>
</body>
</html>'''


CSS = '''@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=Noto+Serif+SC:wght@400;700&display=swap');

:root {
  --bg: #fafafa;
  --bg-sidebar: #1a1b2e;
  --bg-card: #ffffff;
  --bg-code: #f5f5f7;
  --text: #2c2c2e;
  --text-secondary: #6e6e73;
  --text-sidebar: #c7c7cc;
  --text-sidebar-active: #ffffff;
  --accent: #6c5ce7;
  --accent-light: #a29bfe;
  --accent-bg: rgba(108, 92, 231, 0.08);
  --border: #e5e5ea;
  --shadow: 0 2px 8px rgba(0,0,0,0.06);
  --radius: 8px;
  --max-width: 820px;
  --sidebar-width: 260px;
  --header-height: 60px;
}

[data-theme="dark"] {
  --bg: #1c1c1e;
  --bg-sidebar: #0d0d1a;
  --bg-card: #2c2c2e;
  --bg-code: #3a3a3c;
  --text: #f5f5f7;
  --text-secondary: #98989d;
  --text-sidebar: #98989d;
  --text-sidebar-active: #ffffff;
  --accent: #a29bfe;
  --accent-light: #6c5ce7;
  --accent-bg: rgba(162, 155, 254, 0.12);
  --border: #38383a;
  --shadow: 0 2px 8px rgba(0,0,0,0.3);
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  font-size: 16px;
  scroll-behavior: smooth;
}

body {
  font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.8;
  display: flex;
  min-height: 100vh;
}

/* Sidebar */
.sidebar {
  position: fixed;
  top: 0;
  left: 0;
  width: var(--sidebar-width);
  height: 100vh;
  background: var(--bg-sidebar);
  overflow-y: auto;
  z-index: 100;
  transition: transform 0.3s ease;
}

.sidebar-header {
  padding: 24px 20px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}

.site-title {
  color: #fff;
  font-size: 15px;
  font-weight: 700;
  text-decoration: none;
  line-height: 1.4;
  display: block;
}

.sidebar-nav {
  padding: 12px 0;
}

.nav-chapter {
  display: block;
  padding: 10px 20px;
  color: var(--text-sidebar);
  text-decoration: none;
  font-size: 13px;
  line-height: 1.5;
  transition: all 0.2s;
  border-left: 3px solid transparent;
}

.nav-chapter:hover {
  color: var(--text-sidebar-active);
  background: rgba(255,255,255,0.05);
}

.nav-chapter.active {
  color: var(--text-sidebar-active);
  border-left-color: var(--accent);
  background: rgba(255,255,255,0.08);
}

/* Main content */
.main-content {
  margin-left: var(--sidebar-width);
  flex: 1;
  min-width: 0;
  max-width: 100%;
}

/* Article page */
.article {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 48px 40px 80px;
}

.article-header {
  margin-bottom: 40px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}

.article-meta {
  font-size: 13px;
  color: var(--accent);
  font-weight: 500;
  margin-bottom: 8px;
}

.article-title {
  font-family: 'Noto Serif SC', serif;
  font-size: 32px;
  font-weight: 700;
  line-height: 1.3;
  margin-bottom: 12px;
}

.article-summary {
  font-size: 15px;
  color: var(--text-secondary);
  line-height: 1.6;
}

.article-body {
  font-size: 16px;
  line-height: 1.9;
}

.article-body h1 {
  display: none;
}

.article-body h2 {
  font-family: 'Noto Serif SC', serif;
  font-size: 24px;
  margin: 40px 0 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--accent);
}

.article-body h3 {
  font-size: 20px;
  margin: 32px 0 12px;
  color: var(--text);
}

.article-body h4 {
  font-size: 17px;
  margin: 24px 0 8px;
}

.article-body p {
  margin: 12px 0;
}

.article-body blockquote {
  margin: 16px 0;
  padding: 12px 20px;
  background: var(--accent-bg);
  border-left: 4px solid var(--accent);
  border-radius: 0 var(--radius) var(--radius) 0;
  color: var(--text-secondary);
}

.article-body blockquote p {
  margin: 4px 0;
}

.article-body pre {
  margin: 16px 0;
  padding: 20px;
  background: var(--bg-code);
  border-radius: var(--radius);
  overflow-x: auto;
  font-size: 14px;
  line-height: 1.6;
}

.article-body code {
  font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 0.9em;
}

.article-body p code, .article-body li code {
  padding: 2px 6px;
  background: var(--bg-code);
  border-radius: 4px;
}

.article-body img {
  max-width: 100%;
  height: auto;
  margin: 20px 0;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

.article-body figure {
  margin: 20px 0;
  text-align: center;
}

.article-body figure img {
  display: inline-block;
}

.article-body ul, .article-body ol {
  margin: 12px 0;
  padding-left: 28px;
}

.article-body li {
  margin: 6px 0;
}

.article-body table {
  width: 100%;
  margin: 16px 0;
  border-collapse: collapse;
  font-size: 14px;
}

.article-body th, .article-body td {
  padding: 10px 14px;
  border: 1px solid var(--border);
  text-align: left;
}

.article-body th {
  background: var(--bg-code);
  font-weight: 600;
}

.article-body strong {
  font-weight: 600;
  color: var(--text);
}

.article-body a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid var(--accent-light);
}

.article-body a:hover {
  opacity: 0.8;
}

.article-body hr {
  margin: 32px 0;
  border: none;
  border-top: 1px solid var(--border);
}

.article-footer {
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
}

.article-nav {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.article-nav a {
  color: var(--accent);
  text-decoration: none;
  font-size: 14px;
  padding: 8px 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: all 0.2s;
}

.article-nav a:hover {
  background: var(--accent-bg);
  border-color: var(--accent);
}

.nav-prev { margin-right: auto; }
.nav-next { margin-left: auto; }
.nav-toc { text-align: center; }

/* Chapter page */
.chapter-page {
  max-width: 900px;
  margin: 0 auto;
  padding: 48px 40px 80px;
}

.chapter-header {
  margin-bottom: 36px;
}

.chapter-number {
  font-size: 13px;
  color: var(--accent);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 2px;
}

.chapter-header h1 {
  font-family: 'Noto Serif SC', serif;
  font-size: 36px;
  margin: 8px 0 12px;
}

.chapter-desc {
  font-size: 16px;
  color: var(--text-secondary);
}

.article-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.article-card {
  display: flex;
  align-items: center;
  padding: 20px 24px;
  background: var(--bg-card);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  text-decoration: none;
  color: var(--text);
  transition: all 0.2s;
  box-shadow: var(--shadow);
}

.article-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(108, 92, 231, 0.12);
}

.article-order {
  font-size: 28px;
  font-weight: 700;
  color: var(--accent);
  margin-right: 20px;
  min-width: 40px;
}

.article-card-body {
  flex: 1;
  min-width: 0;
}

.article-card-body h3 {
  font-size: 17px;
  font-weight: 600;
  margin-bottom: 4px;
  line-height: 1.4;
}

.article-card-body p {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.arrow {
  font-size: 18px;
  color: var(--text-secondary);
  margin-left: 12px;
}

.chapter-nav {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
}

.chapter-nav a {
  color: var(--accent);
  text-decoration: none;
  font-size: 14px;
  padding: 8px 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: all 0.2s;
}

.chapter-nav a:hover {
  background: var(--accent-bg);
  border-color: var(--accent);
}

/* Home page */
.home-page {
  display: block;
}

.home-content {
  max-width: 960px;
  margin: 0 auto;
  padding: 80px 40px;
}

.home-header {
  text-align: center;
  margin-bottom: 60px;
}

.home-header h1 {
  font-family: 'Noto Serif SC', serif;
  font-size: 42px;
  line-height: 1.3;
  margin-bottom: 16px;
}

.home-subtitle {
  font-size: 18px;
  color: var(--text-secondary);
  margin-bottom: 20px;
}

.home-stats {
  display: flex;
  justify-content: center;
  gap: 24px;
  font-size: 14px;
  color: var(--text-secondary);
}

.toc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}

.toc-chapter {
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  transition: all 0.2s;
}

.toc-chapter:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(108, 92, 231, 0.15);
}

.toc-chapter-link {
  display: block;
  padding: 28px 24px;
  background: var(--bg-card);
  text-decoration: none;
  color: var(--text);
  position: relative;
}

.toc-chapter-num {
  font-size: 12px;
  color: var(--accent);
  font-weight: 600;
  letter-spacing: 1px;
}

.toc-chapter-link h2 {
  font-size: 20px;
  margin: 8px 0;
  font-family: 'Noto Serif SC', serif;
}

.toc-chapter-link p {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.toc-count {
  position: absolute;
  top: 16px;
  right: 16px;
  font-size: 12px;
  color: var(--text-secondary);
  background: var(--bg-code);
  padding: 2px 10px;
  border-radius: 12px;
}

.home-footer {
  text-align: center;
  margin-top: 80px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
  font-size: 13px;
  color: var(--text-secondary);
}

/* Sidebar toggle (mobile) */
.sidebar-toggle {
  display: none;
  position: fixed;
  top: 12px;
  left: 12px;
  z-index: 200;
  width: 40px;
  height: 40px;
  border: none;
  background: var(--bg-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  font-size: 20px;
  cursor: pointer;
  color: var(--text);
}

/* Theme toggle */
.theme-toggle {
  position: fixed;
  top: 12px;
  right: 12px;
  z-index: 200;
  width: 40px;
  height: 40px;
  border: none;
  background: var(--bg-card);
  border-radius: 50%;
  box-shadow: var(--shadow);
  font-size: 18px;
  cursor: pointer;
  color: var(--text);
}

/* Responsive */
@media (max-width: 768px) {
  .sidebar {
    transform: translateX(-100%);
  }

  body.sidebar-open .sidebar {
    transform: translateX(0);
  }

  .main-content {
    margin-left: 0;
  }

  .sidebar-toggle {
    display: block;
  }

  .article {
    padding: 24px 20px 60px;
  }

  .article-title {
    font-size: 24px;
  }

  .chapter-page {
    padding: 24px 20px 60px;
  }

  .home-content {
    padding: 40px 20px;
  }

  .home-header h1 {
    font-size: 28px;
  }

  .toc-grid {
    grid-template-columns: 1fr;
  }
}

/* Print */
@media print {
  .sidebar, .sidebar-toggle, .theme-toggle, .article-footer { display: none; }
  .main-content { margin-left: 0; }
  .article { padding: 0; }
}
'''

JS = '''// Theme toggle
(function() {
  var saved = localStorage.getItem('theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }

  // Close sidebar on click outside (mobile)
  document.addEventListener('click', function(e) {
    if (document.body.classList.contains('sidebar-open')) {
      var sidebar = document.querySelector('.sidebar');
      var toggle = document.querySelector('.sidebar-toggle');
      if (!sidebar.contains(e.target) && !toggle.contains(e.target)) {
        document.body.classList.remove('sidebar-open');
      }
    }
  });

  // Smooth image loading
  document.querySelectorAll('img').forEach(function(img) {
    img.loading = 'lazy';
    img.onerror = function() {
      var dataSrc = this.getAttribute('data-src') || this.getAttribute('data-original');
      if (dataSrc) {
        this.src = dataSrc;
      }
    };
  });
})();
'''


def _find_source_file(filename):
    normalized = filename.replace('\xa0', ' ').replace('\u3000', ' ')
    src_path = SRC_DIR / normalized
    if src_path.exists():
        return src_path
    src_path = SRC_DIR / filename
    if src_path.exists():
        return src_path
    key = normalized[:10].replace(' ', '')
    for f in SRC_DIR.iterdir():
        fn = f.name.replace('\xa0', ' ').replace('\u3000', ' ')
        if fn.replace(' ', '').startswith(key) and f.suffix == '.html':
            return f
    return None


def main():
    print("=== AI Book Builder ===\n")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    OUT_DIR.mkdir(parents=True)
    (OUT_DIR / "chapters").mkdir()
    (OUT_DIR / "articles").mkdir()
    ASSETS_DIR.mkdir()
    IMAGES_DIR.mkdir()

    all_articles = []

    for ch in CHAPTERS:
        ch_id = ch["id"]
        article_defs = ARTICLE_MAP.get(ch_id, [])
        if not article_defs:
            continue

        print(f"Processing Chapter {ch_id}: {ch['title']}")

        ch_articles = []
        for order, (filename, summary) in enumerate(article_defs, 1):
            src_path = _find_source_file(filename)
            if src_path is None:
                print(f"  WARNING: {filename} not found, skipping")
                continue

            article_hash = _safe_filename(str(src_path))
            slug = slugify(filename.replace('.html', ''))

            article = Article(
                slug=slug,
                title="",
                source_file=filename,
                chapter=ch_id,
                order=order,
                summary=summary,
            )

            print(f"  [{order:02d}] Extracting: {filename[:50]}...")

            try:
                title, content = extract_article(str(src_path))
                article.title = title or filename.replace('.html', '')
            except Exception as e:
                print(f"    ERROR: {e}")
                article.title = filename.replace('.html', '')
                content = "<p>内容提取失败，请查看原文。</p>"

            copy_local_images(str(src_path), article_hash)

            article_html = generate_article_html(article, content, ch)
            article_path = OUT_DIR / "articles" / f"{article.slug}.html"
            article_path.write_text(article_html, encoding='utf-8')

            all_articles.append(article)
            ch_articles.append(article)

        ch_html = generate_chapter_html(ch, ch_articles)
        ch_path = OUT_DIR / "chapters" / f"{ch['slug']}.html"
        ch_path.write_text(ch_html, encoding='utf-8')

    print(f"\nGenerating index page...")
    index_html = generate_index_html(all_articles)
    (OUT_DIR / "index.html").write_text(index_html, encoding='utf-8')

    print("Writing CSS and JS...")
    (ASSETS_DIR / "style.css").write_text(CSS, encoding='utf-8')
    (ASSETS_DIR / "theme.js").write_text(JS, encoding='utf-8')

    print(f"\n=== Build Complete ===")
    print(f"Output: {OUT_DIR}")
    print(f"Articles: {len(all_articles)}")
    print(f"Chapters: {len(CHAPTERS)}")
    print(f"Images: {len(list(IMAGES_DIR.iterdir()))} files")

    total_size = sum(f.stat().st_size for f in OUT_DIR.rglob('*') if f.is_file())
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

    format_script = Path(__file__).parent / "scripts" / "format.js"
    if format_script.exists():
        print("\nFormatting HTML files...")
        result = subprocess.run(
            [sys.executable.replace("python", "node"), str(format_script)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"  Format warning: {result.stderr.strip()}")
    else:
        print("\n  Tip: run `npm run format` to beautify HTML files")


if __name__ == '__main__':
    main()
