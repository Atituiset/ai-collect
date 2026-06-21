#!/usr/bin/env python3
"""
AI Book mdBook Builder
从保存的网页构建 mdBook 源文件，然后调用 mdbook build 生成静态站点。
用法: /home/atituiset/Projects/ai-collect/.venv/bin/python3 build_mdbook.py
"""

import os
import re
import glob
import shutil
import hashlib
import base64
import binascii
import urllib.request
import urllib.error
import subprocess
import sys
from pathlib import Path
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify as md
import cairosvg

sys.path.insert(0, str(Path(__file__).parent))
from build import CHAPTERS, ARTICLE_MAP, slugify, _extract_content, _extract_title

COLLECTS_DIR = Path(__file__).parent / "collects" / "收集"
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = SRC_DIR / "assets"
IMAGES_DIR = SRC_DIR / "images"
FORMULAS_DIR = IMAGES_DIR / "formulas"

SITE_TITLE = "AI 技术全景：从 Transformer 到 Agent 工程"


def article_hash(src_file):
    name = os.path.basename(str(src_file))
    return hashlib.md5(name.encode()).hexdigest()[:8]


def extract_full_title(raw_html):
    """Extract the full article title from the HTML <title> tag if available."""
    m = re.search(r'<title[^>]*>([^<]+)</title>', raw_html)
    if m:
        t = m.group(1).strip()
        # Strip common suffixes like site name or "- 微信公众号"
        t = re.sub(r'\s*[-_|]\s*.*$', '', t).strip()
        if t:
            return t
    return ""


def safe_filename(name):
    return re.sub(r'[^\w一-鿿\.\-]', '_', name)


def detect_image_ext(filepath):
    if not os.path.exists(filepath):
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


def is_svg_placeholder(img_tag):
    src = img_tag.get('src', '')
    if src.startswith('data:image/svg+xml'):
        # Typically 1x1 transparent placeholder
        try:
            data = src.split(',', 1)[1]
            if ';base64' in src:
                decoded = base64.b64decode(data).decode('utf-8', errors='ignore')
            else:
                decoded = urllib.parse.unquote(data)
            if 'width="1"' in decoded or 'height="1"' in decoded or decoded.strip().endswith('/>'):
                # Small placeholder
                return True
        except Exception:
            pass
    data_src = img_tag.get('data-src', '')
    if data_src.startswith('data:'):
        return True
    return False


def copy_or_download_image(src_file, img_tag, article_hash_str):
    files_dir = str(src_file).rsplit('.html', 1)[0] + '_files'
    if not os.path.isdir(files_dir):
        base = str(src_file).rsplit('.html', 1)[0]
        for suffix in ['_files', ' _files']:
            candidate = base + suffix
            if os.path.isdir(candidate):
                files_dir = candidate
                break

    src = img_tag.get('src', '')
    data_src = img_tag.get('data-src', '')
    alt = img_tag.get('alt', '') or img_tag.get('data-ratio', '') or '图片'

    # Priority: local src > data-src (WeChat CDN) > src URL
    online_url = None
    local_path = None

    if src and (src.startswith('./') or src.startswith('../')):
        local_path = src
    elif data_src and data_src.startswith('http'):
        online_url = data_src
    elif src and src.startswith('http'):
        online_url = src
    elif src and not src.startswith('data:'):
        local_path = src

    if local_path and not local_path.startswith('data:'):
        basename = os.path.basename(local_path.split('?')[0])
        if not basename:
            basename = 'image'

        candidates = []
        if os.path.isdir(files_dir):
            for f in os.listdir(files_dir):
                if f.startswith(basename):
                    candidates.append(os.path.join(files_dir, f))

        # If no direct prefix match, try the original candidate path
        candidate = os.path.join(files_dir, basename)
        if os.path.exists(candidate):
            candidates.append(candidate)

        if candidates:
            # Prefer non-watermarked versions; then prefer largest file
            non_watermarked = [c for c in candidates if 'watermark' not in c.lower()]
            if non_watermarked:
                candidates = non_watermarked
            source = max(candidates, key=lambda p: os.path.getsize(p))

            ext = detect_image_ext(source)
            base_dest = basename if basename else 'image'
            if ext and not base_dest.endswith(ext):
                dest_name = f"{article_hash_str}_{base_dest}{ext}"
            else:
                dest_name = f"{article_hash_str}_{base_dest}"
            dest_name = safe_filename(dest_name)
            dest = IMAGES_DIR / dest_name
            shutil.copy2(source, dest)
            return f"../images/{dest_name}", alt

    if online_url:
        # Clean WeChat tracking params for a stable URL/filename
        clean_url = re.sub(r'[?&](from|wxfrom|wx_lazy|tp|watermark|scene)=[^&]*', '', online_url)
        clean_url = re.sub(r'#[^#]*$', '', clean_url)  # remove fragment
        clean_url = re.sub(r'\?&', '?', clean_url)
        clean_url = re.sub(r'&$', '', clean_url)

        ext = os.path.splitext(clean_url.split('?')[0])[1]
        if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg']:
            ext = '.webp'
        dest_name = safe_filename(f"{article_hash_str}_{hashlib.md5(clean_url.encode()).hexdigest()[:12]}{ext}")
        dest = IMAGES_DIR / dest_name
        if not dest.exists():
            for attempt in range(2):
                try:
                    req = urllib.request.Request(online_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Referer': 'https://mp.weixin.qq.com/',
                    })
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        data = resp.read()
                        if len(data) < 100:
                            raise ValueError(f'too small ({len(data)} bytes)')
                        dest.write_bytes(data)
                        break
                except Exception as e:
                    if attempt == 1:
                        print(f"  warning: failed to download {online_url}: {e}")
                        return online_url, alt
        return f"../images/{dest_name}", alt

    return None, alt


def is_formula_svg(svg_tag):
    """WeChat/MathJax formula SVGs contain MathML nodes or glyph path data."""
    svg_str = str(svg_tag)
    return 'data-mml-node=' in svg_str or 'data-c=' in svg_str


def process_svg(svg_tag, index, article_hash_str):
    if not is_formula_svg(svg_tag):
        return None, None  # decorative SVG, caller will remove

    svg_str = str(svg_tag)
    # Fix case-sensitive viewBox
    svg_str = svg_str.replace('viewbox=', 'viewBox=')
    # Determine alt text
    alt = svg_tag.get('aria-label', '') or svg_tag.get('alt', '') or '公式'
    # Save as PNG
    dest_name = safe_filename(f"{article_hash_str}_formula_{index}.png")
    dest = FORMULAS_DIR / dest_name
    try:
        if not dest.exists():
            cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=str(dest), scale=2.0)
        if dest.exists() and dest.stat().st_size < 200:
            dest.unlink()
            return None, alt
        return f"../images/formulas/{dest_name}", alt
    except Exception as e:
        print(f"  warning: failed to render SVG formula: {e}")
        return None, alt


def clean_soup(soup):
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    # Remove script/style tags inside content
    for tag in soup.find_all(['script', 'style', 'iframe']):
        tag.decompose()
    # Remove data-attributes and inline styles
    for tag in soup.find_all(True):
        attrs = list(tag.attrs.keys())
        for attr in attrs:
            if attr.startswith('data-') or attr in ('style', 'role', 'aria-hidden', '_width', 'data-imgfileid'):
                del tag.attrs[attr]
        # Unwrap spans with no attributes (leaf wrappers)
        if tag.name == 'span' and not tag.attrs and not tag.find_all():
            tag.unwrap()
    # Convert section to div
    for tag in soup.find_all('section'):
        tag.name = 'div'
    return soup


# Patterns for WeChat/Zhihu platform cruft that should be removed.
CRUFT_EXACT_TEXTS = [
    '预览时标签不可点',
]

CRUFT_CONTAINS_PATTERNS = [
    # Engagement CTAs
    r'点赞.*转发',
    r'欢迎.*点赞',
    r'创作不易',
    r'欢迎.*评论',
    r'欢迎.*转发',
    r'点个.*在看',
    r'记得.*关注',
    r'星标',
    r'点击上方.*关注',
    r'关注.*公众号',
    r'后台回复',
    r'后台私信',
    r'加我微信',
    r'进群',
    r'交流群',
    r'领取',
    r'福利',
    r'扫码',
    r'限席',
    r'快来抢',
    r'订阅',
    r'赠书',
    r'抽奖',
    r'选取.*评论',
    r'送出',
    r'感谢您的阅读',
    r'今天的分享就到这里',
    # Account metadata combos
    r'公众号.*原创内容',
    r'原创内容.*公众号',
    r'篇原创内容',
    # Source/attribution that is not content
    r'^原文来源[：:]',
    r'^声明[：:]',
    r'^版权说明[：:]',
    r'^关于作者[：:]',
    r'^原创作者[：:]',
    r'^作者简介[：:]',
]

CRUFT_BLOCK_TAG_NAMES = {'p', 'div', 'blockquote', 'section', 'li', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b', 'em', 'i'}


def _is_platform_image(img_tag):
    """Heuristics to detect avatars, QR codes, promo banners."""
    src = img_tag.get('src', '') or img_tag.get('data-src', '')
    # WeChat avatars are often small square PNGs on mmbiz.qpic.cn
    if 'mmbiz.qpic.cn' in src:
        # Avatar/QR/promo images often have /300? or /64? size suffix
        if '/300?' in src or '/64?' in src or '/0?' in src:
            return True
    # Very small images are likely icons/avatars
    width = img_tag.get('width', '')
    height = img_tag.get('height', '')
    if width and height:
        try:
            if int(width) <= 80 and int(height) <= 80:
                return True
        except ValueError:
            pass
    return False


def remove_platform_cruft(soup):
    """Remove WeChat/Zhihu platform artifacts from the article body."""
    # 1. Remove small platform images (avatars, QR codes, icons)
    for img in list(soup.find_all('img')):
        if _is_platform_image(img):
            img.decompose()

    # 2. Remove elements whose text exactly matches known artifacts
    for text in CRUFT_EXACT_TEXTS:
        for elem in soup.find_all(string=lambda s: s and text in s):
            parent = elem.parent
            if parent and parent.name in CRUFT_BLOCK_TAG_NAMES:
                parent.decompose()

    # 3. Remove block elements whose text contains platform/CTA patterns
    for tag in list(soup.find_all(CRUFT_BLOCK_TAG_NAMES)):
        text = tag.get_text(strip=True)
        if not text:
            continue
        for pattern in CRUFT_CONTAINS_PATTERNS:
            if re.search(pattern, text):
                tag.decompose()
                break

    # 4. Remove "往期回顾" (past articles) sections and following linked cards
    for tag in soup.find_all(string=lambda s: s and '往期回顾' in s):
        parent = tag.parent
        # Remove the heading and subsequent sibling blocks that are likely article cards
        if parent:
            next_sib = parent.find_next_sibling()
            parent.decompose()
            # Remove up to 10 following siblings (article cards)
            count = 0
            while next_sib and count < 10:
                sib = next_sib
                next_sib = next_sib.find_next_sibling()
                sib.decompose()
                count += 1

    # 5. Remove empty or near-empty paragraphs/divs left behind
    for tag in list(soup.find_all(['p', 'div', 'blockquote', 'section'])):
        text = tag.get_text(strip=True)
        if not text or text in {'', ' ', '​'}:
            tag.decompose()

    return soup


def html_to_markdown(content_html, src_file):
    article_hash_str = article_hash(src_file)
    soup = BeautifulSoup(content_html, 'lxml')

    # Handle SVGs: replace with placeholders, then restore as Markdown images after conversion.
    svgs = list(soup.find_all('svg'))
    svg_replacements = {}
    for i, svg in enumerate(svgs):
        placeholder = f"@@SVG{i}@@"
        img_src, alt = process_svg(svg, i, article_hash_str)
        if img_src:
            svg_replacements[placeholder] = f"![{alt}]({img_src})"
            svg.replace_with(placeholder)
        else:
            svg.decompose()

    # Handle images
    for img in list(soup.find_all('img')):
        if is_svg_placeholder(img):
            img.decompose()
            continue
        new_src, alt = copy_or_download_image(src_file, img, article_hash_str)
        if new_src:
            img.attrs = {'src': new_src, 'alt': alt}
        else:
            img.decompose()

    # Clean markup
    soup = clean_soup(soup)
    soup = remove_platform_cruft(soup)

    # Remove the first H1 (article title will be prepended by caller) and demote
    # any remaining H1 section headings to H2 so each file has exactly one H1.
    first_h1 = soup.find('h1')
    if first_h1:
        first_h1.decompose()
    for h1 in soup.find_all('h1'):
        h1.name = 'h2'

    # Convert to markdown
    html_str = str(soup.body) if soup.body else str(soup)
    markdown = md(html_str, heading_style='ATX', bullets='-')

    # Post-process
    # Restore SVG placeholders
    for placeholder, markdown_img in svg_replacements.items():
        markdown = markdown.replace(placeholder, markdown_img)
    # Fix double-numbered lists like "1. 1. text"
    markdown = re.sub(r'^(\s*)(\d+)\.\s*\2\.\s', r'\1\2. ', markdown, flags=re.MULTILINE)
    # Fix bullet lists with leading bullet char like "- • text"
    markdown = re.sub(r'^(\s*-\s*)[•·]\s+', r'\1', markdown, flags=re.MULTILINE)
    # Convert standalone bullet chars at line start to markdown bullets
    markdown = re.sub(r'^(\s*)[•·●▪▫]\s+', r'\1- ', markdown, flags=re.MULTILINE)
    # Remove empty links
    markdown = re.sub(r'\[([^\]]*)\]\(\s*\)', r'\1', markdown)
    # Render non-standard HTML tags literally to avoid mdBook warnings
    # Handle both escaped (\<void>) and raw (<void>) forms.
    markdown = re.sub(r'\\?<(tick|void|mark|fragment|slot)(\s*[^>]*)?>', r'&lt;\1\2&gt;', markdown, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return clean_markdown(markdown)


def clean_markdown(markdown):
    """Final Markdown-level cleanup for formatting issues."""
    lines = markdown.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Single-column table detection: header row with empty/single cell + separator + rows
        if line.startswith('|') and line.rstrip().endswith('|'):
            table_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].startswith('|') and lines[j].rstrip().endswith('|'):
                table_lines.append(lines[j])
                j += 1
            # A valid table needs at least header + separator + one row
            if len(table_lines) >= 3:
                # Check header and separator each have exactly one cell
                header_parts = [p.strip() for p in table_lines[0].split('|')[1:-1]]
                sep_parts = [p.strip() for p in table_lines[1].split('|')[1:-1]]
                if len(header_parts) == 1 and len(sep_parts) == 1 and re.match(r'^[-:]+$', sep_parts[0]):
                    # Unwrap to plain text; bold non-empty cells, skip separator
                    for idx, tl in enumerate(table_lines):
                        if idx == 1:
                            continue  # separator
                        cells = [p.strip() for p in tl.split('|')[1:-1]]
                        if not cells:
                            continue
                        cell = cells[0]
                        if cell and not re.match(r'^[-:]+$', cell):
                            out.append(f"**{cell}**")
                    i = j
                    continue

        # Remove standalone horizontal rules that are section separators
        if line.strip() == '---':
            # Keep if it follows a heading (true thematic break)
            if out and out[-1].strip().startswith('#'):
                out.append(line)
            else:
                pass  # skip
            i += 1
            continue

        out.append(line)
        i += 1

    markdown = '\n'.join(out)

    # Clean HTML entities
    markdown = markdown.replace('&nbsp;', ' ').replace('&#160;', ' ')

    # Demote all H1 headings to H2; the caller prepends the single article title as H1.
    lines = markdown.split('\n')
    out_lines = []
    for line in lines:
        if line.startswith('# ') and not line.startswith('## '):
            line = '#' + line
        out_lines.append(line)
    markdown = '\n'.join(out_lines)

    # Remove WeChat code-block placeholder bullets (with optional surrounding punctuation/pipes)
    markdown = re.sub(r'^[|>\s]*(\*\*)?[●·▪▫]+(\*\*)?[|\s]*$', '', markdown, flags=re.MULTILINE)

    # Strip trailing images that are only external WeChat CDN URLs (likely promos/QRs)
    lines = markdown.split('\n')
    while lines and lines[-1].strip() == '':
        lines.pop()
    while lines:
        last = lines[-1].strip()
        if last.startswith('![') and last.endswith(')') and 'mmbiz.qpic.cn' in last:
            lines.pop()
            while lines and lines[-1].strip() == '':
                lines.pop()
        else:
            break
    markdown = '\n'.join(lines)

    # Collapse multiple blank lines again
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown


def build():
    # Prepare directories
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    FORMULAS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear existing chapter md files but keep index.md
    for ch_dir in (d for d in SRC_DIR.iterdir() if d.is_dir() and d.name.startswith('chapter')):
        for f in ch_dir.glob('*.md'):
            if f.name != 'index.md':
                f.unlink()

    # Generate/refresh chapter index files
    for ch in CHAPTERS:
        ch_dir = SRC_DIR / f"chapter{ch['id']:02d}"
        ch_dir.mkdir(exist_ok=True)
        lines = [
            f"# 第{ch['id']}章 {ch['title']}\n\n",
            f"{ch['desc']}\n\n",
            "## 本章文章\n\n"
        ]
        for fname, summary in ARTICLE_MAP.get(ch['id'], []):
            slug = slugify(fname.replace('.html', ''))
            title = fname.replace('.html', '')
            lines.append(f"- [{title}](./{slug}.md) — {summary}\n")
        with open(ch_dir / 'index.md', 'w', encoding='utf-8') as f:
            f.writelines(lines)

    # Process articles
    for ch in CHAPTERS:
        ch_dir = SRC_DIR / f"chapter{ch['id']:02d}"
        ch_dir.mkdir(exist_ok=True)
        for fname, summary in ARTICLE_MAP.get(ch['id'], []):
            src_file = COLLECTS_DIR / fname
            if not src_file.exists():
                print(f"  skip missing: {fname}")
                continue

            print(f"Processing: {fname}")
            with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()

            title = extract_full_title(raw) or _extract_title(raw) or fname.replace('.html', '')
            title = title.replace('&nbsp;', ' ').replace('&#160;', ' ').strip()
            content_html = _extract_content(raw)

            markdown = html_to_markdown(content_html, src_file)

            article = f"""# {title}

{markdown}
"""
            slug = slugify(fname.replace('.html', ''))
            out_file = ch_dir / f"{slug}.md"
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(article)

    # Regenerate SUMMARY.md
    summary_lines = ['# Summary\n', '- [简介](README.md)\n']
    for ch in CHAPTERS:
        ch_dir = f"chapter{ch['id']:02d}"
        summary_lines.append(f"- [第{ch['id']}章 {ch['title']}]({ch_dir}/index.md)\n")
        for fname, summary in ARTICLE_MAP.get(ch['id'], []):
            slug = slugify(fname.replace('.html', ''))
            title = fname.replace('.html', '')
            summary_lines.append(f"  - [{title}]({ch_dir}/{slug}.md)\n")
    with open(SRC_DIR / 'SUMMARY.md', 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)

    print("\nBuilding mdBook...")
    result = subprocess.run(['mdbook', 'build'], cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(result.returncode)
    print("Done. Output: book/")


if __name__ == '__main__':
    build()
