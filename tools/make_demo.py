# -*- coding: utf-8 -*-
"""生成 README 里的演示图：输入图 + 模型真实检测框 + 接口实际返回的 JSON。

    python tools/make_demo.py

输入是模型对仓库内合成样例图的**真实原始输出**（docs/demo_raw.txt，一个字没改）。
本脚本不加载模型、不需要 GPU，只负责把它画出来。

不 import ocr_core.py（那个 import torch），所以解析用的是一份等价的最小实现：
正则抠 "text" 字段 —— 正对应 ocr_core.parse_words 的兜底路径。
"""
import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ------------------------------------------------------------------ 配色
BG_TOP = (9, 14, 26)
BG_BOT = (16, 27, 47)
CARD_BG = (14, 24, 42)
CARD_EDGE = (34, 58, 92)
CODE_BG = (7, 12, 22)
CYAN = (125, 211, 252)
AMBER = (252, 211, 77)
WHITE = (240, 245, 252)
MUTED = (139, 156, 179)
INK = (12, 20, 34)

# ------------------------------------------------------------------ 字体
_FONTS = {
    "regular": [
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ],
    "bold": [
        "C:/Windows/Fonts/msyhbd.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ],
}


def _find(kind):
    for p in _FONTS[kind]:
        if Path(p).exists():
            return p
    raise SystemExit("找不到中文字体，见 tools/make_demo.py 里的 _FONTS。")


def font(size, bold=False):
    return ImageFont.truetype(_find("bold" if bold else "regular"), size)


# ------------------------------------------------------------------ 解析
# 完整元素：{"box": [x0,y0,x1,y1], "text": "..."}
_BLOCK_RE = re.compile(
    r'\{"box":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*"text":\s*"((?:[^"\\]|\\.)*)"\}'
)
# 所有 text 字段——包括格式坏掉、坐标读不出来的那一条
_TEXT_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _unescape(s):
    try:
        return json.loads('"' + s + '"')
    except json.JSONDecodeError:
        return s


def parse(raw):
    """返回 (有坐标的文字块, 全部识别出的文字)。"""
    blocks = [
        (int(a), int(b), int(c), int(d), _unescape(t))
        for a, b, c, d, t in _BLOCK_RE.findall(raw)
    ]
    texts = [_unescape(t) for t in _TEXT_RE.findall(raw)]
    return blocks, texts


# ------------------------------------------------------------------ 绘制
def make_background(w, h):
    img = Image.new("RGB", (w, h), BG_TOP)
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        d.line([(0, y), (w, y)],
               fill=tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOT)))
    glow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glow).ellipse([w * 0.30, -h * 0.55, w * 1.2, h * 0.8], fill=46)
    glow = glow.filter(ImageFilter.GaussianBlur(int(w * 0.08)))
    return Image.composite(Image.new("RGB", (w, h), (56, 189, 248)),
                           img, glow).convert("RGBA")


def draw_json_line(d, x, y, key, value, tail="", note=None):
    """一行 {"words": "..."}，键名高亮，值正常色。"""
    f = font(15)
    fk = font(15, bold=True)
    d.text((x, y), key, font=fk, fill=CYAN, anchor="lm")
    d.text((x + d.textlength(key, font=fk), y), value, font=f, fill=WHITE, anchor="lm")
    if tail:
        d.text((x + d.textlength(key, font=fk) + d.textlength(value, font=f), y),
               tail, font=f, fill=MUTED, anchor="lm")
    if note:
        d.text((x + d.textlength(key, font=fk) + d.textlength(value, font=f)
                + d.textlength(tail, font=f) + 16, y),
               note, font=font(13), fill=AMBER, anchor="lm")


def render(out_path, raw_path, image_path, W=1280, H=900):
    raw = Path(raw_path).read_text(encoding="utf-8").strip()
    blocks, texts = parse(raw)
    src = Image.open(image_path).convert("RGB")
    name = Path(image_path).name

    img = make_background(W, H)
    d = ImageDraw.Draw(img)

    # ---------------- 顶部标题
    d.text((40, 44), f"真实输出 · {name}", font=font(26, bold=True),
           fill=WHITE, anchor="lm")
    d.text((W - 40, 48), "tencent/HunyuanOCR · spotting_json · fp16 · scale 1.5",
           font=font(16), fill=MUTED, anchor="rm")

    # ---------------- 两张图
    cw, ch = 580, 400
    cy0 = 84
    for cx0 in (40, 660):
        d.rounded_rectangle([cx0, cy0, cx0 + cw, cy0 + ch], radius=14,
                            fill=CARD_BG, outline=CARD_EDGE, width=2)

    pad = 14
    cap_h = 30
    avail_w = cw - 2 * pad
    avail_h = ch - 2 * pad - cap_h
    scale = min(avail_w / src.width, avail_h / src.height)
    iw, ih = int(src.width * scale), int(src.height * scale)

    for idx, cx0 in enumerate((40, 660)):
        ix = cx0 + (cw - iw) // 2
        iy = cy0 + pad + (avail_h - ih) // 2
        img.paste(src.resize((iw, ih), Image.LANCZOS), (ix, iy))
        d.rectangle([ix - 1, iy - 1, ix + iw, iy + ih], outline=(52, 78, 116), width=1)

        if idx == 1:
            for i, (x0, y0, x1, y1, _t) in enumerate(blocks):
                bx0 = ix + x0 / 1000 * iw
                by0 = iy + y0 / 1000 * ih
                bx1 = ix + x1 / 1000 * iw
                by1 = iy + y1 / 1000 * ih
                d.rectangle([bx0, by0, bx1, by1], outline=CYAN, width=3)
                tag = str(i + 1)
                tw = d.textlength(tag, font=font(12, bold=True))
                d.rounded_rectangle([bx0, by0 - 17, bx0 + tw + 10, by0],
                                    radius=3, fill=CYAN)
                d.text((bx0 + 5, by0 - 8), tag, font=font(12, bold=True),
                       fill=INK, anchor="lm")

    cap_y = cy0 + ch - cap_h + 2
    d.text((40 + pad, cap_y), "① 输入图片（程序生成的合成样例，非真实数据）",
           font=font(16), fill=MUTED, anchor="lm")
    d.text((660 + pad, cap_y),
           f"② 模型检出的 {len(blocks)} 个文字块 · 青色 = 检测框",
           font=font(16), fill=CYAN, anchor="lm")

    # ---------------- 接口返回
    py0 = cy0 + ch + 22
    d.rounded_rectangle([40, py0, W - 40, H - 24], radius=14,
                        fill=CODE_BG, outline=CARD_EDGE, width=2)
    d.text((64, py0 + 26), "③ 接口实际返回", font=font(19, bold=True),
           fill=WHITE, anchor="lm")
    d.text((222, py0 + 27), "POST /ocr/general", font=font(15), fill=MUTED, anchor="lm")
    d.text((W - 64, py0 + 27),
           f"words_result_num = {len(texts)}", font=font(15), fill=AMBER, anchor="rm")

    ly = py0 + 66
    step = 22
    x = 66
    d.text((x, ly), "{", font=font(15), fill=MUTED, anchor="lm")
    d.text((x, ly + step), '"words_result": [', font=font(15, bold=True),
           fill=CYAN, anchor="lm")
    for i, t in enumerate(texts):
        note = None
        if i == len(texts) - 1:
            note = "← 这条的格式是坏的，靠正则兜底救回来"
        draw_json_line(d, x + 16, ly + step * (2 + i), '{"words": "', t, '"}', note)
    d.text((x, ly + step * (2 + len(texts))), "],", font=font(15), fill=MUTED, anchor="lm")
    d.text((x, ly + step * (3 + len(texts))), "}", font=font(15), fill=MUTED, anchor="lm")

    img.convert("RGB").save(out_path, "PNG", optimize=True)
    return out_path, len(blocks), len(texts)


def main():
    ap = argparse.ArgumentParser(description="生成 README 演示图")
    ap.add_argument("--raw", default="docs/demo_raw.txt")
    ap.add_argument("--image", default="data/samples_clean/clean_001.jpg")
    ap.add_argument("--out", default="docs/demo.png")
    args = ap.parse_args()

    path, nbox, ntext = render(args.out, args.raw, args.image)
    print(f"写出：{path}  （{nbox} 个检测框 / {ntext} 段文字）")


if __name__ == "__main__":
    main()
