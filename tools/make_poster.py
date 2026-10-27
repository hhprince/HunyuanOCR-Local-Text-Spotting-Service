# -*- coding: utf-8 -*-
"""生成 README 顶部的海报和 GitHub 社交预览图。

    python tools/make_poster.py                # 输出到 docs/
    python tools/make_poster.py --out-dir docs

只依赖 Pillow：不加载模型、不需要 GPU、不读任何业务数据。
改文案或配色直接改本文件里的常量，然后重跑。
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ------------------------------------------------------------------ 配色
BG_TOP = (10, 17, 31)
BG_BOT = (17, 31, 52)
CYAN = (125, 211, 252)
BLUE = (99, 102, 241)
AMBER = (252, 211, 77)
WHITE = (248, 250, 252)
MUTED = (148, 163, 184)
CARD_BG = (15, 26, 45)
CARD_EDGE = (36, 63, 98)
CHIP_BG = (23, 41, 68)
SURFACE = (231, 236, 244)
BAR = (198, 209, 224)
INK = (12, 20, 34)

# ------------------------------------------------------------------ 字体
_FONT_CANDIDATES = {
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


def _find_font(kind):
    for path in _FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return path
    raise SystemExit(
        "找不到中文字体，装一个 msyh.ttc / NotoSansCJK / wqy-zenhei 再试，"
        "或者改 tools/make_poster.py 里的 _FONT_CANDIDATES。"
    )


def font(size, bold=False):
    return ImageFont.truetype(_find_font("bold" if bold else "regular"), size)


# ------------------------------------------------------------------ 基础绘制
def make_background(w, h):
    """深色渐变底 + 一团青色辉光 + 几道极淡的扫描线。"""
    img = Image.new("RGB", (w, h), BG_TOP)
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        d.line(
            [(0, y), (w, y)],
            fill=tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOT)),
        )

    glow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glow).ellipse(
        [w * 0.42, -h * 0.75, w * 1.3, h * 0.95], fill=52
    )
    glow = glow.filter(ImageFilter.GaussianBlur(int(w * 0.07)))
    img = Image.composite(Image.new("RGB", (w, h), (56, 189, 248)), img, glow)

    d = ImageDraw.Draw(img)
    for i in range(1, 9):
        y = int(h * i / 9)
        d.line([(0, y), (w, y)], fill=(30, 52, 82), width=1)

    return img.convert("RGBA")


def shadow_text(img, xy, s, f, fill, offset=(0, 4), blur=7, anchor="la"):
    """带柔和投影的文字——大标题压在渐变上才不发飘。"""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (xy[0] + offset[0], xy[1] + offset[1]),
        s, font=f, fill=(0, 0, 0, 200), anchor=anchor,
    )
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(img).text(xy, s, font=f, fill=fill, anchor=anchor)


def gradient_rule(img, x, y, w, h, c1=CYAN, c2=BLUE):
    """一小段渐变色条，用作标题下的分隔线。"""
    grad = Image.new("RGB", (w, h))
    gd = ImageDraw.Draw(grad)
    for i in range(w):
        t = i / max(w - 1, 1)
        gd.line(
            [(i, 0), (i, h)],
            fill=tuple(round(a + (b - a) * t) for a, b in zip(c1, c2)),
        )
    img.alpha_composite(grad.convert("RGBA"), (x, y))


def draw_chip(img, x, y, s, f, dot=CYAN):
    """圆角胶囊标签，返回右边界 x。"""
    d = ImageDraw.Draw(img)
    tw = d.textlength(s, font=f)
    h = f.size + 20
    w = tw + 46
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2,
                        fill=CHIP_BG, outline=CARD_EDGE, width=2)
    d.ellipse([x + 17, y + h // 2 - 4, x + 25, y + h // 2 + 4], fill=dot)
    d.text((x + 35, y + h // 2), s, font=f, fill=(203, 213, 225), anchor="lm")
    return x + w


def draw_mock_card(img, box, s=1.0):
    """右侧示意图：一张"照片" + 模型检出的文字块（带编号标签）。

    这是项目最核心的动作——检测并识别图中所有文字行，画出来比讲清楚快。
    """
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=int(18 * s), fill=CARD_BG,
                        outline=CARD_EDGE, width=max(1, int(2 * s)))

    d.ellipse([x0 + 22 * s, y0 + 30 * s, x0 + 32 * s, y0 + 40 * s], fill=CYAN)
    d.text((x0 + 42 * s, y0 + 35 * s), "spotting_json · 一次推理输出全部文字块",
           font=font(int(17 * s)), fill=MUTED, anchor="lm")

    px0, py0 = x0 + 22 * s, y0 + 56 * s
    px1, py1 = x1 - 22 * s, y1 - 22 * s
    pw, ph = px1 - px0, py1 - py0
    d.rounded_rectangle([px0, py0, px1, py1], radius=int(10 * s), fill=SURFACE)

    def rx(a, b):
        return px0 + a * pw, px0 + b * pw

    def ry(a, b):
        return py0 + a * ph, py0 + b * ph

    # 照片里"印"着的灰条（代表真实文字）。宽幅用两栏，竖幅用单栏。
    wide = (pw / ph) > 1.6
    if wide:
        bars = [
            (0.04, 0.42, 0.10, 0.26), (0.50, 0.96, 0.10, 0.26),
            (0.04, 0.42, 0.38, 0.52), (0.50, 0.96, 0.38, 0.52),
            (0.04, 0.32, 0.64, 0.76), (0.50, 0.80, 0.64, 0.76),
        ]
        dets = [(0, 0.028, 0.435, 0.085, 0.275), (1, 0.488, 0.975, 0.085, 0.275)]
        code = (0.04, 0.35, 0.84, 0.97)
    else:
        bars = [
            (0.05, 0.62, 0.09, 0.17),
            (0.05, 0.95, 0.27, 0.34),
            (0.05, 0.33, 0.42, 0.48),
            (0.38, 0.68, 0.42, 0.48),
            (0.72, 0.98, 0.42, 0.48),
            (0.50, 0.95, 0.58, 0.65),
            (0.50, 0.88, 0.72, 0.79),
            (0.05, 0.80, 0.90, 0.96),
        ]
        dets = [(0, 0.038, 0.635, 0.075, 0.185), (1, 0.038, 0.965, 0.255, 0.355)]
        code = (0.05, 0.42, 0.56, 0.84)

    for a, b, c, e in bars:
        x_a, x_b = rx(a, b)
        y_a, y_b = ry(c, e)
        d.rounded_rectangle([x_a, y_a, x_b, y_b], radius=int(3 * s), fill=BAR)

    # 条形码
    x_a, x_b = rx(code[0], code[1])
    y_a, y_b = ry(code[2], code[3])
    bw = (x_b - x_a)
    widths = [0.012, 0.006, 0.018, 0.006, 0.010, 0.006, 0.022, 0.008, 0.014]
    cur = x_a
    for i, wr in enumerate(widths):
        wpx = bw * wr
        if cur + wpx > x_b:
            break
        d.rectangle([cur, y_a, cur + wpx, y_b], fill=(120, 134, 154))
        cur += wpx + bw * 0.014

    # 模型检出的文字块
    for idx, (_, a, b, c, e) in enumerate(dets):
        x_a, x_b = rx(a, b)
        y_a, y_b = ry(c, e)
        d.rectangle([x_a, y_a, x_b, y_b], outline=CYAN, width=max(1, int(2 * s)))
        tag = f"0{idx + 1}"
        tw = d.textlength(tag, font=font(int(12 * s), bold=True))
        th = int(17 * s)
        d.rounded_rectangle([x_a, y_a - th, x_a + tw + 12 * s, y_a],
                            radius=int(3 * s), fill=CYAN)
        d.text((x_a + 6 * s, y_a - th / 2), tag, font=font(int(12 * s), bold=True),
               fill=INK, anchor="lm")


# ------------------------------------------------------------------ 两个版式
def render_poster(path, w=1600, h=600):
    """README 顶部的横幅海报。"""
    img = make_background(w, h)
    d = ImageDraw.Draw(img)

    d.rounded_rectangle([88, 96, 92, 132], radius=2, fill=CYAN)
    d.text((108, 114), "HunyuanOCR-1.5 · Text Spotting",
           font=font(29, bold=True), fill=CYAN, anchor="lm")

    shadow_text(img, (88, 146), "本地文字识别服务", font(70, bold=True), WHITE)

    gradient_rule(img, 90, 236, 520, 4)

    d.text((88, 280), "8GB 显卡起步 · 昇腾多卡扩容 · 替掉按量计费的云 OCR",
           font=font(27), fill=(147, 197, 253), anchor="lm")
    d.text((88, 328), "聚合准确率会骗人 —— 逐字符比对才看得见真差距",
           font=font(26, bold=True), fill=AMBER, anchor="lm")

    x = 90
    for label in ("8GB 显存可跑", "昇腾 8 卡扩容", "逐字符 A/B 验收"):
        x = draw_chip(img, x, 378, label, font(20)) + 18

    d.text((90, 472), "模型权重遵循腾讯 HunyuanOCR 原仓库许可（Apache-2.0） · 本仓库代码 MIT · 不分发任何权重",
           font=font(19), fill=(122, 138, 160), anchor="lm")

    draw_mock_card(img, (940, 85, 1520, 515), s=1.0)

    img.convert("RGB").save(path, "PNG", optimize=True)
    return path


def render_social(path, w=1280, h=640):
    """GitHub 社交预览图（仓库设置页上传，1280×640）。

    居中构图：GitHub 在移动端会把预览图**居中裁成 640×640 的方块**，
    左右两侧会被切掉。所以标题、标语、标签全部收在中间 640px 里。
    """
    img = make_background(w, h)
    d = ImageDraw.Draw(img)
    cx = w // 2

    d.text((cx, 58), "HunyuanOCR-1.5 · Text Spotting",
           font=font(24, bold=True), fill=CYAN, anchor="mm")

    shadow_text(img, (cx, 100), "本地文字识别服务", font(54, bold=True),
                WHITE, anchor="ma")

    gradient_rule(img, cx - 200, 168, 400, 4)

    d.text((cx, 192), "8GB 显卡起步 · 昇腾多卡扩容 · 替掉按量计费的云 OCR",
           font=font(23), fill=(147, 197, 253), anchor="mm")
    d.text((cx, 230), "聚合准确率会骗人 —— 逐字符比对才看得见真差距",
           font=font(23, bold=True), fill=AMBER, anchor="mm")

    labels = ("8GB 显存可跑", "逐字符 A/B 验收", "MIT 开源")
    f = font(19)
    widths = [d.textlength(t, font=f) + 46 for t in labels]
    x = cx - (sum(widths) + 16 * (len(labels) - 1)) / 2
    for t, w_ in zip(labels, widths):
        draw_chip(img, x, 268, t, f)
        x += w_ + 16

    d.text((cx, 328), "Apache-2.0 模型 · MIT 代码 · 不分发权重",
           font=font(18), fill=(100, 116, 139), anchor="mm")

    draw_mock_card(img, (340, 352, 940, 606), s=0.92)

    img.convert("RGB").save(path, "PNG", optimize=True)
    return path


def main():
    ap = argparse.ArgumentParser(description="生成 README 海报和 GitHub 社交预览图")
    ap.add_argument("--out-dir", default="docs")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("写出：", render_poster(str(out / "poster.png")))
    print("写出：", render_social(str(out / "social-preview.png")))


if __name__ == "__main__":
    main()
