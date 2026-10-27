"""生成合成样例图（不含任何真实数据），供冒烟测试、演示和精度实验使用。

样例是程序画出来的：白底、几行中文/数字、一段伪条形码。故意做成
"小字密集"的样子，这样能验证放大倍数、精度档位这些参数是不是真的在起作用。

两种风格：
  clean —— 正常大小的字，用于跑通流程、演示、黄金基准
  hard  —— 小字 + 扫描噪声 + 轻微模糊，用于模拟低质量翻拍
  noisy —— hard 再加压对比度 + 强噪点 + 强 JPEG 压缩，最接近真实脏图

    python tools/make_sample.py --style clean --count 3
    python tools/make_sample.py --style hard  --count 5 --out data/samples_hard
    python tools/make_sample.py --style noisy --count 5 --out data/samples_noisy
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 常见中文字体，按优先级找；都找不到就退回只画 ASCII（用 PIL 内置位图字体）
CJK_FONTS = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

CLEAN_TEXT = [
    "合成样例卡片 SYNTHETIC SAMPLE",
    "本图由 tools/make_sample.py 程序生成，不含任何真实数据。",
    "编号 NO. {n:04d}    批次 BATCH-{n:04d}",
    "收件人栏：示例收件人    电话：000-0000-0000",
    "地址栏：某市某区示例路 {n} 号    邮编：000000",
    "这是一段用于测试小字识别的说明文字，长度接近真实卡片上的印刷内容。",
    "CHINA POST    SAMPLE    80",
]

HARD_TEXT = [
    "样例 {n:04d}",
    "收件人：示例收件人  电话：000-0000-0000",
    "地址：某市某区示例路 {n} 号  邮编：000000",
    "编号 NO.{n:04d}  批次 BATCH-{n:04d}  重量 0.{n:02d}kg",
    "本行故意写得很长很细，用来考察小字号下不同权重精度的差异：",
    "abcdefghijklmnopqrstuvwxyz 0123456789 ABCDEFGHIJKLMNOP",
    "第 {n} 号样例，生成时间基准序号 {n:04d}，请勿当作真实数据使用。",
    "CHINA POST SAMPLE 80  MOCK BARCODE 000000",
]


def load_font(size):
    for path in CJK_FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_barcode(draw, x, y, w, h, rng):
    """伪条形码：随机宽度的黑白竖条，只是长得像，不是任何真实编码。"""
    cx = x
    while cx < x + w:
        bar = rng.choice([1, 1, 2, 3])
        if rng.random() < 0.6:
            draw.rectangle([cx, y, cx + bar, y + h], fill="black")
        cx += bar + rng.choice([1, 1, 2])


def add_scan_noise(img, rng, strength=7):
    """模拟扫描件噪点：先生成噪声图再按强度混合回去。"""
    noise = Image.effect_noise(img.size, 28).convert("L")
    noise = noise.filter(ImageFilter.GaussianBlur(0.4))
    return Image.blend(img, Image.merge("RGB", (noise, noise, noise)), strength / 100.0)


def degrade(img, rng, strength):
    """把图弄脏：压对比度 + 加噪 + 轻微模糊 + 强 JPEG 压缩。

    干净合成图上 bf16 和 fp16 的结果完全一样，测不出精度差别；
    真实的翻拍/扫描图之所以能测出来，就是这层退化。
    """
    from PIL import ImageEnhance

    img = ImageEnhance.Contrast(img).enhance(1.0 - 0.45 * strength)
    img = ImageEnhance.Brightness(img).enhance(1.0 - 0.08 * strength)
    img = img.filter(ImageFilter.GaussianBlur(0.3 + 0.5 * strength))
    img = add_scan_noise(img, rng, strength=6 + 14 * strength)
    return img


def make_one(path, n, rng, style):
    w, h = 1166, 784
    img = Image.new("RGB", (w, h), (247, 245, 238))
    d = ImageDraw.Draw(img)

    hard = style in ("hard", "noisy")
    title_size = 22 if hard else 34
    body_size = 13 if hard else 24
    small_size = 11 if hard else 20
    title_font = load_font(title_size)
    body_font = load_font(body_size)
    small_font = load_font(small_size)
    lines = HARD_TEXT if hard else CLEAN_TEXT

    d.rectangle([18, 18, w - 18, h - 18], outline=(150, 140, 120), width=2)
    d.line([40, 96, w - 40, 96], fill=(160, 150, 130), width=1)

    # 记下画上去的每一行文字，后面导出成 ground_truth.json
    truth = []

    title = lines[0].format(n=n)
    d.text((44, 38), title, fill=(20, 20, 20), font=title_font)
    truth.append(title)

    y = 116 if hard else 150
    step = 20 if hard else 44
    for line in lines[1:]:
        s = line.format(n=n)
        d.text((44, y), s, fill=(40, 40, 40), font=body_font)
        truth.append(s)
        y += step

    d.text((44, y + 14), "0123456789", fill=(40, 40, 40), font=small_font)
    truth.append("0123456789")
    draw_barcode(d, 44, y + 40, 420, 70 if hard else 86, rng)

    # 右下角一个小方块，模拟"小字区域"
    d.rectangle([w - 300, h - 120, w - 60, h - 50], outline=(60, 60, 60), width=1)
    d.text((w - 285, h - 100), "000000", fill=(20, 20, 20), font=small_font)
    truth.append("000000")

    if hard:
        img = img.filter(ImageFilter.GaussianBlur(0.4))
        img = add_scan_noise(img, rng, strength=8)
    if style == "noisy":
        img = degrade(img, rng, 1.0)

    img.save(path, quality=25 if style == "noisy" else (88 if hard else 92))
    return truth


def main():
    ap = argparse.ArgumentParser(description="生成合成样例图")
    ap.add_argument("--out", default="data/samples", help="输出目录")
    ap.add_argument("--count", type=int, default=3, help="生成几张")
    ap.add_argument("--style", default="clean", choices=["clean", "hard", "noisy"],
                    help="样例风格")
    ap.add_argument("--seed", type=int, default=7, help="随机种子，保证可复现")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    truth = {}
    for i in range(1, args.count + 1):
        p = os.path.join(args.out, f"{args.style}_{i:03d}.jpg")
        truth[os.path.basename(p)] = make_one(p, i, rng, args.style)
        print("生成", p)

    # 标注：合成图的 ground truth 是已知的，直接导出，让准确率可以量化，
    # 而不是只能"两张结果一样/不一样"
    gt_path = os.path.join(args.out, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump({"_style": args.style, "results": [
            {"image": k, "lines": v} for k, v in truth.items()]},
            f, ensure_ascii=False, indent=2)
    print("标注", gt_path)

    print(f"\n共 {args.count} 张（{args.style}），输出目录 {args.out}")
    print("用开发机脚本跑一遍：python batch_ocr_dev.py --input-dir", args.out)
    print("然后打分：python tools/score.py <结果.json> --truth", gt_path)


if __name__ == "__main__":
    main()
