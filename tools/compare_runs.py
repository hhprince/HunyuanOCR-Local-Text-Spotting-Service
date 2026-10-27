"""比对两次（或多次）识别结果——准确度 A/B 的正确工具。

**为什么不能只比"命中项数"**：本项目实测过三档权重精度（bf16/fp16/fp32），
在 30 项人工标注上命中数都是 20/30，完全看不出差别；但把每张图的所有文本块
拼起来、去掉所有空白后逐字符比，bf16 和另外两档只有 19/33 相同——
**bf16 会在 14 张里一次都没赢过**。命中数这个指标太粗，会骗人。

用法：
    # 两个结果比
    python tools/compare_runs.py a.json b.json

    # 多个结果两两比（出矩阵）
    python tools/compare_runs.py a.json b.json c.json

    # 只看不一样的图
    python tools/compare_runs.py a.json b.json --diff

结果 JSON 是本项目自己的格式：
    {"results": [{"image": "xxx.jpg", "words_result": [{"words": "..."}, ...]}, ...]}
"""
import argparse
import json
import os
import re
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for r in d.get("results", []):
        text = "".join(w.get("words", "") for w in r.get("words_result", []))
        out[r["image"]] = re.sub(r"\s+", "", text)
    return out


def first_diff(a, b):
    """返回两段文本第一处不同的位置，以及各自前后各 20 个字符。"""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            lo = max(0, i - 20)
            return i, a[lo:i + 20], b[lo:i + 20]
    if len(a) != len(b):
        i = min(len(a), len(b))
        lo = max(0, i - 20)
        return i, a[lo:i + 20], b[lo:i + 20]
    return None, a, b


def main():
    ap = argparse.ArgumentParser(description="比对识别结果（逐字符）")
    ap.add_argument("files", nargs="+", help="结果 JSON，至少两个")
    ap.add_argument("--diff", action="store_true", help="打印不一样的图的差异片段")
    args = ap.parse_args()

    if len(args.files) < 2:
        print("至少给两个结果文件", file=sys.stderr)
        return 1

    runs = {p: load(p) for p in args.files}
    names = sorted({n for r in runs.values() for n in r})
    print(f"共 {len(names)} 张图，{len(runs)} 次运行\n")

    # 两两一致率矩阵
    keys = list(runs)
    if len(keys) > 2:
        print("字符流一致率矩阵（完全相同的张数）：")
        labels = [os.path.basename(os.path.dirname(k)) or os.path.basename(k) for k in keys]
        w = max(len(x) for x in labels)
        print(" " * (w + 2) + "".join(f"{x[:12]:>{w + 2}}" for x in labels))
        for a, la in zip(keys, labels):
            row = "".join(
                f"{sum(1 for n in names if runs[a].get(n) == runs[b].get(n)):>{w + 2}}"
                if a != b else f"{'--':>{w + 2}}"
                for b in keys
            )
            print(f"{la:<{w}} {row}")
        print()

    # 逐对看差异
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            same = sum(1 for n in names if runs[a].get(n) == runs[b].get(n))
            print(f"{os.path.basename(a)}  vs  {os.path.basename(b)}   "
                  f"{same}/{len(names)} 张逐字符相同")
            if args.diff:
                for n in names:
                    x, y = runs[a].get(n, ""), runs[b].get(n, "")
                    if x != y:
                        pos, sx, sy = first_diff(x, y)
                        print(f"    {n}  第 {pos} 字起不同")
                        print(f"      A: ...{sx}...")
                        print(f"      B: ...{sy}...")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
