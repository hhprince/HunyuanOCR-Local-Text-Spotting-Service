"""给识别结果打准确率分（用合成样例的 ground truth）。

两个指标：
  行命中率   —— 每一行标准答案，识别结果里有没有原样出现（整行对才算命中）
  字符准确率 —— 标准答案的字符流 和 识别出的字符流 的最长公共子序列 / 标准答案长度
                 （能反映"错了一个字"这种部分正确的情形）

用法：
    python tools/score.py output/ocr_20260101_120000.json --truth data/samples_noisy/ground_truth.json
"""
import argparse
import json
import re
import sys


def norm(s):
    """比较前先去掉所有空白——OCR 的切块方式不同不该算错。"""
    return re.sub(r"\s+", "", s)


def lcs_len(a, b):
    """最长公共子序列长度（滚动数组版）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[len(b)]


def load_truth(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for r in d.get("results", []):
        out[r["image"]] = r["lines"]
    return out


def load_ocr(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for r in d.get("results", []):
        out[r["image"]] = [w.get("words", "") for w in r.get("words_result", [])]
    return out


def score(truth, ocr):
    rows_hit = rows_total = 0
    chars_ok = chars_total = 0
    per_image = []
    for img, lines in truth.items():
        blocks = ocr.get(img, [])
        got = norm("".join(blocks))
        hit = 0
        for line in lines:
            rows_total += 1
            if norm(line) and norm(line) in got:
                hit += 1
                rows_hit += 1
        ref = norm("".join(lines))
        common = lcs_len(ref, got)
        chars_ok += common
        chars_total += len(ref)
        per_image.append({
            "image": img,
            "line_hit": f"{hit}/{len(lines)}",
            "char_acc": round(common / len(ref), 4) if ref else 0.0,
        })
    return {
        "line_hit": f"{rows_hit}/{rows_total}",
        "line_hit_rate": round(rows_hit / rows_total, 4) if rows_total else 0.0,
        "char_acc": round(chars_ok / chars_total, 4) if chars_total else 0.0,
        "per_image": per_image,
    }


def main():
    ap = argparse.ArgumentParser(description="给识别结果打准确率分")
    ap.add_argument("result", help="结果 JSON")
    ap.add_argument("--truth", required=True, help="ground_truth.json")
    ap.add_argument("--verbose", action="store_true", help="逐张打印")
    args = ap.parse_args()

    truth = load_truth(args.truth)
    ocr = load_ocr(args.result)
    s = score(truth, ocr)
    print(f"文件 {args.result}")
    print(f"  行命中率   {s['line_hit']}  ({s['line_hit_rate'] * 100:.1f}%)")
    print(f"  字符准确率 {s['char_acc'] * 100:.2f}%")
    if args.verbose:
        for p in s["per_image"]:
            print(f"    {p['image']}  行命中 {p['line_hit']}  字符准确率 {p['char_acc'] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
