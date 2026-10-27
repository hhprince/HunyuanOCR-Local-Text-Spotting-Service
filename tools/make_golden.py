"""从一次实际运行结果里生成黄金基准（data/expected/sample_golden.json）。

    python tools/make_golden.py output/pub/bench/clean_fixed/*.json

基准是合成样例的识别结果，不含任何真实数据，可以随仓库发布。
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

src = sorted(glob.glob(sys.argv[1]))[-1]
d = json.load(open(src, encoding="utf-8"))

out = {
    "_comment": "合成样例图的黄金基准。样例由 tools/make_sample.py 程序生成"
                "（不含任何真实数据），本文件是它们在本项目默认配置下的识别结果。"
                "改动默认配置会让这个基准失效——那正是它该起的作用。",
    "_config": {
        "model": d["model"],
        "dtype": d["dtype"],
        "scale": d["scale"],
        "attn_implementation": d["attn_implementation"],
        "sampling": d["sampling"],
    },
    "results": [
        {"image": r["image"], "words_result": r["words_result"]}
        for r in d["results"]
    ],
}

dst = ROOT / "data" / "expected" / "sample_golden.json"
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote", dst, "->", len(out["results"]), "images")
