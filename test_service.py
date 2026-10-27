"""生产服务的 HTTP 层自测（进程内 TestClient，不占端口、不对外暴露）。

会在开发机上跑一遍生产代码的完整链路：起 worker 进程 → 加载模型 → 发请求 →
比对黄金基准。样例是程序合成的（data/samples/），不含任何真实数据。

    python test_service.py
    TEST_DEVICES=cuda:0,cuda:0 python test_service.py   # 模拟多卡

注意：spawn 方式下，调用 pool.start() 的脚本，逻辑必须放进
if __name__ == "__main__": 里面——子进程会重新导入主模块，模块级再 spawn 一次
会报 "An attempt has been made to start a new process before the current process
has finished its bootstrapping phase"，worker 直接 exitcode=1 退出、请求全部超时。
"""
import base64
import json
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from fastapi.testclient import TestClient  # noqa: E402
import serve_ocr_prod  # noqa: E402


def main():
    devices = os.environ.get("TEST_DEVICES", "cuda:0").split(",")
    cfg = {"model": "tencent/HunyuanOCR", "dtype": "float16", "attn": "sdpa", "scale": 1.5}
    pool = serve_ocr_prod.OcrPool(devices, cfg["model"], cfg["dtype"],
                                   cfg["attn"], cfg["scale"])
    app = serve_ocr_prod.build_app(pool, cfg)

    sample = PROJ / "data" / "samples_clean" / "clean_001.jpg"
    if not sample.exists():
        print("没有样例图，先跑：python tools/make_sample.py --style clean --out data/samples_clean")
        return 1
    raw_bytes = sample.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode()

    golden = json.load(open(PROJ / "data" / "expected" / "sample_golden.json",
                            encoding="utf-8"))
    golden_words = [w["words"] for w in golden["results"][0]["words_result"]]

    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-42s %s %s" % (name, "PASS" if cond else "FAIL", extra), flush=True)

    with TestClient(app) as client:
        r = client.get("/")
        check("GET / 返回 200", r.status_code == 200, "-> %s" % r.json().get("service"))

        r = client.get("/health")
        j = r.json()
        check("GET /health 返回 200", r.status_code == 200,
              "-> %s ready=%s/%s" % (j.get("status"), j.get("ready_workers"), j.get("pool_size")))

        r = client.post("/ocr/general", files={"image": (sample.name, raw_bytes, "image/jpeg")})
        check("POST multipart 返回 200", r.status_code == 200, "-> %s" % r.status_code)
        multipart_words = [w["words"] for w in r.json().get("words_result", [])]
        check("识别结果与黄金基准逐字一致", multipart_words == golden_words,
              "%d vs %d 块" % (len(multipart_words), len(golden_words)))
        check("返回 words_result_num", r.json().get("words_result_num") == len(multipart_words),
              "-> %s" % r.json().get("words_result_num"))

        r = client.post("/ocr/general", json={"image_base64": b64})
        check("POST base64 JSON 返回 200", r.status_code == 200)
        b64_words = [w["words"] for w in r.json().get("words_result", [])]
        check("base64 与 multipart 结果一致", b64_words == multipart_words)

        r = client.post("/ocr/general", json={"image_base64": "data:image/jpeg;base64," + b64})
        check("POST data-URL 前缀能解析", r.status_code == 200)

        r = client.post("/ocr/general", json={})
        check("缺字段返回 400", r.status_code == 400, "-> %s" % r.status_code)

        r = client.post("/ocr/general", content=b"not json",
                        headers={"content-type": "application/json"})
        check("非法请求体返回 400", r.status_code == 400, "-> %s" % r.status_code)

        r = client.post("/ocr/general", files={"image": (sample.name, b"", "image/jpeg")})
        check("空图片不 500", r.status_code in (400, 500), "-> %s" % r.status_code)

        r = client.get("/health")
        j = r.json()
        check("health 记录到处理张数", sum(w["images"] for w in j["workers"]) >= 3,
              "-> %s" % [w["images"] for w in j["workers"]])

    print("\n%s" % ("全部通过" if ok else "有失败项"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
