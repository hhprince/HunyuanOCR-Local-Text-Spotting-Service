"""生产入口：昇腾 NPU 常驻 HTTP 服务，8 卡负载均衡。

开发机请用 batch_ocr_dev.py（本地 CUDA 单卡批量），不要用这个。

架构（和 vLLM 官方服务同一种思路）：
    HTTP 进程（uvicorn，1 个）        ← 只做协议解析和转发，不占算力
      └── 8 个 worker 进程           ← 每进程 1 张 Ascend 910B + 1 份模型
            req_q[i] / resp_q[i]     ← 每个 worker 一对私有队列
    请求进来 -> 挑一个空闲 worker -> 投进它的 req_q -> 从它的 resp_q 取结果

用私有队列而不是一条大队列，是为了避免并发时取错别人的响应；挑空闲 worker
就是"负载均衡"——没有 worker 忙时按轮转挑最久没用的那个。

推理逻辑全部来自 ocr_core，和开发机是同一份代码（提示词、放大倍数、精度、采样）。

用法：
    # 常驻服务（生产）
    python serve_ocr_prod.py --host 0.0.0.0 --port 8000

    # 批量跑一遍目录，产出和开发机同构的 JSON（上线前用它和开发机结果逐字符比对）
    python serve_ocr_prod.py --batch --input-dir /data/images --output-dir /data/out

    # 在开发机上调试这个服务（不碰 NPU）
    python serve_ocr_prod.py --devices cuda:0,cuda:0 --port 8001
"""
import argparse
import base64
import io
import itertools
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# 必须在 import torch / transformers 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import ocr_core  # noqa: E402
from PIL import Image  # noqa: E402

log = logging.getLogger("ocr")


def decode_base64(text):
    """base64 图片串 -> 原始字节。兼容 data:image/jpeg;base64, 前缀和缺 padding。"""
    text = text.strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    text += "=" * (-len(text) % 4)
    return base64.b64decode(text)


# --------------------------------------------------------------------------
# worker：一个进程 = 一张 NPU = 一份模型
# --------------------------------------------------------------------------

def _worker(worker_id, device, model_name, dtype_name, attn, scale,
            req_q, resp_q, ready_q):
    """子进程主体。必须能 pickle，所以放模块级。"""
    try:
        if device.startswith("npu"):
            import torch
            import torch_npu  # noqa: F401  导入后 torch 才有 torch.npu

            # 显式绑卡：一进程一卡，不要让它都往 npu:0 上挤。
            torch.npu.set_device(int(device.split(":")[1]))
        ocr = ocr_core.HunyuanOcr(model_name=model_name, device=device,
                                  dtype_name=dtype_name, attn=attn, scale=scale)
    except Exception as e:
        ready_q.put((worker_id, f"{type(e).__name__}: {e}"))
        return

    ready_q.put((worker_id, "ok"))
    log.info("worker %d 就绪，设备 %s", worker_id, device)

    while True:
        job = req_q.get()
        if job is None:          # 收工信号
            break
        req_id, image_bytes = job
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            raw = ocr.recognize_image(image)
            words = ocr_core.parse_words(raw)
            resp = {
                "words_result": words if words is not None else [],
                "elapsed_sec": round(time.time() - t0, 2),
                "raw": None if words is not None else raw,
            }
        except Exception as e:
            resp = {"error": f"{type(e).__name__}: {e}",
                    "elapsed_sec": round(time.time() - t0, 2)}
        resp_q.put((req_id, resp))


# --------------------------------------------------------------------------
# worker 池
# --------------------------------------------------------------------------

class OcrPool:
    """管理 N 个 worker 进程，负责挑空闲 worker 和收发结果。"""

    def __init__(self, devices, model_name, dtype_name, attn, scale):
        # spawn 而不是 fork：CANN 运行时在 fork 出来的进程里会出问题
        ctx = mp.get_context("spawn")
        self.n = len(devices)
        self.devices = list(devices)
        self.req_qs = [ctx.Queue() for _ in range(self.n)]
        self.resp_qs = [ctx.Queue() for _ in range(self.n)]
        self.ready_q = ctx.Queue()
        self.procs = []
        self._busy = set()
        self._cond = threading.Condition()
        self._rr = 0
        self._counter = itertools.count(1)
        self.ready = {}          # worker_id -> "ok" 或错误信息
        self.stats = [{"id": i, "device": d, "images": 0, "alive": False}
                      for i, d in enumerate(devices)]
        self._cfg = (model_name, dtype_name, attn, scale)

    def start(self):
        model_name, dtype_name, attn, scale = self._cfg
        for i, device in enumerate(self.devices):
            p = mp.get_context("spawn").Process(
                target=_worker,
                args=(i, device, model_name, dtype_name, attn, scale,
                      self.req_qs[i], self.resp_qs[i], self.ready_q),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
            self.stats[i]["alive"] = True
        log.info("已拉起 %d 个 worker 进程", self.n)

    def _poll_ready(self):
        """把 ready_q 里已经到达的消息收走，不阻塞。"""
        while True:
            try:
                worker_id, msg = self.ready_q.get_nowait()
            except Exception:
                break
            self.ready[worker_id] = msg
            if msg != "ok":
                log.error("worker %d 启动失败：%s", worker_id, msg)
                self.stats[worker_id]["alive"] = False

    def wait_ready(self, timeout=600):
        deadline = time.time() + timeout
        while len(self.ready) < self.n and time.time() < deadline:
            self._poll_ready()
            if len(self.ready) >= self.n:
                break
            time.sleep(0.2)
        self._poll_ready()
        bad = {k: v for k, v in self.ready.items() if v != "ok"}
        if bad:
            raise RuntimeError(f"有 worker 启动失败：{bad}")
        if len(self.ready) < self.n:
            raise TimeoutError(f"等 worker 就绪超时（{len(self.ready)}/{self.n} 就绪）")

    def _pick_free(self):
        """挑一个空闲 worker，轮转优先。调用方必须持有 self._cond。"""
        for k in range(self.n):
            i = (self._rr + k) % self.n
            if i not in self._busy:
                self._rr = (i + 1) % self.n
                return i
        return None

    def submit(self, image_bytes, timeout=300.0):
        """投一张图，等结果。阻塞调用。"""
        deadline = time.time() + timeout
        with self._cond:
            while True:
                idx = self._pick_free()
                if idx is not None:
                    self._busy.add(idx)
                    break
                if not self._cond.wait(timeout=max(0.1, deadline - time.time())):
                    raise TimeoutError("没有空闲的 NPU，排队超时")
        try:
            req_id = next(self._counter)
            self.req_qs[idx].put((req_id, image_bytes))
            while True:
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError("识别超时")
                got_id, resp = self.resp_qs[idx].get(timeout=left)
                if got_id == req_id:
                    break
            if "error" in resp:
                raise RuntimeError(resp["error"])
            self.stats[idx]["images"] += 1
            return resp
        finally:
            with self._cond:
                self._busy.discard(idx)
                self._cond.notify()

    def status(self):
        self._poll_ready()
        for i, p in enumerate(self.procs):
            self.stats[i]["alive"] = p.is_alive()
        ready = sum(1 for v in self.ready.values() if v == "ok")
        return {
            "status": "ok" if ready == self.n else "loading",
            "ready_workers": ready,
            "pool_size": self.n,
            "devices": self.devices,
            "workers": self.stats,
        }

    def stop(self):
        for q in self.req_qs:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()


# --------------------------------------------------------------------------
# HTTP 层
# --------------------------------------------------------------------------

def build_app(pool, cfg):
    from fastapi import FastAPI, HTTPException, Request
    from starlette.concurrency import run_in_threadpool

    @asynccontextmanager
    async def lifespan(app):
        pool.start()
        yield
        pool.stop()

    app = FastAPI(title="HunyuanOCR 通用文字识别服务", version="1.0", lifespan=lifespan)

    @app.get("/")
    def index():
        return {
            "service": "hunyuan-ocr",
            "model": cfg["model"],
            "endpoints": {
                "POST /ocr/general": "识别一张图，返回云 OCR 同款 words_result",
                "GET /health": "查看各 worker 状态",
            },
            "usage": {
                "multipart": "curl -F 'image=@a.jpg' http://host:port/ocr/general",
                "json": "curl -d '{\"image_base64\":\"...\"}' http://host:port/ocr/general",
            },
        }

    @app.get("/health")
    def health():
        return pool.status()

    @app.post("/ocr/general")
    async def ocr_general(request: Request):
        ctype = request.headers.get("content-type", "")
        try:
            if "multipart/form-data" in ctype:
                form = await request.form()
                up = form.get("image") or form.get("file")
                if up is None:
                    raise HTTPException(400, "缺少 image 字段")
                if hasattr(up, "read"):
                    data = await up.read()
                else:
                    data = decode_base64(str(up))
            else:
                try:
                    body = await request.json()
                except Exception:
                    raise HTTPException(400, "请求体必须是 multipart/form-data 或 JSON")
                b64 = body.get("image_base64") or body.get("image")
                if not b64:
                    raise HTTPException(400, "缺少 image_base64 / image 字段")
                data = decode_base64(b64)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"图片解析失败：{e}")

        if not data:
            raise HTTPException(400, "图片内容为空")

        try:
            resp = await run_in_threadpool(pool.submit, data)
        except TimeoutError as e:
            raise HTTPException(503, f"排队/识别超时：{e}")
        except Exception as e:
            raise HTTPException(500, f"识别失败：{e}")

        words = resp.get("words_result") or []
        out = {
            "words_result": words,
            "words_result_num": len(words),
            "elapsed_sec": resp.get("elapsed_sec"),
        }
        if resp.get("raw") is not None:
            # 解析失败时把原文带出来，方便排查（云 OCR 接口没有这个字段）
            out["raw"] = resp["raw"]
        return out

    return app


# --------------------------------------------------------------------------
# 批量模式：跑一遍目录，产出和开发机同构的 JSON
# --------------------------------------------------------------------------

def run_batch(pool, args, cfg):
    images = ocr_core.find_images(args.input_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"目录里没有图片：{args.input_dir}")
        return 1

    pool.start()
    pool.wait_ready(timeout=args.startup_timeout)
    print(f"待识别 {len(images)} 张，来源 {args.input_dir}，{pool.n} 个 worker 并行")

    started = datetime.now()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ocr_{started.strftime('%Y%m%d_%H%M%S')}.json"

    payload = ocr_core.build_payload(
        model=cfg["model"], scale=args.scale, dtype=args.dtype, attn=args.attn,
        source_dir=args.input_dir, image_count=len(images),
        generated_at=started.strftime("%Y-%m-%d %H:%M:%S"),
        extra={"workers": pool.n, "devices": pool.devices},
    )

    results = [None] * len(images)
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=pool.n) as ex:
        futures = {ex.submit(pool.submit, p.read_bytes()): i
                   for i, p in enumerate(images)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            p = images[i]
            try:
                resp = fut.result()
                entry = {"image": p.name, "elapsed_sec": resp["elapsed_sec"],
                         "words_result": resp["words_result"]}
                if resp.get("raw") is not None:
                    entry["raw"] = resp["raw"]
            except Exception as e:
                entry = {"image": p.name, "elapsed_sec": None, "words_result": [],
                         "raw": "", "error": f"{type(e).__name__}: {e}"}
            results[i] = entry
            done += 1
            payload["results"] = [r for r in results if r is not None]
            ocr_core.write_json(out_file, payload)
            print(f"[{done}/{len(images)}] {entry['image']}  "
                  f"{len(entry['words_result'])} 个文本块  {entry['elapsed_sec']}s")

    print(f"\n全部完成，耗时 {round(time.time() - t_start, 1)}s，结果写入 {out_file}")
    return 0


# --------------------------------------------------------------------------

def parse_devices(args):
    """决定每个 worker 用哪个设备。"""
    if args.devices:
        return [d.strip() for d in args.devices.split(",") if d.strip()]
    return [f"npu:{i}" for i in args.npu_ids]


def main():
    ap = argparse.ArgumentParser(description="HunyuanOCR 生产服务（昇腾 NPU）")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址，默认只听本机；要对外提供服务请显式指定 0.0.0.0")
    ap.add_argument("--port", type=int, default=8000, help="监听端口")
    ap.add_argument("--npu-ids", default="0,1,2,3,4,5,6,7",
                    help="用哪几张卡，逗号分隔")
    ap.add_argument("--devices", default="",
                    help="直接指定设备列表（如 cuda:0,cuda:0），指定后忽略 --npu-ids")
    ap.add_argument("--model", default=ocr_core.MODEL_NAME, help="模型名或本地目录")
    ap.add_argument("--dtype", default=ocr_core.DEFAULT_DTYPE,
                    choices=["bfloat16", "float16", "float32"], help="权重精度")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"], help="注意力实现")
    ap.add_argument("--scale", type=float, default=ocr_core.DEFAULT_SCALE,
                    help="送进模型前放大倍数")
    ap.add_argument("--startup-timeout", type=float, default=600, help="等 worker 就绪的秒数")
    ap.add_argument("--batch", action="store_true", help="批量模式而不是起服务")
    ap.add_argument("--input-dir", default="", help="批量模式的图片目录")
    ap.add_argument("--output-dir", default="", help="批量模式的结果目录")
    ap.add_argument("--limit", type=int, default=0, help="批量模式只跑前 N 张")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    devices = parse_devices(args)
    cfg = {"model": args.model, "dtype": args.dtype, "attn": args.attn,
           "scale": args.scale, "devices": devices}

    if args.batch:
        if not args.input_dir or not args.output_dir:
            print("批量模式需要 --input-dir 和 --output-dir")
            return 1
        pool = OcrPool(devices, args.model, args.dtype, args.attn, args.scale)
        try:
            return run_batch(pool, args, cfg)
        finally:
            pool.stop()

    import uvicorn

    pool = OcrPool(devices, args.model, args.dtype, args.attn, args.scale)
    app = build_app(pool, cfg)
    print(f"启动服务 {args.host}:{args.port}，设备 {devices}，模型 {args.model}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
