"""开发机批量识别（本地 CUDA 单卡）。

生产环境请用 serve_ocr_prod.py（昇腾 NPU + HTTP 服务），不要用这个。

输入：项目内 data/samples/（脚本自动扫描该目录下的图片）
输出：output/ocr_<时间戳>.json，一个文件包含所有图片的结果，
      每张的 words_result 是 [{"words": "..."}, ...]，与主流云 OCR 接口同构。

用法：
    python batch_ocr_dev.py
    python batch_ocr_dev.py --input-dir /path/to/images
    python batch_ocr_dev.py --limit 5          # 只跑前 5 张，调试用
    python batch_ocr_dev.py --scale 1          # 不放大，原图进模型
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 权重已缓存，默认离线，避免镜像站抖动。必须在 import transformers 之前设置。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ocr_core import (  # noqa: E402
    DEFAULT_DTYPE,
    DEFAULT_SCALE,
    MODEL_NAME,
    build_payload,
    find_images,
    recognize_entry,
    write_json,
    HunyuanOcr,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data" / "samples"
DEFAULT_OUTPUT_DIR = ROOT / "output"


def raise_process_priority():
    """把本进程优先级提到 HIGH（仅 Windows）。

    这台机器后台常驻进程不少，推理时 Python 进程按正常优先级拿不到足够 CPU，
    CUDA 内核发射被饿着，NVIDIA 驱动按利用率把 SM 频率压到 400 多 MHz：
    实测同一张图 250 ms/token。提到 HIGH 后 37 ms/token，快 6.8 倍。
    不用 REALTIME——那会和驱动线程抢，反而略慢。

    注意 GetCurrentProcess() 返回的是 64 位伪句柄 (HANDLE)-1，必须显式声明
    c_void_p；不声明的话 ctypes 会按 c_int 截断成 0xFFFFFFFF，调用虽然返回
    成功，实际把优先级设成了 IDLE(0x0)，比正常优先级还慢。

    生产机是 192 核的独占服务器，没有这种抢占，不需要也不应该提优先级，
    所以这段只在开发机入口里。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        HIGH_PRIORITY_CLASS = 0x00000080
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k32.SetPriorityClass(k32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)
    except Exception:
        pass  # 提不了就算了，不影响识别结果


def main():
    ap = argparse.ArgumentParser(description="批量识别（开发机）")
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="图片目录")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="结果输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 表示全部")
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                    help="送进模型前放大倍数，1 表示原图")
    ap.add_argument("--dtype", default=DEFAULT_DTYPE,
                    choices=["bfloat16", "float16", "float32"], help="权重精度")
    ap.add_argument("--device", default="cuda:0",
                    help="设备，如 cuda:0 / cpu（生产机用 serve_ocr_prod.py）")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"找不到输入目录：{input_dir}")
        return 1

    images = find_images(input_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"目录里没有图片：{input_dir}")
        return 1

    print(f"待识别 {len(images)} 张，来源 {input_dir}，放大 {args.scale}x")

    raise_process_priority()
    ocr = HunyuanOcr(device=args.device, scale=args.scale, dtype_name=args.dtype)
    print(f"模型加载完成，设备 {ocr.device}，精度 {ocr.dtype_name}")

    started = datetime.now()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"ocr_{started.strftime('%Y%m%d_%H%M%S')}.json"

    payload = build_payload(
        model=MODEL_NAME,
        scale=args.scale,
        dtype=args.dtype,
        attn=ocr.attn,
        source_dir=input_dir,
        image_count=len(images),
        generated_at=started.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # 每张都重写一遍文件，中途崩了已识别的部分也还在
    for i, image_path in enumerate(images, 1):
        entry = recognize_entry(ocr, image_path)
        payload["results"].append(entry)
        write_json(out_file, payload)

        if "raw" in entry:
            print(f"[{i}/{len(images)}] {entry['image']}  解析失败，原文已存入 raw")
        print(f"[{i}/{len(images)}] {entry['image']}  "
              f"{len(entry['words_result'])} 个文本块  {entry['elapsed_sec']}s")

    print(f"\n全部完成，结果写入 {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
