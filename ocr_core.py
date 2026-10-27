"""HunyuanOCR 推理核心 —— 开发机（CUDA）和生产机（Ascend NPU）共用同一份。

这里只放"与设备无关、且直接影响识别结果"的部分：官方提示词、放大倍数、权重精度、
生成参数、输出解析。开发入口和生产入口都从这里取，保证两边跑的是同一套逻辑——
提示词差一个字，开发机上那 30 项人工标注的实测结论就对生产机失效了。

设备相关的东西（进程优先级、NPU 设备号、HTTP 服务、8 卡负载均衡）分别放在
batch_ocr_dev.py 和 serve_ocr_prod.py。
"""
import json
import os
import re
import sys
import time
from pathlib import Path

# 这几个环境变量必须在 import transformers 之前设置。
# HF_HUB_OFFLINE 不在这里设默认值：开发机权重已缓存（入口脚本里设 1），
# 生产机首次部署可能要联网下载（由部署环境决定）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Windows 控制台默认 GBK，输出中文会 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_NAME = "tencent/HunyuanOCR"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 送进模型前先放大。模型支持的最高输入分辨率远高于本项目回归集的原图尺寸，
# 原图直接进模型只用到约 864 个视觉 token，笔画占的 patch 太少；放大后笔画
# 占更多 patch，实测能找回细小的编号类内容、修正形近字误识。
#
# 上限来自 processor 的 max_pixels = 2048*2048 = 4.19MP（preprocessor_config.json）。
# 以回归集 1166x784 的原图（0.91MP）为例：1.5x -> 2.06MP，还在限制内；
# 2.5x -> 5.71MP 会被 smart_resize 缩回 2492x1652，等于"先 LANCZOS 放大再被内部
# 降采样"，双重重采样把笔画糊掉——这就是 2.5x 反而更差（23/29 vs 24/29）的机制。
# 理论临界点约 2.14x，1.5 稳在限制内且几乎不触发重采样。
# **显存大不是调大 scale 的通行证**，瓶颈在重采样不在显存。
DEFAULT_SCALE = 1.5

# 权重精度。官方默认是 bfloat16，但 bf16 只有 7 位尾数，在小字密集的图上会
# 实实在在掉字。插电状态实测（全量数据集，三档跑同一份配置）：
#   bf16  11.2 min  607 块
#   fp16  10.8~11.1 min  605 块
#   fp32  12.4 min  608 块
# 逐张比对字符流（去空白后）：fp16 与 fp32 有 30/33 完全相同、剩 3 张互有胜负，
# 两者准确度等价；而 bf16 与它们只有 19/33 相同，在另外 14 张里 **bf16 从来没
# 赢过**：7 张是它少认出内容（整块小字漏掉、某个字段漏掉），
# 7 张是它认错了字（形近字互换、人名里的一个字被误识、地名被整个读错）。
# 所以默认用 fp16：比 fp32 更快、显存更省，准确度和 fp32 一样，还比官方默认准。
# （注意：光比"命中项数"看不出差别，三档都是 20/30，必须逐字符比。）
# 昇腾 910B 的原生张量精度也是 fp16，生产端沿用同一档。
DEFAULT_DTYPE = "float16"

# 官方 spotting_json 任务提示词，逐字取自
# Tencent-Hunyuan/HunyuanOCR 的 inference/utils/tasks.py -> TASK_PROMPTS["spotting_json"]。
# 官方明确要求"提示词按任务类型锁死，不让使用者手改"，实测也确实更好：
# 换用官方提示词后，回归集上多找出多个真实文本块（小字字段、零散文字
# 一类），自造提示词整块漏掉。
#
# 模型只认这一种 JSON 输出（带 box），所以 words 是转换出来的，不是模型直接给的。
PROMPT = (
    "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 "
    "输出格式为 JSON 数组，每个元素必须包含："
    '"box": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；'
    '"text": "识别出的文字内容"。 '
    "注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。"
)

# 采样：贪心（do_sample=False），不加 repetition_penalty。
# 官方 client 默认带 repetition_penalty=1.08，但那是为了压住长文档贪心解码的
# 重复退化；本场景下同一个印刷短语重复出现四五次是正常内容，加惩罚实测会
# 掉真块（编号字段、收件人字段）并造出坏块（重复拼接出来的假字段名）。
# 准确率优先，这里刻意偏离官方默认值。
#
# max_new_tokens=2048：本场景 spotting 输出 700~1500 字符（约 400~500 token），
# 2048 有三倍余量；官方 doc_parse 用的 32768 是给长文档的，这里用不上。
MAX_NEW_TOKENS = 2048


def npu_available():
    """torch_npu 是否可用。装了 torch_npu 之后 torch 才会挂上 torch.npu。"""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def resolve_device(requested=None, index=0):
    """决定用哪个设备。

    requested 形如 "npu:3" / "cuda:0" / "cpu"，为 None 时自动挑：
    有 NPU 用 NPU，其次 CUDA，最后 CPU（CPU 上强制 fp32）。
    """
    if requested:
        if requested.startswith("npu") and not npu_available():
            raise RuntimeError(
                "要跑 NPU 但 torch_npu 不可用。生产机需要装 CANN 和 torch_npu；"
                "在开发机上调试请显式指定 --device cuda:0 或 --device cpu。"
            )
        return requested
    if npu_available():
        return f"npu:{index}"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_processor(model_name):
    """加载 processor。5.13.0 上还没有 backend="pil"，退回等价的 use_fast=False。"""
    try:
        return AutoProcessor.from_pretrained(model_name, backend="pil")
    except TypeError:
        return AutoProcessor.from_pretrained(model_name, use_fast=False)


class HunyuanOcr:
    """HunyuanOCR 推理封装：加载一次模型，逐张出文本块。设备无关。"""

    def __init__(self, model_name=MODEL_NAME, device="cuda:0",
                 dtype_name=DEFAULT_DTYPE, attn="sdpa", scale=DEFAULT_SCALE):
        self.device = resolve_device(device)
        self.scale = scale
        self.dtype_name = dtype_name
        self.attn = attn

        dtype = getattr(torch, dtype_name)
        if self.device == "cpu":
            dtype = torch.float32

        self.processor = load_processor(model_name)
        # 必须是 AutoModelForImageTextToText，它才映射到带 .generate() 的
        # HunYuanVLForConditionalGeneration。
        # attention 用 sdpa 而不是 eager：eager 会物化完整注意力矩阵，
        # 放大输入后直接 OOM（2x 时要 12.2GB）。sdpa 峰值显存 2211MiB vs
        # eager 3790MiB，输出逐字相同。
        kwargs = dict(dtype=dtype, attn_implementation=attn)
        if self.device.startswith("cuda"):
            # 开发机路径：交给 accelerate 自动分配，8GB 卡上整个模型进显存
            # （实测峰值 2211MiB）。
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, device_map="auto", **kwargs
            )
        else:
            # 生产机路径：昇腾上不走 accelerate 的 device_map（会按 cuda 猜设备），
            # 显式 .to(device) 最稳。
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, **kwargs
            ).to(self.device)

    def prepare(self, image):
        """原图 -> 送进模型的尺寸。放大用 LANCZOS，和开发机一直一致。"""
        if self.scale != 1:
            image = image.resize(
                (int(image.width * self.scale), int(image.height * self.scale)),
                Image.LANCZOS,
            )
        return image

    def recognize_image(self, image):
        """PIL.Image -> 模型原始输出字符串。"""
        image = self.prepare(image)
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=image, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )

        trimmed = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def recognize_file(self, image_path):
        """图片路径 -> 模型原始输出字符串。"""
        image = Image.open(image_path).convert("RGB")
        return self.recognize_image(image)


def _blocks_to_words(blocks):
    return [
        {"words": str(b.get("text", ""))}
        for b in blocks
        if isinstance(b, dict) and b.get("text")
    ]


def _split_top_level_objects(body):
    """把文本里的顶层 {...} 片段逐个切出来（字符串感知）。"""
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield body[start:i + 1]
                start = None


# 兜底用：直接从原文里抠出所有 "text" 字段的值。
# 匹配字符串值里的转义（\" \\ \n 等），抠出来交给 json.loads 反解。
_TEXT_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_words(raw):
    """把模型输出解析成 [{"words": ...}, ...]；完全解析不了返回 None。

    **一个坏元素不该让整张图的结果全丢。** 模型偶尔会吐出个别格式错误的元素
    （实测过：'"box": [' 被吐成 '"box'，少了冒号和左方括号），整个数组于是不是
    合法 JSON——原来整张图的结果会被一起丢掉。

    所以分两层：
      1. 整段按 JSON 解析（正常路径）；
      2. 失败则退化成"从原文里逐个抠出 "text" 字段"，坏的那个元素自动被跳过，
         其余照常返回。
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        blocks = json.loads(cleaned)
        if isinstance(blocks, list):
            return _blocks_to_words(blocks)
    except json.JSONDecodeError:
        pass

    # 兜底：逐个抠 "text" 字段，坏掉的元素自动被跳过
    words = []
    for m in _TEXT_RE.finditer(cleaned):
        val = m.group(1)
        try:
            val = json.loads('"' + val + '"')
        except json.JSONDecodeError:
            pass
        if val:
            words.append({"words": val})
    return words or None


def find_images(input_dir):
    """扫描目录下的图片，按文件名排序，保证每次跑出来的顺序一致。"""
    return sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def recognize_entry(ocr, image_path):
    """识别一张图，返回结果 JSON 里的一条（开发机批量、生产机批量共用）。"""
    t0 = time.time()
    raw = ocr.recognize_file(image_path)
    elapsed = round(time.time() - t0, 2)

    words = parse_words(raw)
    entry = {"image": Path(image_path).name, "elapsed_sec": elapsed}
    if words is None:
        entry["words_result"] = []
        entry["raw"] = raw
    else:
        entry["words_result"] = words
    return entry


def build_payload(model, scale, dtype, attn, source_dir, image_count,
                  generated_at, extra=None):
    """结果 JSON 的公共头部。开发机和生产机必须完全一致，下游才不用改。"""
    payload = {
        "generated_at": generated_at,
        "model": model,
        "prompt": "spotting_json（官方）",
        "scale": scale,
        "attn_implementation": attn,
        "dtype": dtype,
        "sampling": "greedy",
        "source_dir": str(source_dir),
        "image_count": image_count,
        "results": [],
    }
    if extra:
        payload.update(extra)
    return payload


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
