<div align="center">
  <img src="docs/poster.png" alt="HunyuanOCR 本地文字识别服务" width="100%">
</div>

<div align="center">

# HunyuanOCR 本地文字识别服务

**用 HunyuanOCR-1.5 在自有算力上做中文文字识别，把按量计费的云 OCR 换成自己的一张显卡。**

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-tencent%2FHunyuanOCR-FF6D00.svg)](https://huggingface.co/tencent/HunyuanOCR)
[![Backend](https://img.shields.io/badge/backend-CUDA%20%7C%20Ascend%20NPU-7C3AED.svg)](docs/%E9%83%A8%E7%BD%B2%E8%AF%B4%E6%98%8E.md)
[![Stars](https://img.shields.io/github/stars/hhprince/HunyuanOCR-Local-Text-Spotting-Service?style=social)](https://github.com/hhprince/HunyuanOCR-Local-Text-Spotting-Service)

[快速开始](#五步把字看清) · [踩坑清单](#三个必须知道的坑) · [实测数据](docs/实验记录.md) · [部署说明](docs/部署说明.md)

</div>

---

## 这仓库解决什么

| 痛点 | 这里的答案 |
|---|---|
| 云 OCR 按量计费，业务量上来是一笔持续支出 | 1.12B 权重，**8GB 显卡**起步，成本变成一次性的 |
| 图片要出网——含个人信息的单据/卡片是**合规问题** | 全程本地推理，**图片不出网** |
| 换环境、换参数之后，说不清准确度有没有变差 | 逐字符 A/B 工具 + **全部实测数据公开** |
| 开发机和生产机跑的是两套逻辑，调好的参数一换机就废 | 开发和生产**共用 `ocr_core.py`**，提示词差一个字都不行 |

## 为什么值得看：聚合指标会骗人

> 三档权重精度（bf16 / fp16 / fp32）的**行命中率完全相同**（都是 44/60），
> 字符准确率只差 **0.07 个百分点**。
>
> **光看这两个指标，官方默认的 bf16 和 fp16 毫无区别。**
>
> 但逐字符比，只有 **2/6** 的图片结果完全相同——4 张图的差别是
> "考察/考虑"、"权重/权限"这种**单字替换**。

聚合指标分辨不出"错了一个字"。**本仓库最想说明的就是这件事**，
所以它自带一套逐字符比对工具和一份可以复现的公开基准。

## 五步把字看清

| | 步骤 | 干什么 | 一条命令 |
|---|---|---|---|
| ① | **拉权重** | 装依赖，把 1.12B 权重拉到本地缓存 | `pip install -r requirements.txt` |
| ② | **造字纸** | 程序画出合成样例图，同时导出标准答案 | `python tools/make_sample.py --style noisy --count 6 --out data/samples_noisy` |
| ③ | **试读** | 开发机单卡跑通，产出一份结果 JSON | `python batch_ocr_dev.py --input-dir data/samples_noisy` |
| ④ | **开服务** | 起 HTTP 服务，昇腾多卡数据并行 | `python serve_ocr_prod.py --host 0.0.0.0 --port 8000` |
| ⑤ | **校对** | 逐字符 A/B + 打分，证明改动没让准确度变差 | `python tools/compare_runs.py a.json b.json --diff` |

<details>
<summary><b>展开每一步的细节</b></summary>

### ① 拉权重

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

权重会自动从 HuggingFace 下载并缓存（约 4GB）。国内建议先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> **最低要求**：transformers ≥ 5.13.0（`HunYuanVL` 架构只在 5.x 提供）、显存 ≥ 8GB。

### ② 造字纸

仓库里**不含任何真实图片**，样例是 `tools/make_sample.py` 程序画出来的，
同时导出 `ground_truth.json`（合成图的标注是已知的，直接导出）：

```bash
python tools/make_sample.py --style clean --count 3 --out data/samples_clean
python tools/make_sample.py --style hard  --count 6 --out data/samples_hard
python tools/make_sample.py --style noisy --count 6 --out data/samples_noisy
```

三档风格分别对应：干净 / 小字+噪点 / 小字+噪点+低对比+强压缩（最难）。

### ③ 试读

```bash
python batch_ocr_dev.py --input-dir data/samples_noisy
```

输出 `output/ocr_<时间戳>.json`，头部带全部实际生效的参数（模型、提示词、
放大倍数、精度、采样、attn），**换环境时一眼就能看出配置有没有跑偏**。

### ④ 开服务

```bash
# 开发机调试（用 CUDA 模拟多卡）
python serve_ocr_prod.py --devices cuda:0,cuda:0 --port 8001

# 生产（昇腾 8 卡）
python serve_ocr_prod.py --host 0.0.0.0 --port 8000
```

```bash
curl -F 'image=@data/samples_noisy/noisy_001.jpg' http://127.0.0.1:8000/ocr/general
```

接口与主流云 OCR 同构：`POST /ocr/general` → `{"words_result": [{"words": "..."}], "words_result_num": N}`。

### ⑤ 校对

**光比"命中项数"会骗人，必须逐字符比。**

```bash
# 跑两遍，参数不同
python batch_ocr_dev.py --input-dir 你的图 --output-dir output/a --dtype float16
python batch_ocr_dev.py --input-dir 你的图 --output-dir output/b --dtype bfloat16

# 逐字符比：几张图完全相同 + 每张图第一处不同的位置
python tools/compare_runs.py output/a/*.json output/b/*.json --diff

# 有标注就直接打分：行命中率 + 字符准确率
python tools/score.py output/a/*.json --truth 你的标注.json
```

完整的验收流程和判定阈值见 [docs/准确度验收.md](docs/准确度验收.md)。

</details>

## 三个必须知道的坑

**1. 放大倍数有硬上限，但没有通用最优值。**
processor 的 `max_pixels = 2048² = 4.19MP`，超了会被内部缩回去：

| scale | 进模型的像素 | 行命中率 | 说明 |
|---|---|---|---|
| 1.0 | 1166×784 | 46/60 (76.7%) | 合成图上最好 |
| 1.5 | 1736×1176 | 44/60 (73.3%) | 默认值 |
| 2.5 | 2492×1652 | 37/60 (61.7%) | **超限被缩回**，双重重采样把笔画糊掉 |

> **显存大不是调大 scale 的通行证**，瓶颈在重采样不在显存。
> 临界点 = 4.19MP / 原始像素数。

但 1.0 和 1.5 谁更好**取决于你的图**：合成图本来就清晰，放大只引入插值模糊；
真实翻拍照片的原始细节比模型的 token 网格更细，放大能把细节带进去。
**所以必须用 `tools/score.py` 在自己的数据上测，不要照抄别人的默认值。**

**2. 提示词用官方的，采样反而要偏离官方。**
官方把 12 种任务的提示词锁死不让改，确实有道理——换回官方提示词后回归集上
多找出多个真实文本块，自造提示词整块漏掉。
但官方 client 默认带 `repetition_penalty=1.08`，那是给长文档防贪心重复退化用的；
本场景下同一印刷短语重复出现是正常内容，加惩罚实测会**掉真块、造出假字段名**。
这里刻意用纯贪心，准确率优先。

**3. 输出解析不能"一个坏元素毁掉整张图"。**
实测模型会吐出 `{"box755, 880, 818, 903], "text": "..."}` 这种少冒号少方括号的坏格式，
整个数组于是不是合法 JSON。原来的实现直接判"解析失败"，**整张图的块全丢**
（6 张样例里 4 张中招，clean 集准确率从 100% 掉到 33%）。
现在改成整段解析失败就退化成"逐个抠 `text` 字段"，坏的那个自动跳过。
见 `tools/test_parser.py` 的回归测试（10 条断言，秒级跑完，不用 GPU）。

## 架构

```
HTTP 进程（uvicorn，1 个）          ← 只做协议解析和转发
  └── N 个 worker 进程               ← 每进程 1 张卡 + 1 份模型
        req_q[i] / resp_q[i]         ← 每个 worker 一对私有队列
```

请求进来 → 挑一个空闲 worker → 投进它的队列 → 从它的队列取结果。
worker 之间零通信，挂了不影响其它卡。

模型 1.12B，fp16 权重才 2.24GB，**单卡 64GB 装得下还剩 60GB**——所以用数据并行，
不上张量并行（要拉通信组，收益是负的）。

## 目录结构

```
├── ocr_core.py             共享核心：提示词/放大倍数/精度/采样/输出解析
├── batch_ocr_dev.py        开发机入口：本地 CUDA 单卡批量
├── serve_ocr_prod.py       生产入口：常驻 HTTP 服务 + N 卡负载均衡
├── test_service.py         服务自测（进程内 TestClient，不占端口）
├── tools/
│   ├── make_sample.py      生成合成样例图（仓库里不含任何真实图片）
│   ├── make_poster.py      生成 README 海报 / GitHub 社交预览图
│   ├── compare_runs.py     两次运行逐字符比对——准确度 A/B 工具
│   ├── score.py            用 ground truth 打准确率分
│   └── test_parser.py      输出解析器的单元测试（秒级，不用 GPU）
├── data/
│   ├── samples_clean/      合成样例（干净）
│   ├── samples_hard/       合成样例（小字+噪点）
│   ├── samples_noisy/      合成样例（最难）
│   └── expected/           黄金基准
└── docs/
    ├── 部署说明.md          昇腾/CANN 安装、参数表、验收清单、风险
    ├── 实验记录.md          全部实测数据
    ├── 准确度验收.md        怎么证明改动没让准确度变差
    └── poster.png          README 海报（由 tools/make_poster.py 生成）
```

**开发环境和生产环境共用 `ocr_core.py`**，只有入口文件不同。
这是刻意的：提示词或放大倍数在两边差一点点，开发机上的准确度实测就对生产机失效了。

## 文档

| 文档 | 内容 |
|---|---|
| [需求文档.md](需求文档.md) | 背景、目标、功能/非功能需求、接口契约、验收标准、风险 |
| [使用说明.md](使用说明.md) | 安装、运行、参数、API 示例、准确度验收流程、故障排查 |
| [docs/部署说明.md](docs/部署说明.md) | 昇腾多卡部署细节、版本矩阵、风险清单、踩坑记录 |
| [docs/实验记录.md](docs/实验记录.md) | 全部实测数字 |
| [docs/准确度验收.md](docs/准确度验收.md) | 换环境/换配置后怎么验证没变差 |

## 说明

- 仓库里的样例图是 `tools/make_sample.py` 程序画出来的，**不含任何真实数据**，
  也没有随仓库发布任何真实业务图片或识别结果。
- 公开基准是**合成**样例，能验证方法与相对关系，但**绝对准确度取决于你自己的
  数据**——务必自己测。
- 模型权重来自 [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)（腾讯，Apache-2.0），
  本仓库只包含推理封装、参数调优方法论和部署脚本，**不分发任何权重**。
- 昇腾多卡的生产部署只验证过逻辑（worker 池、负载均衡、HTTP 层），
  **NPU 侧的算子兼容性没有实测**，见 [docs/部署说明.md](docs/部署说明.md) 的风险清单。

## License

本仓库代码 MIT，见 [LICENSE](LICENSE)。
模型权重遵循腾讯 HunyuanOCR 原仓库的许可。
