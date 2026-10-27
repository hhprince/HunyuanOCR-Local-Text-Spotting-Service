"""parse_words 的单元测试（不加载模型，秒级跑完）。

    python tools/test_parser.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_core import parse_words  # noqa: E402

ok = True


def check(name, got, want):
    global ok
    good = got == want
    ok = ok and good
    print("  %-52s %s" % (name, "PASS" if good else "FAIL"))
    if not good:
        print("     got : %r" % (got,))
        print("     want: %r" % (want,))


# 正常输出
check("正常 JSON 数组", parse_words(
    '[{"box": [1, 2, 3, 4], "text": "甲"}, {"box": [5, 6, 7, 8], "text": "乙"}]'),
    [{"words": "甲"}, {"words": "乙"}])

# 模型偶尔把 '"box": [' 吐成 '"box'（少了冒号和左方括号）——这是实测遇到的
# 真实坏格式。修好后整张图的块必须一个都不能少。
check("坏 box 字段被修复", parse_words(
    '[{"box": [1, 2, 3, 4], "text": "甲"}, {"box755, 880, 818, 903], "text": "乙"}]'),
    [{"words": "甲"}, {"words": "乙"}])

# 坏元素在最前面
check("坏元素在开头", parse_words(
    '[{"box9, 1, 2, 3], "text": "甲"}, {"box": [1, 2, 3, 4], "text": "乙"}]'),
    [{"words": "甲"}, {"words": "乙"}])

# 完全无法救的元素被跳过，其余保留
check("无法修复的元素被跳过", parse_words(
    '[{"box": [1, 2, 3, 4], "text": "甲"}, {完全不是JSON}, {"box": [1, 2, 3, 4], "text": "乙"}]'),
    [{"words": "甲"}, {"words": "乙"}])

# 文本里带花括号/方括号，不能被拆错
check("文本里含花括号", parse_words(
    '[{"box": [1, 2, 3, 4], "text": "价格 {备注} [x]"}]'),
    [{"words": "价格 {备注} [x]"}])

# 文本里含转义引号
check("文本里含转义引号", parse_words(
    '[{"box": [1, 2, 3, 4], "text": "他说\\"你好\\""}]'),
    [{"words": '他说"你好"'}])

# markdown 代码块包裹
check("包在 ```json 里", parse_words(
    '```json\n[{"box": [1, 2, 3, 4], "text": "甲"}]\n```'),
    [{"words": "甲"}])

# 完全没有 JSON
check("不是 JSON 返回 None", parse_words("这是一段普通文字"), None)
check("空字符串返回 None", parse_words(""), None)

# 没有 text 字段的元素被丢掉
check("没有 text 的元素被丢掉", parse_words(
    '[{"box": [1, 2, 3, 4]}, {"box": [1, 2, 3, 4], "text": "甲"}]'),
    [{"words": "甲"}])

print("\n%s" % ("全部通过" if ok else "有失败项"))
sys.exit(0 if ok else 1)
