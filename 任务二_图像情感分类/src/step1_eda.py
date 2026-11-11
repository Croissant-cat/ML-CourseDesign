# ============================================================
# 任务二：图像情感分类
# 本文件在做什么：数据探索：统计训练集与测试集的规模、类别分布，核对图像规格与提交格式。
# 需要多久：约 1 分钟。
# 输出什么：控制台打印类别计数、图像尺寸与通道、提交模板的列名与匹配情况。
# ============================================================

# ============================================================
# 第 1 步（精简版）：只打印最关键的十几行信息
# 只读，几秒钟跑完
# ============================================================
import os, glob
import pandas as pd
from PIL import Image

TR = sorted(d for d in glob.glob("/kaggle/input/**/train", recursive=True) if os.path.isdir(d))[0]
TE = sorted(d for d in glob.glob("/kaggle/input/**/test", recursive=True) if os.path.isdir(d))[0]
SUB = sorted(glob.glob("/kaggle/input/**/submission.csv", recursive=True))
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

cls = sorted(d for d in os.listdir(TR) if os.path.isdir(os.path.join(TR, d)))
cnt = {c: len([f for f in os.listdir(os.path.join(TR, c)) if f.lower().endswith(EXTS)]) for c in cls}
print("[1] train 路径:", TR)
print("[2] test  路径:", TE)
print("[3] 类别数量:", cnt, "| 合计:", sum(cnt.values()))

te = [f for f in os.listdir(TE) if f.lower().endswith(EXTS)]
print("[4] test 张数:", len(te), "| 文件名示例:", te[:2])

if SUB:
    df = pd.read_csv(SUB)
    print("[5] submission 路径:", SUB)
    print("[6] submission 形状/列名:", df.shape, list(df.columns))
    print("[7] submission 前 3 行:")
    print(df.head(3).to_string())
    ids = set(df.iloc[:, 0].astype(str))
    names = set(os.path.splitext(f)[0] for f in te)
    print("[8] 第1列 vs test文件名(去后缀) 交集:", len(ids & names), "/", len(names))
    if df.shape[1] > 1:
        print("[9] 第2列取值示例:", df.iloc[:, 1].unique()[:10])
else:
    print("[5] 没找到 submission.csv")

im = Image.open(os.path.join(TR, cls[0], sorted(os.listdir(os.path.join(TR, cls[0])))[0]))
print("[10] train 图片: size=%s mode=%s" % (im.size, im.mode))
im = Image.open(os.path.join(TE, te[0]))
print("[11] test  图片: size=%s mode=%s" % (im.size, im.mode))
print("[12] 类别列表:", cls)
