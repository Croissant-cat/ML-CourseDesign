# -*- coding: utf-8 -*-
"""Step 1: 数据探索与诊断（对应报告 1.2 节）。

用法：
    python src/step1_eda.py
运行后会打印训练/测试规模、背景亮度统计、图像尺寸分布，并保存尺寸直方图。
"""
import os
import glob
from collections import Counter

import numpy as np
from PIL import Image

TRAIN_DIR = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train/"
TEST_DIR = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/test/"
WORK_DIR = "/kaggle/working/plant_data/"


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    classes = sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))
    print("类别:", classes)

    counts = {c: len(glob.glob(os.path.join(TRAIN_DIR, c, "*.png"))) for c in classes}
    print("每类数量:", counts, "训练集总数:", sum(counts.values()))

    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    print("测试集数量:", len(test_files))

    # ---- 图像尺寸 / 通道 ----
    sizes = Counter()
    for c in classes:
        for f in glob.glob(os.path.join(TRAIN_DIR, c, "*.png")):
            with Image.open(f) as im:
                sizes[(im.size, im.mode)] += 1
    print("尺寸/通道分布条数:", len(sizes))

    sides = np.array([k[0][0] for k in sizes.elements()])
    print("图像边长分位 (5/25/50/75/95):",
          np.percentile(sides, [5, 25, 50, 75, 95]).round(0).tolist())

    # ---- 背景亮度统计：判断是否为暗色砂石背景 ----
    border_means, border_stds = [], []
    for c in classes:
        for f in glob.glob(os.path.join(TRAIN_DIR, c, "*.png")):
            with Image.open(f) as im:
                a = np.asarray(im.convert("RGB").resize((64, 64), Image.BILINEAR), dtype=np.float32)
            border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
            border_means.append(border.mean())
            border_stds.append(border.std())

    bm, bs = np.array(border_means), np.array(border_stds)
    print("四边亮度: 均值 %.2f | 中位 %.2f | 最小 %.2f" % (bm.mean(), np.median(bm), bm.min()))
    print("图内背景波动 std: 中位 %.2f | 90%% %.2f" % (np.median(bs), np.percentile(bs, 90)))
    print("暗底图片数(亮度<150): %d / %d" % ((bm < 150).sum(), len(bm)))

    # ---- 保存尺寸直方图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 3))
        plt.hist(sides, bins=30)
        plt.title("image side length")
        plt.xlabel("px")
        plt.ylabel("count")
        plt.tight_layout()
        out = os.path.join(WORK_DIR, "eda_side_hist.png")
        plt.savefig(out, dpi=80)
        print("已保存:", out)
    except Exception as exc:  # 无 matplotlib 时跳过
        print("跳过绘图:", exc)


if __name__ == "__main__":
    main()
