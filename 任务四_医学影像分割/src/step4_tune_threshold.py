# -*- coding: utf-8 -*-
"""step4  在交叉验证概率上搜索阈值与最小连通块

在做什么：
    读 step2 保存的交叉验证概率、真值与视野掩膜，在「阈值 × 最小连通块」的网格上
    计算平均 Dice，选出最优组合。这一步不碰测试集，只用训练图的交叉验证结果。
需要多久：
    约 1 分钟（15 个阈值 × 3 个连通块，45 组二值化）。
输出什么：
    outputs/tune.json   {"thr": ..., "min_size": ..., "oof_dice": ...}
"""
import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seglib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--min-sizes", default="0,20,50")
    a = ap.parse_args()

    oof = {}
    for f in sorted(os.listdir(a.out)):
        if f.startswith("oof_") and f.endswith(".npy"):
            oof[f[4:-4]] = np.load(os.path.join(a.out, f)).astype(np.float32)
    if not oof:
        raise SystemExit("没有找到 oof_*.npy，请先跑 step2")

    gt, fov = {}, {}
    for nm in oof:
        p = os.path.join(a.out, "train_gt", nm + ".npy")
        q = os.path.join(a.out, "fov_train", nm + ".npy")
        if os.path.exists(p) and os.path.exists(q):
            gt[nm] = np.load(p).astype(bool)
            fov[nm] = np.load(q).astype(bool)
    if not gt:
        raise SystemExit("缺少 train_gt/ 或 fov_train/，请先跑 step2")

    best = (-1.0, 0.5, 0)
    rows = []
    for ms in [int(x) for x in a.min_sizes.split(",")]:
        for t in np.arange(0.15, 0.86, 0.05):
            ds = [L.dice_score(L.binarize(oof[n], fov[n], float(t), ms), gt[n]) for n in gt]
            d = float(np.mean(ds))
            rows.append((d, float(t), ms))
            if d > best[0]:
                best = (d, float(t), ms)
    rows.sort(reverse=True)
    print("[step4] 前六名：")
    for d, t, ms in rows[:6]:
        print("    thr=%.2f  min_conn_px=%2d  Dice=%.4f" % (t, ms, d))

    res = {"thr": best[1], "min_size": best[2], "oof_dice": best[0], "top6": rows[:6]}
    with io.open(os.path.join(a.out, "tune.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[step4] 最优 thr=%.2f  min_conn_px=%d  OOF Dice=%.4f"
          % (best[1], best[2], best[0]))


if __name__ == "__main__":
    main()
