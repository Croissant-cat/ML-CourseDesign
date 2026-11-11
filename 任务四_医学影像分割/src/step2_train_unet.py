# -*- coding: utf-8 -*-
"""step2  六折训练残差 U-Net

在做什么：
    读入 20 张训练图与掩膜，做四通道预处理（绿通道 CLAHE、红蓝通道、Frangi 响应），
    按序号取模分成 K 折，逐折训练残差 U-Net，保存每折权重与该折验证图的概率图。
    混合精度在开训前自动探测（fp16 → bf16 → fp32），总时长按 --minutes 自动分配轮数。
需要多久：
    约 130 分钟（T4，6 折，每折 130 轮 × 80 迭代，批大小 8，图块 288）。
输出什么：
    outputs/fold{k}.pth        K 个折模型权重
    outputs/oof_{name}.npy     训练图的交叉验证概率
    outputs/fov_train/{name}.npy  训练图视野掩膜
    outputs/train_gt/{name}.npy   训练图真值（step4 调阈值用）
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seglib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=130)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--patch", type=int, default=288)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--minutes", type=float, default=170.0, help="训练总时间预算，0 表示不限")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.data:
        L.DATA_ROOT = a.data
    for d in ("fov_train", "train_gt"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    L.PATCH = a.patch
    L.BASE_CH = a.base
    L.N_FOLDS = a.folds

    t0 = time.time()
    root = L.find_root()
    pairs, tests, sub_ids, _ = L.discover(root)
    items = []
    for img, gt in pairs:
        X, fov = L.load_features(img)
        y = L.load_gt(gt, fov) & fov
        it = L.Item(img, X, fov, y)
        items.append(it)
        np.save(os.path.join(a.out, "fov_train", it.name + ".npy"), fov.astype(np.uint8))
        np.save(os.path.join(a.out, "train_gt", it.name + ".npy"), y.astype(np.uint8))
    print("[step2] 训练图 %d 张，预处理完成，用时 %.0fs" % (len(items), time.time() - t0))

    L.set_seed(a.seed)
    L.pick_amp(items)

    folds = max(2, min(a.folds, len(items)))
    epochs = a.epochs
    if a.minutes and a.minutes > 0:
        sec = L.probe_speed(items, n=5)
        reserve = 420.0 + 24.0 * max(1, len(tests))
        avail = a.minutes * 60.0 - reserve - (time.time() - t0)
        steps = avail / folds / max(sec, 1e-3)
        epochs = int(np.clip(steps / a.iters, 12, a.epochs))
        print("[step2] 实测 %.0f ms/步 → %d 折 × %d 轮 × %d 迭代，预计 %.0f 分钟"
              % (sec * 1000, folds, epochs, a.iters, sec * epochs * a.iters * folds / 60))

    idx = list(range(len(items)))
    oof = {}
    t_train = time.time()
    for f in range(folds):
        va = [i for i in idx if i % folds == f]
        tr = [i for i in idx if i % folds != f]
        if not va or not tr:
            continue
        print("\n===== fold %d: train %d / val %d =====" % (f, len(tr), len(va)))
        model, probs = L.train_fold(f, [items[i] for i in tr], [items[i] for i in va],
                                    a.seed + f, epochs=epochs, iters=a.iters)
        torch.save(model.state_dict(), os.path.join(a.out, "fold%d.pth" % f))
        for k, v in probs.items():
            oof[k] = v
            np.save(os.path.join(a.out, "oof_%s.npy" % k), v.astype(np.float16))
        if a.minutes and a.minutes > 0 and (folds - f - 1) > 0:
            real = (time.time() - t_train) / max(1, (f + 1) * epochs * a.iters)
            remain = a.minutes * 60.0 - (time.time() - t0) - reserve
            ep_new = int(np.clip(remain / (folds - f - 1) / max(real, 1e-3) / a.iters, 12, a.epochs))
            if ep_new != epochs:
                print("[step2] 实测 %.0f ms/步 → 后面 %d 折改为 %d 轮"
                      % (real * 1000, folds - f - 1, ep_new))
                epochs = ep_new

    ds = [L.dice_score((oof[it.name] >= 0.5) & it.fov, it.y) for it in items if it.name in oof]
    print("\n[step2] 交叉验证 Dice(阈值 0.5): %.4f" % float(np.mean(ds)))
    print("[step2] 总用时 %.1f 分钟，权重与概率图已写入 %s"
          % ((time.time() - t0) / 60, os.path.abspath(a.out)))


if __name__ == "__main__":
    main()
