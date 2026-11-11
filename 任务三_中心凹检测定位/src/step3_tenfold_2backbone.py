# -*- coding: utf-8 -*-
"""step3：10 折 × 双骨干共 20 个模型，热图平均，得到第 2 版的预测与 OOF（即锚点）。

在做什么
    与 step2 同一套网络与数据管线，改成 10 折、每折训练 resnet18 与 resnet34 各一个，
    共 20 个模型。测试集的预测由 20 个模型的双通道热图直接平均后再解码；训练集用留出折
    的预测拼成 OOF。这一版的预测就是第 5 版融合时用的锚点。

需要多久
    Kaggle 单卡 T4 约 92 分钟（20 个模型 × 约 4.6 分钟）。

输出什么
    outputs/heat_v2.npz        20 模型平均后的双通道热图
    outputs/pred_v2.npy        第 2 版测试预测 (20, 2)，原图像素（= 锚点）
    outputs/anchor_v2.npy      与上一行相同，step5 优先读这个文件
    outputs/oof_v2.npy         训练集 OOF (80, 2)
    outputs/sub_v2.csv         加坐标修正后的提交候选
    屏幕输出点/区域/混合三种解码的 OOF 与权重扫描。
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step2_baseline_5fold as S2   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--backbones", default="resnet18,resnet34")
    ap.add_argument("--w-region", type=float, default=0.4)
    args = ap.parse_args()
    bbs = [b.strip() for b in args.backbones.split(",") if b.strip()]
    t0 = time.time()

    train_items, test_items, DATA = S2.load_items(args.size, "none")
    NG, NT = len(train_items), len(test_items)
    X = np.array([[it["gx"], it["gy"]] for it in train_items])
    print("size %d | 折数 %d | 骨干 %s | 模型总数 %d | train %d test %d"
          % (args.size, args.folds, "+".join(bbs), args.folds * len(bbs), NG, NT))

    rng = np.random.default_rng(42)
    perm = rng.permutation(NG)
    fold_of = np.zeros(NG, int)
    for i, j in enumerate(perm):
        fold_of[j] = i % args.folds

    n_tot = args.folds * len(bbs)
    h_te = np.zeros((NT, args.size // 4, args.size // 4), np.float32)
    m_te = np.zeros((NT, args.size // 4, args.size // 4), np.float32)
    P_oof = np.full((NG, 2), np.nan)
    done = 0

    for bi, bb in enumerate(bbs):
        for f in range(args.folds):
            tr = [train_items[i] for i in range(NG) if fold_of[i] != f]
            ev = [train_items[i] for i in range(NG) if fold_of[i] == f]
            seed = 1000 * bi + f
            args.backbone = bb
            h_ev, m_ev = S2.run_model(tr, ev, args, seed, args.size, args.w_region)
            p_ev, _ = S2.mix_points(h_ev, m_ev, ev, args.w_region)
            for j, it in enumerate(ev):
                P_oof[train_items.index(it)] = p_ev[j]
            h, m = S2.run_model(tr, test_items, args, seed, args.size, args.w_region)
            h_te += h / n_tot
            m_te += m / n_tot
            done += 1
            err = float(np.sqrt(np.mean(np.sum((p_ev - np.array([[i2["gx"], i2["gy"]] for i2 in ev])) ** 2, 1))))
            print("  [%2d/%d] %-8s 折%2d 验证 RMSE %6.2f | 累计 %.1f 分钟"
                  % (done, n_tot, bb, f + 1, err, (time.time() - t0) / 60), flush=True)

    np.savez_compressed(os.path.join(OUT, "heat_v2.npz"), heat=h_te, mask=m_te)
    np.save(os.path.join(OUT, "oof_v2.npy"), P_oof)

    ok = ~np.isnan(P_oof[:, 0])
    r = float(np.sqrt(np.mean(np.sum((P_oof[ok] - X[ok]) ** 2, 1))))
    print("OOF RMSE（w_region=%.1f）：%.2f 像素" % (args.w_region, r))
    # 权重扫描要按留出折的热图重算，见报告表 3-11，这里的 OOF 只对应上面这一个权重

    P_te, conf = S2.mix_points(h_te, m_te, test_items, args.w_region)
    np.save(os.path.join(OUT, "pred_v2.npy"), P_te)
    np.save(os.path.join(OUT, "anchor_v2.npy"), P_te)
    print("写出 pred_v2.npy / anchor_v2.npy，峰值置信度中位 %.3f" % float(np.median(conf)))

    sam = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    idx = {int(it["name"]): j for j, it in enumerate(test_items)}
    vals = []
    for s in sam["ImageID"].astype(str):
        axis = 1 if s.strip().lower().endswith("y") else 0
        j = idx[int(s.split("_")[0])]
        vals.append(P_te[j, axis] + (S2.CORRECTION["dy"] if axis else S2.CORRECTION["dx"]))
    pd.DataFrame({"ImageID": sam["ImageID"].values, "value": np.round(vals, 4)}).to_csv(
        os.path.join(OUT, "sub_v2.csv"), index=False, lineterminator="\n")
    print("写出 outputs/sub_v2.csv（含坐标修正）| 总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
