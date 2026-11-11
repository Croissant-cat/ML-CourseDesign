# -*- coding: utf-8 -*-
"""step5  多版本概率等权融合，并在融合后的交叉验证概率上重扫阈值

在做什么：
    把多次训练各自产出的 outputs/test/*.npy 与 oof_*.npy 做等权平均，
    得到融合后的测试概率与交叉验证概率，再在融合后的交叉验证概率上重搜阈值。
    融合后概率分布变窄，最优阈值通常比单模型低，这一步要重新校准。
需要多久：
    约 3 分钟（只做数组平均与网格搜索）。
输出什么：
    outputs/merged/test/{sid}.npy    融合后的测试概率
    outputs/merged/tune.json         融合后的最优阈值
用法：
    python step5_ensemble.py --members runA runB runC
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seglib as L


def load_dir(d, kind):
    out = {}
    for f in glob.glob(os.path.join(d, kind, "*.npy")):
        sid = os.path.splitext(os.path.basename(f))[0]
        out[sid] = np.load(f).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="各次训练的输出目录")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--min-sizes", default="0,20,50")
    a = ap.parse_args()

    dst = os.path.join(a.out, "merged", "test")
    os.makedirs(dst, exist_ok=True)

    acc, cnt = {}, {}
    for d in a.members:
        for sid, pr in load_dir(d, "test").items():
            acc[sid] = pr if sid not in acc else acc[sid] + pr
            cnt[sid] = cnt.get(sid, 0) + 1
    if not acc:
        raise SystemExit("成员目录里没有 test/*.npy")
    for sid in acc:
        np.save(os.path.join(dst, "%s.npy" % sid), (acc[sid] / cnt[sid]).astype(np.float16))
    print("[step5] 融合 %d 个成员目录，%d 张测试图，每张参与 %s 份"
          % (len(a.members), len(acc), sorted(set(cnt.values()))))

    oof_acc, oof_cnt = {}, {}
    for d in a.members:
        for f in glob.glob(os.path.join(d, "oof_*.npy")):
            nm = os.path.basename(f)[4:-4]
            pr = np.load(f).astype(np.float32)
            oof_acc[nm] = pr if nm not in oof_acc else oof_acc[nm] + pr
            oof_cnt[nm] = oof_cnt.get(nm, 0) + 1
    if not oof_acc:
        print("[step5] 成员目录里没有 oof_*.npy，跳过融合后阈值搜索")
        return

    base = a.members[0]
    best = (-1.0, 0.5, 0)
    rows = []
    for nm in list(oof_acc):
        p = os.path.join(base, "train_gt", nm + ".npy")
        q = os.path.join(base, "fov_train", nm + ".npy")
        if not (os.path.exists(p) and os.path.exists(q)):
            continue
        oof_acc[nm] = oof_acc[nm] / oof_cnt[nm]
        oof_acc["%s_gt" % nm] = np.load(p).astype(bool)
        oof_acc["%s_fov" % nm] = np.load(q).astype(bool)
    names = [n for n in list(oof_acc) if not n.endswith("_gt") and not n.endswith("_fov")]
    for ms in [int(x) for x in a.min_sizes.split(",")]:
        for t in np.arange(0.15, 0.86, 0.05):
            ds = [L.dice_score(L.binarize(oof_acc[n], oof_acc[n + "_fov"], float(t), ms),
                               oof_acc[n + "_gt"]) for n in names]
            d = float(np.mean(ds))
            rows.append((d, float(t), ms))
            if d > best[0]:
                best = (d, float(t), ms)
    rows.sort(reverse=True)
    print("[step5] 融合后前六名：")
    for d, t, ms in rows[:6]:
        print("    thr=%.2f  min_conn_px=%2d  Dice=%.4f" % (t, ms, d))
    with io.open(os.path.join(a.out, "merged", "tune.json"), "w", encoding="utf-8") as f:
        json.dump({"thr": best[1], "min_size": best[2], "oof_dice": best[0], "top6": rows[:6]},
                  f, ensure_ascii=False, indent=1)
    print("[step5] 融合后最优 thr=%.2f  min_conn_px=%d  OOF Dice=%.4f"
          % (best[1], best[2], best[0]))


if __name__ == "__main__":
    main()
