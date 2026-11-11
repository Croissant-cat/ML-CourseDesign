# -*- coding: utf-8 -*-
"""step3  测试时增广推理

在做什么：
    载入 step2 的 K 个折模型，对 20 张测试图各做 3 个尺度 × 8 个方向（4 翻转 × 2 姿态）
    的前向，概率取平均，得到每张测试图的血管概率。
需要多久：
    约 5 分钟（20 张图 × K 个模型 × 24 次前向）。
输出什么：
    outputs/test/{sid}.npy      测试图概率（与提交 Id 对应）
    outputs/test_fov/{sid}.npy  测试图视野掩膜
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seglib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--patch", type=int, default=288)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--scales", default="1.0,1.25,1.5")
    a = ap.parse_args()

    if a.data:
        L.DATA_ROOT = a.data
    L.PATCH = a.patch
    L.BASE_CH = a.base
    L.TTA_SCALES = tuple(float(x) for x in a.scales.split(","))

    os.makedirs(os.path.join(a.out, "test"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "test_fov"), exist_ok=True)

    root = L.find_root()
    _, tests, sub_ids, _ = L.discover(root)
    te_items = [L.Item(p, *L.load_features(p)) for p in tests]
    id2item = L.map_ids(te_items, sub_ids)

    wfiles = sorted(glob.glob(os.path.join(a.out, "fold*.pth")))
    if not wfiles:
        raise SystemExit("没有找到 fold*.pth，请先跑 step2")
    models = []
    for wf in wfiles:
        m = L.build_model()
        m.load_state_dict(torch.load(wf, map_location=L.DEV))
        m.eval()
        models.append(m)
    print("[step3] 载入 %d 个折模型，TTA %d 尺度 × 8 方向"
          % (len(models), len(L.TTA_SCALES)))

    for sid in sub_ids:
        it = id2item.get(sid)
        if it is None:
            continue
        pr = L.predict_probs(models, it.X, it.fov, L.TTA_SCALES, tta=True)
        np.save(os.path.join(a.out, "test", "%s.npy" % sid), pr.astype(np.float16))
        np.save(os.path.join(a.out, "test_fov", "%s.npy" % sid), it.fov.astype(np.uint8))
    print("[step3] 已写出 %d 张测试图概率到 %s" % (len(sub_ids), os.path.join(a.out, "test")))


if __name__ == "__main__":
    main()
