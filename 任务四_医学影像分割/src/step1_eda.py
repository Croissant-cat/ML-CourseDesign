# -*- coding: utf-8 -*-
"""step1  数据探索与提交约定核对

在做什么：
    定位比赛数据、逐图统计尺寸与血管占比、把「视野内占比」折算成「整幅图占比」，
    并把每张图的视野掩膜存成 npy，后续步骤直接复用。
    同时读一遍官方 segmentation_to_csv.py 的源码，报告它的 RLE 扫描顺序与极性约定。
需要多久：
    约 1 分钟（40 张图的 Frangi 与视野检测）。
输出什么：
    outputs/fov/{key}.npy        每张图的视野掩膜
    outputs/stats.json           尺寸、占比、提交行数等统计数据
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
    ap.add_argument("--data", default=None, help="比赛数据根目录，默认自动查找 /kaggle/input")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--min-train", type=int, default=5, help="低于这个对数就中止")
    a = ap.parse_args()

    if a.data:
        L.DATA_ROOT = a.data
    os.makedirs(os.path.join(a.out, "fov"), exist_ok=True)

    root = L.find_root()
    pairs, tests, sub_ids, sub_path = L.discover(root)
    print("[step1] 数据根目录:", root)
    print("[step1] 训练对 %d 组 / 测试图 %d 张 / 提交行 %d"
          % (len(pairs), len(tests), len(sub_ids)))
    if len(pairs) < a.min_train:
        raise SystemExit("训练对过少，请用 --data 指定比赛数据根目录")

    cov, sizes = [], []
    for img, gt in pairs:
        X, fov = L.load_features(img)
        y = L.load_gt(gt, fov) & fov
        cov.append(float(y.sum()) / max(1, int(fov.sum())))
        sizes.append(tuple(int(v) for v in fov.shape))
        np.save(os.path.join(a.out, "fov", os.path.splitext(os.path.basename(img))[0] + ".npy"),
                fov.astype(np.uint8))

    fov_ratio = []
    for p in tests:
        X, fov = L.load_features(p)
        fov_ratio.append(float(fov.mean()))
        np.save(os.path.join(a.out, "fov", os.path.splitext(os.path.basename(p))[0] + ".npy"),
                fov.astype(np.uint8))

    mean_cov = float(np.mean(cov))
    ratio = float(np.mean(fov_ratio))
    print("[step1] 训练集血管占比（视野内）: %.2f%%" % (100 * mean_cov))
    print("[step1] 视野面积占整幅图:        %.3f" % ratio)
    print("[step1] 折算到整幅图的占比:      %.2f%%  <- 预测占比应与它同口径比较"
          % (100 * mean_cov * ratio))
    print("[step1] 图像尺寸: %s" % (sizes[0],))

    scripts = L.find_official_script(root)
    info = {}
    if scripts:
        src = io.open(scripts[0], encoding="utf-8", errors="ignore").read()
        info = {"script": str(scripts[0]), "rle_order": L.detect_order(src),
                "mask_is_white": ("== 255" in src or "!= 0" in src)}
        print("[step1] 官方脚本: %s" % scripts[0])
        if info["mask_is_white"]:
            print("[step1] RLE 顺序推断: %s；脚本按白色(255)区域取掩膜" % info["rle_order"])
        else:
            print("[step1] RLE 顺序推断: %s；极性未能从源码判断，以 step6 的覆盖率自检为准"
                  % info["rle_order"])
    else:
        print("[step1] 未找到官方脚本，step6 将使用内置编码器")

    stats = {"root": str(root), "n_train": len(pairs), "n_test": len(tests),
             "n_sub": len(sub_ids), "size": list(sizes[0]),
             "gt_cov_fov": mean_cov, "fov_ratio": ratio,
             "gt_cov_image": mean_cov * ratio, "official": info}
    with io.open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print("[step1] 已写出 %s" % os.path.join(a.out, "stats.json"))


if __name__ == "__main__":
    main()
