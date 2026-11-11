# -*- coding: utf-8 -*-
"""Step 2: ExG 超绿指数定位 + 包围盒裁切 + 5 折分层划分（对应报告 1.3.2 节）。

设计要点（与报告中一致）：
  1. ExG = 2G - R - B，植物为正、砂石/条码接近 0，是天然的判别量；
  2. 只用闭运算连接细叶，不做开运算（开运算会腐蚀掉细叶，是上一版锯齿/断裂的来源）；
  3. 去噪用「连通域面积筛选」代替形态学腐蚀；
  4. 只计算外接矩形，不对像素做任何分割，叶片边缘零损失。

用法：
    python src/step2_preprocess.py
输出：
    /kaggle/working/plant_data/crop256/train/<类别>/*.png
    /kaggle/working/plant_data/crop256/test/*.png
    /kaggle/working/plant_data/train_folds.csv
"""
import glob
import os
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from sklearn.model_selection import StratifiedKFold

TRAIN_DIR = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train/"
TEST_DIR = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/test/"
WORK_DIR = "/kaggle/working/plant_data/"
CROP_DIR = os.path.join(WORK_DIR, "crop256")
OUT_TRAIN = os.path.join(CROP_DIR, "train")
OUT_TEST = os.path.join(CROP_DIR, "test")

CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
S = 256            # 计算掩码用的分辨率
MARGIN = 0.12      # 包围盒四周留白比例


def plant_mask_exg(pil_img, exg_thresh=20):
    """返回保留后的植物掩码（S x S 布尔图）；定位失败返回 None。"""
    small = np.asarray(pil_img.resize((S, S), Image.BILINEAR), dtype=np.float32)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    exg = 2.0 * g - r - b

    mask = exg > exg_thresh
    if mask.sum() < 80:                      # 极暗/极小目标兜底
        mask = exg > max(5.0, exg_thresh / 4.0)

    # 只闭运算：先膨胀再腐蚀，把断开的细叶连起来，但不会抹掉细叶本身
    mask = ndi.binary_closing(mask, np.ones((7, 7)))

    lab, n = ndi.label(mask)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    # 只保留面积 >= 最大连通域 15% 的块：用面积筛选代替腐蚀，既去噪又保住细叶
    keep = np.where(sizes >= 0.15 * sizes.max())[0] + 1
    return np.isin(lab, keep)


def crop_by_mask(pil_img, mask, size=256, margin=MARGIN):
    """按掩码外接框裁切并补成正方形；不修改任何像素内容。"""
    w, h = pil_img.size
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return pil_img.resize((size, size), Image.BILINEAR)

    sy, sx = h / S, w / S
    y0, y1 = ys.min() * sy, ys.max() * sy
    x0, x1 = xs.min() * sx, xs.max() * sx
    mh, mw = (y1 - y0) * margin, (x1 - x0) * margin
    y0, y1 = int(max(0, y0 - mh)), int(min(h - 1, y1 + mh))
    x0, x1 = int(max(0, x0 - mw)), int(min(w - 1, x1 + mw))

    crop = pil_img.crop((x0, y0, x1 + 1, y1 + 1))
    # 用该图自身的背景色（四角中值）补方，避免出现突兀色块
    a = np.asarray(pil_img.resize((64, 64))).reshape(-1, 3)
    bg = tuple(int(v) for v in np.median(np.concatenate([a[:16], a[-16:]]), axis=0))
    crop = ImageOps.pad(crop, (max(crop.size),) * 2, color=bg, method=Image.BILINEAR)
    return crop.resize((size, size), Image.BILINEAR)


def make_crop(path, size=256):
    with Image.open(path) as im:
        img = im.convert("RGB")
        m = plant_mask_exg(img)
        return img.resize((size, size), Image.BILINEAR) if m is None else crop_by_mask(img, m, size)


def main():
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_TEST, exist_ok=True)

    # 先统计各类的植物掩码面积占比（报告表 1-3 的数据来源）
    print("ExG 掩码面积占比中位数（反映目标相对尺度）:")
    for cls in CLASSES:
        ratios = []
        for f in sorted(glob.glob(os.path.join(TRAIN_DIR, cls, "*.png"))):
            with Image.open(f) as im:
                m = plant_mask_exg(im.convert("RGB"))
            ratios.append(0.0 if m is None else float(m.mean()))
        ratios = np.array(ratios)
        print("  %-18s 中位 %.3f | 失败 %d 张" % (cls, np.median(ratios), int((ratios == 0).sum())))

    t0 = time.time()
    records = []
    for label, cls in enumerate(CLASSES):          # 标签映射固定为 0..4
        out_dir = os.path.join(OUT_TRAIN, cls)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(TRAIN_DIR, cls, "*.png")))
        for f in files:
            name = os.path.basename(f)
            dst = os.path.join(out_dir, name)
            make_crop(f, 256).save(dst)
            records.append(dict(path=dst, name=name, cls=cls, label=label))
        print("  %-18s 完成 %d 张" % (cls, len(files)))

    df = pd.DataFrame(records)
    print("训练缓存: %d 张, 耗时 %.1f 秒" % (len(df), time.time() - t0))

    for f in sorted(glob.glob(os.path.join(TEST_DIR, "*.png"))):
        make_crop(f, 256).save(os.path.join(OUT_TEST, os.path.basename(f)))
    print("测试缓存: %d 张" % len(glob.glob(os.path.join(OUT_TEST, "*.png"))))

    # 分层 5 折：每折每类 20 张，类别分布完全一致
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1
    for k, (_, va_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[df.index[va_idx], "fold"] = k
    df.to_csv(os.path.join(WORK_DIR, "train_folds.csv"), index=False)
    print(pd.crosstab(df["fold"], df["cls"]))


if __name__ == "__main__":
    main()
