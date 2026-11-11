# -*- coding: utf-8 -*-
"""step2：512 单骨干 5 折基线，以及三个训练侧的变体（替参数即可复现负结果）。

在做什么
    默认跑第 1 版基线：resnet18、512 输入、5 折交叉验证、双通道热图（中心凹点 + 黄斑掩码），
    100 轮，OOF 与首个提交文件。三个开关用来复现被否定的变体：
      --size 768 --aug strong   第 3 版（输入 768、增强 ±25°/±18%/±7%）
      --full --n-models 5       第 4 版（全量训练、只留 5 个模型，公开榜 80.35）
      --crop fundus             第 7 版（先裁黑边再缩放，检验是否依赖图幅）
    网络、解码、数据管线的实现在 step5 里，这里直接复用，保证各版本同构可比。

需要多久
    512 与 5 折：T4 约 25 分钟；--size 768：约 41 分钟；--full：按模型个数每模型 4~6 分钟。

输出什么
    outputs/pred_step2.npy        测试集预测 (20, 2)，原图像素
    outputs/oof_step2.npy         训练集 OOF 预测 (80, 2)
    outputs/sub_step2.csv         加坐标修正后的提交候选
    屏幕输出 OOF 的 RMSE（点、区域、混合）与逐折验证误差。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step5_full_data_12models import (CORRECTION, Data, Hc, MEAN, STD, build_net,  # noqa: E402
                                      cells_to_px, decode_point, decode_region, gheat,
                                      imread_u, infer, read_boxes, soft_mask)

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def find_data():
    for c in (os.path.join(ROOT, "data", "detection"), os.path.join(ROOT, "data"),
              os.environ.get("FOVEA_DATA", "")):
        if c and os.path.exists(os.path.join(c, "fovea_localization_train_GT.csv")):
            return c
    for base in ("/kaggle/input", os.getcwd()):
        if os.path.isdir(base):
            for r, d, f in os.walk(base):
                if "fovea_localization_train_GT.csv" in f:
                    return r
    raise FileNotFoundError("找不到数据，请放到 data/detection/")


def fundus_crop(im):
    """裁掉四周黑边，只留眼底。"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    m = (g > max(8.0, float(g.max()) * 0.08)).astype(np.uint8)
    ys, xs = np.where(m > 0)
    return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def load_items(size, crop):
    # 热图边长与坐标反算都读 step5 的模块级常量 S / Hc，换尺寸时必须同步改，
    # 否则 768 输入配 512 的目标热图，训练目标与解码全部错位。
    import step5_full_data_12models as S5
    S5.S = size
    S5.Hc = size // 4
    DATA = find_data()
    gt = pd.read_csv(os.path.join(DATA, "fovea_localization_train_GT.csv"))
    gt.columns = [str(c).strip() for c in gt.columns]
    gt["name"] = gt["data"].astype(int).map(lambda i: "%04d" % i)
    gt = gt.rename(columns={"Fovea_X": "x", "Fovea_Y": "y"})
    gt = gt[~((gt["x"] <= 0) & (gt["y"] <= 0))].reset_index(drop=True)
    boxes = read_boxes()

    def prep(im, gx=None, gy=None):
        if crop == "fundus":
            x0, y0 = None, None
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            m = (g > max(8.0, float(g.max()) * 0.08)).astype(np.uint8)
            ys, xs = np.where(m > 0)
            x0, y0 = int(xs.min()), int(ys.min())
            box = (x0, y0, int(xs.max()) + 1, int(ys.max()) + 1)
            im = fundus_crop(im)
            H, W = im.shape[:2]
            if gx is not None:
                gx, gy = gx - box[0], gy - box[1]
        else:
            H, W = im.shape[:2]
            box = (0, 0, W, H)
        sm = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
        return sm, W, H, box, gx, gy

    train_items = []
    for nm, gx, gy in gt[["name", "x", "y"]].itertuples(index=False):
        p = os.path.join(DATA, "train", nm + ".jpg")
        if not os.path.exists(p):
            continue
        sm, W, H, box, lx, ly = prep(imread_u(p), gx, gy)
        mk = np.zeros((size // 4, size // 4), np.float32)
        if nm in boxes:
            b = boxes[nm]
            mk = soft_mask((b[0] - box[0], b[1] - box[1], b[2] - box[0], b[3] - box[1]), W, H)
        train_items.append(dict(name=nm, img=sm, gx=lx, gy=ly, W=W, H=H, mask=mk,
                                px=lx / W * size - 0.5, py=ly / H * size - 0.5))

    tes = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(DATA, "test"))
                 if f.lower().endswith(".jpg"))
    test_items = []
    for nm in tes:
        sm, W, H, box, _, _ = prep(imread_u(os.path.join(DATA, "test", nm + ".jpg")))
        test_items.append(dict(name=nm, img=sm, gx=0.0, gy=0.0, W=W, H=H,
                               mask=np.zeros((size // 4, size // 4), np.float32), px=0.0, py=0.0))
    return train_items, test_items, DATA


def run_model(tr, ev, args, seed, size, w_region):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dl = DataLoader(Data(tr, True, np.random.default_rng(seed)), batch_size=8, shuffle=True,
                    num_workers=0, drop_last=True)
    model = build_net(args.backbone).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                total_steps=max(1, len(dl)) * args.epochs, pct_start=0.25)
    for ep in range(args.epochs):
        model.train()
        for x, t, _ in dl:
            x, t = x.to(DEV), t.to(DEV)
            opt.zero_grad(set_to_none=True)
            o = torch.sigmoid(model(x))
            loss = F.mse_loss(o[:, 0], t[:, 0]) + F.mse_loss(o[:, 1], t[:, 1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
    h, m = infer(model, ev)
    del model
    torch.cuda.empty_cache()
    return h, m


def mix_points(h, m, items, w_region):
    """双通道热图 -> 原图像素坐标：区域质心与点峰值按 w_region 混合。"""
    pk, conf = decode_point(h)
    rg = decode_region(m, pk)
    px, py = cells_to_px(pk, items)
    rx, ry = cells_to_px(rg, items)
    return np.stack([w_region * rx + (1 - w_region) * px,
                     w_region * ry + (1 - w_region) * py], 1), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--aug", default="normal", choices=("normal", "strong"))
    ap.add_argument("--full", action="store_true", help="全量训练，不做交叉验证")
    ap.add_argument("--n-models", type=int, default=1, help="--full 时的模型数")
    ap.add_argument("--crop", default="none", choices=("none", "fundus"))
    ap.add_argument("--w-region", type=float, default=0.4)
    args = ap.parse_args()
    t0 = time.time()
    size = args.size
    print("size %d | epochs %d | folds %d | 骨干 %s | 增强 %s | crop %s | full %s"
          % (size, args.epochs, args.folds, args.backbone, args.aug, args.crop, args.full))

    if args.aug == "strong":
        # 增强幅度在 Data 内部读的是 step5 的默认参数，这里直接改函数默认值
        import step5_full_data_12models as S5
        _orig = S5.geo_aug
        S5.geo_aug = lambda img, pt, mask, rng: _orig(img, pt, mask, rng, rot=25.0, sc=0.18, tr=0.07)

    train_items, test_items, DATA = load_items(size, args.crop)
    X = np.array([[it["gx"], it["gy"]] for it in train_items])
    NG, NT = len(train_items), len(test_items)
    print("train %d | test %d | 数据载入 %.1fs" % (NG, NT, time.time() - t0))

    h_te = np.zeros((NT, size // 4, size // 4), np.float32)
    m_te = np.zeros((NT, size // 4, size // 4), np.float32)
    P_oof = np.full((NG, 2), np.nan)
    if args.full:
        for k in range(args.n_models):
            h, m = run_model(train_items, test_items, args, 42 + 1000 * k, size, args.w_region)
            h_te += h / args.n_models
            m_te += m / args.n_models
            print("  [%d/%d] 全量模型完成，累计 %.1f 分钟"
                  % (k + 1, args.n_models, (time.time() - t0) / 60), flush=True)
    else:
        rng = np.random.default_rng(42)
        perm = rng.permutation(NG)
        fold_of = np.zeros(NG, int)
        for i, j in enumerate(perm):
            fold_of[j] = i % args.folds
        folds_px = []
        for f in range(args.folds):
            tr = [train_items[i] for i in range(NG) if fold_of[i] != f]
            ev = [train_items[i] for i in range(NG) if fold_of[i] == f]
            h_ev, m_ev = run_model(tr, ev, args, 42 + 1000 * f, size, args.w_region)
            p_ev, _ = mix_points(h_ev, m_ev, ev, args.w_region)
            for j, it in enumerate(ev):
                P_oof[train_items.index(it)] = p_ev[j]
            h, m = run_model(tr, test_items, args, 42 + 1000 * f, size, args.w_region)
            h_te += h / args.folds
            m_te += m / args.folds
            e = float(np.sqrt(np.mean(np.sum((p_ev - np.array([[i2["gx"], i2["gy"]] for i2 in ev])) ** 2, 1))))
            folds_px.append(e)
            print("  折 %d 验证 RMSE %.2f 像素 | 累计 %.1f 分钟"
                  % (f + 1, e, (time.time() - t0) / 60), flush=True)
        ok = ~np.isnan(P_oof[:, 0])
        oof = float(np.sqrt(np.mean(np.sum((P_oof[ok] - X[ok]) ** 2, 1))))
        print("\nOOF RMSE（原图像素）：%.2f | 逐折：%s"
              % (oof, ", ".join("%.2f" % v for v in folds_px)))
        np.save(os.path.join(OUT, "oof_step2.npy"), P_oof)

    P_te, conf = mix_points(h_te, m_te, test_items, args.w_region)
    print("热图平均后解码，峰值置信度中位 %.3f" % float(np.median(conf)))
    np.save(os.path.join(OUT, "pred_step2.npy"), P_te)
    sam = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    idx = {int(it["name"]): j for j, it in enumerate(test_items)}
    vals = []
    for s in sam["ImageID"].astype(str):
        axis = 1 if s.strip().lower().endswith("y") else 0
        j = idx[int(s.split("_")[0])]
        vals.append(P_te[j, axis] + (CORRECTION["dy"] if axis else CORRECTION["dx"]))
    pd.DataFrame({"ImageID": sam["ImageID"].values, "value": np.round(vals, 4)}).to_csv(
        os.path.join(OUT, "sub_step2.csv"), index=False, lineterminator="\n")
    print("写出 outputs/pred_step2.npy 与 outputs/sub_step2.csv | 总耗时 %.1f 分钟"
          % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
