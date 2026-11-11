# -*- coding: utf-8 -*-
"""Step 6: 专题实验（对应报告 1.7.4 与 1.7.5 节）。

包含两个可独立运行的诊断：
  (A) --detail   细节量诊断：把样本按「植物包围盒在原图分辨率下的边长」分组，
                 统计各组 OOF 正确率，检验「分辨率是否为瓶颈」。
  (B) --spec     两细叶类（Black-grass vs Loose Silky-bent）专家模型 5 折训练，
                 并在主模型 OOF 上搜索融合权重 w。

用法：
    python src/step6_specialist_diag.py --detail
    python src/step6_specialist_diag.py --spec

结论（实测）：
  (A) 正确率与细节量无单调关系（0.920/0.911/0.885/0.927）-> 分辨率不是瓶颈；
  (B) 专家模型二分类准确率均值仅 0.7650，低于主模型在同一对类别上的 0.810，
      权重搜索给出 w=0（即完全采用主模型），该方案被弃用。
"""
import argparse
import gc
import glob
import math
import os
import random

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

WORK_DIR = "/kaggle/working/plant_data/"
MODEL_DIR = os.path.join(WORK_DIR, "models")
TRAIN_DIR = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train/"
CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
PAIR = [0, 2]                 # 0 = Black-grass, 2 = Loose Silky-bent
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE, BS, EPOCHS = 256, 16, 40
DEVICE = torch.device("cuda")
S = 256


# ---------------------------------------------------------------- 公共部分
class PathDS(Dataset):
    def __init__(self, paths, ys, tf):
        self.paths, self.ys, self.tf = paths, ys, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.ys[i]


train_tf = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(30)], p=0.5),
    T.ColorJitter(0.25, 0.25, 0.25, 0.05),
    T.ToTensor(), T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
val_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------- (A) 细节量诊断
def bbox_side_orig(path):
    """返回原图分辨率下，ExG 包围盒（含 12% 边距）的最长边像素数。"""
    with Image.open(path) as im:
        img = im.convert("RGB")
        w, h = img.size
        a = np.asarray(img.resize((S, S), Image.BILINEAR), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2 * g - r - b
    mask = exg > 20
    if mask.sum() < 80:
        mask = exg > 5
    mask = ndi.binary_closing(mask, np.ones((7, 7)))
    lab, n = ndi.label(mask)
    if n == 0:
        return float(w)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.where(sizes >= 0.15 * sizes.max())[0] + 1
    ys, xs = np.where(np.isin(lab, keep))
    sy, sx = h / S, w / S
    side_h = (ys.max() - ys.min() + 1) * sy * 1.24
    side_w = (xs.max() - xs.min() + 1) * sx * 1.24
    return float(max(side_h, side_w))


def run_detail():
    df = pd.read_csv(os.path.join(WORK_DIR, "train_folds.csv"))
    df["orig"] = [os.path.join(TRAIN_DIR, r["cls"], r["name"]) for _, r in df.iterrows()]
    df["side"] = [bbox_side_orig(p) for p in df["orig"]]

    prob = np.load(os.path.join(WORK_DIR, "oof_prob.npy"))
    label = np.load(os.path.join(WORK_DIR, "oof_label.npy"))
    df["correct"] = (prob.argmax(1) == label).astype(int)

    print("包围盒边长分位 (5/25/50/75/95):",
          np.percentile(df["side"], [5, 25, 50, 75, 95]).round(0).tolist())
    print(">256px: %.1f%% | >512px: %.1f%% | <128px: %.1f%%"
          % ((df["side"] > 256).mean() * 100, (df["side"] > 512).mean() * 100,
             (df["side"] < 128).mean() * 100))

    df["bin"] = pd.cut(df["side"], bins=[0, 128, 256, 512, 1e6],
                       labels=["<128", "128-256", "256-512", ">512"])
    print("\n全体 5 类：不同细节量区间的 OOF 正确率")
    print(df.groupby("bin", observed=True)["correct"].agg(["mean", "count"]))

    pair = df[df["label"].isin(PAIR)]
    print("\n仅两个细叶类：不同细节量区间的 OOF 正确率")
    print(pair.groupby("bin", observed=True)["correct"].agg(["mean", "count"]))


# ---------------------------------------------------------------- (B) 专家模型
def tta_bin(model, paths, bs=32):
    q = np.zeros(len(paths), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            views = []
            for p in batch:
                im = Image.open(p).convert("RGB")
                big = im.resize((320, 320), Image.BILINEAR)
                views.append([im, im.transpose(Image.FLIP_LEFT_RIGHT),
                              im.transpose(Image.FLIP_TOP_BOTTOM),
                              TF.center_crop(big.rotate(10.0, resample=Image.BILINEAR), SIZE),
                              TF.center_crop(big.rotate(-10.0, resample=Image.BILINEAR), SIZE)])
            for vi in range(5):
                x = torch.stack([val_tf(v[vi]) for v in views]).to(DEVICE)
                with torch.amp.autocast("cuda"):
                    o = model(x)
                q[i:i + len(batch)] += torch.softmax(o.float(), 1)[:, 1].cpu().numpy() / 5.0
    return q


def train_bin_fold(fold, df):
    set_seed(100 + fold)
    pair = df[df["label"].isin(PAIR)].copy()
    pair["y"] = (pair["label"] == 2).astype(int)          # 1 = Loose Silky-bent
    tr = pair[pair["fold"] != fold].reset_index(drop=True)
    va = pair[pair["fold"] == fold].reset_index(drop=True)

    tr_loader = DataLoader(PathDS(tr["path"].tolist(), tr["y"].tolist(), train_tf),
                           batch_size=BS, shuffle=True, num_workers=2,
                           pin_memory=True, drop_last=True)

    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=2,
                              drop_rate=0.2, drop_path_rate=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    steps = len(tr_loader)
    total, warm = EPOCHS * steps, 2 * steps
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")

    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if not (scaler.get_scale() < before):
                sch.step()
        qv = tta_bin(model, va["path"].tolist())
        acc = float(((qv > 0.5).astype(int) == va["y"].values).mean())
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    qv = tta_bin(model, va["path"].tolist())
    torch.save(best_state, os.path.join(MODEL_DIR, "spec_fold%d.pth" % fold))
    print("  spec fold %d | 二分类 val acc %.4f" % (fold, best))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return va.index.to_numpy(), qv, best


def fuse(p, q, w):
    """只重分配 Black-grass 与 Loose Silky-bent 之间的概率质量，其余三类保持不变。"""
    out = p.copy()
    mass = out[:, 0] + out[:, 2]
    m = mass > 1e-8
    ratio = np.zeros(len(out))
    ratio[m] = out[m, 2] / mass[m]
    ls = mass * (w * q + (1 - w) * ratio)
    out[:, 2] = ls
    out[:, 0] = mass - ls
    return out


def run_spec():
    df = pd.read_csv(os.path.join(WORK_DIR, "train_folds.csv"))
    pair_idx = df.index[df["label"].isin(PAIR)].to_numpy()

    oof_q = np.zeros(len(df), dtype=np.float32)
    accs = []
    for k in range(5):
        idx, qv, acc = train_bin_fold(k, df)
        oof_q[pair_idx[idx]] = qv
        accs.append(acc)
    print("专家模型 5 折平均二分类准确率: %.4f" % float(np.mean(accs)))

    prob = np.load(os.path.join(WORK_DIR, "oof_prob.npy"))
    y = np.load(os.path.join(WORK_DIR, "oof_label.npy"))
    mask = np.isin(y, PAIR)
    print("主模型在这两类上的 OOF 准确率: %.4f"
          % float((prob[mask].argmax(1) == y[mask]).mean()))
    print("融合前 OOF macro F1: %.4f" % f1_score(y, prob.argmax(1), average="macro"))

    print("\nw 搜索（0=完全听主模型，1=完全听专家）:")
    best_w, best_f1 = 0.0, -1.0
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        fp = fuse(prob, oof_q, w)
        f1 = f1_score(y, fp.argmax(1), average="macro")
        pa = float((fp[mask].argmax(1) == y[mask]).mean())
        print("  w=%.1f | OOF macroF1 %.4f | 两类准确率 %.4f" % (w, f1, pa))
        if f1 > best_f1:
            best_f1, best_w = f1, w
    print("最佳 w=%.1f | OOF macro F1 %.4f" % (best_w, best_f1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--spec", action="store_true")
    args = ap.parse_args()
    if args.detail:
        run_detail()
    if args.spec:
        run_spec()
    if not (args.detail or args.spec):
        ap.print_help()
