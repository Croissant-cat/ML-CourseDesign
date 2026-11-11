# -*- coding: utf-8 -*-
"""Step 3: 单折受控对比实验（对应报告 1.4 节表 1-5、1.7 节）。

统一在第 0 折上训练并比较，每次只改一个变量：
    E1  EfficientNet-B0 / 224 / Mixup 0.5       -> 0.7840
    E2  同上，关闭 Mixup                        -> 0.8709
    E3  同上，只训练最后 stage + 分类头        -> 0.8198
    E4  同上，改用整图（不裁切）               -> 0.8027
    E5  同上，训练轮数 60                       -> 0.8618
    E6  同上，输入 256                          -> 0.8821
    E7  ConvNeXt-Tiny / 224 / 40 轮             -> 0.9107

用法：
    python src/step3_ablation.py --exp E2
    python src/step3_ablation.py --all
"""
import argparse
import gc
import math
import os
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

WORK_DIR = "/kaggle/working/plant_data/"
ORIG_DIR = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train/"
CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class PathDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


class ListDS(Dataset):
    """整图变体：把预缩放到 320 的原图放进内存，避免每轮重复解码大图。"""

    def __init__(self, imgs, labels, tf):
        self.imgs, self.labels, self.tf = imgs, labels, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), self.labels[i]


def make_tf(size, strong=True):
    ops = [
        T.RandomResizedCrop(size, scale=(0.65, 1.0) if strong else (0.8, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),          # 叶片上下翻转物理合理，等同扩充样本
    ]
    if strong:
        ops += [T.RandomApply([T.RandomRotation(30)], p=0.5),
                T.ColorJitter(0.25, 0.25, 0.25, 0.05)]
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if strong:
        ops += [T.RandomErasing(p=0.25, scale=(0.02, 0.15))]
    return T.Compose(ops)


def val_tf(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def set_trainable(model, mode):
    for p in model.parameters():
        p.requires_grad = False
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "last":
        for n, p in model.named_parameters():
            if n.startswith(("classifier", "conv_head", "bn2", "blocks.6")):
                p.requires_grad = True
    elif mode == "head":
        for n, p in model.named_parameters():
            if n.startswith("classifier"):
                p.requires_grad = True


def freeze_bn(model):
    """被冻结的 BN 层必须切到 eval，否则其统计量会被当前小批次改写。"""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def mix_batch(x, y, p=0.5, mixup_a=0.2, cutmix_a=1.0):
    if random.random() > p:
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:
        lam = float(np.random.beta(mixup_a, mixup_a))
        x = lam * x + (1 - lam) * x[perm]
    else:
        lam = float(np.random.beta(cutmix_a, cutmix_a))
        h, w = x.shape[2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return x, y, y[perm], lam


def run(name, backbone="efficientnet_b0", use_crop=True, size=224, mix_p=0.0, ls=0.05,
        mode="full", lr_back=2e-4, lr_head=1e-3, epochs=35, bs=32):
    set_seed(42)
    df = pd.read_csv(os.path.join(WORK_DIR, "train_folds.csv"))
    tr_df = df[df["fold"] != 0].reset_index(drop=True)     # 400 张
    va_df = df[df["fold"] == 0].reset_index(drop=True)     # 100 张

    if use_crop:
        tr_ds = PathDS(tr_df["path"].tolist(), tr_df["label"].tolist(), make_tf(size))
        va_ds = PathDS(va_df["path"].tolist(), va_df["label"].tolist(), val_tf(size))
    else:
        pre = lambda frm: [Image.open(os.path.join(ORIG_DIR, r["cls"], r["name"]))
                           .convert("RGB").resize((320, 320), Image.BILINEAR) for _, r in frm.iterrows()]
        tr_ds = ListDS(pre(tr_df), tr_df["label"].tolist(), make_tf(size))
        va_ds = ListDS(pre(va_df), va_df["label"].tolist(), val_tf(size))

    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=2,
                           pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    model = timm.create_model(backbone, pretrained=True, num_classes=len(CLASSES),
                              drop_rate=0.2, drop_path_rate=0.1).to(DEVICE)
    set_trainable(model, mode)
    head_ids = {id(p) for n, p in model.named_parameters()
                if p.requires_grad and n.startswith(("classifier", "head"))}
    head = [p for p in model.parameters() if id(p) in head_ids]
    back = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = ([{"params": back, "lr": lr_back}] if back else []) + [{"params": head, "lr": lr_head}]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)

    steps = len(tr_loader)
    total, warm = epochs * steps, 2 * steps
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))),
    )
    crit = nn.CrossEntropyLoss(label_smoothing=ls)
    scaler = torch.amp.GradScaler("cuda")

    def evaluate():
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in va_loader:
                with torch.amp.autocast("cuda"):
                    o = model(x.to(DEVICE))
                preds += o.argmax(1).cpu().tolist()
                gts += y.tolist()
        return f1_score(gts, preds, average="macro")

    print("  [%s] %s | %dpx | %d 轮 | mode=%s | mix=%0.1f"
          % (name, backbone, size, epochs, mode, mix_p))
    best, best_ep, t0 = -1.0, -1, time.time()
    for ep in range(epochs):
        model.train()
        freeze_bn(model)
        tot, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x, ya, yb, lam = mix_batch(x, y, p=mix_p)
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if not (scaler.get_scale() < before):   # 只在参数真的被更新时推进调度器
                sch.step()
            tot += loss.item()
            n += 1
        f1 = evaluate()
        if f1 > best:
            best, best_ep = f1, ep + 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print("    ep %2d/%d loss %.3f | val macroF1 %.4f (best %.4f @ep%d)"
                  % (ep + 1, epochs, tot / n, f1, best, best_ep))
    print("  [%s] 完成 %.1f 分钟 | best %.4f @ep%d" % (name, (time.time() - t0) / 60, best, best_ep))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best


EXPS = {
    "E1": dict(name="E1 Mixup 0.5", mix_p=0.5, epochs=35),
    "E2": dict(name="E2 关闭 Mixup", mix_p=0.0, epochs=35),
    "E3": dict(name="E3 只训末段", mix_p=0.0, epochs=35, mode="last", lr_back=5e-4),
    "E4": dict(name="E4 整图", use_crop=False, mix_p=0.0, epochs=35),
    "E5": dict(name="E5 训练 60 轮", mix_p=0.0, epochs=60),
    "E6": dict(name="E6 输入 256", size=256, mix_p=0.0, epochs=60),
    "E7": dict(name="E7 ConvNeXt-Tiny", backbone="convnext_tiny", size=224,
               mix_p=0.0, epochs=40, bs=16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2", choices=sorted(EXPS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    results = {}
    todo = sorted(EXPS) if args.all else [args.exp]
    for key in todo:
        results[key] = run(**EXPS[key])

    print("\n===== 单折（fold 0）结果 =====")
    for k, v in results.items():
        print("  %-20s %.4f" % (EXPS[k]["name"], v))


if __name__ == "__main__":
    main()
