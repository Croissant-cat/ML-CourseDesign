# -*- coding: utf-8 -*-
"""Step 4: ConvNeXt-Tiny 5 折正式训练（对应报告 1.3.3 / 1.5 / 1.6.2 节）。

配方（由 step3 的受控实验选出）：
    ConvNeXt-Tiny / 256px / bs16 / AdamW / 差分学习率 2e-4 + 1e-3 /
    2 轮 warmup + 余弦退火 / 50 轮 / 标签平滑 0.05 / AMP / 无 Mixup

输出：
    /kaggle/working/plant_data/models/cvx_fold0-4.pth
    /kaggle/working/plant_data/oof_prob.npy, oof_label.npy
    /kaggle/working/plant_data/fold_metrics.csv
"""
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

WORK_DIR = "/kaggle/working/plant_data/"
MODEL_DIR = os.path.join(WORK_DIR, "models")
CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda")

CFG = dict(backbone="convnext_tiny", size=256, epochs=50, bs=16,
           lr_back=2e-4, lr_head=1e-3, ls=0.05)


def set_seed(seed):
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


def train_tf(size):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(30)], p=0.5),
        T.ColorJitter(0.25, 0.25, 0.25, 0.05),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def val_tf(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def train_one_fold(fold, df, cfg):
    set_seed(42 + fold)                      # 每折不同种子，5 个模型有差异，集成才有增益
    size, bs, epochs = cfg["size"], cfg["bs"], cfg["epochs"]

    va_idx = df.index[df["fold"] == fold].to_numpy()
    tr_df = df[df["fold"] != fold].reset_index(drop=True)
    va_df = df[df["fold"] == fold].reset_index(drop=True)

    tr_loader = DataLoader(PathDS(tr_df["path"].tolist(), tr_df["label"].tolist(), train_tf(size)),
                           batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    va_loader = DataLoader(PathDS(va_df["path"].tolist(), va_df["label"].tolist(), val_tf(size)),
                           batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    model = timm.create_model(cfg["backbone"], pretrained=True, num_classes=len(CLASSES),
                              drop_rate=0.2, drop_path_rate=0.1).to(DEVICE)
    head = [p for n, p in model.named_parameters() if n.startswith(("head", "classifier", "fc"))]
    hid = {id(p) for p in head}
    back = [p for p in model.parameters() if id(p) not in hid]
    opt = torch.optim.AdamW([{"params": back, "lr": cfg["lr_back"]},
                             {"params": head, "lr": cfg["lr_head"]}], weight_decay=0.05)

    steps = len(tr_loader)
    total, warm = epochs * steps, 2 * steps
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))),
    )
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["ls"])
    scaler = torch.amp.GradScaler("cuda")

    def infer(net, loader):
        net.eval()
        prob, gts, preds = [], [], []
        with torch.no_grad():
            for x, y in loader:
                with torch.amp.autocast("cuda"):
                    o = net(x.to(DEVICE))
                prob.append(torch.softmax(o.float(), 1).cpu().numpy())
                preds += o.argmax(1).cpu().tolist()
                gts += y.tolist()
        return np.concatenate(prob), np.array(gts), np.array(preds)

    best, best_ep, best_state, t0 = -1.0, -1, None, time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
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
            tot += loss.item()
            n += 1
        _, gt, pr = infer(model, va_loader)
        f1 = f1_score(gt, pr, average="macro")
        if f1 > best:                          # 以 macro F1（而非 accuracy）选优
            best, best_ep = f1, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print("  fold %d | ep %2d/%d loss %.3f | val macroF1 %.4f (best %.4f @ep%d)"
                  % (fold, ep + 1, epochs, tot / n, f1, best, best_ep))

    model.load_state_dict(best_state)
    prob, gt, pr = infer(model, va_loader)
    torch.save(best_state, os.path.join(MODEL_DIR, "cvx_fold%d.pth" % fold))
    print("  fold %d 完成 %.1f 分钟 | best %.4f @ep%d | acc %.4f"
          % (fold, (time.time() - t0) / 60, best, best_ep, float((pr == gt).mean())))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best, best_ep, va_idx, prob, gt


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(WORK_DIR, "train_folds.csv"))
    print("配方:", CFG)

    t_all = time.time()
    oof_prob = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    oof_label = np.zeros(len(df), dtype=int)
    rows = []

    for k in range(5):
        best, best_ep, idx, prob, gt = train_one_fold(k, df, CFG)
        oof_prob[idx] = prob
        oof_label[idx] = gt
        rows.append(dict(fold=k, best_f1=best, best_epoch=best_ep))

    oof_pred = oof_prob.argmax(1)
    print("\n===== 5 折汇总 =====")
    print(pd.DataFrame(rows).to_string(index=False))
    print("单折平均: %.4f" % np.mean([r["best_f1"] for r in rows]))
    print("OOF（全部 500 张）macro F1: %.4f"
          % f1_score(oof_label, oof_pred, average="macro"))
    print("\nOOF 分类报告:")
    print(classification_report(oof_label, oof_pred, target_names=CLASSES, digits=4))
    print("OOF 混淆矩阵（行=真实，列=预测）:")
    print(pd.DataFrame(confusion_matrix(oof_label, oof_pred), index=CLASSES, columns=CLASSES))

    np.save(os.path.join(WORK_DIR, "oof_prob.npy"), oof_prob)
    np.save(os.path.join(WORK_DIR, "oof_label.npy"), oof_label)
    pd.DataFrame(rows).to_csv(os.path.join(WORK_DIR, "fold_metrics.csv"), index=False)
    print("\n5 折总耗时 %.1f 分钟" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
