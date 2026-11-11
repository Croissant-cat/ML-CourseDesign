# -*- coding: utf-8 -*-
"""step5：全量训练 12 个模型，坐标层取中位数，与第 2 版锚点融合，得到最终提交的预测。

在做什么
    12 个模型（8 个 resnet18 + 4 个 resnet34，不同随机种子）各自用**全部 80 张训练图**
    训练 100 轮，不做交叉验证。每个模型单独解码出 20 张测试图的坐标，12 个模型取中位数，
    再与第 2 版（10 折乘双骨干的 20 模型热图平均）的预测等权融合；单张分歧超过 200 像素
    时回退到第 2 版。最后加固定的坐标修正，写出提交候选。

需要多久
    Kaggle 单卡 T4 约 59 分钟（resnet18 每个约 4.2 分钟，resnet34 每个约 6.0 分钟）。

输出什么
    outputs/heat_v5.npz          12 个模型累加后的双通道热图（1024×128×128 太大，这里只存均值）
    outputs/pred_v5_models.npy   每个模型独立解码的坐标，形状 (12, 20, 2)，原图像素
    outputs/pred_v5.npy          融合后的测试预测，形状 (20, 2)，原图像素（未加坐标修正）
    outputs/sub_v5_raw.csv       融合结果，未加坐标修正
    outputs/sub_v5_blend.csv     融合结果 + 坐标修正（step6 用它写最终 submission.csv）
    屏幕输出最后 20 行坐标，可与第 5 版运行日志逐行比对。
"""
import os
import re
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------- 配置
CFG = dict(
    size=512,
    epochs=100,
    batch=8,
    lr=1e-3,
    sigma=1.5,
    tta_flip=True,
    n_workers=0,
    w_region=0.4,
    # (骨干, 种子)：8 个 resnet18 + 4 个 resnet34
    models=(("resnet18", 42), ("resnet18", 7), ("resnet18", 2024), ("resnet18", 1993),
            ("resnet18", 555), ("resnet18", 1234), ("resnet18", 88), ("resnet18", 31),
            ("resnet34", 42), ("resnet34", 2024), ("resnet34", 7), ("resnet34", 555)),
)
CORRECTION = dict(dx=15.2, dy=-5.3)     # 拟合值 (+15.15, −5.28) 四舍五入到 0.1 像素
GUARD_PX = 200.0                        # 与锚点分歧超过此值则回退锚点

# 第 2 版锚点（10 折乘双骨干、20 模型热图平均的预测，原图像素，未加修正）
# step3 会把同样的数写到 outputs/anchor_v2.npy，存在时优先读文件
ANCHOR_V2 = {81: (1695.7, 1149.9), 82: (1686.8, 1102.0), 83: (1483.5, 1048.1),
             84: (1228.2, 1116.1), 85: (1198.1, 986.2), 86: (1247.9, 1131.1),
             87: (1473.7, 1022.7), 88: (1248.3, 1128.1), 89: (1475.4, 1053.3),
             90: (1663.6, 1099.4), 91: (1473.7, 1040.3), 92: (1692.0, 1109.7),
             93: (1690.9, 1163.7), 94: (1437.9, 1023.4), 95: (1668.3, 1101.6),
             96: (1694.8, 1082.4), 97: (1496.3, 1209.2), 98: (1463.3, 1070.3),
             99: (1455.5, 1034.8), 100: (1684.0, 1098.3)}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
S = CFG["size"]
Hc = S // 4


def find_data():
    """数据目录：优先 <任务三>/data/detection，找不到再在 /kaggle/input 下递归找。"""
    cand = [os.path.join(ROOT, "data", "detection"), os.path.join(ROOT, "data"),
            os.environ.get("FOVEA_DATA", "")]
    for c in cand:
        if c and os.path.exists(os.path.join(c, "fovea_localization_train_GT.csv")):
            return c
    for base in ("/kaggle/input", os.getcwd()):
        if os.path.isdir(base):
            for r, d, f in os.walk(base):
                if "fovea_localization_train_GT.csv" in f:
                    return r
    raise FileNotFoundError("找不到 fovea_localization_train_GT.csv，请把数据放到 data/detection/")


DATA = None
TRAIN_DIR = TEST_DIR = BOX_DIR = None


def init_paths():
    """把数据路径装进全局。导入本模块时不触发，方便单独核对融合逻辑。"""
    global DATA, TRAIN_DIR, TEST_DIR, BOX_DIR
    DATA = find_data()
    TRAIN_DIR = os.path.join(DATA, "train")
    TEST_DIR = os.path.join(DATA, "test")
    BOX_DIR = os.path.join(DATA, "train_location")
    return DATA


def imread_u(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return im


def read_boxes():
    out = {}
    if not os.path.isdir(BOX_DIR):
        return out
    for f in sorted(os.listdir(BOX_DIR)):
        if not f.endswith(".xml"):
            continue
        r = ET.parse(os.path.join(BOX_DIR, f)).getroot()
        nm = r.find("filename").text.split(".")[0]
        b = r.find("object/bndbox")
        out[nm] = tuple(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
    return out


def soft_mask(box, W, H, blur=1.0):
    """把黄斑区域框画成 Hc×Hc 的软掩码（框内 1、外 0，高斯模糊后峰值归一到 1）。"""
    x0, y0, x1, y1 = box
    m = np.zeros((Hc, Hc), np.float32)
    ix0 = max(0, int(round(x0 / W * Hc - 0.5)))
    ix1 = min(Hc, int(round(x1 / W * Hc - 0.5)) + 1)
    iy0 = max(0, int(round(y0 / H * Hc - 0.5)))
    iy1 = min(Hc, int(round(y1 / H * Hc - 0.5)) + 1)
    m[iy0:iy1, ix0:ix1] = 1.0
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur)
        if m.max() > 1e-6:
            m /= m.max()
    return m


def gheat(cx, cy):
    yy, xx = np.mgrid[0:Hc, 0:Hc].astype(np.float32)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * CFG["sigma"] ** 2)).astype(np.float32)


def geo_aug(img, pt, mask, rng, rot=12.0, sc=0.12, tr=0.04):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rng.uniform(-rot, rot), rng.uniform(1 - sc, 1 + sc))
    M[0, 2] += rng.uniform(-tr, tr) * w
    M[1, 2] += rng.uniform(-tr, tr) * h
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y = M @ np.array([pt[0], pt[1], 1.0])
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
        x = w - 1.0 - x
    return img, (x, y), mask


def photo_aug(img, rng):
    img = img.astype(np.float32)
    if rng.random() < 0.7:
        img *= rng.uniform(0.85, 1.18)
        m = img.mean()
        img = (img - m) * rng.uniform(0.85, 1.15) + m
    np.clip(img, 0, 255, out=img)
    return img


class Data(Dataset):
    def __init__(self, items, train, rng=None):
        self.items, self.train = items, train
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img, pt, mask = it["img"].copy(), (it["px"], it["py"]), it["mask"].copy()
        if self.train:
            img, pt, mask = geo_aug(img, pt, mask, self.rng)
            img = photo_aug(img, self.rng)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        th = torch.from_numpy(gheat((pt[0] + 0.5) / S * Hc, (pt[1] + 0.5) / S * Hc))
        tm = torch.from_numpy(cv2.resize(mask, (Hc, Hc), interpolation=cv2.INTER_AREA))
        return x, torch.stack([th, tm], 0), torch.zeros(1)


class ConvBNR(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                               nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


def build_net(kind):
    import torchvision.models as M
    if kind == "resnet18":
        net, ch = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1), (64, 128, 256, 512)
    else:
        net, ch = M.resnet34(weights=M.ResNet34_Weights.IMAGENET1K_V1), (64, 128, 256, 512)
    ch1, ch2, ch3, ch4 = ch

    class HeatNet(nn.Module):
        """四级特征自底向上拼接，输出 stride 4 的双通道热图。"""

        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.l1, self.l2, self.l3, self.l4 = net.layer1, net.layer2, net.layer3, net.layer4
            self.u4 = ConvBNR(ch4, ch4 // 2)
            self.c3 = ConvBNR(ch4 // 2 + ch3, ch3 // 2)
            self.c2 = ConvBNR(ch3 // 2 + ch2, 64)
            self.c1 = ConvBNR(64 + ch1, 64)
            self.out = nn.Sequential(ConvBNR(64, 64), nn.Conv2d(64, 2, 1))
            nn.init.constant_(self.out[-1].bias, -4.0)

        def forward(self, x):
            f1 = self.l1(self.stem(x))
            f2 = self.l2(f1)
            f3 = self.l3(f2)
            f4 = self.l4(f3)
            d = self.u4(F.interpolate(f4, scale_factor=2, mode="nearest"))
            d = self.c3(torch.cat([d, f3], 1))
            d = self.c2(torch.cat([F.interpolate(d, scale_factor=2, mode="nearest"), f2], 1))
            d = self.c1(torch.cat([F.interpolate(d, scale_factor=2, mode="nearest"), f1], 1))
            return self.out(d)

    return HeatNet()


def decode_point(hm, winsz=9, power=2.0):
    """点热图 -> 峰值 9×9 窗口内的 power 加权质心（热图格坐标）。"""
    N, Hh, Ww = hm.shape
    out = np.zeros((N, 2), np.float32)
    conf = np.zeros(N, np.float32)
    r = winsz // 2
    for b in range(N):
        py, px_ = divmod(int(np.argmax(hm[b])), Ww)
        conf[b] = float(hm[b, py, px_])
        y0, y1 = max(0, py - r), min(Hh, py + r + 1)
        x0, x1 = max(0, px_ - r), min(Ww, px_ + r + 1)
        w = np.clip(hm[b, y0:y1, x0:x1], 0, None) ** power
        s = w.sum()
        if s <= 1e-12:
            out[b] = (px_ + 0.5, py + 0.5)
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        out[b, 0] = (w * xx).sum() / s
        out[b, 1] = (w * yy).sum() / s
    return out, conf


def decode_region(mask, centers, winsz=25, thr=0.35):
    """区域掩码 -> 在点峰值邻域内、减去阈值后的加权质心（热图格坐标）。"""
    N, Hh, Ww = mask.shape
    out = np.zeros((N, 2), np.float32)
    r = winsz // 2
    yy0, xx0 = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    for b in range(N):
        cx, cy = int(round(float(centers[b, 0]))), int(round(float(centers[b, 1])))
        y0, y1 = max(0, cy - r), min(Hh, cy + r + 1)
        x0, x1 = max(0, cx - r), min(Ww, cx + r + 1)
        w = np.clip(mask[b, y0:y1, x0:x1] - thr, 0, None)
        s = w.sum()
        if s <= 1e-9:
            out[b] = (cx + 0.5, cy + 0.5)
            continue
        out[b, 0] = (w * xx0[y0:y1, x0:x1]).sum() / s
        out[b, 1] = (w * yy0[y0:y1, x0:x1]).sum() / s
    return out


def cells_to_px(hc, items):
    """热图格坐标 -> 原图像素，逐图用各自的宽高。"""
    W = np.array([it["W"] for it in items], np.float64)
    H = np.array([it["H"] for it in items], np.float64)
    return (hc[:, 0] * 4.0 + 0.5) / S * W - 0.5, (hc[:, 1] * 4.0 + 0.5) / S * H - 0.5


@torch.no_grad()
def infer(model, items, bs=16):
    dl = DataLoader(Data(items, False), batch_size=bs, shuffle=False, num_workers=0)
    model.eval()
    hs, ms = [], []
    for x, _, _ in dl:
        xg = x.to(DEV)
        o = torch.sigmoid(model(xg)).float().cpu().numpy()
        if CFG["tta_flip"]:
            o2 = torch.sigmoid(model(torch.flip(xg, dims=[3]))).float().cpu().flip(3).numpy()
            o = (o + o2) / 2
        hs.append(o[:, 0])
        ms.append(o[:, 1])
    return np.concatenate(hs, 0), np.concatenate(ms, 0)


# ---------------------------------------------------------------- 数据
def load_items():
    if DATA is None:
        init_paths()
    gt = pd.read_csv(os.path.join(DATA, "fovea_localization_train_GT.csv"))
    gt.columns = [str(c).strip() for c in gt.columns]
    low = {c: c.lower().replace(" ", "").replace("_", "") for c in gt.columns}
    cn = next(c for c, k in low.items() if k in ("data", "image", "imagename", "filename", "name", "id", "imageid"))
    cx_ = next(c for c, k in low.items() if k in ("foveax", "x", "centerx"))
    cy_ = next(c for c, k in low.items() if k in ("foveay", "y", "centery"))
    gt = gt.rename(columns={cn: "name", cx_: "x", cy_: "y"})
    gt["name"] = gt["name"].astype(str).str.strip().map(lambda s: os.path.splitext(os.path.basename(s))[0])
    gt["name"] = gt["name"].map(lambda s: s if not s.isdigit() else "%04d" % int(s))
    gt["x"], gt["y"] = gt["x"].astype(float), gt["y"].astype(float)
    gt = gt[~((gt["x"] <= 0) & (gt["y"] <= 0))].reset_index(drop=True)
    boxes = read_boxes()

    train_items = []
    for nm, gx, gy in gt[["name", "x", "y"]].itertuples(index=False):
        p = os.path.join(TRAIN_DIR, nm + ".jpg")
        if not os.path.exists(p):
            continue
        im = imread_u(p)
        H, W = im.shape[:2]
        sm = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (S, S), interpolation=cv2.INTER_AREA)
        train_items.append(dict(
            name=nm, img=sm, gx=gx, gy=gy, W=W, H=H,
            px=gx / W * S - 0.5, py=gy / H * S - 0.5,
            mask=soft_mask(boxes[nm], W, H) if nm in boxes else np.zeros((Hc, Hc), np.float32)))

    te_names = sorted(os.path.splitext(f)[0] for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg"))
    test_items = []
    for nm in te_names:
        im = imread_u(os.path.join(TEST_DIR, nm + ".jpg"))
        H, W = im.shape[:2]
        sm = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (S, S), interpolation=cv2.INTER_AREA)
        test_items.append(dict(name=nm, img=sm, gx=0.0, gy=0.0, W=W, H=H, px=0.0, py=0.0,
                               mask=np.zeros((Hc, Hc), np.float32)))
    return train_items, test_items


def main():
    t0 = time.time()
    init_paths()
    print("torch %s | device %s | OUT %s | 模型数 %d"
          % (torch.__version__, DEV, OUT, len(CFG["models"])))
    print("DATA:", DATA)
    train_items, test_items = load_items()
    NG, NT = len(train_items), len(test_items)
    print("train %d | test %d | 预处理 %.1fs" % (NG, NT, time.time() - t0))

    n_models = len(CFG["models"])
    heat_acc = np.zeros((NT, Hc, Hc), np.float32)
    mask_acc = np.zeros((NT, Hc, Hc), np.float32)
    P_all = np.zeros((n_models, NT, 2), np.float32)

    for k, (bb, seed) in enumerate(CFG["models"]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        dl = DataLoader(Data(train_items, True, np.random.default_rng(seed)),
                        batch_size=CFG["batch"], shuffle=True, num_workers=CFG["n_workers"], drop_last=True)
        model = build_net(bb).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=CFG["lr"], total_steps=max(1, len(dl)) * CFG["epochs"], pct_start=0.25)
        for ep in range(CFG["epochs"]):
            model.train()
            tot = 0.0
            for x, t, _ in dl:
                x, t = x.to(DEV), t.to(DEV)
                opt.zero_grad(set_to_none=True)
                o = torch.sigmoid(model(x))
                loss = F.mse_loss(o[:, 0], t[:, 0]) + F.mse_loss(o[:, 1], t[:, 1])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(x)
            if ep % 40 == 0 or ep == CFG["epochs"] - 1:
                print("  [%2d/%d] %-8s seed%-4d ep%3d loss=%.5f (%.0fs)"
                      % (k + 1, n_models, bb, seed, ep, tot / max(1, len(train_items)),
                         time.time() - t0), flush=True)
        h, m = infer(model, test_items)
        heat_acc += h / n_models
        mask_acc += m / n_models
        pk, _ = decode_point(h)
        rg = decode_region(m, pk)
        pxm, pym = cells_to_px(pk, test_items)
        rxm, rym = cells_to_px(rg, test_items)
        P_all[k, :, 0] = CFG["w_region"] * rxm + (1 - CFG["w_region"]) * pxm
        P_all[k, :, 1] = CFG["w_region"] * rym + (1 - CFG["w_region"]) * pym
        print("  [%2d/%d] 完成，累计 %.1f 分钟" % (k + 1, n_models, (time.time() - t0) / 60), flush=True)
        del model
        torch.cuda.empty_cache()

    np.savez_compressed(os.path.join(OUT, "heat_v5.npz"), heat=heat_acc, mask=mask_acc,
                        te_names=np.array([it["name"] for it in test_items]))
    np.save(os.path.join(OUT, "pred_v5_models.npy"), P_all)

    # ---------------- 融合 ----------------
    B = np.median(P_all, axis=0)
    anchor_path = os.path.join(OUT, "anchor_v2.npy")
    if os.path.exists(anchor_path):
        anchor = np.load(anchor_path)
        print("锚点：读 outputs/anchor_v2.npy")
    else:
        anchor = np.array([ANCHOR_V2[int(re.search(r"\d+", it["name"]).group())] for it in test_items],
                          np.float64)
        print("锚点：使用脚本内置的第 2 版预测（未找到 outputs/anchor_v2.npy）")

    gap = np.hypot(*(B - anchor).T)
    guard = gap > GUARD_PX
    Bg = B.copy()
    Bg[guard] = anchor[guard]
    final = 0.5 * (Bg + anchor)

    np.save(os.path.join(OUT, "pred_v5.npy"), final)
    print("\n逐模型中位数与锚点的分歧：中位 %.1f、最大 %.1f 像素；回退 %d 张"
          % (np.median(gap), gap.max(), int(guard.sum())))

    # ---------------- 写候选 ----------------
    sam = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ID = sam.columns[0]
    idx = {int(re.search(r"\d+", it["name"]).group()): j for j, it in enumerate(test_items)}

    def write(tag, Q, dx, dy):
        vals = []
        for s in sam[ID].astype(str):
            axis = 1 if s.strip().lower().endswith("y") else 0
            m = re.search(r"\d+", s)
            j = idx.get(int(m.group())) if m else None
            v = (Q[j, axis] if j is not None else (1450.0 if axis == 0 else 1050.0)) + (dy if axis else dx)
            vals.append(v)
        o = pd.DataFrame({ID: sam[ID].values, "value": np.round(vals, 4)})
        p = os.path.join(OUT, "sub_v5_%s.csv" % tag)
        o.to_csv(p, index=False)
        print("写出 %-42s %d 行" % (p, len(o)))
        return p

    write("raw", final, 0.0, 0.0)
    write("blend", final, CORRECTION["dx"], CORRECTION["dy"])

    print("\n最终预测（原图像素，未加修正）：")
    for j, it in enumerate(test_items):
        print("  %-5s x=%8.1f y=%8.1f" % (it["name"], final[j, 0], final[j, 1]))
    print("\n最终预测（加修正 x%+.1f / y%+.1f）：" % (CORRECTION["dx"], CORRECTION["dy"]))
    for j, it in enumerate(test_items):
        print("  %-5s x=%8.1f y=%8.1f"
              % (it["name"], final[j, 0] + CORRECTION["dx"], final[j, 1] + CORRECTION["dy"]))
    print("\n总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
