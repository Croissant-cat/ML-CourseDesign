# -*- coding: utf-8 -*-
"""任务四公共库。step1 至 step6 共用。

包含：比赛数据定位与配对、视野掩膜、四通道预处理（绿通道 CLAHE、红蓝通道、Frangi 响应）、
残差 U-Net、三项损失（BCE + Dice + 中心线 Dice）、图块采样与增广、测试时增广推理、
阈值与连通块二值化、run-length 编解码、调用官方转码脚本。
"""
import os, re, sys, glob, json, copy, tempfile, time, math, random, shutil, subprocess, warnings
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy import ndimage as ndi

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)          # 避免和 torch 抢线程


# ================================ 0. 参数区（一般只需改这里） ================================
DATA_ROOT = None       # None = 自动在 /kaggle/input 下查找；也可直接填 "/kaggle/input/2026task-4"
OUT_DIR   = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

SEED        = 42       # 换个种子可以再训一套做集成，能再涨一点点
N_FOLDS     = 6        # 交叉验证折数（=集成模型个数）。想更快就改 3~5
EPOCHS      = 130      # 每折 epoch
ITERS       = 80       # 每 epoch 迭代数（每轮重新随机采样 patch）
BATCH       = 8        # batch size
PATCH       = 288      # 训练 patch 尺寸（v2 从 256 提到 288，细血管更清楚）
BASE_CH     = 32       # U-Net 基础通道数（小显存可改 24/16）
LR          = 3e-4
MIN_LR      = 1e-6
WD          = 1e-4
CL_DICE_W   = 0.2      # 中心线 Dice 权重（帮助细血管连通），0 = 关闭
USE_FRANGI  = True     # 把 Frangi 血管增强图作为额外输入通道（约 +0.005~0.01）
TTA_SCALES  = (1.0, 1.25, 1.5)   # 推理多尺度
TTA_ROT90   = True     # TTA 再加 90° 旋转（4 翻转 x 2 = 8 路）
USE_SWA     = True     # 最后几轮权重平均（免费涨点，且自动和最后一轮比，取好的）
SWA_EPOCHS  = 5
AMP_MODE    = "auto"   # 混合精度: "auto"(推荐,自动探测防NaN) / "fp16" / "bf16" / "off"
THR_FIXED   = None     # None = 用 OOF(交叉验证)自动选最佳阈值；也可手填 0.5
MAX_MINUTES = 170      # 训练总时间预算(分钟)，会根据实测速度自动缩放 epoch/折数；0=不限
FAST        = False    # True = 极简模式(约2分钟)，只用来确认流程能跑通，分数很低
PNG_VESSEL_WHITE = True   # 官方脚本按 PNG 中 255 的区域取 mask（血管）
DUMP_PROBS  = True        # 存测试集/OOF 概率图，便于不重训就换阈值试投
SUB_THRS    = (0.35, 0.40, 0.45)   # 额外生成的阈值版本（实测 thr 越低越好，所以往低侧探）

# ---- RETUNE 模式：不训练，直接用上一版存的概率图重扫阈值（约3分钟，不占GPU）----
RETUNE_ONLY = False           # True = 跳过训练，只重扫阈值
RETUNE_PROB_DIR = None        # 手填单个 probs 目录；None = 自动找
RETUNE_PROB_DIRS = None       # 手填多个目录（列表），例如 ['/kaggle/input/a/probs', '/kaggle/input/b/probs']
RETUNE_MERGE_ALL = True       # 自动模式下，把找到的**所有** probs 目录做概率平均（跨版本集成）
RETUNE_THRS = (0.15, 0.25, 0.35, 0.45)   # 要试的阈值
RETUNE_MIN_SIZE = 20          # 最小连通块像素数（去掉碎点）
RETUNE_MAIN_THR = 0.45        # 用哪个阈值写 submission.csv（主提交）

# ---- 伪标签自训练（v4 新增）----
USE_PSEUDO     = False  # 实测：这份数据上伪标签没有收益（0.85654 -> 0.85595），默认关闭
PSEUDO_HI      = 0.85   # prob >= HI  → 当作血管
PSEUDO_LO      = 0.05   # prob <= LO  → 当作背景；中间地带不计损失
PSEUDO_EPOCHS  = 12
PSEUDO_ITERS   = 40
PSEUDO_LR      = 3e-5   # 微调用小学习率，别把原模型带偏
PSEUDO_BLEND   = 0.5    # 最终概率 = (1-BLEND)*原预测 + BLEND*微调后预测
PSEUDO_MAX_DROP = 0.004 # 若微调后 OOF 掉超过这个值，就放弃伪标签

# 自动识别失败时可手动指定（把路径填上再跑）
TRAIN_DIR, GT_DIR, TEST_DIR = None, None, None
# ===========================================================================================

def p(*a):
    print(*a, flush=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if FAST:
    N_FOLDS, EPOCHS, ITERS, BASE_CH, PATCH, TTA_SCALES = 2, 3, 8, 16, 192, (1.0,)
elif DEV.type != "cuda":
    N_FOLDS, EPOCHS, ITERS, BASE_CH, PATCH, TTA_SCALES = 2, 30, 35, 16, 192, (1.0,)
    p("!! 没有检测到 GPU，已自动降级为 CPU 小模型（分数会低一些）。"
      "建议右上角 Accelerator 选择 GPU T4 后重跑。")



# ==========================================================================================
#  1. 数据发现：自动找出 训练图/训练标签/测试图/sample_submission
# ==========================================================================================
IMG_EXT  = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".ppm")
HINT_GT  = ("manual", "label", "gt", "vessel", "annot", "1st", "seg", "truth", "mask_gt")
HINT_BAD = ("mask", "fov", "roi", "contour", "border", "circle")

def _classify(path):
    """判断一张图是 fundus 彩色眼底图 还是 二值标签图"""
    try:
        im = Image.open(path); im.load()
    except Exception:
        return None
    try:
        a = np.array(im)
    except Exception:
        return None
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = a[..., :3]
        if a.shape[2] >= 3 and np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]):
            a = a[..., 0]
        else:
            return "photo"
    if a.ndim == 2:
        return "binary" if np.unique(a).size <= 8 else "photo"
    return None

def _numkey(path):
    m = re.search(r"(\d+)", Path(path).stem)
    return m.group(0) if m else Path(path).stem.lower()

def _has(path, words):
    s = str(path).lower()
    return any(w in s for w in words)

OUTPUT_DIR_NAMES = ("_official_run", "pred_png", "probs", ".ipynb_checkpoints",
                    "notebooks", "__pycache__", "_png_thr")

def _is_output_dir(p):
    """判断是否是"notebook 输出"之类的目录（不能当比赛数据用）"""
    s = str(p).lower().replace("\\", "/")
    parts = s.split("/")
    if any(x in OUTPUT_DIR_NAMES for x in parts):
        return True
    return any(x.startswith("_png_thr") for x in parts) or any(x.startswith("_official_run") for x in parts)

def _walk_dirs(root, maxdepth=4):
    """遍历 /kaggle/input，跳过 notebook 输出目录"""
    root = Path(root)
    if not root.exists():
        return []
    out = []
    base_len = len(root.parts)
    for dp, dns, _fns in os.walk(root):
        d = Path(dp)
        depth = len(d.parts) - base_len
        dns[:] = [x for x in dns if not _is_output_dir(d / x)]
        if depth >= maxdepth:
            dns[:] = []
        out.append(d)
    return out

def _image_files(root):
    """收集目录下的所有图片（跳过输出目录）"""
    out = []
    for d in _walk_dirs(root, maxdepth=6):
        try:
            names = os.listdir(d)
        except Exception:
            continue
        for f in names:
            if Path(f).suffix.lower() in IMG_EXT:
                out.append(d / f)
    return out

def find_root():
    if DATA_ROOT:
        return Path(DATA_ROOT)
    dirs = _walk_dirs("/kaggle/input", maxdepth=4)
    # (1) 优先：目录里直接有 sample_submission.csv
    for d in dirs:
        try:
            if (d / "sample_submission.csv").exists():
                return d
        except Exception:
            pass
    # (2) 退一步：目录下有 train / test 之类子目录
    for d in dirs:
        try:
            names = {x.lower() for x in os.listdir(d)}
        except Exception:
            continue
        if (("train" in names or "test" in names) and len(names) <= 40
                and any((d / x).is_dir() for x in names if x in ("train", "test"))):
            return d
    # (3) 兜底：第一个不是纯输出目录的
    for d in dirs:
        if str(d).rstrip("/") not in ("/kaggle/input", "") and not _is_output_dir(d):
            return d
    return Path(".")

def discover(root):
    """返回 (train_pairs, test_paths, sub_ids, sub_path)"""
    root = Path(root)
    files = _image_files(root)
    photos, bins = [], []
    for q in files:
        k = _classify(q)
        if k == "photo":
            photos.append(q)
        elif k == "binary":
            bins.append(q)
    p(f"[数据] 候选眼底图 {len(photos)} 张, 候选二值标签 {len(bins)} 张")

    sub_path, sub_ids = None, []
    for pat in ["sample_submission.csv", "*sample*submission*.csv"]:
        c = sorted(root.rglob(pat))
        if c:
            sub_path = c[0]; break
    if sub_path is None:
        c = sorted(glob.glob("/kaggle/input/*/sample_submission.csv"))
        sub_path = c[0] if c else None
    if sub_path is not None:
        df = pd.read_csv(sub_path, encoding="utf-8-sig", dtype=str)
        sub_ids = [str(x).strip() for x in df.iloc[:, 0].tolist()]
        p(f"[数据] sample_submission: {sub_path}  共 {len(sub_ids)} 行, 例: {sub_ids[:5]}")
    else:
        p("[数据] 未找到 sample_submission.csv，将从测试图文件名生成 Id")

    # ---- 训练图 <-> 标签 配对 ----
    bin_by_key = defaultdict(list)
    for b in bins:
        bin_by_key[_numkey(b)].append(b)

    def gt_score(img, b):
        sc = 0.0
        if _has(b, HINT_GT):
            sc += 3
        elif _has(b, HINT_BAD):
            sc -= 3
        if b.parent == img.parent:
            sc += 1
        if b.stem.startswith(img.stem) or img.stem.startswith(b.stem):
            sc += 0.5
        return sc

    pairs = {}
    for c in photos:
        cands = [b for b in bin_by_key.get(_numkey(c), []) if b != c]
        if not cands:
            continue
        cands.sort(key=lambda b: gt_score(c, b), reverse=True)
        if len(cands) == 1 or gt_score(c, cands[0]) >= 0:
            pairs[c] = cands[0]

    # ---- 手动指定目录 ----
    if TEST_DIR:
        test = [c for c in photos if str(c).lower().startswith(str(Path(TEST_DIR).resolve()).lower())]
    else:
        def test_score(c):
            s = 0.0
            parts = [q for q in str(c).lower().split(os.sep)]
            if any("test" in q for q in parts):
                s += 3
            if c not in pairs:
                s += 2.5
            if _numkey(c) in sub_ids:
                s += 0.7
            if any("train" in q for q in parts):
                s -= 3
            return s
        N = len(sub_ids) if sub_ids else 0
        scored = sorted(photos, key=test_score, reverse=True)
        if N:
            test = [c for c in scored if test_score(c) > 0][:N]
        else:
            test = [c for c in scored if c not in pairs and test_score(c) > 0]
            if not test:
                test = [c for c in scored if c not in pairs]

    if TRAIN_DIR:
        tdir = Path(TRAIN_DIR).resolve()
        pairs = {k: v for k, v in pairs.items() if str(k).lower().startswith(str(tdir).lower())}
    if GT_DIR:
        gdir = Path(GT_DIR)
        fixed = {}
        gts = [q for q in bins if str(q).lower().startswith(str(gdir).resolve()).lower()]
        for img in pairs:
            c = [g for g in gts if _numkey(g) == _numkey(img)]
            if c:
                fixed[img] = c[0]
        pairs = fixed

    tset = set(test)
    train_pairs = sorted([(c, g) for c, g in pairs.items() if c not in tset])
    test_sorted = sorted(test)

    p(f"[数据] 训练对 {len(train_pairs)} 组 | 测试图 {len(test_sorted)} 张 | 提交行数 {len(sub_ids)}")
    if len(train_pairs) < 5:
        p("!!! 训练图识别过少，请手动设置 TRAIN_DIR / GT_DIR / TEST_DIR 后重跑")
    if sub_ids and len(test_sorted) != len(sub_ids):
        p(f"!!! 测试图数({len(test_sorted)}) != 提交行数({len(sub_ids)})，将在生成提交时自动补齐/裁剪")
    for c, g in train_pairs[:5]:
        p(f"    img: {c.name:28s} -> gt: {g.name}")
    return train_pairs, test_sorted, sub_ids, sub_path


# ==========================================================================================
#  2. 预处理：绿通道 CLAHE / Frangi 血管增强 / FOV 视野掩膜
# ==========================================================================================
FEAT_CH = 4 if USE_FRANGI else 3

def fov_of(rgb):
    """眼底圆形视野掩膜"""
    g = rgb.max(axis=2).astype(np.float32)
    m = (g > 12).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    m = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=1)
    return m.astype(bool)

def frangi_map(gray_u8):
    try:
        from skimage.filters import frangi
        v = frangi(gray_u8.astype(np.float32) / 255.0, sigmas=np.arange(1, 5, 1.0), black_ridges=False)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(v / (np.percentile(v, 99.5) + 1e-8), 0, 1).astype(np.float32)
    except Exception as e:
        p(f"[warn] frangi 不可用({e})，用零通道代替")
        return np.zeros_like(gray_u8, dtype=np.float32)

def load_features(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    g = rgb[..., 1]
    gc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g).astype(np.float32) / 255.0
    chans = [gc, rgb[..., 0].astype(np.float32) / 255.0, rgb[..., 2].astype(np.float32) / 255.0]
    if USE_FRANGI:
        chans.append(frangi_map(g))
    return np.stack(chans, 2).astype(np.float32), fov_of(rgb)

def load_gt(path, fov):
    """读标签并自动判定极性（血管是少数类）"""
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0] if a.shape[2] < 3 else cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    vals = np.unique(a)
    if vals.size <= 2:
        b = (a == vals.max())
    else:
        b = a > (int(vals.min()) + int(vals.max())) // 2
    if b.shape != fov.shape:
        b = cv2.resize(b.astype(np.uint8), (fov.shape[1], fov.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    inside = b[fov]
    if inside.size and inside.mean() > 0.5:
        b = ~b
    return b

class Item:
    __slots__ = ("path", "name", "X", "fov", "y", "w", "fy", "fx", "vy", "vx", "ident")
    def __init__(self, path, X, fov, y=None, ident=None, w=None):
        self.path, self.name, self.X, self.fov, self.y, self.ident = str(path), Path(path).stem, X, fov, y, ident
        # w: 参与损失计算的区域（训练图 = 整个 FOV；伪标签图 = 高置信区域）
        self.w = fov.astype(np.uint8) if w is None else np.asarray(w, np.uint8)
        rs = np.random.RandomState(7)
        ys, xs = np.nonzero(fov)
        if ys.size > 40000:
            idx = rs.choice(ys.size, 40000, replace=False); ys, xs = ys[idx], xs[idx]
        if ys.size == 0:
            ys, xs = np.array([fov.shape[0] // 2]), np.array([fov.shape[1] // 2])
        self.fy, self.fx = ys, xs
        if y is not None and y.any():
            vy, vx = np.nonzero(y)
            if vy.size > 25000:
                idx = rs.choice(vy.size, 25000, replace=False); vy, vx = vy[idx], vx[idx]
            self.vy, self.vx = vy, vx
        else:
            self.vy = self.vx = None


# ==========================================================================================
#  3. 数据增广（旋转/缩放/平移 + 翻转 + 强度扰动）
# ==========================================================================================
def sample_patch(item, P, rng, p_vessel=0.55):
    fov = item.fov
    if item.vy is not None and rng.random() < p_vessel:
        k = rng.randint(len(item.vy))
        cy = float(item.vy[k]) + rng.randint(-P // 3, P // 3 + 1)
        cx = float(item.vx[k]) + rng.randint(-P // 3, P // 3 + 1)
    else:
        k = rng.randint(len(item.fy))
        cy, cx = float(item.fy[k]), float(item.fx[k])

    ang, sc = rng.uniform(-180, 180), rng.uniform(0.85, 1.20)
    M = cv2.getRotationMatrix2D((cx, cy), ang, sc)
    M[0, 2] += P / 2.0 - cx
    M[1, 2] += P / 2.0 - cy

    img = cv2.warpAffine(item.X, M, (P, P), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if item.y is not None:
        lab = np.dstack([item.y.astype(np.uint8), fov.astype(np.uint8), item.w])
        lab = cv2.warpAffine(lab, M, (P, P), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        y = (lab[..., 0] > 0).astype(np.float32)
        m = (lab[..., 2] > 0).astype(np.float32)
    else:
        y = np.zeros((P, P), np.float32)
        m = np.ones((P, P), np.float32)

    if rng.random() < 0.5:
        img, y, m = img[:, ::-1], y[:, ::-1], m[:, ::-1]
    if rng.random() < 0.5:
        img, y, m = img[::-1], y[::-1], m[::-1]
    img = np.ascontiguousarray(img, dtype=np.float32)

    img = img * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.5:
        img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(0.01, 0.05), img.shape).astype(np.float32)
    if rng.random() < 0.15:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(np.ascontiguousarray(img), (k, k), 0)
    return np.clip(img, 0, 1).astype(np.float32), y, m

def make_batch(items, B, P, rng):
    xs, ys, ms = [], [], []
    for _ in range(B):
        it = items[rng.randint(len(items))]
        a, b, c = sample_patch(it, P, rng)
        xs.append(np.ascontiguousarray(a.transpose(2, 0, 1)))
        ys.append(b[None]); ms.append(c[None])
    x = torch.from_numpy(np.stack(xs)).float().to(DEV, non_blocking=True)
    y = torch.from_numpy(np.stack(ys)).float().to(DEV, non_blocking=True)
    m = torch.from_numpy(np.stack(ms)).float().to(DEV, non_blocking=True)
    return x, y, m


# ==========================================================================================
#  4. 模型与损失
# ==========================================================================================
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sk = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        h = self.b2(self.c2(h))
        return F.relu(h + self.sk(x), inplace=True)

class UNet(nn.Module):
    def __init__(self, cin, base=32, depth=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.depth = depth
        self.enc = nn.ModuleList(); c = cin
        for ch in chs:
            self.enc.append(ResBlock(c, ch)); c = ch
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.dec.append(ResBlock(chs[i] + chs[i - 1], chs[i - 1]))
        self.out = nn.Conv2d(chs[0], 1, 1)
    def forward(self, x):
        feats, h = [], x
        for i, blk in enumerate(self.enc):
            h = blk(h)
            if i < self.depth:
                feats.append(h); h = F.max_pool2d(h, 2)
        for i, blk in enumerate(self.dec):
            skip = feats[self.depth - 1 - i]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, skip], 1))
        return self.out(h)

def soft_erode(x):
    a = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    b = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.min(a, b)

def soft_open(x):
    return F.max_pool2d(soft_erode(x), 3, 1, 1)

def soft_skel(x, iters=3):
    x1 = soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = soft_erode(x)
        x1 = soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def cl_dice_loss(pred, target, iters=3):
    sp, st = soft_skel(pred, iters), soft_skel(target, iters)
    tprec = (sp * target).sum() / (sp.sum() + 1e-6)
    tsens = (st * pred).sum() / (st.sum() + 1e-6)
    return 1 - 2 * tprec * tsens / (tprec + tsens + 1e-6)

def criterion(logits, y, m, cl_w=0.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * m).sum() / (m.sum() + 1e-6)
    pr = torch.sigmoid(logits) * m
    tg = y * m
    dice = 1 - (2 * (pr * tg).sum() + 1.0) / (pr.sum() + tg.sum() + 1.0)
    loss = 0.5 * bce + 0.5 * dice
    if cl_w > 0:
        loss = loss + cl_w * cl_dice_loss(pr, tg)
    return loss


# ==========================================================================================
#  5. 训练 / 推理
# ==========================================================================================
@torch.no_grad()
def predict_probs(models, X, fov, scales=(1.0,), tta=True):
    H, W, _ = X.shape
    acc = np.zeros((H, W), np.float32); n = 0
    for s in scales:
        Xi = X if abs(s - 1.0) < 1e-6 else cv2.resize(X, (int(round(W * s)), int(round(H * s))),
                                                     interpolation=cv2.INTER_LINEAR)
        h, w = Xi.shape[:2]
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        if ph or pw:
            Xi = cv2.copyMakeBorder(Xi, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        t = torch.from_numpy(np.ascontiguousarray(Xi.transpose(2, 0, 1)))[None].float().to(DEV)
        if tta:
            modes = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
            if TTA_ROT90:
                modes += [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        else:
            modes = [(0, 0, 0)]
        for fv, fh, rt in modes:
            tt = t
            if rt: tt = torch.rot90(tt, 1, [2, 3])
            if fv: tt = torch.flip(tt, [2])
            if fh: tt = torch.flip(tt, [3])
            o = 0
            for mdl in models:
                o = o + torch.sigmoid(mdl(tt))
            o = (o / len(models))[:, 0]
            if fh: o = torch.flip(o, [2])          # 逆变换：按相反顺序还原
            if fv: o = torch.flip(o, [1])
            if rt: o = torch.rot90(o, -1, [1, 2])
            o = o[:, :h, :w][0].float().cpu().numpy()
            if o.shape != (H, W):
                o = cv2.resize(o, (W, H), interpolation=cv2.INTER_LINEAR)
            acc += o; n += 1
    prob = acc / max(n, 1)
    return prob * fov.astype(np.float32)

def build_model():
    return UNet(FEAT_CH, BASE_CH).to(DEV)

# ---- 混合精度：某些 GPU(如 Turing/GTX16 系) 的 fp16 卷积会整片输出 NaN，必须自动探测 ----
AMP_DTYPE = None

def _amp_ctx():
    if AMP_DTYPE is None or DEV.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=AMP_DTYPE)

def _make_scaler():
    use = (AMP_DTYPE == torch.float16)
    try:
        return torch.amp.GradScaler("cuda", enabled=use)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use)

def amp_is_stable(items, dtype, steps=6):
    """用真实数据跑几步，检查 fp16/bf16 会不会出 NaN"""
    if not items:
        return False
    old = globals()["AMP_DTYPE"]
    globals()["AMP_DTYPE"] = dtype
    ok = True
    try:
        mdl = build_model(); mdl.train()
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-6)
        scaler = _make_scaler()
        rng = np.random.RandomState(1)
        for i in range(steps):
            x, y, m = make_batch(items, BATCH, PATCH, rng)
            with _amp_ctx():
                loss = criterion(mdl(x), y, m, CL_DICE_W)
            if not torch.isfinite(loss):
                ok = False; break
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        del mdl, opt
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        p(f"[AMP] {dtype} 测试异常: {e}")
        ok = False
    globals()["AMP_DTYPE"] = old
    return ok

def pick_amp(items):
    """返回所选精度；fp16 → bf16 → fp32 依次尝试"""
    global AMP_DTYPE
    if DEV.type != "cuda":
        AMP_DTYPE = None
        return AMP_DTYPE
    want = str(AMP_MODE).lower()
    if want in ("off", "fp32", "none"):
        AMP_DTYPE = None
    elif want == "fp16":
        AMP_DTYPE = torch.float16
    elif want == "bf16":
        AMP_DTYPE = torch.bfloat16
    else:
        AMP_DTYPE = None
        for dt, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            if amp_is_stable(items, dt):
                AMP_DTYPE = dt
                p(f"[AMP] 自动选择 {name} 混合精度")
                break
        if AMP_DTYPE is None:
            p("[AMP] fp16/bf16 都不可靠（可能出现 NaN），改用 fp32 训练（更稳，稍慢）")
    return AMP_DTYPE

def probe_speed(items, n=5):
    """先跑几步测速度（取中位数，避开首次 kernel 自动调优的毛刺）"""
    if not items:
        return 1.0
    set_seed(0)
    mdl = build_model()
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-5)
    scaler = _make_scaler()
    rng = np.random.RandomState(0)
    times = []
    for i in range(n + 2):
        x, y, m = make_batch(items, BATCH, PATCH, rng)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        with _amp_ctx():
            loss = criterion(mdl(x), y, m, CL_DICE_W)
        if torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        if i >= 2:
            times.append(time.time() - t)
    del mdl, opt
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return float(np.median(times)) if times else 1.0

def make_pseudo_items(te_items, probs_by_name, hi, lo):
    """把测试集高置信区域变成伪标签训练样本；不确定的区域不参与损失"""
    out = []
    for it in te_items:
        p = probs_by_name.get(it.name)
        if p is None:
            continue
        yp = (p >= hi)
        if yp.sum() < 200:                 # 半点血管都没预测出来，跳过
            continue
        w = (((p >= hi) | (p <= lo)) & it.fov).astype(np.uint8)
        if w.sum() < 0.3 * it.fov.sum():   # 置信区域太少，跳过
            continue
        out.append(Item(it.path, it.X, it.fov, yp, w=w))
    return out

def finetune_pseudo(models_by_fold, train_items_by_fold, pseudo_items, seed, epochs, iters):
    """用 (该折自己的训练图) + 伪标签图 混合微调；先深拷贝，保证"不合格可以原样丢弃"

    注意：train_items_by_fold 必须只包含该折的训练图（不含验证图），
          否则 OOF Dice 会被泄漏污染，兜底检查就失去意义了。
    """
    rng = np.random.RandomState(seed)
    out = {}
    for f, model0 in models_by_fold.items():
        model = copy.deepcopy(model0)          # 关键：不动原模型，坏了大不了丢掉
        mix = list(train_items_by_fold.get(f, [])) + list(pseudo_items)
        set_seed(seed + f)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=PSEUDO_LR, weight_decay=WD)
        scaler = _make_scaler()
        total = max(1, epochs * iters)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda st: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, st / total))))
        bad = 0
        for e in range(epochs):
            for _ in range(iters):
                x, y, m = make_batch(mix, BATCH, PATCH, rng)
                with _amp_ctx():
                    loss = criterion(model(x), y, m, CL_DICE_W)
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True); bad += 1; sched.step(); continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
        model.eval()
        out[f] = model
    return out

def train_fold(fold, tr_items, va_items, seed, epochs=None, iters=None):
    EPOCHS_F = int(EPOCHS if epochs is None else epochs)
    ITERS_F = int(ITERS if iters is None else iters)
    set_seed(seed)
    rng = np.random.RandomState(seed + 100 * fold)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total = max(1, EPOCHS_F * ITERS_F)
    warm = max(1, int(0.05 * total))

    def lr_fn(step):
        if step < warm:
            return (step + 1) / warm
        q = min(1.0, (step - warm) / max(1, total - warm))
        return MIN_LR / LR + (1 - MIN_LR / LR) * 0.5 * (1 + math.cos(math.pi * q))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    scaler = _make_scaler()
    bad = 0

    swa_state, swa_n = None, 0
    t0 = time.time()
    for ep in range(EPOCHS_F):
        model.train()
        run, nb = 0.0, 0
        for it in range(ITERS_F):
            x, y, m = make_batch(tr_items, BATCH, PATCH, rng)
            with _amp_ctx():
                logits = model(x)
                loss = criterion(logits, y, m, CL_DICE_W)
            if not torch.isfinite(loss):          # NaN/Inf 守卫：跳过这一步，绝不让权重被污染
                opt.zero_grad(set_to_none=True)
                bad += 1
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss.item()); nb += 1
        if bad > 0 and (ep + 1) % 10 == 0:
            p(f"  fold{fold} 注意: 累计 {bad} 步出现 NaN 已跳过")
        if (ep + 1) % 10 == 0 or ep == EPOCHS_F - 1:
            p(f"  fold{fold} ep{ep+1:3d}/{EPOCHS_F}  loss={run/max(nb,1):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
        if USE_SWA and EPOCHS_F >= 25 and ep >= EPOCHS_F - SWA_EPOCHS:
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state, swa_n = sd, 1
            else:
                for k in swa_state:
                    if swa_state[k].dtype.is_floating_point:
                        swa_state[k].mul_(swa_n / (swa_n + 1.0)).add_(sd[k].float(), alpha=1.0 / (swa_n + 1.0))
                swa_n += 1

    model.eval()
    probs = {}
    for it in va_items:
        probs[it.name] = predict_probs([model], it.X, it.fov, TTA_SCALES, tta=True)
    d = np.mean([dice_score((probs[it.name] >= 0.5) & it.fov, it.y) for it in va_items])
    p(f"  fold{fold} 验证 Dice@0.5 = {d:.4f}  (用时 {time.time()-t0:.0f}s)")

    if swa_state is not None and swa_n >= 2:
        try:
            swa_model = build_model()
            swa_model.load_state_dict(swa_state)
            swa_model.train()
            with torch.no_grad():
                for _ in range(24):            # 用训练分布重新统计 BN
                    x, y, m = make_batch(tr_items, BATCH, PATCH, rng)
                    with _amp_ctx():
                        swa_model(x)
            swa_model.eval()
            probs2 = {it.name: predict_probs([swa_model], it.X, it.fov, TTA_SCALES, tta=True)
                      for it in va_items}
            d2 = np.mean([dice_score((probs2[it.name] >= 0.5) & it.fov, it.y) for it in va_items])
            p(f"  fold{fold} SWA({swa_n}个快照) 验证 Dice@0.5 = {d2:.4f}")
            if d2 > d:
                p(f"  fold{fold} -> 采用 SWA 权重 (+{d2 - d:.4f})")
                model, probs, d = swa_model, probs2, d2
            del swa_model
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            p(f"  fold{fold} SWA 失败已跳过: {e}")
    return model, probs


# ==========================================================================================
#  6. 评价/后处理/编码
# ==========================================================================================
def dice_score(pred, gt, eps=1e-6):
    pred = np.asarray(pred, bool); gt = np.asarray(gt, bool)
    inter = np.count_nonzero(pred & gt)
    return float((2.0 * inter + eps) / (np.count_nonzero(pred) + np.count_nonzero(gt) + eps))

def binarize(prob, fov, thr, min_size=0):
    m = ((prob >= thr) & fov).astype(np.uint8)
    if min_size > 0 and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros(n, np.uint8)
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_size).astype(np.uint8)
        m = keep[lab]
    return m.astype(bool)

def rle_encode(mask01, order="C"):
    flat = np.asarray(mask01, np.uint8).flatten(order=order)
    if flat.size == 0 or not flat.any():
        return ""
    padded = np.concatenate(([0], flat, [0]))
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0]
    lengths = ends - starts + 1
    return " ".join(f"{s} {l}" for s, l in zip(starts.tolist(), lengths.tolist()))

def rle_decode(s, shape, order="C"):
    m = np.zeros(int(np.prod(shape)), np.uint8)
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s).strip()
    if s:
        v = np.array(s.split(), dtype=np.int64)
        for a, b in zip(v[0::2].tolist(), v[1::2].tolist()):
            m[a - 1:a - 1 + b] = 1
    return m.reshape(shape, order=order)

def detect_order(src):
    s = re.sub(r"\s+", "", src)
    if re.search(r"order=['\"]F['\"]", s) or ".T.flatten" in s or re.search(r"flatten\(order=.?F", s):
        return "F"
    return "C"

def to_png(vessel01):
    """血管区域写 255（官方脚本就是按 255 区域取 mask）"""
    v = np.where(vessel01, 255, 0) if PNG_VESSEL_WHITE else np.where(vessel01, 0, 255)
    return v.astype(np.uint8)

def imwrite_u(path, arr):
    """cv2.imwrite 遇到非 ASCII 路径会静默失败，这里做兜底"""
    path = str(path)
    try:
        if cv2.imwrite(path, arr):
            return True
    except Exception:
        pass
    try:
        Image.fromarray(arr).save(path)
        return True
    except Exception as e:
        p(f"[warn] 写图失败 {path}: {e}")
        return False


# ==========================================================================================
#  7. 提交：优先调用官方 segmentation_to_csv.py，失败则用内置编码器
# ==========================================================================================
def find_official_script(root):
    seen, out = set(), []
    for base in [str(root), "/kaggle/input", "."]:
        if not os.path.isdir(base):
            continue
        for q in Path(base).rglob("*to_csv*.py"):
            r = q.resolve()
            if r not in seen:
                seen.add(r); out.append(q)
    return out

def valid_sub(path, sub_ids):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    except Exception:
        return False
    if len(df) != len(sub_ids):
        return False
    if sub_ids and set(df.iloc[:, 0].str.strip()) != set(sub_ids):
        return False
    if df.shape[1] < 2:
        return False
    return bool(df.iloc[:, 1].str.len().max() > 0)

def run_official(script, png_dir, sub_ids, out_csv, masks_shapes):
    """把 png 放进脚本可能读取的各个目录，运行它，检查产物是否合规"""
    try:
        src = script.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    p(f"[提交] 发现官方脚本 {script}（RLE 顺序推断: {detect_order(src)}）")
    # 放到系统临时目录：这些中间图不值得出现在 Output 里（会让附件面板塞满上千个文件）
    work = Path(tempfile.mkdtemp(prefix="official_run_"))
    shutil.copy2(script, work / script.name)

    cand = {"pred", "preds", "prediction", "predictions", "predicted", "result", "results",
            "mask", "masks", "output", "outputs", "submission", "csv", "test_pred", "test"}
    for lit in re.findall(r"[\"']([^\"'\n]{1,60})[\"']", src):
        lit = lit.strip().strip("./")
        if lit and not lit.endswith(".py") and "." not in os.path.basename(lit):
            cand.add(lit)
    n_copy = 0
    for d in cand:
        try:
            dd = (work / d)
            dd.mkdir(parents=True, exist_ok=True)
            for f in Path(png_dir).glob("*.png"):
                shutil.copy2(f, dd / f.name); n_copy += 1
        except Exception:
            pass
    p(f"[提交] 已把预测图复制到 {len(cand)} 个候选目录，开始运行官方脚本 …")

    before = {q.resolve() for q in work.rglob("*.csv")}
    try:
        r = subprocess.run([sys.executable, script.name], cwd=str(work), capture_output=True,
                           text=True, timeout=900, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        tail = (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:]
        p("[提交] 脚本输出(节选):", tail.replace("\n", " ")[:1200])
    except Exception as e:
        p("[提交] 运行官方脚本失败:", e)
        return None

    csvs = [q for q in work.rglob("*.csv") if q.resolve() not in before]
    csvs = [q for q in csvs if "sample" not in q.name.lower()]
    csvs.sort(key=lambda q: (not q.name.lower().startswith("submission"), -q.stat().st_size))
    for q in csvs:
        if valid_sub(q, sub_ids):
            # 覆盖率自检：防止 0/255 极性反了
            try:
                df = pd.read_csv(q, encoding="utf-8-sig", dtype=str).fillna("")
                i = 0
                cov = []
                for _, row in df.iterrows():
                    sid = str(row.iloc[0]).strip()
                    if sid not in masks_shapes:
                        continue
                    shp = masks_shapes[sid]
                    m = rle_decode(row.iloc[1], shp, detect_order(src))
                    cov.append(m.mean())
                    i += 1
                    if i >= 5:
                        break
                if cov and float(np.mean(cov)) > 0.5:
                    p(f"[提交] 注意：官方脚本输出覆盖率 {np.mean(cov):.2%} (>50%)，"
                      f"说明脚本编码的是 255 区域，正在用反极性重跑 …")
                    return "INVERTED"
            except Exception as e:
                p("[提交] 覆盖率自检跳过:", e)
            shutil.copy2(q, out_csv)
            p(f"[提交] 已使用官方脚本产物 -> {out_csv}")
            shutil.rmtree(work, ignore_errors=True)
            return str(out_csv)
    shutil.rmtree(work, ignore_errors=True)
    return None

def build_submission_orderF(records, masks_shapes, sub_ids):
    """备用：另一种 RLE 扫描顺序（若榜上分数异常低可换这个）"""
    rows = []
    for sid in sub_ids:
        if sid in records:
            rows.append((sid, rle_encode(records[sid], order="F")))
        else:
            rows.append((sid, ""))
    df = pd.DataFrame({"Id": sub_ids, "Predicted": [r[1] for r in rows]})
    path = Path(OUT_DIR) / "submission_orderF.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    p(f"[提交] 备用文件(列优先 RLE) -> {path}")
    return path


# ==========================================================================================
#  7.5 RETUNE：不训练，直接用已存概率图重扫阈值
# ==========================================================================================
def find_probs_dirs():
    """返回概率图目录列表；多个目录会被做概率平均（跨版本集成）"""
    if RETUNE_PROB_DIRS:
        return [Path(d) for d in RETUNE_PROB_DIRS]
    if RETUNE_PROB_DIR:
        return [Path(RETUNE_PROB_DIR)]
    cnt = {}
    for q in Path("/kaggle/input").rglob("test_*.npy"):
        k = str(q.parent)
        cnt[k] = cnt.get(k, 0) + 1
    for extra in ["probs", OUT_DIR + "/probs"]:
        if os.path.isdir(extra) and glob.glob(extra + "/test_*.npy"):
            cnt[str(Path(extra))] = len(glob.glob(extra + "/test_*.npy"))
    dirs = [Path(k) for k, v in cnt.items() if v >= 3]
    dirs.sort(key=lambda d: -cnt[str(d)])
    if not dirs:
        raise RuntimeError("找不到 probs 目录：请在 Add Data 里把上一版 notebook 的 Output 挂上")
    return dirs if RETUNE_MERGE_ALL else dirs[:1]

def retune(root, te_items, sub_ids, id2item):
    pdirs = find_probs_dirs()
    p(f"[RETUNE] 找到 {len(pdirs)} 个概率图目录:")
    for d in pdirs:
        p(f"    {d}  ({len(list(d.glob('test_*.npy')))} 个文件)")
    if len(pdirs) > 1:
        p("[RETUNE] 将对它们做概率平均（跨版本集成）")
    probs_test, fovs_test, used = {}, {}, {}
    for sid in sub_ids:
        it = id2item.get(sid)
        if it is None:
            continue
        acc, cnt = None, 0
        for d in pdirs:
            f = d / f"test_{sid}.npy"
            if not f.exists():
                continue
            pr = np.load(f).astype(np.float32)
            if pr.shape != it.fov.shape:
                continue
            acc = pr if acc is None else acc + pr
            cnt += 1
        if acc is None:
            continue
        probs_test[sid] = acc / max(cnt, 1)
        fovs_test[sid] = it.fov
        used[sid] = cnt
    if not probs_test:
        raise RuntimeError("没有可用的概率图，检查 Id 是否与 sample_submission 对得上")
    mix = sorted(set(used.values()))
    p(f"[RETUNE] 载入 {len(probs_test)} 张测试图概率；每张参与融合的目录数 {mix}")

    thrs = sorted(set([float(t) for t in RETUNE_THRS] + [float(RETUNE_MAIN_THR)]))
    rows = []
    for t in thrs:
        rec = {s_: binarize(probs_test[s_], fovs_test[s_], t, RETUNE_MIN_SIZE) for s_ in probs_test}
        shp = {s_: rec[s_].shape for s_ in rec}
        cv = float(np.mean([rec[s_].mean() for s_ in rec]))
        tag = "thr%03d" % int(round(t * 100))
        out_csv = Path(OUT_DIR) / f"submission_{tag}.csv"
        d = Path(tempfile.mkdtemp(prefix="png_%s_" % tag))
        for sid, m01 in rec.items():
            imwrite_u(d / f"{sid}.png", to_png(m01))
            it = id2item.get(sid)
            if it is not None and Path(it.path).stem != sid:
                imwrite_u(d / f"{Path(it.path).stem}.png", to_png(m01))
        ok = False
        for script in find_official_script(root):
            if run_official(script, d, sub_ids, out_csv, shp) and os.path.exists(out_csv):
                ok = True
                break
        if not ok:
            pd.DataFrame({"Id": sub_ids,
                          "Predicted": [rle_encode(rec.get(s_, np.zeros((8, 8), bool)), order="C")
                                        for s_ in sub_ids]}).to_csv(out_csv, index=False, encoding="utf-8")
        rows.append((t, cv, out_csv.name))
        p(f"[RETUNE] thr={t:.2f}  平均血管占比 {cv:.2%}  ->  {out_csv.name}")
        if abs(t - float(RETUNE_MAIN_THR)) < 1e-6:
            shutil.copy2(out_csv, Path(OUT_DIR) / "submission.csv")

    p("\n[RETUNE] 全部候选（按血管占比排序）:")
    for t, cv, nm in sorted(rows, key=lambda r: r[1]):
        p(f"    thr={t:.2f}  占比 {cv:.2%}  {nm}")
    p(f"[RETUNE] 主提交 = thr {RETUNE_MAIN_THR} -> {Path(OUT_DIR)/'submission.csv'}")
    p("[RETUNE] 提示：每天 5 次提交额度，按占比从低到高逐个交，找出分数最高的阈值")


# ==========================================================================================
#  8. 主流程
# ==========================================================================================


def map_ids(te_items, sub_ids):
    """把测试图文件名对应到 sample_submission 的 Id"""
    used, left = {}, []
    for it in te_items:
        stem = Path(it.path).stem
        cand = None
        if stem in sub_ids:
            cand = stem
        else:
            for d in re.findall(r"\d+", stem):
                if d in sub_ids and d not in used:
                    cand = d; break
            if cand is None:
                for d in re.findall(r"\d+", stem):
                    d2 = d.lstrip("0") or "0"
                    if d2 in sub_ids and d2 not in used:
                        cand = d2; break
        if cand:
            used[cand] = it
        else:
            left.append(it)
    free = [s for s in sub_ids if s not in used]
    for s, it in zip(free, left):
        used[s] = it
    if len(used) != len(sub_ids):
        p(f"[警告] Id 映射不完整: {len(used)}/{len(sub_ids)}")
    return used


