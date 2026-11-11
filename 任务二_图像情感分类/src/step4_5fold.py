# ============================================================
# 任务二：图像情感分类
# 本文件在做什么：5 折交叉验证：多骨干统一流程，逐折保存验证集与测试集概率，支持断点续跑与逐折打包。
# 需要多久：每个骨干约 50~60 分钟（5 折 × 14 轮）。
# 输出什么：pv_*/pt_*.npy 概率矩阵、OOF 指标、ckpt_all.zip。
# ============================================================

# ============================================================
# 第 9 步（保守版）：补完 convnext_small 第 5 折 + 训练 efficientnet_b1，然后融合所有骨干
# 安全设计：
#   1) 时间预算 140 分钟 —— 超时立刻停止训练，用已完成的部分出结果（保证输出非空）
#   2) AMP 报错自动回退 FP32 重跑该折
#   3) 每折结束都重写 ckpt_all.zip
# 前提：把 ckpt_v10.zip 上传成 Dataset 并 Add Input
# ============================================================
import os, glob, time, random, copy, gc, shutil, zipfile, traceback
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import (resnet18, resnet34, efficientnet_b0, efficientnet_b1,
                               convnext_tiny, convnext_small, densenet121)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "没开GPU！", flush=True)

ROOT = "/kaggle/input/competitions/2026task2/fer_data/fer_data"
TRAIN_DIR, TEST_DIR = ROOT + "/train", ROOT + "/test"
SUB_PATH = "/kaggle/input/competitions/2026task2/submission.csv"
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
WORK = "/kaggle/working"

# ---------------- 从挂载的 Input 恢复 ----------------
n_restored = 0
for p in glob.glob("/kaggle/input/*/*.zip") + glob.glob("/kaggle/input/*/*/*.zip"):
    try:
        with zipfile.ZipFile(p) as z:
            for nm in z.namelist():
                base = os.path.basename(nm)
                dst = os.path.join(WORK, base)
                if base.endswith(".npy") and not os.path.exists(dst):
                    with z.open(nm) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
                    n_restored += 1
    except Exception as e:
        print("跳过第三方文件", p, repr(e), flush=True)
for pat in ("pv_*.npy", "pt_*.npy"):
    for p in glob.glob("/kaggle/input/*/" + pat) + glob.glob("/kaggle/input/*/*/" + pat):
        dst = os.path.join(WORK, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy(p, dst); n_restored += 1
print("从 Input 恢复的中间文件:", n_restored, flush=True)
if n_restored == 0 and not os.path.exists(WORK + "/pt_resnet18_f0.npy"):
    raise SystemExit("！！没找到存档：请先上传 ckpt_v10.zip 成 Dataset 并 Add Input")

# ---------------- 配置 ----------------
ALL_BACKBONES = ['resnet18', 'efficientnet_b0', 'convnext_tiny', 'convnext_small', 'efficientnet_b1']
TRAIN_NOW     = ['convnext_small', 'efficientnet_b1']   # 只训这两个（其余从存档恢复）
TIME_BUDGET_MIN = 140          # 超过这个时间就不再开新折，直接用已有结果出提交
NFOLD, EPOCHS, IMG, BS, LR, WD = 5, 14, 128, 128, 6e-4, 1e-4
BS_BY_BB = {'convnext_small': 96}
NW = 2                         # 回到 Version 9/10 那个成功跑完的配置
TTA_SCALES = [IMG, int(IMG * 1.15)]
USE_AMP = True

try:
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
def autocast_ctx(enabled):
    try:    return torch.amp.autocast('cuda', enabled=enabled)
    except Exception: return torch.cuda.amp.autocast(enabled=enabled)

# ---------------- 数据 ----------------
files, labels = [], []
for i, c in enumerate(CLASSES):
    for f in sorted(os.listdir(os.path.join(TRAIN_DIR, c))):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(os.path.join(TRAIN_DIR, c, f)); labels.append(i)
labels = np.array(labels)
te_paths = sorted(p for p in glob.glob(TEST_DIR + "/*") if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
te_names = [os.path.basename(p) for p in te_paths]
print("训练集 %d 张 / 测试集 %d 张" % (len(files), len(te_paths)), flush=True)

class FerSet(Dataset):
    def __init__(self, files, labels=None, tf=None):
        self.files, self.labels, self.tf = files, labels, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = self.tf(Image.open(self.files[i]).convert("L"))
        return x if self.labels is None else (x, self.labels[i])

def make_tf(img, is_train):
    if is_train:
        return T.Compose([
            T.Grayscale(3),
            T.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(12),
            T.ColorJitter(brightness=0.25, contrast=0.25),
            T.ToTensor(), T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    return T.Compose([T.Grayscale(3), T.Resize((img, img)), T.ToTensor(), T.Normalize(MEAN, STD)])

def make_model(name):
    if name == 'resnet18':
        m = resnet18(weights='IMAGENET1K_V1'); nf = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(nf, len(CLASSES)))
    elif name == 'resnet34':
        m = resnet34(weights='IMAGENET1K_V1'); nf = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(nf, len(CLASSES)))
    elif name in ('efficientnet_b0', 'efficientnet_b1'):
        m = (efficientnet_b0 if name == 'efficientnet_b0' else efficientnet_b1)(weights='IMAGENET1K_V1')
        nf = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(nf, len(CLASSES))
    elif name in ('convnext_tiny', 'convnext_small'):
        m = (convnext_tiny if name == 'convnext_tiny' else convnext_small)(weights='IMAGENET1K_V1')
        nf = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(nf, len(CLASSES))
    elif name == 'densenet121':
        m = densenet121(weights='IMAGENET1K_V1'); nf = m.classifier.in_features
        m.classifier = nn.Linear(nf, len(CLASSES))
    else:
        raise ValueError(name)
    return m.to(DEVICE)

@torch.no_grad()
def tta_predict(model, paths, amp):
    model.eval()
    acc = None
    for s in TTA_SCALES:
        dl = DataLoader(FerSet(paths, None, make_tf(s, False)), batch_size=256, shuffle=False, num_workers=0)
        parts = []
        for x in dl:
            x = x.to(DEVICE)
            with autocast_ctx(amp):
                p = (torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, dims=[3])), 1)) / 2
            parts.append(p.float().cpu().numpy())
        sp = np.concatenate(parts)
        acc = sp if acc is None else acc + sp
    return acc / len(TTA_SCALES)

def save_ckpt(tag=""):
    zp = os.path.join(WORK, "ckpt_all.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in ("pv_*.npy", "pt_*.npy", "validx_*.npy", "test_*.npy", "oof_blend.npy", "submission.csv"):
            for p in sorted(glob.glob(os.path.join(WORK, pat))):
                z.write(p, os.path.basename(p))
    print("   [存档] ckpt_all.zip 已更新 (%.1f MB) %s" % (os.path.getsize(zp) / 1e6, tag), flush=True)

def train_fold(bb, k, tr_idx, va_idx, amp):
    bs = BS_BY_BB.get(bb, BS)
    tr_f = [files[i] for i in tr_idx]; tr_y = labels[tr_idx]
    va_f = [files[i] for i in va_idx]; va_y = labels[va_idx]
    tr_dl = DataLoader(FerSet(tr_f, tr_y, make_tf(IMG, True)), batch_size=bs, shuffle=True,
                       num_workers=NW, pin_memory=True, drop_last=True, persistent_workers=True)
    va_dl = DataLoader(FerSet(va_f, va_y, make_tf(IMG, False)), batch_size=256, shuffle=False,
                       num_workers=NW, pin_memory=True, persistent_workers=True)
    model = make_model(bb)
    opt = torch.optim.AdamW(model.parameters(), lr=LR / 2, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(tr_dl), pct_start=0.25)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    sc = scaler if amp else torch.cuda.amp.GradScaler(enabled=False)
    best_acc, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        model.train(); t0 = time.time(); tot = cor = 0; ls = 0.0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(amp):
                logits = model(x); loss = crit(logits, y)
            sc.scale(loss).backward()
            sc.step(opt); sc.update(); sched.step()
            ls += loss.item() * y.size(0)
            cor += (logits.argmax(1) == y).sum().item(); tot += y.size(0)
        model.eval(); vp = []
        with torch.no_grad():
            for x, y in va_dl:
                with autocast_ctx(amp):
                    o = model(x.to(DEVICE))
                vp.append(torch.softmax(o.float(), 1).cpu().numpy())
        vp = np.concatenate(vp)
        acc = (vp.argmax(1) == va_y).mean()
        print("  ep %2d | loss %.4f train %.4f | val %.4f | %.0fs%s"
              % (ep, ls / tot, cor / tot, acc, time.time() - t0, "" if amp else "  [FP32]"), flush=True)
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    vp = tta_predict(model, va_f, amp); tp = tta_predict(model, te_paths, amp)
    del model, tr_dl, va_dl; gc.collect(); torch.cuda.empty_cache()
    return vp, tp, best_acc

# ---------------- 逐折训练 ----------------
skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
splits = list(skf.split(files, labels))
for k in range(NFOLD):
    np.save(WORK + "/validx_f%d.npy" % k, splits[k][1])
np.save(WORK + "/test_names.npy", np.array(te_names, dtype=object))

T0 = time.time()
stop_all = False
for bb in TRAIN_NOW:
    if stop_all: break
    for k, (tr_idx, va_idx) in enumerate(splits):
        fp_val, fp_test = WORK + "/pv_%s_f%d.npy" % (bb, k), WORK + "/pt_%s_f%d.npy" % (bb, k)
        if os.path.exists(fp_val) and os.path.exists(fp_test):
            print("跳过已完成:", bb, "fold", k, flush=True); continue
        el = (time.time() - T0) / 60
        if el > TIME_BUDGET_MIN:
            print("!! 已用 %.0f 分钟，超过时间预算 %d 分钟，停止开新折（用已有结果出提交）"
                  % (el, TIME_BUDGET_MIN), flush=True)
            stop_all = True; break
        print("\n" + "=" * 60 + "\n%s | fold %d/%d  (累计 %.0f 分钟)\n" % (bb, k + 1, NFOLD, el) + "=" * 60, flush=True)
        ok = False
        for amp in ([True, False] if USE_AMP else [False]):
            try:
                vp, tp, best = train_fold(bb, k, tr_idx, va_idx, amp)
                np.save(fp_val, vp); np.save(fp_test, tp)
                print("fold %d 最佳 val %.4f（含TTA %.4f）已存盘%s"
                      % (k, best, (vp.argmax(1) == labels[va_idx]).mean(),
                         "" if amp else "  [FP32 重跑]"), flush=True)
                ok = True; break
            except Exception:
                print("!! 用 %s 跑这一折失败：%s" % ("AMP" if amp else "FP32", "见下方回溯"), flush=True)
                traceback.print_exc()
                gc.collect(); torch.cuda.empty_cache()
        if not ok:
            print("!! 这一折放弃了，继续下一折", flush=True); continue
        save_ckpt("(%s fold %d 完成)" % (bb, k))

# ---------------- 汇总：所有可用骨干，等权融合 ----------------
print("\n" + "=" * 60 + "\n汇总（累计 %.0f 分钟）\n" % ((time.time() - T0) / 60) + "=" * 60, flush=True)
def load_oof(bb):
    oof = np.full((len(files), len(CLASSES)), np.nan)
    for k in range(NFOLD):
        p = WORK + "/pv_%s_f%d.npy" % (bb, k)
        if os.path.exists(p):
            oof[np.load(WORK + "/validx_f%d.npy" % k)] = np.load(p)
    return oof

avail, blend_test = [], 0
for bb in ALL_BACKBONES:
    tp_list = [WORK + "/pt_%s_f%d.npy" % (bb, k) for k in range(NFOLD)]
    tp_list = [p for p in tp_list if os.path.exists(p)]
    if not tp_list:
        print("缺少", bb, "，跳过", flush=True); continue
    oof = load_oof(bb)
    m = ~np.isnan(oof).any(1)
    acc = (oof[m].argmax(1) == labels[m]).mean() if m.sum() else float('nan')
    print("%-16s 可用 %d/5 折 | 覆盖 %.0f%% | 单模型 acc=%.4f"
          % (bb, len(tp_list), 100 * m.mean(), acc), flush=True)
    blend_test = blend_test + np.mean([np.load(p) for p in tp_list], axis=0)
    avail.append(bb)

if avail:
    blend_test = blend_test / len(avail)
    mask = np.ones(len(files), bool)
    for bb in avail:
        mask &= ~np.isnan(load_oof(bb)).any(1)
    print("\n所有骨干都有 OOF 的样本: %d (%.0f%%)" % (mask.sum(), 100 * mask.mean()), flush=True)
    if mask.sum() > 1000:
        oof_blend = np.mean([load_oof(bb)[mask] for bb in avail], axis=0)
        print("等权融合(%d 个骨干) 在该子集 OOF 准确率: %.4f"
              % (len(avail), (oof_blend.argmax(1) == labels[mask]).mean()), flush=True)
        print(classification_report(labels[mask], oof_blend.argmax(1), target_names=CLASSES, digits=3))
        np.save(WORK + "/oof_blend.npy", oof_blend)

    sub = pd.read_csv(SUB_PATH)
    name2pred = dict(zip(te_names, blend_test.argmax(1)))
    sub["Emotion"] = sub["ID"].map(lambda n: int(name2pred[n]))
    sub.to_csv(WORK + "/submission.csv", index=False)
    print("已保存提交文件:", WORK + "/submission.csv", "| 用到的骨干:", avail, flush=True)

save_ckpt("(全部完成)")
print("\n结束，累计 %.0f 分钟。→ 去 Output 下载 ckpt_all.zip 和 submission.csv" % ((time.time() - T0) / 60), flush=True)
