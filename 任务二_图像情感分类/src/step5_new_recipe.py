# ============================================================
# 任务二：图像情感分类
# 本文件在做什么：新配方训练：160 像素、22 轮、RandAugment 与 EMA 权重平均，标签 cnxs2 / cnxT2。
# 需要多久：ConvNeXt-Small 约 210 分钟，ConvNeXt-Tiny 约 133 分钟。
# 输出什么：新配方骨干的验证与测试概率、逐折准确率。
# ============================================================

# ============================================================
# 阶段 B（修正版）：convnext_small 新配方 5 折
# 160px / 22轮 / RandAugment / EMA，标签 cnxs2
# 修复：TTA 前强制 model.float() 并禁用 autocast（AMP 权重缓存导致的 dtype 冲突）
# 兜底：TTA 失败则用单尺度概率存盘，不浪费该折
# ============================================================
import os, glob, time, random, copy, gc, shutil, zipfile, traceback
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import convnext_small
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
USE_AMP = True
try:    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
except Exception: scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
def autocast_ctx(enabled=True):
    try:    return torch.amp.autocast('cuda', enabled=enabled)
    except Exception: return torch.cuda.amp.autocast(enabled=enabled)

ROOT = "/kaggle/input/competitions/2026task2/fer_data/fer_data"
TRAIN_DIR, TEST_DIR = ROOT + "/train", ROOT + "/test"
SUB_PATH = "/kaggle/input/competitions/2026task2/submission.csv"
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
WORK = "/kaggle/working"

RUN_TAG = "cnxs2"
IMG, EPOCHS, BS = 160, 22, 80
LR, WD, NW = 6e-4, 5e-4, 2
EMA_DECAY = 0.999
NFOLD = 5
TIME_BUDGET_MIN = 260
OLD = ['resnet18', 'efficientnet_b0', 'convnext_tiny', 'convnext_small', 'efficientnet_b1']

print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "没开GPU", flush=True)

# ---------- 恢复存档 ----------
n_restored = 0
for pat in ("pv_*.npy", "pt_*.npy"):
    cands = []
    for d in range(1, 5):
        cands += glob.glob("/kaggle/input/" + "*/" * d + pat)
    cands += glob.glob("/kaggle/input/datasets/**/" + pat, recursive=True)
    for p in cands:
        dst = os.path.join(WORK, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy(p, dst); n_restored += 1
for pat in ("validx_*.npy", "test_names.npy"):
    for p in glob.glob("/kaggle/input/datasets/**/" + pat, recursive=True):
        dst = os.path.join(WORK, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy(p, dst); n_restored += 1
print("从 Input 恢复的中间文件:", n_restored, flush=True)

# ---------- 数据与完整性检查 ----------
files, labels = [], []
for i, c in enumerate(CLASSES):
    for f in sorted(os.listdir(os.path.join(TRAIN_DIR, c))):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(os.path.join(TRAIN_DIR, c, f)); labels.append(i)
labels = np.array(labels)
te_paths = sorted(p for p in glob.glob(TEST_DIR + "/*") if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
te_names = [os.path.basename(p) for p in te_paths]
print("训练集 %d 张 / 测试集 %d 张" % (len(files), len(te_paths)), flush=True)
if len(files) != 28275 or len(te_paths) != 7065:
    raise SystemExit("！！数据不完整（应为 28275 / 7065，实际 %d / %d），请重启会话后重跑"
                     % (len(files), len(te_paths)))

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
            T.RandomResizedCrop(img, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(), T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    return T.Compose([T.Grayscale(3), T.Resize((img, img)), T.ToTensor(), T.Normalize(MEAN, STD)])

def make_model():
    m = convnext_small(weights='IMAGENET1K_V1')
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(CLASSES))
    return m.to(DEVICE)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for pe, pm in zip(self.ema.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for be, bm in zip(self.ema.buffers(), model.buffers()):
            be.copy_(bm)

@torch.no_grad()
def predict_probs(model, paths, scales, hflip):
    """关键修复：强制 float32 + 关闭 autocast，避免 AMP 缓存半精度权重导致的 dtype 冲突"""
    model = model.float().eval()
    acc = None
    for s in scales:
        dl = DataLoader(FerSet(paths, None, make_tf(s, False)), batch_size=256, shuffle=False, num_workers=0)
        parts = []
        for x in dl:
            x = x.to(DEVICE)
            p = torch.softmax(model(x), 1)
            if hflip:
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])), 1)) / 2
            parts.append(p.float().cpu().numpy())
        sp = np.concatenate(parts)
        acc = sp if acc is None else acc + sp
    return acc / len(scales)

def save_ckpt(tag=""):
    zp = os.path.join(WORK, "ckpt_all.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in ("pv_*.npy", "pt_*.npy", "validx_*.npy", "test_*.npy", "submission.csv"):
            for p in sorted(glob.glob(os.path.join(WORK, pat))):
                z.write(p, os.path.basename(p))
    print("   [存档] ckpt_all.zip 已更新 (%.1f MB) %s" % (os.path.getsize(zp) / 1e6, tag), flush=True)

skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
splits = list(skf.split(files, labels))
for k in range(NFOLD):
    np.save(WORK + "/validx_f%d.npy" % k, splits[k][1])
np.save(WORK + "/test_names.npy", np.array(te_names, dtype=object))

T0 = time.time()
for k, (tr_idx, va_idx) in enumerate(splits):
    fp_val, fp_test = WORK + "/pv_%s_f%d.npy" % (RUN_TAG, k), WORK + "/pt_%s_f%d.npy" % (RUN_TAG, k)
    if os.path.exists(fp_val) and os.path.exists(fp_test):
        print("跳过已完成: %s fold %d" % (RUN_TAG, k), flush=True); continue
    el = (time.time() - T0) / 60
    if el > TIME_BUDGET_MIN:
        print("!! 已用 %.0f 分钟，超过预算，停止开新折" % el, flush=True); break
    print("\n" + "=" * 60 + "\n%s | fold %d/%d  (累计 %.0f 分钟)\n" % (RUN_TAG, k + 1, NFOLD, el) + "=" * 60, flush=True)
    try:
        random.seed(SEED + k); np.random.seed(SEED + k)
        torch.manual_seed(SEED + k); torch.cuda.manual_seed_all(SEED + k)
        tr_f = [files[i] for i in tr_idx]; tr_y = labels[tr_idx]
        va_f = [files[i] for i in va_idx]; va_y = labels[va_idx]
        tr_dl = DataLoader(FerSet(tr_f, tr_y, make_tf(IMG, True)), batch_size=BS, shuffle=True,
                           num_workers=NW, pin_memory=True, drop_last=True, persistent_workers=True)
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=LR / 2, weight_decay=WD)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(tr_dl), pct_start=0.15)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        ema = ModelEMA(model, EMA_DECAY)
        te0 = time.time()
        for ep in range(1, EPOCHS + 1):
            model.train(); tot = cor = 0; ls = 0.0
            for x, y in tr_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with autocast_ctx():
                    logits = model(x); loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                ema.update(model)
                ls += loss.item() * y.size(0)
                cor += (logits.argmax(1) == y).sum().item(); tot += y.size(0)
            print("  ep %2d | loss %.4f | train %.4f | %.0fs" % (ep, ls / tot, cor / tot, time.time() - te0), flush=True)
            te0 = time.time()
        # 选 raw / EMA
        raw_v = predict_probs(model, va_f, [IMG], False)
        ema_v = predict_probs(ema.ema, va_f, [IMG], False)
        a_raw = float((raw_v.argmax(1) == va_y).mean())
        a_ema = float((ema_v.argmax(1) == va_y).mean())
        print("  单尺度验证: raw %.4f | EMA %.4f" % (a_raw, a_ema), flush=True)
        use = ema.ema if a_ema >= a_raw else model
        pick = "EMA" if a_ema >= a_raw else "raw"
        try:
            vp = predict_probs(use, va_f, [IMG, int(IMG * 1.15)], True)
            tp = predict_probs(use, te_paths, [IMG, int(IMG * 1.15)], True)
            print("  TTA 成功", flush=True)
        except Exception:
            print("!! TTA 失败，退回单尺度概率（该折不浪费）：", flush=True); traceback.print_exc()
            vp = ema_v if a_ema >= a_raw else raw_v
            tp = predict_probs(use, te_paths, [IMG], False)
        np.save(fp_val, vp); np.save(fp_test, tp)
        print("fold %d 采用 %s | 验证 %.4f" % (k, pick, (vp.argmax(1) == va_y).mean()), flush=True)
        del model, tr_dl, use; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        print("!! 这一折失败：", flush=True); traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache(); continue
    save_ckpt("(%s fold %d 完成)" % (RUN_TAG, k))

# ---------- 融合 ----------
print("\n" + "=" * 60 + "\n汇总（累计 %.0f 分钟）\n" % ((time.time() - T0) / 60) + "=" * 60, flush=True)
def load_oof(bb):
    oof = np.full((len(files), len(CLASSES)), np.nan)
    for k in range(NFOLD):
        p = WORK + "/pv_%s_f%d.npy" % (bb, k)
        if os.path.exists(p):
            oof[np.load(WORK + "/validx_f%d.npy" % k)] = np.load(p)
    return oof

ALL = OLD + [RUN_TAG]
avail, blend = [], 0
for bb in ALL:
    tps = [WORK + "/pt_%s_f%d.npy" % (bb, k) for k in range(NFOLD)]
    tps = [p for p in tps if os.path.exists(p)]
    if not tps:
        print("缺少", bb, "跳过", flush=True); continue
    oof = load_oof(bb); m = ~np.isnan(oof).any(1)
    print("%-16s %d/5 折 | 覆盖 %.0f%% | 单模型 acc=%.4f"
          % (bb, len(tps), 100 * m.mean(), (oof[m].argmax(1) == labels[m]).mean()), flush=True)
    blend = blend + np.mean([np.load(p) for p in tps], axis=0)
    avail.append(bb)

if avail:
    blend = blend / len(avail)
    mask = np.ones(len(files), bool)
    for bb in avail:
        mask &= ~np.isnan(load_oof(bb)).any(1)
    if mask.sum() > 1000:
        ob = np.mean([load_oof(bb)[mask] for bb in avail], axis=0)
        print("\n等权融合(%d 个) OOF: %.4f" % (len(avail), (ob.argmax(1) == labels[mask]).mean()), flush=True)
        only_old = [b for b in avail if b in OLD]
        if only_old:
            ob_old = np.mean([load_oof(bb)[mask] for bb in only_old], axis=0)
            print("（对照）旧骨干 OOF: %.4f" % (ob_old.argmax(1) == labels[mask]).mean(), flush=True)
        print(classification_report(labels[mask], ob.argmax(1), target_names=CLASSES, digits=3))
    sub = pd.read_csv(SUB_PATH)
    name2pred = dict(zip(te_names, blend.argmax(1)))
    sub["Emotion"] = sub["ID"].map(lambda n: int(name2pred[n]))
    sub.to_csv(WORK + "/submission.csv", index=False)
    print("已保存 submission.csv | 骨干:", avail, flush=True)

save_ckpt("(全部完成)")
print("\n结束，累计 %.0f 分钟 → 下载 ckpt_all.zip 和 submission.csv" % ((time.time() - T0) / 60), flush=True)
