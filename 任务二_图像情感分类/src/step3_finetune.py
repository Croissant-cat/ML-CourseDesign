# ============================================================
# 任务二：图像情感分类
# 本文件在做什么：单模型微调：ResNet18 加载 ImageNet 权重，输入 112 像素，12 轮 OneCycle。
# 需要多久：约 8 分钟（12 轮 × 38 秒）。
# 输出什么：最佳验证准确率、TTA 验证准确率、submission.csv。
# ============================================================

# ============================================================
# 第 3 步A：ImageNet 预训练 ResNet18 微调（单折，约 20 分钟）
# 对比第 2 步的小 CNN：验证准确率预期从 0.65 提升到 0.69~0.72
# ============================================================
import os, glob, time, random, copy
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

ROOT = "/kaggle/input/competitions/2026task2/fer_data/fer_data"
TRAIN_DIR, TEST_DIR = ROOT + "/train", ROOT + "/test"
SUB_PATH = "/kaggle/input/competitions/2026task2/submission.csv"

# 训练顺序 = 提交编号：0愤怒 1恐惧 2快乐 3悲伤 4惊讶 5中性
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

IMG = 112                      # 输入分辨率（48 -> 112）
MEAN = [0.485, 0.456, 0.406]   # ImageNet 统计量
STD  = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Grayscale(3),                                     # 单通道灰度 -> 3 通道
    T.RandomResizedCrop(IMG, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(12),
    T.ColorJitter(brightness=0.25, contrast=0.25),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([
    T.Grayscale(3),
    T.Resize((IMG, IMG)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# ---------------- 数据 ----------------
files, labels = [], []
for i, c in enumerate(CLASSES):
    for f in sorted(os.listdir(os.path.join(TRAIN_DIR, c))):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(os.path.join(TRAIN_DIR, c, f)); labels.append(i)
tr_f, va_f, tr_y, va_y = train_test_split(files, labels, test_size=0.1, random_state=SEED, stratify=labels)
print("训练 %d / 验证 %d" % (len(tr_f), len(va_f)))

class FerSet(Dataset):
    def __init__(self, files, labels=None, tf=None):
        self.files, self.labels, self.tf = files, labels, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = self.tf(Image.open(self.files[i]).convert("L"))
        return x if self.labels is None else (x, self.labels[i])

tr_dl = DataLoader(FerSet(tr_f, tr_y, train_tf), batch_size=128, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
va_dl = DataLoader(FerSet(va_f, va_y, eval_tf),  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# ---------------- 模型 ----------------
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)   # 下载 ImageNet 预训练权重（需要 Internet 开着）
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(DEVICE)

EPOCHS = 12
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=EPOCHS * len(tr_dl), pct_start=0.25)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)

def tta_probs(paths, bs=256):
    """水平翻转 TTA：原图和镜像图的概率取平均"""
    model.eval()
    dl = DataLoader(FerSet(paths, None, eval_tf), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    out = []
    with torch.no_grad():
        for x in dl:
            x = x.to(DEVICE)
            p = torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, dims=[3])), 1)
            out.append((p / 2).cpu().numpy())
    return np.concatenate(out)

best_acc, best_state = 0.0, None
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); tot = cor = 0; loss_sum = 0.0
    for x, y in tqdm(tr_dl, desc="epoch %d/%d" % (ep, EPOCHS), leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward(); opt.step(); sched.step()
        loss_sum += loss.item() * y.size(0)
        cor += (logits.argmax(1) == y).sum().item(); tot += y.size(0)
    model.eval(); vp = []
    with torch.no_grad():
        for x, y in va_dl:
            vp.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
    vp = np.concatenate(vp)
    va_acc = (vp.argmax(1) == np.array(va_y)).mean()
    print("epoch %2d | loss %.4f train_acc %.4f | val_acc %.4f | %.0fs" %
          (ep, loss_sum / tot, cor / tot, va_acc, time.time() - t0))
    if va_acc > best_acc:
        best_acc = va_acc
        best_state = copy.deepcopy(model.state_dict())
        print("   ↑ 新的最佳，已保存")

print("最佳验证准确率: %.4f" % best_acc)
model.load_state_dict(best_state)

# ---------------- 逐类表现（看哪些表情最容易混） ----------------
va_probs = tta_probs(va_f)
print("\n逐类报告（验证集，含 TTA）:")
print(classification_report(np.array(va_y), va_probs.argmax(1), target_names=CLASSES, digits=3))
np.save("/kaggle/working/val_probs_step3a.npy", va_probs)
np.save("/kaggle/working/val_y_step3a.npy", np.array(va_y))

# ---------------- 生成提交文件 ----------------
te_paths = sorted(p for p in glob.glob(TEST_DIR + "/*") if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
te_names = [os.path.basename(p) for p in te_paths]
test_probs = tta_probs(te_paths)
np.save("/kaggle/working/test_probs_step3a.npy", test_probs)   # 留给第 4 步做集成
np.save("/kaggle/working/test_names.npy", np.array(te_names, dtype=object))

sub = pd.read_csv(SUB_PATH)
name2pred = dict(zip(te_names, test_probs.argmax(1)))
sub["Emotion"] = sub["ID"].map(lambda n: int(name2pred[n]))
out = "/kaggle/working/submission.csv"
sub.to_csv(out, index=False)
print("已保存:", out)
print(sub.head())
print("预测分布:\n", sub["Emotion"].value_counts().sort_index())
