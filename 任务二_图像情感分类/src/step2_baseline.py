# ============================================================
# 任务二：图像情感分类
# 本文件在做什么：基线：4 层小 CNN + 单次 9:1 划分，用来打通训练到提交的链路并验证标签映射。
# 需要多久：约 4 分钟（15 轮 × 16 秒）。
# 输出什么：最佳验证准确率、submission.csv（7065 行）。
# ============================================================

# ============================================================
# 第 2 步：最小 CNN baseline
# 目标：把「读图 -> 训练 -> 预测 -> 生成 submission.csv」整条链路跑通
# 分数不是重点，格式正确才是重点（预计验证准确率 55%~62%）
# 大约 10 分钟（GPU T4）
# ============================================================
import os, glob, time, random, copy
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "（没开 GPU，会很慢）")

ROOT = "/kaggle/input/competitions/2026task2/fer_data/fer_data"
TRAIN_DIR, TEST_DIR = ROOT + "/train", ROOT + "/test"
SUB_PATH = "/kaggle/input/competitions/2026task2/submission.csv"
# 官方提交编号：0愤怒 1恐惧 2快乐 3悲伤 4惊讶 5中性
# 就让训练顺序 = 提交编号，模型输出的下标直接就是要提交的数字
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
for c in CLASSES:
    assert os.path.isdir(os.path.join(TRAIN_DIR, c)), "目录名对不上: " + c
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
print("类别(顺序=提交编号):", CLASSES)
print("标签映射:", LABEL_MAP)

# ---------------- 数据 ----------------
files, labels = [], []
for i, c in enumerate(CLASSES):
    for f in sorted(os.listdir(os.path.join(TRAIN_DIR, c))):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(os.path.join(TRAIN_DIR, c, f)); labels.append(i)
tr_f, va_f, tr_y, va_y = train_test_split(files, labels, test_size=0.1, random_state=SEED, stratify=labels)
print("训练 %d 张 / 验证 %d 张" % (len(tr_f), len(va_f)))

class FerSet(Dataset):
    def __init__(self, files, labels=None, is_train=False):
        self.files, self.labels = files, labels
        tf = [T.Grayscale(1)]
        if is_train:
            tf += [T.RandomHorizontalFlip(),
                   T.RandomAffine(12, translate=(0.08, 0.08), scale=(0.9, 1.1))]
        tf += [T.ToTensor()]
        self.tf = T.Compose(tf)
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = self.tf(Image.open(self.files[i]).convert("L"))
        return x if self.labels is None else (x, self.labels[i])

tr_dl = DataLoader(FerSet(tr_f, tr_y, True),  batch_size=256, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
va_dl = DataLoader(FerSet(va_f, va_y, False), batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

# ---------------- 模型：4 层小 CNN ----------------
def blk(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))

class Net(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.body = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, n))
    def forward(self, x): return self.head(self.body(x))

model = Net(len(CLASSES)).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
EPOCHS = 15
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * len(tr_dl), pct_start=0.2)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)

@torch.no_grad()
def predict(paths, bs=512):
    """返回概率矩阵；带水平翻转 TTA"""
    model.eval()
    dl = DataLoader(FerSet(paths), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    out = []
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
    va_probs = predict(va_f)
    va_acc = (va_probs.argmax(1) == np.array(va_y)).mean()
    print("epoch %2d | train loss %.4f acc %.4f | val acc %.4f | %.0fs" %
          (ep, loss_sum / tot, cor / tot, va_acc, time.time() - t0))
    if va_acc > best_acc:
        best_acc = va_acc; best_state = copy.deepcopy(model.state_dict())
        np.save("/kaggle/working/val_probs_step2.npy", va_probs)
print("最佳验证准确率: %.4f" % best_acc)

# ---------------- 预测测试集并生成提交文件 ----------------
model.load_state_dict(best_state)
te_paths = sorted(p for p in glob.glob(TEST_DIR + "/*") if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
te_names = [os.path.basename(p) for p in te_paths]
probs = predict(te_paths)
np.save("/kaggle/working/test_probs_step2.npy", probs)   # 存下来，之后换映射/做集成不用重训
pred_cls = probs.argmax(1)

sub = pd.read_csv(SUB_PATH)
name2pred = dict(zip(te_names, pred_cls))
sub["Emotion"] = sub["ID"].map(lambda n: LABEL_MAP[CLASSES[name2pred[n]]] if n in name2pred else -1)
print("没匹配上的行数（应该是 0）:", int((sub["Emotion"] == -1).sum()))

out = "/kaggle/working/submission.csv"
sub.to_csv(out, index=False)
print("已保存:", out)
print(sub.head())
print("预测分布:\n", sub["Emotion"].value_counts().sort_index())
print("\n下一步：右上角 Save Version（Commit）跑完后，去比赛页 Submit 这个 submission.csv，把 LB 分数告诉我")
