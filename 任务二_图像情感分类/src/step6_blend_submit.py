# ============================================================
# 任务二：图像情感分类
# 本文件在做什么：融合与提交：合并多个骨干的概率存档，等权与 Caruana 权重对照，按留出一半选方案，生成提交文件。
# 需要多久：约 1 分钟（纯 CPU，读概率文件）。
# 输出什么：submission.csv、融合方案对照表、OOF 混淆矩阵与逐类指标。
# ============================================================

# -*- coding: utf-8 -*-
"""七骨干最终融合：合并 ckpt11/13/14 → 权重搜索 → 生成提交"""
import numpy as np, os

# 概率存档目录：把 step4 / step5 产出的 pv_*.npy、pt_*.npy、validx_*.npy、test_names.npy 放到这里
KC = os.environ.get("FER_CKPT_DIR", "./ckpt")
if not os.path.isdir(KC):
    raise SystemExit("未找到概率目录 %s；请把 step4/step5 产出放进去，或用环境变量 FER_CKPT_DIR 指定" % KC)
print("概率目录:", KC, "| 文件数:", len(os.listdir(KC)))
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COUNTS = [3963, 4097, 7192, 4862, 3202, 4959]
NFOLD = 5
BBS = ['resnet18', 'efficientnet_b0', 'convnext_tiny', 'convnext_small', 'efficientnet_b1', 'cnxs2', 'cnxT2']
SHORT = {'resnet18': 'r18', 'efficientnet_b0': 'effb0', 'convnext_tiny': 'cnxT', 'convnext_small': 'cnxS',
         'efficientnet_b1': 'effb1', 'cnxs2': 'cnxs2*', 'cnxT2': 'cnxT2*'}
labels = np.concatenate([np.full(c, i) for i, c in enumerate(COUNTS)])
N = len(labels)

def load_oof(bb):
    o = np.full((N, 6), np.nan)
    for k in range(NFOLD):
        p = os.path.join(KC, "pv_%s_f%d.npy" % (bb, k))
        if os.path.exists(p):
            o[np.load(os.path.join(KC, "validx_f%d.npy" % k))] = np.load(p)
    return o

def load_test(bb):
    ps = [os.path.join(KC, "pt_%s_f%d.npy" % (bb, k)) for k in range(NFOLD)]
    ps = [p for p in ps if os.path.exists(p)]
    return np.mean([np.load(p) for p in ps], axis=0).astype(np.float32)

OOF_l, TST_l, ok = [], [], []
for bb in BBS:
    o = load_oof(bb)
    if np.isnan(o).any():
        print("跳过（覆盖不全）:", bb); continue
    OOF_l.append(o); TST_l.append(load_test(bb)); ok.append(bb)
OOF = np.stack(OOF_l).astype(np.float32)
TST = np.stack(TST_l)
n = len(ok)
print("可用 %d 个骨干:" % n, [SHORT[b] for b in ok])

def acc_p(p): return float((p.argmax(1) == labels).mean())
def acc_w(w): return acc_p(np.tensordot(w.astype(np.float32), OOF, axes=1))

print("\n=== 单模型 OOF ===")
for i, bb in enumerate(ok):
    w = np.zeros(n); w[i] = 1
    print("  %-8s %.4f" % (SHORT[bb], acc_w(w)))

eq = np.ones(n) / n
old4 = np.array([1/4 if bb in ('resnet18', 'efficientnet_b0', 'convnext_tiny', 'convnext_small') else 0 for bb in ok])
old6 = np.array([1/6 if bb != 'cnxT2' else 0 for bb in ok])
print("\n=== 融合对比 ===")
print("  旧 4 骨干（LB 0.741）   OOF %.4f" % acc_w(old4))
print("  六骨干（LB 0.744）      OOF %.4f" % acc_w(old6))
print("  七骨干等权              OOF %.4f  (%+.4f vs 六骨干)" % (acc_w(eq), acc_w(eq) - acc_w(old6)))

# ---------- 诚实权重搜索 ----------
rng = np.random.RandomState(2028)
perm = rng.permutation(N); A, B = perm[:N // 2], perm[N // 2:]
def acc_idx(w, idx):
    return float((np.tensordot(w.astype(np.float32), OOF, axes=1)[idx].argmax(1) == labels[idx]).mean())

def greedy(idx, iters=50):
    ysub = labels[idx]; cnt = np.zeros(n); cur = np.zeros((len(idx), 6), dtype=np.float32)
    for t in range(iters):
        bi, ba = 0, -1
        for i in range(n):
            a = float((((cur * t + OOF[i][idx]) / (t + 1)).argmax(1) == ysub).mean())
            if a > ba: ba, bi = a, i
        cnt[bi] += 1
        cur = (cur * t + OOF[bi][idx]) / (t + 1)
    return cnt / cnt.sum()

bags = [greedy(rng.choice(A, size=len(A) // 2, replace=False)) for _ in range(30)]
w_car = np.mean(bags, axis=0)
best_wr, best_ar = None, -1
for _ in range(5000):
    w = rng.dirichlet(np.ones(n) * 0.7); a = acc_idx(w, A)
    if a > best_ar: best_ar, best_wr = a, w

cands = [("等权7", eq), ("六骨干", old6), ("Caruana", w_car), ("随机搜索", best_wr)]
print("\n=== B 半（未参与选择）排名 ===")
for nm, w in sorted(cands, key=lambda t: -acc_idx(t[1], B)):
    print("  %-10s B半 %.4f | 全量 %.4f" % (nm, acc_idx(w, B), acc_w(w)))
print("  Caruana 权重:", dict(zip([SHORT[b] for b in ok], np.round(w_car, 3).tolist())))
print("  随机搜索权重:", dict(zip([SHORT[b] for b in ok], np.round(best_wr, 3).tolist())))

base_b = acc_idx(eq, B)
top_nm, top_w = max(cands, key=lambda t: acc_idx(t[1], B))
use_nm, use_w = (top_nm, top_w) if acc_idx(top_w, B) > base_b + 0.0015 else ("等权7", eq)
print("\n>>> 采用: %s" % use_nm)
print("    权重:", dict(zip([SHORT[b] for b in ok], np.round(use_w, 3).tolist())))

final = np.tensordot(use_w.astype(np.float32), OOF, axes=1)
print("\n最终 OOF: %.4f  (比六骨干 %+.4f)" % (acc_p(final), acc_p(final) - acc_w(old6)))
po = final.argmax(1)
print("%-10s %8s %8s %8s" % ("类别", "prec", "recall", "f1"))
for c in range(6):
    tp = int(((po == c) & (labels == c)).sum()); fp = int(((po == c) & (labels != c)).sum())
    fn = int(((po != c) & (labels == c)).sum())
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    print("%-10s %8.3f %8.3f %8.3f" % (CLASSES[c], pr, rc, 2 * pr * rc / max(1e-9, pr + rc)))

tp_test = np.tensordot(use_w.astype(np.float32), TST, axes=1)
names = np.load(os.path.join(KC, "test_names.npy"), allow_pickle=True)
pred = tp_test.argmax(1)
import pandas as pd
out = r"D:\1\submission_v14.csv"
pd.DataFrame({"ID": names, "Emotion": pred}).to_csv(out, index=False)
print("\n已写出:", out, "| 行数", len(pred))
prev = pd.read_csv(r"D:\1\submission_v13.csv").sort_values("ID").Emotion.values
new = pd.DataFrame({"ID": names, "Emotion": pred}).sort_values("ID").Emotion.values
print("与六骨干版本(v13, LB0.744) 预测不同:", int((prev != new).sum()), "/", len(new))
print("预测分布:", np.bincount(pred, minlength=6).tolist())
