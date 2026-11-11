# -*- coding: utf-8 -*-
"""step1：数据盘点与关键诊断，产出报告表 3-1～3-4 与图 3-1 的全部数字。

在做什么
    统计训练/测试的张数与两种尺寸；真值分布；真值点相对区域框中心的偏移；取景（非黑区域
    左上角与宽高、眼底面积占比）；画质（平均亮度、灰度标准差、Laplacian 方差、高频能量）；
    中心凹局部对比度（内环与环带的亮度差，测试集用提交文件里的预测点）；几何先验（视盘
    到中心凹的距离、视盘等效半径、图幅相对位置）；以及 SIFT + RANSAC 的重复图检索。

需要多久
    约 3 分钟（80×80 与 20×80 对匹配占大头），不需要 GPU。

输出什么
    outputs/eda_frame.csv     每张图的取景与画质指标
    outputs/eda_disc.csv      每张图的视盘中心与等效半径
    屏幕输出表 3-1～3-4 与重复图检索结果。
"""
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(0)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def find_data():
    for c in (os.path.join(ROOT, "data", "detection"), os.path.join(ROOT, "data"),
              os.environ.get("FOVEA_DATA", "")):
        if c and os.path.exists(os.path.join(c, "fovea_localization_train_GT.csv")):
            return c
    for base in ("/kaggle/input", os.getcwd()):
        if os.path.isdir(base):
            for r, d, f in os.walk(base):
                if "fovea_localization_train_GT.csv" in f:
                    return r
    raise FileNotFoundError("找不到数据，请放到 data/detection/")


DATA = find_data()


def imread_u(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return im


def load_gt():
    gt = pd.read_csv(os.path.join(DATA, "fovea_localization_train_GT.csv"))
    gt.columns = [str(c).strip() for c in gt.columns]
    low = {c: c.lower().replace(" ", "").replace("_", "") for c in gt.columns}
    cn = next(c for c, k in low.items() if k in ("data", "image", "name", "filename", "id"))
    cx = next(c for c, k in low.items() if k in ("foveax", "x", "centerx"))
    cy = next(c for c, k in low.items() if k in ("foveay", "y", "centery"))
    gt = gt.rename(columns={cn: "name", cx: "x", cy: "y"})
    gt["name"] = gt["name"].astype(str).str.strip().map(lambda s: os.path.splitext(os.path.basename(s))[0])
    gt["name"] = gt["name"].map(lambda s: s if not s.isdigit() else "%04d" % int(s))
    gt["x"], gt["y"] = gt["x"].astype(float), gt["y"].astype(float)
    return gt


def load_boxes():
    boxes = {}
    d = os.path.join(DATA, "train_location")
    if not os.path.isdir(d):
        return boxes
    for f in sorted(os.listdir(d)):
        if f.endswith(".xml"):
            r = ET.parse(os.path.join(d, f)).getroot()
            nm = r.find("filename").text.split(".")[0]
            b = r.find("object/bndbox")
            boxes[nm] = tuple(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
    return boxes


def framing(im):
    """非黑区域外接框、眼底面积占比，以及只在眼底内部统计的画质指标。"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = (g > max(8.0, float(g.max()) * 0.08)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    sel = mask > 0
    val = g[sel].astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return dict(x0=x0, y0=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
                fundus_frac=float(sel.mean()), mean=float(val.mean()), std=float(val.std()),
                lapvar=float(lap[sel].var()), hf=float(np.sqrt(gx ** 2 + gy ** 2)[sel].mean()))


def fovea_contrast(im, x, y):
    """内环（半径 40 内）与环带（60~120）的平均亮度差，以及环带的梯度能量。"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    if not (0 <= x < W and 0 <= y < H):
        return np.nan, np.nan
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot(xx - x, yy - y)
    valid = g > 5
    inner = (r <= 40) & valid
    ring = (r >= 60) & (r <= 120) & valid
    if inner.sum() < 50 or ring.sum() < 200:
        return np.nan, np.nan
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    return float(g[ring].mean() - g[inner].mean()), float(gm[ring].mean())


def disc(img, work=512):
    """视盘：在 1/12 边长的模糊图上取眼底内最亮点，再以该点邻域（0.18 边长）的亮度
    加权质心与加权二阶矩估计中心与等效半径。"""
    H, W = img.shape[:2]
    sc = work / float(max(W, H))
    small = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    fov = gray > max(12.0, gray.max() * 0.08)
    fovm = cv2.morphologyEx(fov.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    k = max(9, int(round(min(h, w) / 12.0)) | 1)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    blur[fovm == 0] = -1e6
    py, px = np.unravel_index(int(np.argmax(blur)), blur.shape)
    R = int(min(h, w) * 0.18)
    y0, y1 = max(0, py - R), min(h, py + R)
    x0, x1 = max(0, px - R), min(w, px + R)
    patch = np.clip(blur[y0:y1, x0:x1], 0, None)
    base = np.percentile(patch, 40)
    wgt = np.clip(patch - base, 0, None)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    s = max(wgt.sum(), 1e-6)
    mx = (wgt * xx).sum() / s
    my = (wgt * yy).sum() / s
    sxx = (wgt * (xx - mx) ** 2).sum() / s
    syy = (wgt * (yy - my) ** 2).sum() / s
    rad = float(np.sqrt((sxx + syy) / 2.0)) / 0.7071
    return mx / sc, my / sc, rad / sc


def sift_feats(paths, nfeatures=1200):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = [], []
    for p in paths:
        g = cv2.cvtColor(imread_u(p), cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        k, d = sift.detectAndCompute(g, None)
        kps.append(k if k is not None else [])
        desc.append(d)
    return kps, desc


def match_pairs(kps_a, desc_a, kps_b, desc_b, same_list=False, min_good=40):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    out = []
    for i in range(len(desc_a)):
        js = range(i + 1, len(desc_b)) if same_list else range(len(desc_b))
        for j in js:
            if desc_a[i] is None or desc_b[j] is None or len(kps_a[i]) < 2 or len(kps_b[j]) < 2:
                continue
            m = bf.knnMatch(desc_a[i], desc_b[j], k=2)
            good = [a for a, b in m if a.distance < 0.75 * b.distance]
            if len(good) < min_good:
                out.append((i, j, 0, len(good)))
                continue
            src = np.float32([kps_a[i][g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
            dst = np.float32([kps_b[j][g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
            Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            out.append((i, j, int(inl.sum()) if inl is not None else 0, len(good)))
    return sorted(out, key=lambda t: -t[2])


def transfer_error(path_a, path_b, pt_a, nfeatures=1200):
    """用 a→b 的单应把 a 的真值搬到 b 上，返回 (搬过去的点, 内点数)。"""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    gs = []
    for p in (path_a, path_b):
        g = cv2.cvtColor(imread_u(p), cv2.COLOR_BGR2GRAY)
        gs.append(cv2.resize(g, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    ka, da = sift.detectAndCompute(gs[0], None)
    kb, db = sift.detectAndCompute(gs[1], None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [a for a, b in bf.knnMatch(da, db, k=2) if a.distance < 0.75 * b.distance]
    src = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    p = np.array([pt_a[0] * 0.25, pt_a[1] * 0.25, 1.0])
    q = Hm @ p
    return (q[0] / q[2] / 0.25, q[1] / q[2] / 0.25), int(inl.sum())


def load_sub_points(test_names):
    """提交文件里的测试集坐标，用于在预测点处量中心凹对比度。"""
    p = os.path.join(ROOT, "submission.csv")
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p)
    out = {}
    for s, v in zip(d["ImageID"].astype(str), d["value"].astype(float)):
        num = int("".join(c for c in s.split("_")[0] if c.isdigit()))
        axis = 1 if s.strip().lower().endswith("y") else 0
        out.setdefault(num, [0.0, 0.0])[axis] = v
    return {int(n): tuple(out[int(n)]) for n in test_names if int(n) in out}


def main():
    gt = load_gt()
    boxes = load_boxes()
    train = [(n, x, y) for n, x, y in gt[["name", "x", "y"]].itertuples(index=False)
             if os.path.exists(os.path.join(DATA, "train", n + ".jpg"))]
    tes = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(DATA, "test"))
                 if f.lower().endswith(".jpg"))
    sub_pts = load_sub_points(tes)

    print("=" * 78)
    print("表 3-1  数据规模与真值分布")
    print("=" * 78)
    print("训练 %d 张 | 测试 %d 张" % (len(train), len(tes)))
    sizes, tsz = {}, {}
    for n, _, _ in train:
        im = imread_u(os.path.join(DATA, "train", n + ".jpg"))
        k = (im.shape[1], im.shape[0])
        sizes[k] = sizes.get(k, 0) + 1
    for n in tes:
        im = imread_u(os.path.join(DATA, "test", n + ".jpg"))
        k = (im.shape[1], im.shape[0])
        tsz[k] = tsz.get(k, 0) + 1
    for k in sorted(set(list(sizes) + list(tsz)), key=lambda s: -s[0]):
        print("  %d×%d：训练 %d 张，测试 %d 张" % (k[0], k[1], sizes.get(k, 0), tsz.get(k, 0)))
    xs = np.array([t[1] for t in train], np.float64)
    ys = np.array([t[2] for t in train], np.float64)
    print("Fovea_X  %.1f ~ %.1f，均值 %.1f，标准差 %.1f" % (xs.min(), xs.max(), xs.mean(), xs.std()))
    print("Fovea_Y  %.1f ~ %.1f，均值 %.1f，标准差 %.1f" % (ys.min(), ys.max(), ys.mean(), ys.std()))
    print("不可见样本 (0,0)：%d" % int(((xs <= 0) & (ys <= 0)).sum()))
    if boxes:
        wh = np.array([[b[2] - b[0], b[3] - b[1]] for b in boxes.values()])
        print("区域框  中位 %.0f×%.0f，宽 %.0f~%.0f，高 %.0f~%.0f"
              % (np.median(wh[:, 0]), np.median(wh[:, 1]), wh[:, 0].min(), wh[:, 0].max(),
                 wh[:, 1].min(), wh[:, 1].max()))

    if boxes:
        dx, dy = [], []
        for n, x, y in train:
            if n in boxes:
                b = boxes[n]
                dx.append(x - (b[0] + b[2]) / 2.0)
                dy.append(y - (b[1] + b[3]) / 2.0)
        dx, dy = np.array(dx), np.array(dy)
        print("\n" + "=" * 78)
        print("表 3-2  真值点相对区域框中心的偏移（像素，n=%d）" % len(dx))
        print("=" * 78)
        for nm, v in (("X", dx), ("Y", dy)):
            print("  %s  均值 %+.2f | 标准差 %.2f | 中位绝对值 %.1f | 最大绝对值 %.2f"
                  % (nm, v.mean(), v.std(), np.median(np.abs(v)), np.abs(v).max()))

    rows = []
    for split, items in (("train", train), ("test", [(n, np.nan, np.nan) for n in tes])):
        for n, x, y in items:
            im = imread_u(os.path.join(DATA, split, n + ".jpg"))
            r = framing(im)
            if split == "test" and int(n) in sub_pts:
                px, py = sub_pts[int(n)]
            else:
                px, py = x, y
            if np.isnan(px):
                c, ge = np.nan, np.nan
            else:
                c, ge = fovea_contrast(im, px, py)
            r.update(split=split, name=n, x=x, y=y, contrast=c, grad=ge)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "eda_frame.csv"), index=False)

    print("\n" + "=" * 78)
    print("表 3-3  取景与画质对比（取景取 2992×2000 子集的中位与四分位；画质取全体均值±标准差）")
    print("=" * 78)
    big = df[df.w > 1500]
    for key in ("x0", "y0", "w", "h"):
        t = big[big.split == "train"][key].astype(float)
        s = big[big.split == "test"][key].astype(float)
        print("  %-4s 训练 %.0f（%.0f~%.0f） | 测试 %.0f（%.0f~%.0f）"
              % (key, t.median(), t.quantile(.25), t.quantile(.75),
                 s.median(), s.quantile(.25), s.quantile(.75)))
    for key in ("fundus_frac", "mean", "std", "lapvar", "hf", "contrast", "grad"):
        t = df[df.split == "train"][key].astype(float).dropna()
        s = df[df.split == "test"][key].astype(float).dropna()
        print("  %-11s 训练 %.2f±%.2f | 测试 %.2f±%.2f | 相对差 %+.1f%%"
              % (key, t.mean(), t.std(), s.mean(), s.std(), (s.mean() - t.mean()) / abs(t.mean()) * 100))

    disc_rows = []
    for split, names in (("train", [n for n, _, _ in train]), ("test", tes)):
        for n in names:
            im = imread_u(os.path.join(DATA, split, n + ".jpg"))
            cx, cy, rad = disc(im)
            disc_rows.append(dict(split=split, name=n, cx=cx, cy=cy, r=rad))
    dd = pd.DataFrame(disc_rows)
    dd.to_csv(os.path.join(OUT, "eda_disc.csv"), index=False)

    print("\n" + "=" * 78)
    print("表 3-4  中心凹的几何先验（2992×2000 子集）")
    print("=" * 78)
    m = df.merge(dd, on=["split", "name"])
    m = m[m.w > 1500]
    tr = m[(m.split == "train") & m.x.notna()]
    te = m[m.split == "test"].copy()
    print("  视盘等效半径：训练 %.1f±%.1f（n=%d）| 测试 %.1f±%.1f（n=%d）| 相对差 %+.1f%%"
          % (tr.r.mean(), tr.r.std(), len(tr), te.r.mean(), te.r.std(), len(te),
             (te.r.mean() - tr.r.mean()) / tr.r.mean() * 100))
    dist = np.hypot(tr.x - tr.cx, tr.y - tr.cy)
    ratio = dist / tr.r
    print("  视盘到中心凹距离：%.0f±%.0f 像素" % (dist.mean(), dist.std()))
    print("  距离/等效半径：%.2f±%.2f（标准差折算 %.0f 像素）"
          % (ratio.mean(), ratio.std(), ratio.std() * tr.r.mean()))
    u = (tr.x - tr.x0) / tr.w
    v = (tr.y - tr.y0) / tr.h
    print("  相对横坐标 u：%.3f±%.3f（折算 %.0f 像素）" % (u.mean(), u.std(), u.std() * tr.w.mean()))
    print("  相对纵坐标 v：%.3f±%.3f（折算 %.0f 像素）" % (v.mean(), v.std(), v.std() * tr.h.mean()))

    print("\n" + "=" * 78)
    print("重复图检索（SIFT + RANSAC，1/4 尺度、每张最多 1200 个关键点）")
    print("=" * 78)
    tr_paths = [os.path.join(DATA, "train", n + ".jpg") for n, _, _ in train]
    te_paths = [os.path.join(DATA, "test", n + ".jpg") for n in tes]
    ktr, dtr = sift_feats(tr_paths)
    kte, dte = sift_feats(te_paths)
    for i, j, inl, good in match_pairs(ktr, dtr, ktr, dtr, same_list=True)[:5]:
        print("  训练 %s ↔ %s：内点 %d，好匹配 %d" % (train[i][0], train[j][0], inl, good))
    i, j, inl, good = match_pairs(ktr, dtr, ktr, dtr, same_list=True)[0]
    (qx, qy), _ = transfer_error(tr_paths[i], tr_paths[j], (train[i][1], train[i][2]))
    err = float(np.hypot(qx - train[j][1], qy - train[j][2]))
    print("  用这对的单应把 %s 的真值搬到 %s，误差 %.2f 像素" % (train[i][0], train[j][0], err))
    cross = match_pairs(kte, dte, ktr, dtr)
    b = cross[0]
    print("  测试对训练的最高内点：%d（%s ↔ %s，好匹配 %d）"
          % (b[2], tes[b[0]], train[b[1]][0], b[3]))
    print("  内点 ≥100 的跨集对数：%d" % sum(1 for t in cross if t[2] >= 100))
    print("\n完成，明细见 outputs/eda_frame.csv 与 outputs/eda_disc.csv")


if __name__ == "__main__":
    main()
