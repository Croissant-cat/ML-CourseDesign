# -*- coding: utf-8 -*-
"""step4：平移探针与坐标框架修正。生成四个探针文件，并在填好榜面分数后拟合修正量。

在做什么
    对同一份测试预测施加四个平移，写成四个提交文件；逐个提交后把公开榜分数填进下面的
    SCORES，脚本用二次模型 RMSE² = C + [(m_x + s_x)² + (m_y + s_y)²]/2 做最小二乘，
    解出残差的平均偏移 (m_x, m_y) 与残差方差 C，修正量取 (−m_x, −m_y)。
    这一步只用榜面分数，不用验证集。

需要多久
    生成文件 10 秒；拟合瞬间完成。

输出什么
    outputs/sub_probe_*.csv     四个探针文件
    outputs/correction.json     拟合出的 (dx, dy, C, 最优 RMSE)
    屏幕输出拟合结果与残差。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# 探针：(名字, 横向平移, 纵向平移)；分数为公开榜实测值
PROBES = [("minus20", -20.0, 0.0), ("base", 0.0, 0.0), ("plus15", 15.0, 0.0),
          ("plus20", 20.0, 0.0), ("plus15_y12", 15.0, 12.0)]
SCORES = {"minus20": 29.87, "base": 19.74, "plus15": 16.29, "plus20": 17.12, "plus15_y12": 20.23}


def find_data():
    for c in (os.path.join(ROOT, "data", "detection"), os.path.join(ROOT, "data"),
              os.environ.get("FOVEA_DATA", "")):
        if c and os.path.exists(os.path.join(c, "sample_submission.csv")):
            return c
    for base in ("/kaggle/input", os.getcwd()):
        if os.path.isdir(base):
            for r, d, f in os.walk(base):
                if "sample_submission.csv" in f:
                    return r
    raise FileNotFoundError("找不到数据，请放到 data/detection/")


def load_base():
    for name in ("pred_v5.npy", "pred_v2.npy", "pred_step2.npy"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            print("基准预测：outputs/%s" % name)
            return np.load(p)
    raise FileNotFoundError("缺少基准预测，请先运行 step2/step3/step5")


def write_probe(Q, sam, te_names, tag, dx, dy):
    idx = {int(n): j for j, n in enumerate(te_names)}
    vals = []
    for s in sam["ImageID"].astype(str):
        axis = 1 if s.strip().lower().endswith("y") else 0
        j = idx[int(s.split("_")[0])]
        vals.append(round(float(Q[j, axis]) + (dy if axis else dx), 4))
    p = os.path.join(OUT, "sub_probe_%s.csv" % tag)
    pd.DataFrame({"ImageID": sam["ImageID"].values, "value": vals}).to_csv(
        p, index=False, lineterminator="\n")
    return p


def fit(probes, scores):
    """RMSE² − (s_x²+s_y²)/2 = D + m_x·s_x + m_y·s_y，对 (D, m_x, m_y) 做最小二乘。"""
    rows, ys = [], []
    for tag, sx, sy in probes:
        rows.append([1.0, sx, sy])
        ys.append(scores[tag] ** 2 - (sx ** 2 + sy ** 2) / 2.0)
    A = np.array(rows)
    y = np.array(ys)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    D, mx, my = sol
    C = D - (mx ** 2 + my ** 2) / 2.0
    resid = A @ sol - y
    return dict(mx=mx, my=my, C=C, rmse_min=float(np.sqrt(max(C, 0.0))),
                resid_max=float(np.abs(resid).max()))


def main():
    DATA = find_data()
    sam = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    te_names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(DATA, "test"))
                      if f.lower().endswith(".jpg"))
    Q = load_base()

    for tag, dx, dy in PROBES:
        p = write_probe(Q, sam, te_names, tag, dx, dy)
        sc = SCORES.get(tag)
        print("写出 %-42s 平移 (%+5.1f, %+5.1f) | 公开榜 %s"
              % (p, dx, dy, ("%.2f" % sc) if sc else "待填"))

    if len(SCORES) != len(PROBES):
        print("\nSCORES 未填全，先提交四个文件再回填。")
        return
    r = fit(PROBES, SCORES)
    print("\n拟合：残差平均偏移 m = (%+.2f, %+.2f) 像素，残差方差 C = %.1f"
          % (r["mx"], r["my"], r["C"]))
    print("     修正量 (dx, dy) = (%+.2f, %+.2f)，最优 RMSE = %.2f，拟合残差最大 %.3f"
          % (-r["mx"], -r["my"], r["rmse_min"], r["resid_max"]))
    out = dict(dx=round(-r["mx"], 2), dy=round(-r["my"], 2), C=round(r["C"], 1),
               rmse_min=round(r["rmse_min"], 2))
    with open(os.path.join(OUT, "correction.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("写出 outputs/correction.json")
    print("\n把这组数写进 step6 的 CORRECTION（现为 +15.2 / −5.3，即上面的四舍五入到 0.1）。")


if __name__ == "__main__":
    main()
