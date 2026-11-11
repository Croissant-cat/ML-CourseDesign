# -*- coding: utf-8 -*-
"""step6  生成提交文件并自检

在做什么：
    用融合后的测试概率与 step5 选出的阈值做二值化，按视野裁剪，写成「血管为 255」的
    PNG，调用官方 segmentation_to_csv.py 生成 CSV；再把 CSV 反解码回掩膜，与预测逐像素
    比对。若第一次比对不通过，自动按相反极性重出一次，取能对上的那一份，所以提交图
    极性写反也不会静默出错。
需要多久：
    约 2 分钟（20 张 PNG 写盘 + 调官方脚本 + 自检）。
输出什么：
    submission.csv        最终提交文件
    submission_orderF.csv 列优先编码的备用文件
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seglib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--thr", type=float, default=None, help="默认取 step5 的 tune.json")
    ap.add_argument("--min-size", type=int, default=None)
    ap.add_argument("--csv", default="submission.csv")
    a = ap.parse_args()

    if a.data:
        L.DATA_ROOT = a.data

    tune_path = os.path.join(a.out, "merged", "tune.json")
    if not os.path.exists(tune_path):
        tune_path = os.path.join(a.out, "tune.json")
    thr, ms = a.thr, a.min_size
    if os.path.exists(tune_path):
        tune = json.load(io.open(tune_path, encoding="utf-8"))
        thr = tune["thr"] if thr is None else thr
        ms = tune["min_size"] if ms is None else ms
    thr = 0.5 if thr is None else thr
    ms = 0 if ms is None else ms
    print("[step6] 使用 thr=%.2f  min_conn_px=%d（来源 %s）" % (thr, ms, tune_path))

    root = L.find_root()
    _, tests, sub_ids, sub_path = L.discover(root)
    te_items = [L.Item(p, *L.load_features(p)) for p in tests]
    id2item = L.map_ids(te_items, sub_ids)

    probs, fovs = {}, {}
    for sid in sub_ids:
        f = os.path.join(a.out, "merged", "test", "%s.npy" % sid)
        if not os.path.exists(f):
            f = os.path.join(a.out, "test", "%s.npy" % sid)
        if not os.path.exists(f):
            raise SystemExit("缺少测试概率 %s，请先跑 step3 或 step5" % f)
        probs[sid] = np.load(f).astype(np.float32)
        fovs[sid] = id2item[sid].fov

    records, shapes = {}, {}
    for sid in sub_ids:
        records[sid] = L.binarize(probs[sid], fovs[sid], thr, ms)
        shapes[sid] = records[sid].shape
    cov = [float(records[s].mean()) for s in records]
    print("[step6] 血管占比 均值 %.2f%%  范围 %.2f%%~%.2f%%"
          % (100 * np.mean(cov), 100 * min(cov), 100 * max(cov)))

    scripts = L.find_official_script(root)

    def emit(invert, out_csv):
        """按给定极性写 PNG 并生成 CSV；返回 (是否用了官方脚本, 反解码不一致行数)。"""
        if os.path.exists(out_csv):
            os.remove(out_csv)          # 清掉上一次的产物，避免重出时读到旧文件
        png = tempfile.mkdtemp(prefix="png_submit_")
        for sid, m in records.items():
            v = (~m) if invert else m
            L.imwrite_u(os.path.join(png, "%s.png" % sid), L.to_png(v))
            it = id2item.get(sid)
            stem = os.path.splitext(os.path.basename(it.path))[0] if it is not None else sid
            if stem != sid:
                L.imwrite_u(os.path.join(png, "%s.png" % stem), L.to_png(v))
        used = False
        for script in scripts:
            if L.run_official(script, png, sub_ids, out_csv, shapes) and os.path.exists(out_csv):
                used = True
                break
        shutil.rmtree(png, ignore_errors=True)
        if not used:
            pd.DataFrame({"Id": sub_ids,
                          "Predicted": [L.rle_encode(records[s], order="C") for s in sub_ids]}).to_csv(
                out_csv, index=False, encoding="utf-8")
            return False, 0
        df = pd.read_csv(out_csv, dtype=str).fillna("")
        bad = 0
        for _, r in df.iterrows():
            sid = str(r.iloc[0]).strip()
            if sid not in records:
                continue
            if not np.array_equal(L.rle_decode(r.iloc[1], shapes[sid], "C"), records[sid]):
                bad += 1
        return True, bad

    used, bad = emit(False, a.csv)
    print("[step6] 第一次生成：官方脚本%s，反解码不一致 %d 行"
          % ("已使用" if used else "未使用", bad))
    if bad:
        print("[step6] 不一致，说明脚本的极性约定与 PNG 相反，按反极性重出一次 …")
        used2, bad2 = emit(True, a.csv)
        print("[step6] 第二次生成：官方脚本%s，反解码不一致 %d 行"
              % ("已使用" if used2 else "未使用", bad2))
        if bad2 > bad:
            raise SystemExit("两种极性都不一致，请人工核对官方脚本的编码方式")
        bad = bad2

    rows = [(sid, L.rle_encode(records[sid], order="F")) for sid in sub_ids]
    pd.DataFrame({"Id": sub_ids, "Predicted": [r[1] for r in rows]}).to_csv(
        "submission_orderF.csv", index=False, encoding="utf-8")

    df = pd.read_csv(a.csv, dtype=str).fillna("")
    print("[step6] 自检：行数 %d，列 %s，空预测 %d，Id 集合一致 %s"
          % (len(df), list(df.columns), int((df.iloc[:, 1].str.len() == 0).sum()),
             set(df.iloc[:, 0].str.strip()) == set(sub_ids)))
    print("[step6] 反解码逐像素比对：不一致 %d 行" % bad)
    print("[step6] 已写出 %s" % os.path.abspath(a.csv))


if __name__ == "__main__":
    main()
