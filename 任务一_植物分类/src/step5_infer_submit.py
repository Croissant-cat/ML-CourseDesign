# -*- coding: utf-8 -*-
"""Step 5: 5 折 × 5 视图 TTA 推理，生成 submission.csv（对应报告 1.3.4 / 1.6.3 节）。

TTA 视图：原图 / 水平翻转 / 垂直翻转 / +10° / -10°
（旋转前先放大到 320，再中心裁回 256，避免黑角落入视野）

重要：比赛模板要求 ID 带 .png（submission-for-task2.csv 的 ID 形如 xguwzbbzwk.png），
因此提交文件的 ID 必须与模板严格一致，否则会被判为全部匹配失败。

用法：
    python src/step5_infer_submit.py
输出：
    /kaggle/working/submission.csv
    /kaggle/working/plant_data/test_prob_5fold_tta.npy
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

WORK_DIR = "/kaggle/working/plant_data/"
MODEL_DIR = os.path.join(WORK_DIR, "models")
TEST_CROP = os.path.join(WORK_DIR, "crop256", "test")
TEMPLATE = "/kaggle/input/competitions/2026task1/submission-for-task2.csv"
CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = 256
DEVICE = torch.device("cuda")

val_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def tta_probs(model, paths, bs=32):
    probs = np.zeros((len(paths), len(CLASSES)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            views = []
            for p in batch:
                im = Image.open(p).convert("RGB")
                big = im.resize((320, 320), Image.BILINEAR)
                views.append([
                    im,
                    im.transpose(Image.FLIP_LEFT_RIGHT),
                    im.transpose(Image.FLIP_TOP_BOTTOM),
                    TF.center_crop(big.rotate(10.0, resample=Image.BILINEAR), SIZE),
                    TF.center_crop(big.rotate(-10.0, resample=Image.BILINEAR), SIZE),
                ])
            for vi in range(5):                       # 逐视图前向，概率累加后除以视图数
                x = torch.stack([val_tf(v[vi]) for v in views]).to(DEVICE)
                with torch.amp.autocast("cuda"):
                    out = model(x)
                probs[i:i + len(batch)] += torch.softmax(out.float(), 1).cpu().numpy() / 5.0
    return probs


def main():
    test_paths = sorted(glob.glob(os.path.join(TEST_CROP, "*.png")))
    assert len(test_paths) == 378, "测试缓存数量不是 378，请先运行 step2_preprocess.py"
    print("测试图:", len(test_paths))

    all_prob = np.zeros((len(test_paths), len(CLASSES)), dtype=np.float32)
    for k in range(5):
        model = timm.create_model("convnext_tiny", pretrained=False,
                                  num_classes=len(CLASSES), drop_rate=0.2,
                                  drop_path_rate=0.1).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "cvx_fold%d.pth" % k),
                                        map_location="cpu"))
        t0 = time.time()
        all_prob += tta_probs(model, test_paths) / 5.0   # 跨折等权平均
        print("fold %d TTA 完成 %.1f 分钟" % (k, (time.time() - t0) / 60))
        del model
        torch.cuda.empty_cache()

    np.save(os.path.join(WORK_DIR, "test_prob_5fold_tta.npy"), all_prob)
    pred = all_prob.argmax(1)
    print("预测分布:")
    print(pd.Series([CLASSES[i] for i in pred]).value_counts())
    print("平均最大概率: %.4f" % all_prob.max(1).mean())

    # ---- 按官方模板的 ID 顺序与格式组装提交文件 ----
    names = [os.path.basename(p) for p in test_paths]          # 形如 abkieurfrr.png
    stems = [os.path.splitext(n)[0] for n in names]
    lut = {n: i for i, n in enumerate(names)}
    lut.update({s: i for i, s in enumerate(stems)})

    tpl = pd.read_csv(TEMPLATE)
    tids = tpl.iloc[:, 0].astype(str).tolist()
    idx = [lut.get(t) for t in tids]
    assert all(i is not None for i in idx), "模板 ID 与测试图无法匹配"

    sub = pd.DataFrame({"ID": tids, "Category": [CLASSES[pred[i]] for i in idx]})
    out = "/kaggle/working/submission.csv"
    sub.to_csv(out, index=False)

    print("\n已写入:", out, "| 形状:", sub.shape)
    print(sub.head())
    print("校验: 行数==378 ->", len(sub) == 378,
          "| ID 全部带 .png ->", bool(sub["ID"].astype(str).str.endswith(".png").all()),
          "| 列名 ->", list(sub.columns))


if __name__ == "__main__":
    main()
