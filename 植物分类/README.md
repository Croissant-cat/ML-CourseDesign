# 任务一：植物分类（Kaggle 2026task1）

东北大学《机器学习》课程设计 · 实验一。

5 类植物幼苗图像分类，评价指标为 **macro F1**。训练集 500 张（每类 100 张），测试集 378 张。
本仓库包含完整可复现的训练与推理流程，以及全部消融实验脚本。

| 指标 | 数值 |
| --- | --- |
| 5 折 OOF macro F1（本地，500 张） | **0.9122** |
| 单折最佳 macro F1 平均 | 0.9114 |
| Kaggle Public 榜单 | **0.9417** |

> 上一版方案（HSV 阈值抠图 + MobileNetV2 两段式微调）的本地验证分为 0.8989、榜单为 0.8597。
> 本方案相对上一版在榜单上提升 **+0.082**。

---

## 1. 环境

```
Python 3.12
PyTorch 2.10.0+cu128
timm 1.0.26
torchvision / numpy / pandas / scikit-learn / scipy / Pillow / matplotlib
```

Kaggle Notebook，Accelerator = GPU T4 × 2，Internet 开启（首次运行需下载 ImageNet 预训练权重）。

## 2. 数据路径

```
/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train/   # 5 类 × 100 张
/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/test/    # 378 张
/kaggle/input/competitions/2026task1/submission-for-task2.csv                     # 官方模板
```

所有中间产物写到 `/kaggle/working/plant_data/`，提交文件写到 `/kaggle/working/submission.csv`。

**注意**：`/kaggle/working` 不跨会话保留。切换 Accelerator 或重启会话后需重新运行 `step2_preprocess.py`
（约 50 秒）重建缓存；若会话中途断开，重跑即可，无需改代码。

## 3. 运行顺序

```bash
python src/step1_eda.py            # 数据规模 / 背景亮度 / 尺寸分布诊断
python src/step2_preprocess.py     # ExG 定位 + 包围盒裁切(256) + 分层 5 折划分
python src/step3_ablation.py --all # 【可选】复现单折消融实验（约 15 分钟）
python src/step4_train_5fold.py    # ConvNeXt-Tiny 5 折训练（约 17 分钟）
python src/step5_infer_submit.py   # 5 折 × 5 视图 TTA 推理，生成 submission.csv
```

专题实验（对应报告 1.7.4 / 1.7.5 节）：

```bash
python src/step6_specialist_diag.py --detail   # 细节量诊断（正确率 vs 目标像素边长）
python src/step6_specialist_diag.py --spec     # 两细叶类专家模型 + 融合权重搜索
```

## 4. 方法要点

**预处理：只定位、不抠图。**

上一版用 HSV 阈值做像素级抠图，细叶断裂、锯齿严重。本方案改为：

1. 在 256×256 缩略图上计算超绿指数 `ExG = 2G − R − B`（植物为正，棕灰砂石为负，黑白条码≈0）；
2. 阈值 20（前景过少时回退 5）得到掩码，**只做 7×7 闭运算**连接细叶（不做开运算，避免腐蚀掉细叶）；
3. 用「连通域面积 ≥ 最大面积 15%」筛选代替形态学腐蚀来去噪；
4. 取全部保留连通域的外接矩形，四周留 12% 边距，按图像自身背景色补成正方形，缩放到 256×256。

整个过程不修改任何像素，叶片边缘零损失。500 张训练图与 378 张测试图均无定位失败。

**数据诊断结论（决定了全部设计）**

| 诊断项 | 结果 |
| --- | --- |
| 背景类型 | 100% 暗底，四边亮度均值 75.85，四角 RGB≈(88,72,55)，砂石纹理 std 24.3 |
| 目标占比（ExG 掩码） | 中位 10.2%；Black-grass 4.0%、Loose Silky-bent 4.7%、Sugar beet 14.0%、Scentless Mayweed 20.6% |
| 图像边长 | 66 ~ 2030 px（5/25/50/75/95 分位 = 84/150/315/590/1141） |

**模型与训练**

ConvNeXt-Tiny（timm，ImageNet 预训练，27.8M 参数）/ 输入 256 / bs 16 / AdamW（wd 0.05）/
差分学习率（主干 2e-4、分类头 1e-3）/ 2 轮 warmup + 余弦退火 / 50 轮 / 标签平滑 0.05 / AMP /
增强：RandomResizedCrop(0.65~1.0) + 水平/垂直翻转 + ±30° 旋转 + ColorJitter + RandomErasing。
按验证集 **macro F1**（而非 accuracy）保存最优权重。

**推理**

5 折等权 + 每折 5 视图 TTA（原图 / 水平翻转 / 垂直翻转 / ±10° 旋转，旋转前放大到 320 再中心裁回 256），
概率平均后取最大类。不对测试集搜索融合权重，避免在 378 张小测试集上过拟合。

## 5. 消融实验结论

| 编号 | 方案 | 最佳 val macro F1 |
| --- | --- | --- |
| E1 | EfficientNet-B0 / 224 / Mixup 0.5 | 0.7840 |
| E2 | 同 E1，关闭 Mixup | **0.8709** |
| E3 | 同 E2，只训练最后 stage + 分类头 | 0.8198 |
| E4 | 同 E2，改用整图（不裁切） | 0.8027 |
| E5 | 同 E2，训练 60 轮 | 0.8618 |
| E6 | 同 E2，输入 256 | 0.8821 |
| E7 | ConvNeXt-Tiny / 224 / 40 轮 | **0.9107** |
| E8 | ConvNeXt-Tiny / 256 / 5 折 + TTA | OOF 0.9122（榜单 0.9417） |

三条主要结论：

1. **Mixup/CutMix 有害**（0.7840 → 0.8709）。它把两个细叶类的像素线性混合，人为制造了它们之间不存在的
   中间样本，破坏了唯一需要精细区分的决策边界；且每折仅 400 张样本，混合使有效监督量减半。
2. **冻结主干有害**（0.8709 → 0.8198）。暗色砂石背景与 ImageNet 统计差异大，BN 统计量必须重新适应；
   且细叶判别特征位于主干后段。当前瓶颈是欠拟合而非过拟合。
3. **提高分辨率无效**。按「原图分辨率下包围盒边长」把样本分四组，OOF 正确率为
   0.920 / 0.911 / 0.885 / 0.927，与细节量无单调关系 —— 说明模型依赖的是整体形态而非像素级纹理，
   「重建高分辨率数据」这条路线被数据否决。

## 6. 错误分析

OOF 共 500 张、错 44 张。其中 **36 张为 Black-grass ↔ Loose Silky-bent 互相误判（占全部错误的 81.8%）**，
其余三类 F1 分别为 0.9645 / 0.9950 / 0.9900。为此训练的「两细叶类二分类专家模型」5 折平均准确率仅 0.7650，
低于主模型在同一对类别上的 0.810，融合权重搜索给出 w = 0（完全采用主模型），该方案被弃用。
失败原因：专家模型每折仅 160 张训练样本（主模型 400 张）、验证集仅 40 张噪声大，且两个模型依赖同一批像素信息。

## 7. 目录结构

```
.
├── README.md
├── requirements.txt
├── submission.csv                        # 榜单 0.9417 的提交文件（378 行，ID 带 .png）
└── src/
    ├── step1_eda.py                      # 数据探索与诊断
    ├── step2_preprocess.py               # ExG 裁切 + 5 折划分
    ├── step3_ablation.py                 # 单折受控对比实验（E1~E7）
    ├── step4_train_5fold.py              # ConvNeXt-Tiny 5 折训练 + OOF 评估
    ├── step5_infer_submit.py             # 5 折 × 5 视图 TTA + 生成 submission.csv
    └── step6_specialist_diag.py          # 细节量诊断 / 细叶类专家模型
```

## 8. 说明

- 本仓库不包含比赛数据与模型权重（体积与版权原因）。数据请从 Kaggle 比赛页下载；
  权重可通过重新运行 `step4_train_5fold.py` 在 17 分钟内复现。
- 全部随机过程固定种子：5 折划分 `random_state=42`，第 k 折训练 `seed = 42 + k`，
  专家模型第 k 折 `seed = 100 + k`。
- 报告中的全部数字均来自上述脚本在 Kaggle T4 × 2 上的实际运行输出。
- 仓库内的 submission.csv 即 Kaggle 榜单 0.9417 的原始提交文件，可与提交记录逐行核对；重新运行 step4、step5 应能得到内容等价的文件（受 GPU 非确定性影响，个别行可能有出入）。
