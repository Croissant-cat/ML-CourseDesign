# 机器学习课程设计 任务二：图像情感分类

东北大学机器学习课程设计第二个实验，Kaggle 竞赛 `2026task2`。任务是对 48×48 的灰度人脸图做六类情绪分类（愤怒、恐惧、快乐、悲伤、惊讶、中性）。

- 训练集 28275 张，测试集 7065 张，6 个类别
- 评测口径：测试集准确率
- 最终成绩：**Kaggle Public 0.744**（5 折 OOF 0.7294）
- 详细过程与全部实验数据见课程报告第 2 章（报告单独提交，不放入仓库）

## 目录

| 文件 | 作用 | 主要输出 |
| --- | --- | --- |
| `src/step1_eda.py` | 数据规模、类别分布、图像规格与提交格式核对 | 类别计数、映射核对结论 |
| `src/step2_baseline.py` | 4 层小 CNN 基线，打通训练到提交的链路 | val 0.6524、submission.csv |
| `src/step3_finetune.py` | ResNet18 单模型微调（112 像素、12 轮） | val 0.6736、TTA 0.6800 |
| `src/step4_5fold.py` | 多骨干 5 折交叉验证，逐折保存概率 | `pv_*.npy`、`pt_*.npy`、OOF |
| `src/step5_new_recipe.py` | 160 像素 + RandAugment + EMA 配方 | 新配方骨干概率 |
| `src/step6_blend_submit.py` | 合并概率、权重搜索、生成提交 | submission.csv、融合对照表 |
| `submission.csv` | 榜单 0.744 对应的提交文件 | 7065 行，md5 `9844410550b20f63cdb6a6694fa0deb2` |

## 运行顺序

1. `step1_eda.py` 核对数据（约 1 分钟）
2. `step2_baseline.py` 可选，用于复现基线并验证标签映射（约 4 分钟）
3. `step4_5fold.py` 训练 5 折并保存概率（每个骨干 50~60 分钟）
4. `step5_new_recipe.py` 训练新配方骨干（ConvNeXt-Small 约 210 分钟，Tiny 约 133 分钟）
5. `step6_blend_submit.py` 融合并生成提交（约 1 分钟）

`step3_finetune.py` 用于复现单模型阶段的结论，不是最终流程的必需步骤。

## 环境

Kaggle Notebook，加速器 GPU T4 × 2（代码只用 1 张），Python 3.13，PyTorch 2.x 与 torchvision，预训练权重从 PyTorch 官方仓库自动下载。依赖见 `requirements.txt`。

## 结果汇总

| 阶段 | 方案 | 本地指标 | 榜单 |
| --- | --- | --- | --- |
| E0 | 4 层小 CNN | val 0.6524 | 0.657 |
| E1 | ResNet18 / 112 像素 | val 0.6736 | 0.694 |
| E2 | ResNet18 5 折 + TTA | OOF 0.6913 | 0.711 |
| E3 | 加 EfficientNet-B0 | OOF 0.7047 | 0.718 |
| E4 | 加 ConvNeXt-Tiny | OOF 0.7182 | 0.734 |
| E5 | 加 ConvNeXt-Small | OOF 0.7246 | 0.741 |
| E6 | 加 EfficientNet-B1 | OOF 0.7243 | 0.737 |
| E9 | 新配方两骨干 + Caruana 加权（最终） | OOF 0.7294 | **0.744** |

## 复现注意

- 概率文件必须与训练在同一会话内产生，`/kaggle/working` 不跨会话保留；`step4`、`step5` 每折结束都会重写 `ckpt_all.zip`，可在下一轮训练前挂载为输入数据集，代码会自动跳过已完成的折。
- 推理阶段必须先执行 `model.float()` 并关闭 autocast，否则混合精度会残留半精度权重缓存，触发输入类型与偏置类型不一致。
- 数据挂载是流式的，代码开头会检查 28275 / 7065 两个数量，不符时每 30 秒重试一次，最多等 20 分钟。
