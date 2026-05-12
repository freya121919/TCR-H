# TCR-H 复现报告

## 概述

复现 TCR-H 论文论文提出 SVM-RBF 模型，基于 CDR3β 和抗原表位的 194 维理化特征预测 TCR-表位结合。

**核心思路：** TCR 的 CDR3β 区与抗原表位（epitope）的物理化学互补性决定了结合特异性。将 CDR3β 和 epitope 各编码为 97 维特征向量（共 194 维），拼接后训练分类模型。首先基于epitope hard 分割对RF,GBT,XGB,SVM-RBF进行性能比较，发现SVM-RBF最好，后

**SHAP 分析：** 使用 KernelExplainer（背景样本 30，解释样本 100）分析特征重要性。结果表明各分割方式下最重要特征一致：Kidera factors（KF3/5/7/9/10）、FASGAI（F1/2/4/5/6）、MSWHIM、BLOSUM、SVGER、疏水性矩、分子量、不稳定指数。

---

## 数据

- **来源：** VDJdb 数据库
- **正样本：** VDJdb 已验证结合的 TCR-表位对
- **负样本：** 实验验证过的不可信结合对
- **总样本量：** ~25 万条
- **特征编码（194 维）：**  `scripts/peptide_features.py`

## 模型参数

### RF / GBT / XGB（默认参数，与论文一致）

| 模型 | 类 | 关键参数 |
|------|---|---------|
| RF | `RandomForestClassifier` | n_estimators=100, random_state=42, n_jobs=-1 |
| GBT | `GradientBoostingClassifier` | n_estimators=100, random_state=42 |
| XGB | `XGBClassifier` | n_estimators=100, eval_metric=logloss, random_state=42 |

**文档：**
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
- https://xgboost.readthedocs.io/en/stable/parameter.html

### SVM-RBF（核心模型）

| 参数 | 值 | 含义 |
|------|:---:|------|
| kernel | rbf | 径向基核，捕捉非线性关系 |
| C | 1.0 | 正则化系数，默认值 |
| gamma | scale | 核宽度，自动适配特征维度 |
| class_weight | balanced | 正负样本自动加权 |
| 子采样 | 30K | 分层采样，避免全量 O(n²) 计算 |

**文档：** https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

### 相关特征去除

| 参数 | 值 |
|------|:---:|
| 阈值 | Pearson r > 0.8 的对中随机删除一个 |
| 结果 | 194 → 73（121 个去除） |

### SHAP

| 参数 | 值 |
|------|:---:|
| Explainer | KernelExplainer |
| n_bg / n_explain / nsamples | 30 / 100 / 50 |

---

## 数据集分割

| 分割方式 | 约束 | 训练集 | 测试集 |
|---------|------|:-----:|:-----:|
| Epitope hard | 测试表位训练集未见 | 199,988 | 54,449 |
| TCR hard | 测试 CDR3β 训练集未见 | ~203K | ~51K |
| Strict | 两者皆未见 | ~173K | ~54K |
| Random | 80/20 随机 | ~203K | ~51K |

---

## 结果

### 与论文对比

| 模型 | 特征 | 论文 AUC | 复现 AUC | TP | TN | FP | FN | Acc | Prec | Recall | Spec | F1 |
|------|:----:|:-------:|:--------:|:--:|:--:|:--:|:--:|:---:|:----:|:-----:|:----:|:--:|
| RF | 全 194 | 0.50 | 0.5000 | 31511 | 0 | 22938 | 0 | .5787 | .5787 | 1.0 | 0.0 | .7332 |
| GBT | 全 194 | 0.54 | 0.4969 | 30841 | 347 | 22591 | 670 | .5728 | .5772 | .9787 | .0151 | .7262 |
| XGB | 全 194 | 0.51 | 0.5002 | 31461 | 44 | 22894 | 50 | .5786 | .5788 | .9984 | .0019 | .7328 |
| SVM-RBF | 全 194 | 0.80 | 0.8596 | 26713 | 19991 | 2947 | 4798 | .8578 | .9006 | .8477 | .8715 | .8734 |
| TCR-Hβ | 去相关 | 0.92 | 0.9204 | 16260 | 31247 | 974 | 2406 | .9336 | .9435 | .8711 | .9698 | .9058 |
| TCR-HβE | 去相关 | 0.89 | 0.8760 | 27894 | 19883 | 3055 | 3617 | .8775 | .9013 | .8852 | .8668 | .8932 |
| TCR-RS | 去相关 | 0.92 | 0.9133 | 18217 | 28772 | 642 | 3257 | .9234 | .9660 | .8483 | .9782 | .9033 |
| TCR-HE | 去相关 | 0.87 | 0.8735 | 27578 | 19999 | 2939 | 3933 | .8738 | .9037 | .8752 | .8719 | .8892 |

### 与已发表模型对比（epitope hard split）

| 指标 | 模型 | 数值 |
|------|------|:----:|
| AUC | ATM-TCR / ImRex / epiTCR / Pan-Peptide / **TCR-HE** | 0.47 / 0.55 / 0.75 / 0.78 / **0.87** |
| Precision | NetTCR / ERGO-LSTM / ERGO-AE / ATM-TCR / **TCR-HE** | 0.53 / 0.52 / 0.57 / 0.51 / **0.90** |
| Recall | NetTCR / ERGO-LSTM / ERGO-AE / ATM-TCR / **TCR-HE** | 0.62 / 0.70 / 0.51 / 0.86 / **0.88** |

---

## 图

| 图 | 文件 |
|----|------|
| ROC 曲线 | `results/final/roc_curves.png` |
| 基线模型柱状图（RF/GBT/XGB/SVM-RBF） | `results/final/performance_bar_chart_baseline.png` |
| TCR 模型柱状图（Hβ/HβE/RS/HE） | `results/final/performance_bar_chart_tcr.png` |
| 文献对比图 | `results/final/figure4_comparison.png` |
| SHAP 条形图（×4） | `results/final/{TCR-Hβ,TCR-HβE,TCR-RS,TCR-HE}_shap_bar.png` |
| SHAP 蜂群图（×4） | `results/final/{TCR-Hβ,TCR-HβE,TCR-RS,TCR-HE}_shap_beeswarm.png` |

---

## 关键结论

1. 树模型（RF/GBT/XGB）在 epitope-hard split 下 AUC≈0.50（随机），SVM-RBF 显著更好（0.86-0.92）
2. 去相关后 194→73 特征，SVM 性能不变，特征冗余度 62%
3. 分割越严格 AUC 越低：TCR hard(0.92) > random(0.91) > strict(0.88) ≈ epitope hard(0.87)
4. TCR-HE 在所有指标上超过 Pan-Peptide、epiTCR、NetTCR、ERGO 等已发表模型
5. SHAP 显示不同分割方式下最重要特征一致（Kidera, FASGAI, MSWHIM, BLOSUM 等）

---

## 文件结构

```
scripts/train_final.py          — 训练 + SHAP
scripts/figure4_comparison.py   — 文献对比图
scripts/peptide_features.py     — 特征编码
src/data/loaders.py + splits.py — 数据加载与分割
src/models/train_svm.py         — SVM 训练
results/final/                  — 所有结果（表、图、模型）
report.md / CLAUDE.md           — 文档
```
