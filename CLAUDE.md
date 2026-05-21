# Project: TCR-H Reproduction (Epitope-Hard Split)

Reproduce TCR-H paper (Front Immunol 2024) epitope-hard split results.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
TCR-H/
├── CLAUDE.md                       # 项目说明
├── requirements.txt                # 依赖项
├── submit.sh                       # SLURM 提交脚本（HPC）
├── scripts/
│   ├── train_final.py              # 主训练脚本（8 个模型+对应数据分割）
│   ├── peptide_features.py         # 特征工程
│   ├── figure4_comparison.py       # 模仿论文Figure4,与先前模型的性能对比
│   ├── robustness_test.py          # 鲁棒性测试（多种 seed + split）
│   ├── imbalance_stress_test.py    # 类别不平衡压力测试
│   ├── shap_top50.py               # SHAP Top 50 特征重要性
│   ├── test_equal_prob_split.py    # 等概率分割 vs sqrt-weighted greedy
│   └── run_one_model.py            # 单模型运行器（SLURM array jobs）
├── src/
│   ├── data/                       # 数据加载与分割（含_greedy_hard_split）
│   ├── features/                   # 特征构建
│   ├── models/                     # 模型定义
│   └── evaluation/                 # 评估指标
├── data/splits_tchard/             # Epitope-hard 分割数据
├── results/
│   ├── final/                      # 旧版结果（30K 子采样）
│   │   ├── results_table.csv
│   │   ├── roc_curves.png
│   │   ├── performance_bar_chart*.png
│   │   ├── figure4_comparison.png
│   │   ├── imbalance_stress_test.*
│   │   ├── robustness_test.*
│   │   ├── tcr_he_features_removed.json
│   │   ├── equal_prob_split_comparison.txt
│   │   ├── shap_*.png              # SHAP 可解释性图
│   │   └── models/*.pkl            # 8 个训练好的模型
│   └── final_hpc/                  # HPC 结果（全量 + GridSearchCV）
│       ├── results_table.csv
│       ├── roc_curves.png
│       ├── performance_bar_chart*.png
│       ├── tcr_he_features_removed.json
│       ├── *_shap_*.png            # SHAP 图（KernelExplainer + link='logit'）
│       └── models/*.pkl            # 全量训练的模型文件
└── reports/
    ├── report_full.md              # 完整报告（英文/Chinese）
    └── report_cn.md
```

## Run

### Local (30K subsample, fast)
```bash
python scripts/train_final.py        # 输出到 results/final/
```

### HPC (full data + GridSearchCV, matching paper)
```bash
sbatch submit.sh                     # 输出到 results/final_hpc/
```

### 其他分析
```bash
python scripts/robustness_test.py    # 鲁棒性测试
python scripts/imbalance_stress_test.py  # 不平衡测试
python scripts/shap_top50.py         # SHAP Top-50 分析
python scripts/figure4_comparison.py # Figure4 复现
```

## Models

| Name | Split | Features | Training | Description |
|------|-------|----------|----------|-------------|
| RF | Epitope-hard | All 194 | Full | Random Forest (100 trees) |
| GBT | Epitope-hard | All 194 | Full | Gradient Boosting (100 estimators) |
| XGB | Epitope-hard | All 194 | Full | XGBoost (100 estimators) |
| SVM-RBF | Epitope-hard | All 194 | **Full + GridSearchCV** | SVM RBF, C=1.0 (paper baseline) |
| TCR-HE | Epitope-hard | Removed corr>0.8 | **Full + GridSearchCV** | SVM RBF, reduced features |
| TCR-Hβ | TCR-hard | Removed corr>0.8 | **Full + GridSearchCV** | SSR hard split |
| TCR-HβE | Strict | Removed corr>0.8 | **Full + GridSearchCV** | Epitope + TCR hard |
| TCR-RS | Random | Removed corr>0.8 | **Full + GridSearchCV** | Random 80/20 split |

Note: All SVM models use `SVC(probability=True)` + `GridSearchCV(cv=None)` (5-fold CV → refit on full data), matching the paper.

## ROC-AUC Methodology

ROC-AUC is computed using `predict()` (class labels 0/1) to match the paper's methodology. `AUC (proba)` using `predict_proba()` / `decision_function()` is also reported for reference.

## Data Splits

| Split | Train | Train Epitopes | Test | Test Epitopes |
|-------|-------|--------------|------|--------------|
| Epitope-hard | 199,988 | 816 | 54,449 | **65** (unseen, no overlap)|
| TCR-hard | 203,550 | — | 50,887 | — |
| Strict | 150,714 | — | 54,449 | — |
| Random | 203,549 | — | 50,888 | — |

## Key Results vs Paper (HPC — full data + GridSearchCV)

| Model | Paper | Ours (HPC) | Match |
|-------|-------|:----------:|:-----:|
| RF | 0.50 | **0.5000** | ✅ |
| GBT | 0.54 | **0.4969** | 🟡 slight ↓ |
| XGB | 0.51 | **0.5002** | ✅ |
| SVM-RBF | 0.80 | **0.8709** | ↑ higher |
| TCR-HE | 0.87 | **0.8785** | ✅ |
| TCR-Hβ | 0.92 | **0.9224** | ✅ |
| TCR-HβE | 0.89 | **0.8718** | 🟡 slight ↓ |
| TCR-RS | 0.92 | **0.9170** | ✅ |

## Methodology (vs original 30K code)

| Aspect | Original (results/final/) | HPC (results/final_hpc/) |
|--------|--------------------------|--------------------------|
| SVM training | 30K subsample | **Full 200K** |
| SVM method | `SVC(C=1.0).fit()` | **GridSearchCV(cv=None) + refit** |
| `probability=True` | No | **Yes** |
| Correlation removal | `rng.choice` random | **Deterministic (keep lower-index)** |
| SHAP | PermutationExplainer | **KernelExplainer + link='logit'** |
| TCR-RS seed | `random_state=42` | **`random_state=1` (paper's) |
