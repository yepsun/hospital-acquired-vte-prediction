# 临床评分条目 → 数据源映射表（M2 Task 6）

人群：MIMIC-IV 入院队列 N = 422,408（`results_vte/vte_labels.parquet`）。脚本：`scripts/36_clinical_scores.py`。
产出：`results_vte/clinical_scores.parquet`（hadm_id, padua_score, padua_high, improve_score, improve_high, improve_dd, improve_dd_high）。

高危阈值：Padua ≥ 4；IMPROVE / IMPROVE-DD ≥ 4（均按原版定义）。

## Padua Prediction Score（11 条，≥4 高危）

| 条目 | 分值 | 数据源 / 映射 | 类型 | 局限说明 |
|---|---|---|---|---|
| 活动性癌症 | 3 | `cancer_active`（本次入院 ICD C 前缀/ICD-9 140–208 等） | 直接 | 出院编码覆盖住院全程，无法确认入院时"活动性"；Padua 原定义要求活动/近期治疗，此处偏宽松 |
| 既往 VTE | 3 | `prior_vte_any`（既往入院 VTE 编码 + 本次入院 present-on-admission VTE 编码） | 直接 | ICD 编码敏感性有限，既往史低估 |
| 制动 ≥3 天 | 3 | **代理**：`icu_los_first24h > 0`（入院 24h 内入 ICU） | **代理** | MIMIC 无卧床/制动医嘱数据。入 ICU 者多为重症/制动，但 ① ICU≠全部制动（轻症 ICU 监护），② 普通病房制动患者完全漏检。方向：制动条目仅在 ICU 亚人群可识别，选择性偏倚明显 |
| 已知易栓症 | 3 | `thrombophilia`（ICD：D68.5x/D68.6x 等） | 直接 | ICD 对遗传性易栓症敏感性低，低估 |
| 近期（≤1 月）手术/创伤 | 2 | `surgery_flag or trauma_flag`（本次入院手术/创伤编码） | **代理** | 无法区分"入院前 1 个月内"与本次住院期间手术/创伤；时间窗不符合原定义，QC 注明 |
| 年龄 ≥70 岁 | 2 | `age` | 直接 | 无 |
| 心衰 / 呼衰 | 1 | `heart_failure or copd`（ICD） | 直接 | COPD 代呼衰，无急性呼衰特异编码区分 |
| 急性心梗 / 缺血性卒中 | 1 | `mi or stroke`（本次入院 ICD） | 直接 | 未区分急性 vs 陈旧，偏宽松 |
| 急性感染 / 风湿病 | 1 | `infection_severe or rheumatologic`（ICD） | 直接 | infection_severe 仅重症感染（败血症等），普通急性感染漏检 |
| 肥胖（BMI ≥30） | 1 | `bmi >= 30 or obesity_icd` | 直接+回退 | bmi 缺失 32%（ICU 子人群才有身高体重），以 ICD 肥胖编码回退；非 ICU 且无编码者漏检 |
| 激素治疗 | 1 | `rx_hormone`（住院用药：雌激素/孕激素/他莫昔芬等） | 直接 | 仅院内给药记录，院外长期激素（如 OCP）漏检 |

## IMPROVE VTE RAM（7 条 + D-二聚体扩展，≥4 高危）

| 条目 | 分值 | 数据源 / 映射 | 类型 | 局限说明 |
|---|---|---|---|---|
| 既往 VTE | 3 | `prior_vte_any` | 直接 | 同 Padua |
| 已知易栓症 | 2 | `thrombophilia` | 直接 | 同 Padua |
| 下肢瘫痪/制动 | 2 | **代理**：`icu_los_first24h > 0`（同 Padua 制动代理） | **代理** | 无瘫痪/制动直接数据，同 Padua 局限 |
| 活动性癌症 | 2 | `cancer_active` | 直接 | 同 Padua |
| ICU/CCU 住院 | 1 | `icu_los_first24h > 0` | 直接 | 仅取首个 24h 内 ICU；原版含整个住院期 ICU，略保守 |
| 中心静脉导管 | 1 | `proc_cvc`（ICD 操作码） | 直接 | 操作编码可能低估床旁置管 |
| 年龄 >60 岁 | 1 | `age` | 直接 | 无 |
| D-二聚体 ≥2×ULN（IMPROVE-DD 扩展） | 2 | `lab_ddimer_median >= 1000 ng/mL`（FEU，ULN=500） | 直接但**高缺失** | **ddimer 24h 窗覆盖仅 ~0.7–0.8%（3,320/422,408）；缺失按 0 分处理，improve_dd 与 improve 几乎相同，基本不可推广，仅供有测量者子集参考** |

## 总体说明

1. **制动代理是最大局限**：Padua 制动（3 分）与 IMPROVE 瘫痪/制动（2 分）共用 `icu_los_first24h>0` 代理。Padua 高危比例因此达 39.8%，高于内科原始队列文献值（~10–25% 预期区间）——本队列为全入院人群（含外科/ICU），手术/创伤条目阳性率 36%、高龄 30%，本身风险谱高于 Padua 原始内科队列，制动代理进一步放大了高危比例。
2. **合并症均取自本次入院出院 ICD 编码**（VTE 相关码已按 M2 黑名单剔除，防结局泄漏），时间属性为"住院全程"，作为入院基线合并症系文献惯例，但"活动性/急性"属性不可严格验证。
3. 两评分高危组 VTE 事件率均显著高于低危组（Padua 4.6×、IMPROVE 3.5×），区分度方向正确。
