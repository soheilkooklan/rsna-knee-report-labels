# Tables recomputed from the released files

Differences are observed values on the 58 gold studies; intervals: 1000 bootstrap resamples of the studies (seed 0); p: paired permutation test with 1000 within-study swaps.

## Targets against gold-58 (target agreement AUC)

| Finding | Labels | Teacher | Flat | Silent only |
|---|---|---|---|---|
| ACL | 0.996 | 0.912 | 0.993 | 0.996 |
| MCL | 0.992 | 0.846 | 0.975 | 0.988 |
| Medial Meniscus | 0.955 | 0.934 | 0.968 | 0.959 |
| Lateral Meniscus | 0.875 | 0.852 | 0.929 | 0.879 |
| Medial OA | 0.905 | 0.947 | 0.952 | 0.889 |
| Lateral OA | 0.847 | 0.857 | 0.858 | 0.797 |
| PF OA | 0.945 | 0.809 | 0.915 | 0.893 |
| Effusion | 0.820 | 0.957 | 0.928 | 0.820 |
| Synovitis | 0.783 | 0.840 | 0.857 | 0.800 |
| Baker's | 0.831 | 0.913 | 0.895 | 0.848 |
| Contusion | 0.865 | 0.947 | 0.921 | 0.864 |
| Fracture | 0.890 | 0.932 | 0.935 | 0.899 |
| Macro | 0.892 | 0.896 | 0.927 | 0.886 |

- teacher-labels: +0.0036 [-0.0256, +0.0333], p 0.814
- flat-labels: +0.0352 [+0.0192, +0.0520], p <0.001
- flat-teacher: +0.0315 [+0.0127, +0.0508], p 0.003
- silent_only-labels: -0.0060 [-0.0188, +0.0067], p 0.286
- grouped-labels: +0.0435 [+0.0263, +0.0629], p <0.001

## Out-of-fold, ours and teacher

- gold macro ours 0.8761, teacher 0.8955, blend 0.8902
- ours-teacher: -0.0194 [-0.0387, -0.0020], p 0.045
- blend-teacher: -0.0053 [-0.0154, +0.0036], p 0.254
- round 2 target 0.9227 vs round 1 0.9270: -0.0044 [-0.0124, +0.0028]
- labels-only student: gold 0.8574, explicit 0.8591
- mixed - labels-only, gold: +0.0187 [+0.0033, +0.0361], p 0.006
- mixed - labels-only, explicit: +0.0036 [+0.0011, +0.0062]
- fold-0 runs (explicit, gold, explicit minus the seeded v3 prior run with 95% interval):
  - v1_noprior_unseeded: 0.8492, 0.8948, -0.0066 [-0.0144, +0.0002]
  - v3_noprior_seeded: 0.8602, 0.9235, +0.0043 [-0.0008, +0.0100]
  - v3_prior_seeded: 0.8559, 0.9052, +0.0000 [+0.0000, +0.0000]
  - v3_prior_seeded_repeat: 0.8559, 0.9052, +0.0000 [+0.0000, +0.0000]
  - v3_prior_seed2027: 0.8568, 0.9226, +0.0010 [-0.0037, +0.0056]
  - v3_prior_seed2028: 0.8567, 0.9085, +0.0008 [-0.0037, +0.0058]
  - v3_prior_labels_only_seeded: 0.8564, 0.9020, +0.0006 [-0.0049, +0.0057]
  - v3_prior_unseeded: 0.8606, 0.9283, +0.0048 [+0.0001, +0.0098]
  - teacher: 0.8710, 0.9206, +0.0151 [+0.0059, +0.0243]

## Explaining away (observed gold difference, bootstrap interval, permutation p, Holm over the 15 negative settings)

| Scores | lambda | explicit | gold | gold diff [95% CI] | p | p Holm | p DeLong | DeLong Holm |
|---|---|---|---|---|---|---|---|---|
| teacher | -0.3 | 0.8787 | 0.8984 | +0.0029 [-0.0063, +0.0125] | 0.449 | 1.000 | 0.514 | 1.000 |
| teacher | -0.2 | 0.8800 | 0.8989 | +0.0034 [-0.0040, +0.0109] | 0.262 | 1.000 | 0.335 | 1.000 |
| teacher | -0.15 | 0.8802 | 0.8997 | +0.0042 [-0.0019, +0.0103] | 0.056 | 0.539 | 0.121 | 1.000 |
| teacher | -0.1 | 0.8800 | 0.8990 | +0.0035 [-0.0003, +0.0073] | 0.007 | 0.091 | 0.038 | 0.493 |
| teacher | -0.05 | 0.8797 | 0.8969 | +0.0014 [-0.0013, +0.0041] | 0.054 | 0.539 | 0.242 | 1.000 |
| teacher | 0.0 | 0.8791 | 0.8955 |  |  |  |  |  |
| teacher | 0.05 | 0.8782 | 0.8946 |  |  |  |  |  |
| teacher | 0.1 | 0.8772 | 0.8926 |  |  |  |  |  |
| teacher | 0.2 | 0.8750 | 0.8868 |  |  |  |  |  |
| teacher | 0.3 | 0.8725 | 0.8832 |  |  |  |  |  |
| ours | -0.3 | 0.8649 | 0.8804 | +0.0043 [-0.0066, +0.0170] | 0.379 | 1.000 | 0.458 | 1.000 |
| ours | -0.2 | 0.8650 | 0.8800 | +0.0038 [-0.0043, +0.0129] | 0.263 | 1.000 | 0.357 | 1.000 |
| ours | -0.15 | 0.8647 | 0.8784 | +0.0023 [-0.0040, +0.0099] | 0.410 | 1.000 | 0.480 | 1.000 |
| ours | -0.1 | 0.8642 | 0.8794 | +0.0033 [-0.0017, +0.0096] | 0.096 | 0.767 | 0.183 | 1.000 |
| ours | -0.05 | 0.8635 | 0.8783 | +0.0022 [-0.0016, +0.0074] | 0.226 | 1.000 | 0.303 | 1.000 |
| ours | 0.0 | 0.8627 | 0.8761 |  |  |  |  |  |
| ours | 0.05 | 0.8617 | 0.8730 |  |  |  |  |  |
| ours | 0.1 | 0.8606 | 0.8698 |  |  |  |  |  |
| ours | 0.2 | 0.8582 | 0.8634 |  |  |  |  |  |
| ours | 0.3 | 0.8558 | 0.8598 |  |  |  |  |  |
| blend | -0.3 | 0.8823 | 0.8965 | +0.0063 [-0.0049, +0.0174] | 0.197 | 1.000 | 0.242 | 1.000 |
| blend | -0.2 | 0.8821 | 0.8972 | +0.0070 [-0.0005, +0.0156] | 0.039 | 0.429 | 0.068 | 0.750 |
| blend | -0.15 | 0.8816 | 0.8968 | +0.0066 [+0.0001, +0.0135] | 0.011 | 0.132 | 0.040 | 0.493 |
| blend | -0.1 | 0.8808 | 0.8962 | +0.0060 [+0.0015, +0.0109] | <0.001 | 0.015 | 0.009 | 0.132 |
| blend | -0.05 | 0.8799 | 0.8936 | +0.0034 [+0.0005, +0.0068] | <0.001 | 0.015 | 0.009 | 0.132 |
| blend | 0.0 | 0.8789 | 0.8902 |  |  |  |  |  |
| blend | 0.05 | 0.8777 | 0.8896 |  |  |  |  |  |
| blend | 0.1 | 0.8764 | 0.8872 |  |  |  |  |  |
| blend | 0.2 | 0.8736 | 0.8841 |  |  |  |  |  |
| blend | 0.3 | 0.8707 | 0.8772 |  |  |  |  |  |

## Corrections: change in explicit and gold macro AUC

| Correction | teacher d_explicit | teacher d_gold | ours d_explicit | ours d_gold | blend d_explicit | blend d_gold |
|---|---|---|---|---|---|---|
| none | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| sex | +0.0000 | -0.0047 | -0.0003 | -0.0049 | -0.0005 | -0.0041 |
| model_centring_0.25 | -0.0018 | -0.0062 | -0.0014 | -0.0081 | -0.0015 | -0.0055 |
| model_centring_1.0 | -0.0207 | -0.0381 | -0.0171 | -0.0489 | -0.0193 | -0.0456 |
| model_prior_0.1 | +0.0029 | +0.0020 | +0.0033 | -0.0031 | +0.0029 | +0.0008 |
| manufacturer_centring_0.25 | -0.0011 | -0.0020 | -0.0014 | -0.0020 | -0.0011 | +0.0005 |
| manufacturer_centring_1.0 | -0.0089 | -0.0127 | -0.0094 | -0.0159 | -0.0095 | -0.0134 |
| manufacturer_prior_0.1 | +0.0030 | +0.0006 | +0.0032 | +0.0002 | +0.0030 | +0.0014 |
| field_centring_0.25 | -0.0015 | +0.0008 | -0.0009 | -0.0010 | -0.0012 | +0.0015 |
| field_centring_1.0 | -0.0084 | -0.0026 | -0.0044 | -0.0003 | -0.0064 | +0.0016 |
| boost_0.1 | -0.0017 | -0.0021 | -0.0022 | -0.0063 | -0.0025 | -0.0025 |
| boost_0.2 | -0.0040 | -0.0080 | -0.0046 | -0.0124 | -0.0053 | -0.0054 |
| ea_-0.1 | +0.0010 | +0.0035 | +0.0015 | +0.0033 | +0.0020 | +0.0060 |
| ea_-0.2 | +0.0009 | +0.0034 | +0.0023 | +0.0038 | +0.0032 | +0.0070 |

## DeLong-type check (difference, standard error, two-sided p)

- teacher-labels: +0.0036, se 0.0146, p 0.804
- flat-labels: +0.0352, se 0.0078, p <0.001
- flat-teacher: +0.0315, se 0.0099, p 0.001
- ours-teacher: -0.0194, se 0.0095, p 0.041
- blend-teacher: -0.0053, se 0.0046, p 0.250
- mixed-labels_only_student: +0.0187, se 0.0073, p 0.010

## Report state on the gold studies (expert positive: present/absent/silent | expert negative: present/absent/silent)

- ACL: 23/1/0 | 0/30/4
- MCL: 7/2/0 | 0/42/7
- Medial Meniscus: 24/1/1 | 5/25/2
- Lateral Meniscus: 19/3/1 | 5/23/7
- Medial OA: 11/4/0 | 2/27/14
- Lateral OA: 5/5/1 | 5/21/21
- PF OA: 10/6/5 | 0/27/10
- Effusion: 18/13/4 | 2/19/2
- Synovitis: 11/2/14 | 3/1/27
- Baker's: 8/3/1 | 2/15/29
- Contusion: 17/1/1 | 15/15/9
- Fracture: 13/3/2 | 5/21/14

## Omission rate and teacher advantage: Spearman 0.538 (p 0.076) over all reports, 0.336 (p 0.283) over the gold reports

## Side audit: 1772 of 24371 series without a side; 172 of 4407 studies with conflicting sides; gold: 3 conflicting, 9 with an unresolved series, 12 either

| Gold subset | n | labels | teacher | flat | flat-labels [CI] | flat-teacher [CI] | teacher-labels [CI] |
|---|---|---|---|---|---|---|---|
| all | 58 | 0.8919 | 0.8955 | 0.9270 | +0.0352 [+0.0192, +0.0520] | +0.0315 [+0.0127, +0.0508] | +0.0036 [-0.0256, +0.0333] |
| no_conflict | 55 | 0.8924 | 0.8972 | 0.9273 | +0.0348 [+0.0188, +0.0517] | +0.0300 [+0.0106, +0.0509] | +0.0048 [-0.0256, +0.0351] |
| no_conflict_no_unresolved | 46 | 0.8820 | 0.8874 | 0.9231 | +0.0411 [+0.0239, +0.0572] | +0.0356 [+0.0116, +0.0607] | +0.0055 [-0.0290, +0.0395] |
