# Statistical Significance: Full Fine-Tuning vs. LoRA

Paired comparison across the same 3 seeds (42, 123, 2024) used for both conditions. Paired t-test is the primary test; Wilcoxon signed-rank is reported as a non-parametric cross-check given the small sample size.

**Caveat: with only n=3 seeds per condition, statistical power is low. A 'not significant' result here means the difference could not be distinguished from noise at this sample size -- it does not prove the two methods are equivalent.**

| Model | Shot | Metric | Baseline Mean | LoRA Mean | Diff | t-test p | Wilcoxon p | Result |
|---|---|---|---|---|---|---|---|---|
| mbert | 5 | accuracy | 0.3779 | 0.3629 | -0.0150 | 0.3074 | 0.5000 | not significant at alpha=0.05 |
| mbert | 5 | f1 | 0.3662 | 0.3330 | -0.0331 | 0.1645 | 0.2500 | not significant at alpha=0.05 |
| mbert | 10 | accuracy | 0.3929 | 0.3807 | -0.0122 | 0.6662 | 0.7500 | not significant at alpha=0.05 |
| mbert | 10 | f1 | 0.3806 | 0.3780 | -0.0027 | 0.9242 | 1.0000 | not significant at alpha=0.05 |
| mbert | 20 | accuracy | 0.4462 | 0.4174 | -0.0288 | 0.2214 | 0.5000 | not significant at alpha=0.05 |
| mbert | 20 | f1 | 0.4412 | 0.4086 | -0.0326 | 0.3216 | 0.5000 | not significant at alpha=0.05 |
| xlmr | 5 | accuracy | 0.4324 | 0.3864 | -0.0460 | 0.3056 | 0.5000 | not significant at alpha=0.05 |
| xlmr | 5 | f1 | 0.4042 | 0.3352 | -0.0691 | 0.3879 | 0.5000 | not significant at alpha=0.05 |
| xlmr | 10 | accuracy | 0.5175 | 0.4750 | -0.0424 | 0.3496 | 0.5000 | not significant at alpha=0.05 |
| xlmr | 10 | f1 | 0.5094 | 0.4690 | -0.0404 | 0.3877 | 0.5000 | not significant at alpha=0.05 |
| xlmr | 20 | accuracy | 0.5735 | 0.5786 | +0.0051 | 0.7937 | 0.7500 | not significant at alpha=0.05 |
| xlmr | 20 | f1 | 0.5659 | 0.5760 | +0.0100 | 0.6028 | 0.7500 | not significant at alpha=0.05 |
| mt5 | 5 | accuracy | 0.3491 | 0.3657 | +0.0166 | 0.1503 | 0.2500 | not significant at alpha=0.05 |
| mt5 | 5 | f1 | 0.2157 | 0.2686 | +0.0529 | 0.0535 | 0.2500 | not significant at alpha=0.05 |
| mt5 | 10 | accuracy | 0.3635 | 0.3430 | -0.0205 | 0.0979 | 0.2500 | not significant at alpha=0.05 |
| mt5 | 10 | f1 | 0.2666 | 0.2069 | -0.0597 | 0.0291 | 0.2500 | significant at alpha=0.05 |
| mt5 | 20 | accuracy | 0.3856 | 0.3300 | -0.0557 | 0.1900 | 0.2500 | not significant at alpha=0.05 |
| mt5 | 20 | f1 | 0.3087 | 0.1794 | -0.1292 | 0.1851 | 0.2500 | not significant at alpha=0.05 |