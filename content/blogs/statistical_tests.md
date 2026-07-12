---
title: "Which Statistical Test Should You Actually Use?"
date: December 2025
description: A practical guide to picking the right statistical test based on group count, distribution shape, and sample size, with Python snippets for each.
tags:
  - Statistics
  - Hypothesis Testing
  - Python
---

## Introduction

Choosing a test is less about memorizing names than about answering a few structured questions: how many groups you compare, whether the measurement scale and distribution suit parametric assumptions (normal vs non-normal), how large $n$ is per group, and how you control false positives across multiple tests when you compare many pairs.

Parametric tests (typical $t$-test and ANOVA) assume, among other things, that you are comparing means of roughly continuous data whose residuals (from mean) are *approximately normal* [1, 2]. Non-parametric rank tests relax the normality assumption; they suit *ordinal* data, heavy skew, or outliers [3, 4]. Omnibus tests answer whether any group differs (looking at the big picture); post-hoc tests say *which* pairs differ after an omnibus result (investigating the details).

---

## Comparing Two Groups

With two samples (for example, control versus treatment), you ask whether the difference is larger than you would expect from chance alone.

### Welch’s $t$-Test (Parametric)

The classical **Student $t$-test** assumes *equal variances* in both groups [2]. **Welch’s $t$-test** is the usual default for two independent groups because it does not assume equal variance; it remains valid when group sizes or spreads differ [5].

Use it when outcomes are continuous (or treated as such), approximately normal within each group (or $n$ is large enough that the logic below applies), and intervals between values are meaningful.

Welch’s statistic uses separate variance estimates in the denominator:

$$
t=\frac{\bar{X}_1-\bar{X}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}
$$

### Mann–Whitney $U$ Test (Non-Parametric)

When distributions are strongly skewed, have heavy tails, or are ordinal, comparing means with a $t$-test can mislead. The **Mann–Whitney $U$ test** ranks all observations together and compares rank sums between groups [3]. It does not assume a bell-shaped distribution; it asks whether one group tends to produce larger values than the other. For details, see [here](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test).

---

## Comparing Three or More Groups: Omnibus, Then Post-Hoc

With three or more groups (for example, placebo, low dose, high dose), running every pairwise $t$-test without adjustment inflates the chance of at least one false positive (**Type I error**).

The probability of at least one false positive across $k$ independent tests follows the formula below:

$$P(\text{at least one error}) = 1 - (1 - \alpha)^k$$

where $\alpha$ is the per-test Type I error rate and $k$ is the number of pairwise tests. If you have $n$ groups, the number of pairwise comparisons is $k = \frac{n(n-1)}{2}$. The table below shows how quickly this family-wise error rate grows past the nominal 5% ($\alpha = 0.05$) as the number of groups increases.

| Number of Groups | Number of Pairwise Tests ($k$) | Prob. of at least one False Positive |
| :--- | :--- | :--- |
| **2 Groups** | 1 | **5%** (Standard) |
| **3 Groups** | 3 | **14.2%** |
| **5 Groups** | 10 | **40.1%** |
| **10 Groups** | 45 | **90.1%** |

By the time you have 10 groups, you have a **90% chance** of finding a "significant" $p < 0.05$ result even if all your data is just random numbers.

The standard pattern for testing multiple groups is a two-stage analysis:

### Stage 1: “Is Anything Different?” (Omnibus)

**One-way ANOVA (parametric)** tests whether any group mean differs from the others [1]. A significant result means “not all means are equal”; it does not name which pairs differ.

**Kruskal–Wallis** is the rank-based analogue for three or more groups when normality is doubtful or data are ordinal [4]. Like the Mann–Whitney test, it works on ranks.

### Stage 2: “Which Pairs Differ?” (Post-Hoc)

If Stage 1 is significant at your chosen $\alpha$ (often $0.05$), use post-hoc comparisons to localize differences.

* After ANOVA, **Tukey’s HSD** (“Honestly Significant Difference”) compares all pairs while controlling the familywise error rate [6].
* After Kruskal–Wallis, run **pairwise Mann–Whitney** tests and adjust for multiplicity. A simple adjustment is **Bonferroni**: with $m$ comparisons, treat a result as significant only if $p < \alpha / m$ [7, 8]. For example, with $m = 3$ comparisons and $\alpha = 0.05$, require $p < 0.05/3 \approx 0.0167$.

**When to avoid Bonferroni correction**: With many groups, $m$ grows quickly (ten groups imply forty-five pairs), so Bonferroni becomes very strict unless $n$ is large or effects are strong. Bonferroni is conservative: it controls Type I errors tightly, but that same strictness often costs *power*, causing you to miss real effects (Type II errors). Reserve it for cases where a single false positive is genuinely costly. Otherwise, a less punishing correction like **FDR (False Discovery Rate, via Benjamini–Hochberg)** is usually preferable — it's the standard choice in omics data analysis, where thousands of features are tested at once.

FDR—often implemented with Benjamini–Hochberg (BH)—bounds the *expected fraction* of false positives among all rejections, unlike Bonferroni, which controls the chance of *any* false positive across the family. FDR is stricter than no correction but less conservative than Bonferroni when $m$ is large; it fits screening settings (many tests, a ranked list of candidates) where a few false leads are acceptable.

**Using FDR in Stage 2:** After a significant Kruskal–Wallis, run pairwise Mann–Whitney tests for all pairs, collect the $m$ raw $p$-values, and apply BH at level $Q$ (commonly $0.05$). Sort $p_{(1)} \le \cdots \le p_{(m)}$; find the largest $i$ such that $p_{(i)} \le (i/m)Q$, and reject the hypotheses corresponding to $p_{(1)},\ldots,p_{(i)}$ (standard step-up BH).

To sum up, prefer Bonferroni when a single false positive in the family is unacceptable; prefer BH when you want more power across many comparisons and can interpret results as *proportion* of noise among claimed hits.

---

## Quick Reference: Which Test?


| Situation                   | Parametric (roughly normal, continuous scale) | Non-parametric (skew, outliers, or ordinal)                                       |
| :-------------------------- | :-------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Two groups**              | Welch’s $t$-test                              | Mann–Whitney $U$                                                                  |
| **Three or more (Stage 1)** | One-way ANOVA                                 | Kruskal–Wallis                                                                    |
| **Three or more (Stage 2)** | Tukey HSD                                     | Pairwise Mann–Whitney with Bonferroni (or other multiplicity control like FDR-BH) |


---

## How Sample Size Changes the Decision

Per-group sample size **$n$** changes what you can assume and how much you should trust a normality check. There is no universal maximum $n$; the important issues are minimum $n$, power, and when the Central Limit Theorem (CLT) makes parametric tests on means reasonable [9].

### Very Small $n$

With roughly **$n < 5$** per group, estimates of mean and variance are unstable, normality tests are uninformative, and standard asymptotic $p$-values are hard to interpret. Prefer descriptive statistics, exact or permutation methods (see below) where available [12, 13], or collecting more data. Rank tests also need enough distinct ranks to attain small $p$-values; with **$n < 4$** per group in Mann–Whitney, the discrete null distribution may never reach conventional significance.

### Moderate $n$ (About 10–30 per Group)

This is the band where a normality diagnostic (for example Shapiro–Wilk on residuals [10]) is most often used as a tie-breaker: it has some power to detect clear non-normality without CLT having fully “smoothed” the sampling distribution of the mean. If normality is plausible, use Welch’s $t$-test or ANOVA; if not, prefer Mann–Whitney or Kruskal–Wallis. For borderline $n$, aim for at least **$n > 10$–$15$** per group before leaning hard on parametric tests when shapes are visibly odd.

### Large $n$ (Often Cited as $n \gtrsim 30$ per Group)

The CLT says that the **sampling distribution of the sample mean** tends toward normal as $n$ grows, even when individual observations are not normal. With large $n$, Welch’s $t$-test and ANOVA on means are often robust to moderate non-normality, and parametric tests typically have higher power than rank tests when means truly differ on a continuous scale. Extreme outliers can still distort means; in those cases, robust or non-parametric approaches remain attractive.

### Design Details That Interact With $n$

* **Unbalanced groups** (very different $n_k$): prefer Welch’s $t$-test over the equal-variance Student test; ANOVA assumptions should still be checked.
* **Many ties** (for example, many identical values): rank-based tests carry less information; investigate censoring, rounding, or zero-inflation.
* **Many post-hoc pairs**: Bonferroni’s threshold $\alpha/m$ shrinks as $m$ grows; alternative correction methods (like FDR-BH), large studies, or hierarchical modeling are sometimes used instead of dozens of pairwise tests.

### Ordinal Data and When Large $n$ Does *Not* Default to $t$-Tests

Ordinal scales (1–5 satisfaction, pain grades) do not have equal intervals between levels: the step from 1 to 2 is not necessarily comparable to 4 to 5 [11]. Means and $t$-tests assume a scale where averaging is meaningful. For ordinal outcomes, Mann–Whitney and Kruskal–Wallis remain appropriate even at very large $n$; CLT on the mean does not fix an inappropriate mean for the measurement scale.

---

## Sample-Size-Aware Summary

Use the table below together with the quick reference: pick the test family from group count and scale, then refine with $n$ and data type.


| Per-group $n$ (rule of thumb)    | Distribution / scale                           | Practical default                                                                           |
| :------------------------------- | :--------------------------------------------- | :------------------------------------------------------------------------------------------ |
| **Tiny ($n < 5$)**               | Unknown                                        | Exact, permutation, or descriptive; avoid strong conclusions from standard asymptotic tests |
| **Small ($n \approx 5$–$15$)**   | Normality hard to verify                       | Favor non-parametric (Mann–Whitney, Kruskal–Wallis)                                         |
| **Medium ($n \approx 15$–$30$)** | Passes normality check                         | Welch’s $t$-test or ANOVA                                                                   |
| **Medium ($n \approx 15$–$30$)** | Fails normality or heavy tails                 | Mann–Whitney or Kruskal–Wallis                                                              |
| **Large ($n \gtrsim 30$)**       | Continuous, interval-scale; mild non-normality | Welch’s $t$-test or ANOVA (CLT)                                                             |
| **Any $n$**                      | Ordinal                                        | Mann–Whitney or Kruskal–Wallis                                                              |


**Rules of thumb:** With small $n$, stay conservative and prefer non-parametric or exact methods. With large $n$ on continuous outcomes, parametric tests on means are often justified despite mild non-normality. For two-group parametric comparisons with unequal variance or unequal $n$, use Welch’s formulation. Always align the test with the scale of measurement (continuous vs ordinal), not only with $n$ and $p$-values.

For normality checking, Shapiro–Wilk is common on small-to-moderate samples [10]; on large $n$, trivial departures from normality may test “significant” while still being irrelevant for mean-based inference—use judgment, plots (histograms, Q–Q), and domain knowledge. Permutation tests [12, 13] and bootstrap confidence intervals [14] are useful complements when assumptions are doubtful or samples are small; see **Permutation tests** and **Bootstrapping** below.

---

## Permutation Tests

### What Permutation Tests Do

Under the null hypothesis, some labels or assignments are exchangeable—for example, which observations belong to group A versus B might not matter if both groups are draws from the same distribution. A permutation test reshuffles those assignments (or applies permutations consistent with the stated null), recomputes your statistic each time (difference of means, rank sum, etc.), and builds an empirical null distribution [12, 13]. That distribution answers how extreme your *observed* statistic would be if the null were true. The goal is usually a $p$-value, not a confidence interval.

### Permutation $p$-Values

The $p$-value is the proportion of permutations whose statistic is at least as extreme as the one from your actual data (one- or two-sided, depending on how you define extremity). With very small samples you can sometimes enumerate all distinct permutations (**exact** test); otherwise **Monte Carlo** permutation uses many random shuffles and introduces Monte Carlo error.

### How This Complements Test Choice

Permutation reasoning aligns with this guide’s emphasis on doubtful normality or small $n$: you avoid relying on a closed-form sampling distribution when it is untrusted. Rank tests such as Mann–Whitney connect to permutation logic for a particular statistic; *general permutation tests* let you specify other contrasts (mean difference, robust summaries) while retaining the same label-randomization logic.

### Limitations

Inference is only as plausible as *exchangeability under the null*. Mis-specified pairing, ignored blocking, or dependence across units can invalidate naive label shuffling—similar to bootstrap pitfalls. Exact enumeration grows combinatorially with $n$; Monte Carlo mitigates cost but leaves simulation noise.

**Bootstrapping** (next section) targets uncertainty about estimates (intervals); permutation targets *null distributions for $p$-values*. See **Bootstrapping** below for more detail.

An implementation for a *difference of means* using `scipy.stats.permutation_test` appears under **Python snippets**, in *Permutation test for a difference of means (optional)*.

---

## Bootstrapping

### What Bootstrapping Does

The bootstrap treats your observed sample as a stand-in for the population. You draw many *bootstrap samples* by **resampling with replacement** from the original data (same sample size per group, or a structure that matches your design—for example independent resampling within each group for a two-sample comparison). For each replicate you compute your statistic (difference of means, median, correlation, etc.). The resulting distribution approximates *sampling variability* of that estimator under repeated sampling [14].

### Bootstrap Confidence Intervals

The usual *percentile bootstrap interval* takes many bootstrap values of the statistic and uses empirical quantiles as bounds: for example, the 2.5th and 97.5th percentiles of the bootstrap distribution give an approximate *95% confidence interval*. Refinements exist (e.g. bias-corrected accelerated, BCa) when skew or bias make plain percentiles unreliable but those are beyond the scope of this guide.

### How This Complements Test Choice

Like permutation tests (see **Permutation tests** above), the bootstrap earns its keep when normality or other parametric assumptions are doubtful, or per-group $n$ is small [12, 13]. Reach for it specifically when the question is about an *interval* — how uncertain is this effect size — rather than a $p$-value. Neither method replaces careful design or a scale-appropriate test; both extend the toolkit for when classical assumptions don't hold.

### Limitations

Bootstrap intervals do not create information absent from the data; with very small $n$, intervals can be unstable. Complex dependence (clusters, time series, spatial structure) usually requires resampling schemes that respect that structure rather than naive independent-unit resampling.

An implementation for a **difference of means** using `scipy.stats.bootstrap` appears under **Python snippets**, in *Bootstrap confidence interval for a difference of means (optional)*.

---

## Python Snippets

The examples below use **SciPy** (`scipy.stats`) for the core tests and **statsmodels** for **Tukey HSD** and for **Bonferroni / Benjamini–Hochberg** adjustment of a vector of $p$-values. Install if needed: `pip install scipy statsmodels`.

```python
import numpy as np
from scipy import stats  # t-tests, ANOVA, ranks, normality, bootstrap, permutation_test
from itertools import combinations  # unique group pairs for post-hoc loops
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # Tukey HSD after ANOVA
from statsmodels.stats.multitest import multipletests  # Bonferroni, BH-FDR, etc.

# Toy data: three independent samples (replace with your measurements)
rng = np.random.default_rng(123)  # fixed seed so snippets are reproducible
g0 = rng.normal(0, 1, 40)  # “control”
g1 = rng.normal(0.8, 1.1, 35)  # different n and variance → illustrates Welch vs Student
g2 = rng.normal(0.2, 0.9, 50)  # third group for ANOVA / Kruskal–Wallis examples
```

### Welch and Student Two-Sample $t$-Tests

Welch is `equal_var=False` (recommended default for two independent groups).

```python
# Welch: does not assume equal variance across groups (default choice in the article)
tw = stats.ttest_ind(g0, g1, equal_var=False, alternative="two-sided")
print("Welch t:", tw.statistic, "p-value:", tw.pvalue)  # tw.statistic = t; tw.pvalue = two-sided p

# Student: classical equal-variance t-test (only if variances plausibly equal)
ts = stats.ttest_ind(g0, g1, equal_var=True, alternative="two-sided")
print("Student t:", ts.statistic, "p-value:", ts.pvalue)

# Use alternative="less" or "greater" for one-sided tests if your hypothesis is directional
```

### Mann–Whitney $U$ (Two Groups)

```python
# Non-parametric comparison of two independent samples (ranks; robust to skew/outliers)
mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
# mw.statistic is the U statistic (SciPy’s convention); mw.pvalue is asymptotic or exact depending on size
print("Mann–Whitney U:", mw.statistic, "p-value:", mw.pvalue)
```

### One-Way ANOVA and Kruskal–Wallis (Stage 1)

```python
# Stage 1 (parametric): any mean different? Significant F → proceed to Tukey (or contrasts)
f_res = stats.f_oneway(g0, g1, g2)
print("ANOVA F:", f_res.statistic, "p-value:", f_res.pvalue)

# Stage 1 (non-parametric): omnibus difference in distributions across groups
kw = stats.kruskal(g0, g1, g2)
print("Kruskal–Wallis H:", kw.statistic, "p-value:", kw.pvalue)
# If kw.pvalue < alpha, run pairwise Mann–Whitney + multiplicity adjustment (next section)
```

### Tukey HSD After ANOVA (Stage 2, Parametric)

```python
# Tukey needs one response vector + a parallel group label vector (same length as y)
y = np.concatenate([g0, g1, g2])
grp = np.array(["g0"] * len(g0) + ["g1"] * len(g1) + ["g2"] * len(g2))

# pairwise_tukeyhsd: all pairwise mean differences with familywise error at alpha
tukey = pairwise_tukeyhsd(endog=y, groups=grp, alpha=0.05)
print(tukey)  # columns: meandiff, p-adj, lower, upper, reject (True = pair differs)
```

### Pairwise Mann–Whitney With Bonferroni or BH-FDR (Stage 2, Non-Parametric)

Collect raw $p$-values for each pair, then adjust with `multipletests`.

```python
groups = {"g0": g0, "g1": g1, "g2": g2}
pairs, pvals = [], []

# Run every pairwise Mann–Whitney; order of pvals matches order of `pairs`
for a, b in combinations(groups.keys(), 2):
    _, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
    pairs.append((a, b))
    pvals.append(p)

# multipletests returns: reject, pvals_corrected, alphacSidak, alphacBonf (last two depend on method)
rej_b, p_bonf, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")  # familywise; stricter as m grows
rej_bh, p_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")  # Benjamini–Hochberg FDR at Q=alpha

for ab, p, rb, pb, rfb, pfb in zip(pairs, pvals, rej_b, p_bonf, rej_bh, p_bh):
    # rb / rfb = True if null for that pair is rejected after the chosen adjustment
    print(ab, "raw p =", p, "| Bonf reject", rb, "adj p =", pb, "| BH reject", rfb, "adj p =", pfb)
```

### Shapiro–Wilk Normality (Per Group or on Residuals)

SciPy’s `shapiro` accepts up to 5000 points per call. For ANOVA-style residuals, center each group by its mean before pooling, or use a dedicated model diagnostic.

```python
# H0: sample comes from a normal distribution; small p → evidence against normality
# SciPy limits length to 5000; for ANOVA residuals use model-based tests or pooled residuals
for name, x in [("g0", g0), ("g1", g1)]:
    w_stat, p = stats.shapiro(x)  # w_stat = Shapiro–Wilk W; p = p-value
    print(name, "Shapiro–Wilk W =", w_stat, "p =", p)
```

### Permutation Test for a Difference of Means (Optional)

As in the **Permutation tests** section above, this uses random label reassignment under the null that both samples come from the same distribution; the $p$-value compares the observed difference of means to the *null distribution*.

```python
def diff_means_perm(x, y, axis=-1):
    """Difference of sample means (vectorized along `axis` for resampling)."""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

perm_res = stats.permutation_test(
    (g0, g1),
    diff_means_perm,
    permutation_type="independent",  # pool and split between groups; unequal n allowed
    n_resamples=10_000,
    random_state=0,  # reproducible permutations (SciPy 1.15+ prefers `rng=` on new code)
)
print("Permutation test: statistic =", perm_res.statistic, "p-value =", perm_res.pvalue)
```

### Bootstrap Confidence Interval for a Difference of Means (Optional)

As in the **Bootstrapping** section above, this builds a percentile interval for the same contrast — but targets the sampling variability of the estimate rather than a null distribution.

```python
# SciPy passes each sample as a separate argument to `statistic(..., axis=-1)` when paired=False
def diff_means(a, b, axis=-1):
    """Difference of sample means (vectorized along `axis` for resampling)."""
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)

boot = stats.bootstrap(
    (g0, g1),  # tuple of arrays; resampling is independent across groups when paired=False
    diff_means,
    confidence_level=0.95,  # percentile interval for the statistic
    n_resamples=10_000,  # more resamples → smoother Monte Carlo error (slower)
    random_state=0,  # reproducible bootstrap draws
    paired=False,  # set True only if observations are matched pairs (same length required)
)
# boot.confidence_interval: (low, high) for mean(g0) - mean(g1)
print("Bootstrap 95% CI for mean(g0) - mean(g1):", boot.confidence_interval)
```

I hope the sections above serve as a concise, practical guide to choosing a statistical test that fits your question and your data. The numbered references keep author names and publication years in view—deliberately so—as a small acknowledgment of the researchers whose work, spanning more than a century, still underpins how we design and interpret analyses today.

---

## References

1. Fisher, R. A. (1925). *Statistical methods for research workers*. Oliver and Boyd.
2. Student. (1908). The probable error of a mean. *Biometrika*, *6*(1), 1–25. [https://doi.org/10.2307/2331554](https://doi.org/10.2307/2331554)
3. Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics*, *18*(1), 50–60. [https://doi.org/10.1214/aoms/1177730491](https://doi.org/10.1214/aoms/1177730491)
4. Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, *47*(260), 583–621. [https://doi.org/10.1080/01621459.1952.10483441](https://doi.org/10.1080/01621459.1952.10483441)
5. Welch, B. L. (1947). The generalization of “Student’s” problem when several different population variances are involved. *Biometrika*, *34*(1–2), 28–35. [https://doi.org/10.1093/biomet/34.1-2.28](https://doi.org/10.1093/biomet/34.1-2.28)
6. Tukey, J. W. (1949). Comparing individual means in the analysis of variance. *Biometrics*, *5*(2), 99–114. [https://doi.org/10.2307/3001913](https://doi.org/10.2307/3001913)
7. Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle probabilità. *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*, *8*, 3–62.
8. Miller, R. G. (1981). *Simultaneous statistical inference* (2nd ed.). Springer-Verlag.
9. Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury Press.
10. Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika*, *52*(3–4), 591–611. [https://doi.org/10.1093/biomet/52.3-4.591](https://doi.org/10.1093/biomet/52.3-4.591)
11. Stevens, S. S. (1946). On the theory of scales of measurement. *Science*, *103*(2684), 677–680. [https://doi.org/10.1126/science.103.2684.677](https://doi.org/10.1126/science.103.2684.677)
12. Fisher, R. A. (1935). *The design of experiments*. Oliver and Boyd.
13. Good, P. I. (2005). *Permutation, parametric, and bootstrap tests of hypotheses* (3rd ed.). Springer.
14. Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman and Hall.

