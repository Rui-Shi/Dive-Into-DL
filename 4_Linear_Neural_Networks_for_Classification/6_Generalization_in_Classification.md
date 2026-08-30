# 4.6 Generalization in Classification

We routinely report a test accuracy and treat it as *the* answer. But a test
set is a finite sample, so that number is an estimate with error bars. This
section makes the estimate quantitative and asks three questions:

1. How many test examples do we need before we believe the number?
2. What breaks when the same test set is reused across many models?
3. Why should fitting *training* data tell us anything about *unseen* data at
   all?

The first two are answered with elementary probability; the third is the
subject of statistical learning theory, which gives a real (if pessimistic)
answer.

---

## 4.6.1 The Test Set

### Two errors

For a classifier $f$ and a dataset $\mathcal{D}$ of $n$ examples, the
**empirical error** is the fraction it gets wrong:

$$\epsilon_{\mathcal{D}}(f) = \frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}\!\left(f(\mathbf{x}^{(i)}) \neq y^{(i)}\right)$$

The quantity we actually care about is the **population error** (the
**generalization error**) — the probability of a mistake on a fresh draw from
the underlying distribution $P$:

$$\epsilon(f) = \mathbb{E}_{(\mathbf{x}, y) \sim P}
\left[\mathbf{1}\!\left(f(\mathbf{x}) \neq y\right)\right]$$

We can never compute $\epsilon(f)$: it is an expectation over a distribution
we only ever see samples from. The empirical error on a **test set** is our
estimate of it.

### The test error is a sample mean of coin flips

Suppose $\mathcal{D}$ was drawn iid from $P$ and — crucially — was **not used
to fit or select** $f$. Then each indicator
$\mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)})$ is an independent **Bernoulli**
random variable with success probability $\epsilon(f)$, and $\epsilon_{\mathcal{D}}(f)$
is their sample mean. Two consequences follow immediately:

- The estimate is **unbiased**: $\mathbb{E}[\epsilon_{\mathcal{D}}(f)] = \epsilon(f)$.
- The number of mistakes $n\,\epsilon_{\mathcal{D}}(f)$ is
  $\mathrm{Binomial}(n, \epsilon(f))$.

So estimating test accuracy is exactly the problem of estimating the bias of a
coin from $n$ flips — a problem statistics solved long ago.

### How noisy is it?

A Bernoulli variable with parameter $\epsilon$ has variance
$\epsilon(1-\epsilon)$, so the sample mean has

$$\sigma^2 = \frac{\epsilon(f)\,(1 - \epsilon(f))}{n}
\quad\Longrightarrow\quad
\sigma = \sqrt{\frac{\epsilon(f)(1-\epsilon(f))}{n}} \le \frac{1}{2\sqrt{n}}$$

The bound uses $\epsilon(1-\epsilon) \le 1/4$, maximized at $\epsilon = 1/2$ —
a classifier at chance level is the hardest one to measure, while a very good
(or very bad) classifier is measured more precisely for free.

| test set size $n$ | worst-case std. dev. $1/(2\sqrt{n})$ |
| --- | --- |
| 100 | 0.050 |
| 1,000 | 0.016 |
| 2,500 | 0.010 |
| 10,000 | 0.005 |
| 100,000 | 0.0016 |

The headline is the $\mathcal{O}(1/\sqrt{n})$ rate: **to halve the error bar
you must quadruple the test set.**

### The central limit theorem view

The CLT says the standardized error is asymptotically normal:

$$\sqrt{n}\,\big(\epsilon_{\mathcal{D}}(f) - \epsilon(f)\big)
\ \xrightarrow{\ d\ }\ \mathcal{N}\big(0,\ \epsilon(f)(1-\epsilon(f))\big)$$

For a normal, roughly 95% of the mass sits within $\pm 2\sigma$. Plugging in
the worst case $\sigma \le 1/(2\sqrt{n})$:

- $n = 2500 \Rightarrow \sigma \le 0.01$: the estimate lands within $\pm 0.01$
  at least about 68% of the time (one standard deviation).
- $n = 10000 \Rightarrow \sigma \le 0.005$: the estimate lands within
  $2\sigma = \pm 0.01$ at least about **95%** of the time.

This is where the folklore "**a test set of ~10,000 examples pins accuracy to
about $\pm 1\%$**" comes from — and it is no accident that Fashion-MNIST and
CIFAR-10 both ship test sets of exactly 10,000 examples.

### A finite-sample guarantee: Hoeffding's inequality

The CLT is an *asymptotic* statement; it says nothing rigorous at a fixed $n$,
and normal approximations are least trustworthy exactly in the tails we are
quoting. **Hoeffding's inequality** gives a bound that holds for every $n$.
For iid random variables bounded in $[0,1]$ with sample mean $\bar{X}$:

$$P\big(\bar{X} - \mathbb{E}[\bar{X}] \ge t\big) \le \exp(-2nt^2),
\qquad
P\big(|\bar{X} - \mathbb{E}[\bar{X}]| \ge t\big) \le 2\exp(-2nt^2)$$

Indicators live in $\{0,1\} \subset [0,1]$, so this applies directly to our
test error:

$$P\big(\epsilon_{\mathcal{D}}(f) - \epsilon(f) \ge t\big) \le \exp(-2nt^2)$$

Set the right-hand side to a failure probability $\delta$ and solve for $n$:

$$n \ge \frac{\log(1/\delta)}{2t^2}
\qquad\text{(one-sided)},
\qquad
n \ge \frac{\log(2/\delta)}{2t^2}
\qquad\text{(two-sided)}$$

**Worked example.** Take $t = 0.01$ and $\delta = 0.05$ (95% confidence):

$$n \ge \frac{\log 20}{2 (0.01)^2} = \frac{2.996}{0.0002} \approx 15{,}000
\qquad\text{one-sided}$$

$$n \ge \frac{\log 40}{2 (0.01)^2} = \frac{3.689}{0.0002} \approx 18{,}500
\qquad\text{two-sided}$$

Read the other way round, with $n = 10{,}000$ fixed Hoeffding certifies
$|\epsilon_{\mathcal{D}}(f) - \epsilon(f)| \le 0.0136$ at 95% confidence — a
bit looser than the CLT's $\pm 0.01$, because Hoeffding is a worst-case,
distribution-free, non-asymptotic bound while the CLT is an approximation that
happens to be very good here.

**Why one-sided is often enough.** In practice we mostly fear one direction of
error: that the test set *flatters* the model and the true error is worse than
reported. Guarding only against that costs $\log(1/\delta)$ instead of
$\log(2/\delta)$ — worth about 3,500 examples in the calculation above.

**The $1/\sqrt{n}$ tax is brutal.** Because $n$ scales like $1/t^2$, pushing
the resolution from $\pm 10^{-2}$ to $\pm 10^{-4}$ at 99.9% confidence needs

$$n \ge \frac{\log 1000}{2 (10^{-4})^2} \approx 3.5 \times 10^{8}$$

hundreds of millions of labeled test examples. Distinguishing two models whose
accuracies differ in the fourth decimal place is usually not statistically
possible with the test sets we have.

> **The fine print that matters most:** every bound above assumes $f$ was
> fixed *before* $\mathcal{D}$ was touched. That assumption is what the next
> section destroys.

---

## 4.6.2 Test Set Reuse

### The multiple hypothesis testing problem

Suppose you evaluate $k$ classifiers on the same test set and give each one a
95% confidence interval. Each interval individually fails 5% of the time, but
the probability that *at least one* of them is misleading grows fast. If the
tests were independent:

$$P(\text{at least one interval fails}) = 1 - (1 - 0.05)^k$$

For $k = 20$ models this is $1 - 0.95^{20} \approx 0.64$. In other words, with
20 models on the board you have essentially **no power** to rule out the
possibility that one of the scores is a fluke. The model with the best test
number is systematically likely to be one that got lucky, so its test score is
an **optimistically biased** estimate of its population error — the same
"winner's curse" that plagues clinical trials and A/B testing.

### Correcting for $k$: the union bound

If the $k$ classifiers are fixed *in advance*, the fix is a union bound over
Hoeffding:

$$P\big(\exists\, j \le k:\ \epsilon_{\mathcal{D}}(f_j) - \epsilon(f_j) \ge t\big)
\le k \exp(-2nt^2)$$

Setting the right-hand side to $\delta$ and solving for $t$:

$$t = \sqrt{\frac{\log k + \log(1/\delta)}{2n}}$$

The extra cost enters only as $\log k$, which is remarkably cheap. With
$n = 10{,}000$ and $\delta = 0.05$:

| number of models $k$ | valid interval $t$ |
| --- | --- |
| 1 | 0.0122 |
| 10 | 0.0163 |
| 100 | 0.0195 |
| 1,000 | 0.0223 |

Evaluating a *thousand* models instead of one does not even double the width
of the interval. That is the good news.

### Adaptive overfitting

The bad news is that real research does not evaluate a pre-registered list of
$k$ models. We look at the test score, then decide what to try next: tweak the
architecture, change the learning rate, add augmentation. Information from the
test set leaks back into the model through **the researcher**, and after
enough rounds the test set is being fitted — slowly, by hand. This is called
**adaptive overfitting**, and it is worse than plain multiple testing because:

- $k$ is not fixed in advance, so the union bound does not apply. The relevant
  hypothesis class is everything you *might* have tried, which is far larger
  than what you did try.
- Whole *communities* share a benchmark. The effective $k$ for ImageNet is not
  one lab's experiments but a decade of them.

Empirically the damage seems real but not catastrophic. When researchers built
brand-new test sets for CIFAR-10 and ImageNet following the original
collection protocols, every model's absolute accuracy dropped noticeably — but
the *ranking* of models was largely preserved, which suggests the community had
been tracking genuine progress rather than pure test-set noise.

### Practical hygiene

- **Select on a validation set, report on a test set.** All hyperparameter
  tuning, early stopping, and architecture search go through validation data.
- **Touch the test set as few times as possible**, ideally once, at the end.
- **Widen your intervals by $\log k$** if you did compare many models on it.
- **Refresh test sets** between rounds of a benchmark when you can afford it.
- **Be extra careful** when the test set is small, when the differences you
  care about are fractions of a percent, or when the stakes are high (medicine,
  safety, credit).

---

## 4.6.3 Statistical Learning Theory

### Why the training error is not an estimate

Everything in 4.6.1 hinged on $f$ being independent of the evaluation data.
The classifier $f_{\mathcal{S}}$ produced by training on a set $\mathcal{S}$ is
the opposite: it was chosen *because* it does well on $\mathcal{S}$. So

$$\mathbb{E}\big[\epsilon_{\mathcal{S}}(f_{\mathcal{S}})\big]
\ \le\ \mathbb{E}\big[\epsilon(f_{\mathcal{S}})\big]$$

is the typical situation — training error is **optimistically biased**, and the
gap $\epsilon(f_{\mathcal{S}}) - \epsilon_{\mathcal{S}}(f_{\mathcal{S}})$ is the
**generalization gap** we want to bound.

### Uniform convergence

The standard move is to stop trying to bound the error of the *one* function
the learner picked, and instead bound *all* of them at once. We want: with
probability at least $1 - \delta$ over the draw of $\mathcal{S}$,

$$\sup_{f \in \mathcal{F}} \big|\epsilon_{\mathcal{S}}(f) - \epsilon(f)\big| \le \alpha$$

This is **uniform convergence**. If it holds, then whichever $f_{\mathcal{S}}$
the training algorithm returns — even one chosen adversarially — its training
error is within $\alpha$ of its population error. The learner's data-dependent
choice no longer matters.

For a **finite** class $\mathcal{F}$ the union bound from the previous section
does the job immediately, with $k = |\mathcal{F}|$:

$$\alpha = \sqrt{\frac{\log |\mathcal{F}| + \log(2/\delta)}{2n}}$$

So what governs generalization is the *logarithm* of the number of candidate
functions, relative to $n$.

### From Glivenko–Cantelli to VC dimension

Interesting function classes are infinite, so $\log|\mathcal{F}| = \infty$ and
the above is vacuous. The inspiration for doing better is the
**Glivenko–Cantelli theorem**, which says the empirical CDF of $n$ iid samples
converges to the true CDF *uniformly over all thresholds*:

$$\sup_{z} \big|\hat{F}_n(z) - F(z)\big| \ \longrightarrow\ 0
\quad\text{almost surely}$$

and the Dvoretzky–Kiefer–Wolfowitz inequality makes it quantitative with
exactly the Hoeffding shape,
$P\big(\sup_z |\hat{F}_n(z) - F(z)| > t\big) \le 2\exp(-2nt^2)$. There are
infinitely many thresholds, yet the bound is no worse than for a single one.
The reason is combinatorial: on $n$ fixed points, the infinitely many
thresholds produce only $n+1$ distinct labelings.

**Vapnik and Chervonenkis** turned that counting argument into a general
theory. The **VC dimension** of a class $\mathcal{F}$ is the largest number of
points $m$ such that *some* set of $m$ points can be **shattered** — labeled in
all $2^m$ possible ways by members of $\mathcal{F}$.

- Half-lines $\mathbf{1}(x \ge a)$ on $\mathbb{R}$: VC dimension $1$.
- Intervals on $\mathbb{R}$: VC dimension $2$.
- Linear classifiers on $\mathbb{R}^d$ (with bias): VC dimension $d + 1$. In
  the plane, a line realizes all $2^3 = 8$ labelings of three points in general
  position, but no set of four points can be shattered — for four points in
  convex position the XOR-style labeling of opposite corners is unachievable.
- Axis-aligned rectangles in the plane: VC dimension $4$.
- $f(x) = \mathrm{sign}(\sin(\omega x))$: **infinite** VC dimension with a
  single parameter — capacity is not parameter count.

### The VC generalization bound

The central result: with probability at least $1 - \delta$, simultaneously for
every $f \in \mathcal{F}$,

$$\epsilon(f) \le \epsilon_{\mathcal{S}}(f) + \alpha,
\qquad
\alpha \le c\sqrt{\frac{\mathrm{VC}(\mathcal{F}) - \log \delta}{n}}$$

for a universal constant $c$. What it tells us is worth stating plainly:

- The gap shrinks as $\mathcal{O}\!\big(\sqrt{\mathrm{VC}/n}\big)$, so you need
  roughly **as many examples as the VC dimension** before the bound says
  anything at all.
- It is the formal version of "more complex models need more data", and it
  motivates **structural risk minimization**: choose the model class by
  balancing training error against a capacity penalty.
- It is **distribution-free** — it holds for *every* data distribution $P$,
  with no assumptions whatsoever.

### What it does not explain

That last strength is also the fatal weakness. A bound that must survive the
worst conceivable distribution will be loose for the benign ones we actually
encounter:

- **VC bounds are wildly pessimistic.** They typically demand orders of
  magnitude more data than practitioners find sufficient.
- **Deep networks break the bound outright.** A modern network has more
  parameters than the dataset has examples, and can fit *randomly shuffled*
  labels to zero training error — which means its VC dimension is at least $n$.
  Plugging $\mathrm{VC} \ge n$ into $\sqrt{\mathrm{VC}/n}$ yields $\alpha \ge c$,
  a vacuous bound (error rates are at most 1 anyway).
- **The trend goes the wrong way.** Making a network *bigger* often makes it
  generalize *better*, whereas VC theory predicts the opposite.

The resolution cannot come from the hypothesis class alone. It must involve
the actual data distribution (real labels are not random), the *algorithm*
(SGD has an implicit bias toward particular low-complexity solutions), and
capacity measures sensitive to the **size** of the weights — margins and norms
— rather than to how many of them there are. That is precisely why weight
decay and other norm-based regularizers work, and why the generalization
question stays open in later chapters.

---

## 4.6.4 Summary

- The **empirical error** $\epsilon_{\mathcal{D}}(f)$ on a held-out test set is
  an **unbiased** estimate of the **generalization error** $\epsilon(f)$ —
  provided $f$ was fixed before the test set was touched.
- Test error is a sample mean of iid Bernoulli draws, so its standard deviation
  is $\sqrt{\epsilon(1-\epsilon)/n} \le 1/(2\sqrt{n})$: an
  $\mathcal{O}(1/\sqrt{n})$ rate, where **quadrupling the test set halves the
  error bar**.
- By the CLT, $n \approx 10{,}000$ test examples put the estimate within about
  $\pm 0.01$ at roughly 95% confidence; **Hoeffding's inequality**
  $P(|\bar{X} - \mathbb{E}[\bar{X}]| \ge t) \le 2\exp(-2nt^2)$ gives the same
  guarantee rigorously at any finite $n$, asking for about 15,000 examples
  one-sided (18,500 two-sided).
- **Reusing a test set** across $k$ models invites false discoveries: with 20
  models at 95% confidence, the chance that some interval is misleading is
  about 64%. A union bound restores validity at a cost of only
  $t = \sqrt{(\log k + \log(1/\delta))/2n}$ — but it only applies to a list of
  models fixed in advance.
- **Adaptive overfitting** — choosing the next model after seeing test results —
  escapes that correction entirely. Select on validation data, consult the test
  set as rarely as possible, and refresh benchmarks when you can.
- **Statistical learning theory** bounds the training-to-population gap via
  **uniform convergence** over a function class, generalizing the
  **Glivenko–Cantelli** theorem; the **VC dimension** measures capacity as the
  largest shatterable point set (e.g. $d+1$ for linear classifiers in
  $\mathbb{R}^d$).
- The **VC bound** $\epsilon(f) \le \epsilon_{\mathcal{S}}(f) + \alpha$ with
  $\alpha \le c\sqrt{(\mathrm{VC} - \log\delta)/n}$ is distribution-free but far too
  pessimistic for practice, and **vacuous for deep networks**, which can
  memorize random labels yet still generalize — and generalize better as they
  grow. Explaining that requires the data distribution, the optimizer, and
  norm-based notions of complexity, not parameter counting.
