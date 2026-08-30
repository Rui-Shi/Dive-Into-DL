# 3.6 Generalization

Fitting the training set is the easy part — a lookup table does that
perfectly. What actually matters is whether the pattern a model found holds on
data it hasn't seen: the book frames this with two students cramming for
finals, one who memorizes last year's exact answers, one who learns the
underlying pattern instead — the memorizer wins only if the new exam recycles
old questions, and loses badly the moment it doesn't. This section makes that
goal precise:

1. What exactly separates training error from generalization error, and why
   is the latter something we can estimate but never actually compute?
2. What do underfitting and overfitting look like, and how do model
   complexity and dataset size determine which one you land in?
3. Since generalization error is inaccessible, how do you choose among
   competing models without quietly cheating?

---

## 3.6.1 Training Error and Generalization Error

In the standard supervised-learning setup we assume the training data and
future (test) data are drawn **independently** from **identical**
distributions — the **iid assumption**. It's a strong assumption, and it's
false more often than we'd like (data drifts, collection is biased, deployment
conditions differ from collection conditions) — but absent it, or some
substitute for it, there's no reason to think anything learned from a training
sample says anything about data we haven't seen. We take
$P(\mathbf{x}, y) = Q(\mathbf{x}, y)$ as given for the rest of this section;
distribution shift gets its own treatment later in the book.

Given that assumption, define two error quantities for a hypothesis $f$ and
loss $l$.

**Training error** ($R_\text{emp}$) is a *statistic*, computed on the $n$
examples actually in hand:

$$R_\text{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n}\sum_{i=1}^{n}
l\big(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})\big)$$

**Generalization error** ($R$) is an *expectation* over the entire
distribution $P$, most of which we will never observe:

$$R[p, f] = \mathbb{E}_{(\mathbf{x}, y) \sim P}\big[l(\mathbf{x}, y,
f(\mathbf{x}))\big]
= \int\!\!\int l(\mathbf{x}, y, f(\mathbf{x}))\, p(\mathbf{x}, y)\,
d\mathbf{x}\, dy$$

$R$ is the number we actually care about — it's what you'd see if you ran $f$
against an infinite stream of fresh data from $P$ — and it is exactly the
number we can never compute: nobody hands us $p(\mathbf{x}, y)$, and we can't
sample infinitely. In practice we *estimate* $R$ by evaluating the same
formula as $R_\text{emp}$, but on a **test set** $(\mathbf{X}', \mathbf{y}')$
that was withheld from training.

That substitution is legitimate for the test set in a way it is not for the
training set. A test set is evaluated against a **fixed** classifier — one
that doesn't depend on which particular test examples were drawn — so
estimating its error is a textbook mean-estimation problem, and
$R_\text{emp}$ computed there is unbiased for $R$. (Section 4.6 makes this
precise: how many test examples you need, and what happens once the same test
set gets reused across many models.) The training set enjoys no such luxury:
$f$ was *chosen* because it fit that particular sample, so training error is
systematically **optimistic** — biased low relative to the true population
error. The central question of generalization is exactly when, and how much,
you should expect that bias to matter.

One more wrinkle worth flagging early: low training error does not, by
itself, certify low generalization error — but it doesn't certify *high*
generalization error either. A model expressive enough to fit any assignment
of labels tells you nothing about the population error from its training
error alone. Deep networks are exactly such models: they generalize well in
practice despite having enough capacity to memorize randomly shuffled labels,
and precisely because training error is so uninformative for them, holdout
data — the **validation error** — carries more of the weight in certifying
that they've actually generalized.

### Model Complexity

What makes one hypothesis class more *complex* than another has no single
scalar answer, but two rules of thumb recur:

- **More parameters** usually (not always) means more complexity — a model
  class with more free coefficients can typically fit more of the $2^n$
  possible label assignments on $n$ points. The exception matters: kernel
  methods have infinitely many parameters yet keep their complexity in check
  through other means, so parameter count is a heuristic, not a definition.
- **Wider parameter ranges** add complexity even at a fixed parameter count —
  a weight vector allowed to take any value in $\mathbb{R}^d$ is a more
  complex hypothesis class than the same vector restricted to a bounded
  region. This is exactly the lever weight decay pulls (Section 3.7): shrink
  the *range* the parameters may occupy, without touching how many of them
  there are.

The underlying reason any of this matters traces back to a piece of
philosophy of science: Karl Popper's **falsifiability**. A theory compatible
with *every* possible observation has told you nothing about the world — it
never rules anything out. A hypothesis class expressive enough to fit *any*
labeling of the data is exactly such a theory: fitting the training data with
it is uninformative, because it would have fit almost anything. A hypothesis
class narrow enough that it *could* have failed to fit the data — and didn't
— has actually told you something. That is the sense in which restricting
model complexity is not a nuisance imposed by finite data, but the very thing
that makes "it fit" mean anything at all.

---

## 3.6.2 Underfitting or Overfitting?

Given a training error and a validation error, there are two failure patterns
worth naming, distinguished by the size of the **generalization gap**
$R - R_\text{emp}$:

- **Underfitting** — training and validation error are both high, and close
  together (a small gap). The model isn't managing to reduce even its
  training error, which means it's too simple (insufficiently expressive) to
  capture the pattern in the data at all. Because the gap is small, there's
  reason to believe a more complex model would do better on *both* errors.
- **Overfitting** — training error sits significantly below validation error
  (a large gap). The model has captured some of the noise or idiosyncrasies
  specific to the training sample, in addition to — or instead of — the
  underlying pattern.

Overfitting is not automatically a problem. What we actually want minimized
is $R$ itself; the gap only matters insofar as it stands in the way of that.
In fact, the best-performing deep networks routinely have training error far
below their validation error and still generalize better than smaller models
with a narrower gap — a fact that sits uneasily with the classical intuition
above, and one the rest of the book returns to more than once. One clean
identity is worth keeping in mind (see also Exercise 3 below): if training
error reaches exactly zero, the gap becomes *equal to* the generalization
error, $R - 0 = R$, so at that point the only way to make progress is to
close the gap directly.

### Polynomial Curve Fitting

To make the trade-off concrete, follow the book's running example: a single
feature $x$, a real-valued label $y$, fit with a degree-$d$ polynomial

$$\hat{y} = \sum_{i=0}^{d} x^i w_i$$

This is ordinary linear regression in disguise — the "features" are just the
powers $x^0, x^1, \dots, x^d$, with $w_0$ playing the role of the bias since
$x^0 = 1$ — so squared error and everything from Section 3.1 applies
directly.

Two facts pin down the trade-off:

- **Training error falls monotonically in $d$.** A degree-$d$ hypothesis
  class is nested inside degree-$(d+1)$ (just set the new coefficient to
  $0$), so more degree can only ever help fit the sample already in hand —
  never hurt.
- **Training error can be driven to exactly zero.** Given $n$ examples with
  $n$ *distinct* $x$ values, a polynomial with enough coefficients
  interpolates them exactly — Exercise 1 works out precisely how many degrees
  that takes, via the Vandermonde matrix.

Neither fact says anything about validation error, and that's the entire
point: too low a degree underfits — both errors stay high, because the true
pattern isn't representable at all. Too high a degree pushes training error
toward zero while validation error, measured on a fresh noise draw, gets
*worse* — the spare capacity is spent fitting noise specific to the training
sample rather than signal that would transfer. Plotted against $d$,
validation error traces the classic U-shape: falling as capacity starts to
capture real structure, then rising once capacity outpaces what the data can
constrain. The valley of that U — not the vanishing training error to its
right — is where model selection should be aiming; Section 3.6.3 covers how
to actually locate it without cheating by looking at the test set.

### Dataset Size

Fix the model and vary the amount of training data instead, and the same
U-shaped tension reappears from the other axis: the fewer samples in the
training set, the more likely — and the more severely — a fixed model
overfits, because there are fewer constraints pinning down which of the many
hypotheses consistent with the data is the "right" one. As training data
grows, generalization error typically falls, and **more data essentially
never hurts**. The practical corollary is the rule of thumb already stated in
the summary below: model complexity should not grow faster than the data
available to constrain it. Given abundant data, a more complex model becomes
affordable; with only a few hundred or a few thousand examples, a simpler
model is often hard to beat regardless of what you suspect the "true"
relationship looks like. This is also, in large part, why deep learning
didn't overtake linear models until datasets reached into the
thousands-to-millions of examples — the capacity was always available; the
data needed to safely use it wasn't.

---

## 3.6.3 Model Selection

Real projects rarely fit one model — they compare several (polynomial degree,
regularization strength, architecture, learning rate) and need to pick a
winner. The obvious way to choose is by comparing generalization error, and
the obvious problem is that $R$ is exactly what Section 3.6.1 established we
cannot compute.

Two things you *cannot* do:

- **Select using the test set.** If test-set performance drives which model,
  which hyperparameters, or which features you choose, the test set stops
  measuring generalization error and starts measuring how well you've fit the
  test set — and unlike overfitting the training set, there's no held-out
  check left to catch you. Overfit the training data and evaluation on test
  data keeps you honest; overfit the test data, and how would you ever know?
- **Select using only the training set.** You cannot estimate generalization
  error on the very data used to fit the model — that's the biased-estimator
  problem from Section 3.6.1 all over again.

The standard resolution is a **three-way split**: train on the training set,
compare candidates and tune hyperparameters on a **validation set**, and
touch the **test set** exactly once, at the very end, to report a final
number. In practice the line between "validation" and "test" blurs —
benchmark datasets get reused for years across an entire research community,
so a nominal test set often ends up behaving like an unusually public,
unusually expensive validation set. (D2L flags exactly this: unless stated
otherwise, the "test accuracy" reported in the book's own experiments is
honestly validation accuracy, for precisely this reason.)
Section 4.6, *Generalization in Classification*, works out just how much
damage that reuse does, and how many times a test set can be consulted before
its number stops meaning what you think it means.

### Cross-Validation

Setting aside a validation set is wasteful when data is scarce — every
example moved into validation is one the model doesn't get to train on.
**$K$-fold cross-validation** recovers most of that data while still keeping
training and validation disjoint on every round:

1. Split the training data into $K$ non-overlapping subsets ("folds") of
   roughly equal size.
2. For each fold $k = 1, \dots, K$: train on the other $K-1$ folds, validate
   on fold $k$.
3. Average the $K$ validation errors (and, if useful, the $K$ training
   errors) to get the final estimate.

Every example is used for training $K-1$ times out of $K$ and for validation
exactly once, so nearly the whole dataset informs both quantities. The price
is computational — $K$ full training runs instead of one, which is why
Exercise 4 below flags it as expensive to repeat across a hyperparameter
search, and why Exercise 5 flags the resulting estimate as still slightly
pessimistic (each fold trains on only $\frac{K-1}{K}$ of the data).

---

## 3.6.4 Summary

Generalization gets murky with deeper models: they can overfit badly, and
complexity becomes implicit and counterintuitive — bigger architectures
sometimes generalize *better*.

Rules of thumb:

1. **Model selection** — use a validation set or *K*-fold cross-validation.
2. **Complexity vs. data** — more complex models generally need more data.
3. **What counts as complexity** — not just parameter count, but also the
   range of values those parameters may take.
4. **More data helps** — all else equal, it almost always improves
   generalization.
5. **IID is load-bearing** — if the train and test distributions shift, none
   of this reasoning holds without some further assumption.

---

## 3.6.5 Exercises

### 1. When can you solve the problem of polynomial regression exactly?

When the design matrix is invertible (or the system is underdetermined but
consistent). For $n$ examples with **distinct** $x$ values, a polynomial of
degree $d \ge n-1$ interpolates them exactly; at $d = n-1$ the Vandermonde
matrix is square with determinant $\prod_{i<j}(x_j - x_i) \ne 0$, so the
solution is unique. It fails if two examples share an $x$ but disagree on $y$,
and it becomes numerically hopeless for large $n$ — Vandermonde matrices are
notoriously ill-conditioned.

### 2. Give at least five examples where dependent random variables make treating the problem as IID data inadvisable.

1. **Time series** — stock prices, weather, sensor streams; consecutive
   observations are strongly autocorrelated.
2. **Text** — words within a sentence, sentences within a document.
3. **Video frames** — adjacent frames are near-duplicates, so a random
   train/test split leaks.
4. **Repeated measures** — multiple visits from the same patient, or multiple
   photos of the same person.
5. **Network / clustered data** — friends in a social graph share labels;
   students in the same classroom share a teacher.
6. **Spatial data** — nearby locations have correlated soil, income, pollution.

Naive splitting here inflates test performance, because the test set is partly
a copy of the training set.

### 3. Can you ever expect to see zero training error? Under which circumstances would you see zero generalization error?

**Training error:** routinely, yes. Any model with capacity at least the size
of the dataset can interpolate it — 1-nearest-neighbor, a degree-$(n-1)$
polynomial, or an overparameterized network can memorize even random labels.

**Generalization error:** essentially never. It requires that the true
labeling be deterministic (no label noise, no Bayes error), that it lie inside
your hypothesis class, that you recover it exactly, and that the test
distribution match the training one. Only toy problems satisfy all four.

### 4. Why is $K$-fold cross-validation very expensive to compute?

You train $K$ separate models from scratch, so the cost is roughly $K\times$ a
single fit — and that multiplies against every hyperparameter configuration in
your search. For deep networks costing hours or days per run, this is usually
prohibitive.

### 5. Why is the $K$-fold cross-validation error estimate biased?

Each model sees only $\frac{K-1}{K}$ of the data, so it is systematically
weaker than the final model trained on everything. The estimate is therefore
**pessimistic**, and the bias grows as $K$ shrinks. Separately, the $K$
training sets overlap heavily, so the fold estimates are correlated and their
spread understates the true variance.

### 6. The VC dimension is defined as the maximum number of points that can be classified with arbitrary labels $\{\pm 1\}$ by a function of a class of functions. Why might this not be a good idea for measuring how complex the class of functions is?

*Hint: consider the magnitude of the functions.*

It counts only *which label patterns are achievable*, ignoring the magnitude
and smoothness of the functions producing them. Consequences:

- $f(x) = \mathrm{sign}(\sin(\omega x))$ has one parameter and **infinite** VC
  dimension.
- Two classes can share a VC dimension while one separates points by a wide
  margin and the other by a hair — very different generalization behavior.
- It is worst-case and distribution-free, so its bounds are vacuous for modern
  networks, whose parameter counts exceed their datasets yet which still
  generalize.

Norm- and margin-based measures (weight decay, spectral norms) capture the
missing scale information.

### 7. Your manager gives you a difficult dataset on which your current algorithm does not perform so well. How would you justify to him that you need more data?

*Hint: you cannot increase the data but you can decrease it.*

Build a **learning curve**: you can't add data, but you *can* subsample it.
Train on 10%, 20%, ..., 100% of the set and plot validation error against
training size. Two readings:

- If the curve is still falling at 100%, you are **variance-limited** —
  extrapolating it predicts how much more data buys how much error, which is a
  concrete, quantified ask.
- If it has flattened while train and validation error sit close together, you
  are **bias-limited** and more data won't help; the model or features need
  changing instead.

The same plot honestly answers "no" when the answer is no, which is what makes
it persuasive when the answer is yes.
