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

# 3.6.5 Exercises

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