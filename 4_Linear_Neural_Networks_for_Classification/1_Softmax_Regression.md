# 4.1 Softmax Regression

Linear regression predicts a *number*. **Softmax regression** predicts a
*category* — it is the standard linear model for **classification**, where the
output is one of $q$ discrete classes. It's "regression" only in name: the
machinery is a linear layer followed by a softmax that turns raw scores into a
probability distribution over the classes.

---

## 4.1.1 Classification

### One-hot encoding

Labels have no natural ordering (cat/dog/chicken are not $1 < 2 < 3$), so we
encode each label as a **one-hot vector**: a length-$q$ vector that is $1$ at
the true class and $0$ elsewhere.

$$y \in \{(1,0,0),\ (0,1,0),\ (0,0,1)\}$$

### Linear model

We need one output per class, so the model has $q$ **affine functions** — one
weight vector and bias per class. For a single example $\mathbf{x} \in
\mathbb{R}^d$:

$$\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}, \qquad
\mathbf{o} \in \mathbb{R}^q,\ \ \mathbf{W} \in \mathbb{R}^{q \times d},\ \
\mathbf{b} \in \mathbb{R}^q$$

The outputs $\mathbf{o}$ are called **logits** — unbounded real scores, one per
class. This is a single-layer, fully-connected network ($d$ inputs, $q$
outputs).

### The softmax

Logits can be any real number, but we want **probabilities**: non-negative and
summing to $1$. The **softmax** function does exactly this:

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}), \qquad
\hat{y}_j = \frac{\exp(o_j)}{\sum_{k=1}^{q} \exp(o_k)}$$

- $\exp(\cdot)$ forces non-negativity; the denominator normalizes to sum $1$.
- Softmax is **monotonic**, so it preserves ordering:
  $$\underset{j}{\arg\max}\ \hat{y}_j = \underset{j}{\arg\max}\ o_j$$
  We predict the class with the largest logit — no need to compute the softmax
  just to pick the winner (though we need it for the loss).

### Vectorization for minibatches

For a minibatch $\mathbf{X} \in \mathbb{R}^{n \times d}$ ($n$ examples), stack
everything into matrix form (here $\mathbf{W} \in \mathbb{R}^{d \times q}$,
$\mathbf{b} \in \mathbb{R}^{1 \times q}$ broadcast over rows):

$$\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}, \qquad
\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{O})$$

Softmax is applied **row-wise**: each of the $n$ rows becomes its own
probability distribution over the $q$ classes.

---

## 4.1.2 Loss Function

### Log-likelihood

The model outputs $\hat{\mathbf{y}} = P(\mathbf{y} \mid \mathbf{x})$. Over a
dataset we maximize the likelihood of the observed labels, which (taking the
negative log) is the same as minimizing:

$$-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^{n} l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
\qquad
l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^{q} y_j \log \hat{y}_j$$

This per-example loss is the **cross-entropy loss**. Because $\mathbf{y}$ is
one-hot, the sum collapses to a single term:

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_{\text{true class}}$$

i.e. *the negative log-probability the model assigned to the correct answer*.
Confident and right → loss near $0$; confident and wrong → loss blows up.

### Softmax + cross-entropy together

Plugging the softmax definition into the loss simplifies beautifully:

$$l(\mathbf{y}, \hat{\mathbf{y}})
= -\sum_j y_j \log \frac{\exp(o_j)}{\sum_k \exp(o_k)}
= \log \sum_{k} \exp(o_k) - \sum_j y_j\, o_j$$

Taking the derivative w.r.t. the logit $o_j$ gives a remarkably clean gradient:

$$\partial_{o_j}\, l(\mathbf{y}, \hat{\mathbf{y}})
= \frac{\exp(o_j)}{\sum_k \exp(o_k)} - y_j
= \mathrm{softmax}(\mathbf{o})_j - y_j
= \hat{y}_j - y_j$$

**The gradient is just (predicted probability − actual).** This is the exact
same form as linear regression's gradient — the error signal is the gap between
what the model predicted and what actually happened, which makes optimization
well-behaved.

### Numerical stability (LogSumExp)

Computing $\exp(o_k)$ directly can **overflow** for large logits. The fix is to
subtract the max logit $\bar{o} = \max_k o_k$ before exponentiating — this
doesn't change the result (it cancels in the ratio):

$$\hat{y}_j = \frac{\exp(o_j - \bar{o})}{\sum_k \exp(o_k - \bar{o})}$$

In practice, frameworks fuse softmax and the log into one **cross-entropy**
operator (passing logits directly) so the $\log$ and $\exp$ cancel and the
computation stays stable.

---

## 4.1.3 Information Theory Basics

Cross-entropy comes from information theory — here's the intuition.

### Entropy

The **entropy** of a distribution $P$ measures its inherent uncertainty — the
minimum number of *nats* (or *bits*, base 2) needed to encode data drawn from
it:

$$H(P) = -\sum_j P(j) \log P(j) = \mathbb{E}_{x \sim P}[-\log P(x)]$$

Predictable distribution (one likely outcome) → low entropy; uniform (all
outcomes equally likely) → maximum entropy.

### Surprisal

The term $-\log P(j)$ is the **surprisal** (self-information) of event $j$: a
rare event ($P$ small) is very surprising (large value); a certain event
($P = 1$) carries zero surprise. Entropy is just the *expected* surprisal.

### Cross-entropy

**Cross-entropy** $H(P, Q)$ is the expected surprisal when the world truly
follows $P$ but we model it with $Q$:

$$H(P, Q) = -\sum_j P(j) \log Q(j) = \mathbb{E}_{x \sim P}[-\log Q(x)]$$

- It is minimized when $Q = P$, where it equals $H(P)$.
- The gap $H(P,Q) - H(P)$ is the extra cost of using the wrong model — this is
  the **KL divergence**.

**The connection:** our classification loss *is* the cross-entropy between the
true label distribution $P$ (the one-hot vector) and the model's prediction
$Q = \hat{\mathbf{y}}$. So minimizing cross-entropy loss = making the predicted
distribution match the data distribution = maximizing likelihood. Three views,
one objective.

---

## 4.1.4 Summary

- **Softmax regression** = a linear layer ($\mathbf{o} = \mathbf{Wx} + \mathbf{b}$)
  followed by **softmax**, mapping $d$ inputs to a probability distribution over
  $q$ classes.
- **Softmax** exponentiates and normalizes logits into non-negative
  probabilities summing to $1$; being monotonic, $\arg\max$ of logits gives the
  prediction directly.
- The natural loss is **cross-entropy** $-\log \hat{y}_{\text{true class}}$,
  equivalent to maximum likelihood.
- Its gradient is the clean **$\hat{y}_j - y_j$** (prediction − truth), just
  like linear regression.
- **Information theory** frames the loss as cross-entropy between the true and
  predicted distributions; entropy = expected surprisal, minimized when the
  model matches reality.
- For stability, subtract the max logit (**LogSumExp**) and let the framework
  fuse softmax with the log by operating on logits directly.

---

## 4.1.5 Exercises

A few representative ones (the others in the book are variations on these).

### 1. Why is softmax called "soft"? Show $\mathrm{RealSoftMax}(a,b) = \log(e^a + e^b)$ is a smooth approximation of $\max(a,b)$.

Take WLOG $a \ge b$. Factor out $e^a$:

$$\log(e^a + e^b) = a + \log\!\big(1 + e^{b-a}\big)$$

Since $e^{b-a} > 0$, the extra term is positive, so $\mathrm{RealSoftMax}(a,b)
> \max(a,b)$ always. The gap $\log(1 + e^{b-a})$ ranges in $(0, \log 2]$:
largest ($\log 2$) when $a = b$, shrinking to $0$ as $a \gg b$. Adding a
**temperature** $\lambda$ sharpens it:

$$\tfrac{1}{\lambda}\,\mathrm{RealSoftMax}(\lambda a, \lambda b)
= \max(a,b) + \tfrac{1}{\lambda}\log\!\big(1 + e^{-\lambda|a-b|}\big)
\ \xrightarrow{\ \lambda \to \infty\ }\ \max(a,b)$$

So it's a *soft* (differentiable) max that becomes the hard max as
$\lambda \to \infty$.

### 2. Show the log-partition $g(\mathbf{x}) = \log\sum_i e^{x_i}$ is convex and translation-invariant, and use it to justify the LogSumExp trick.

- **Gradient** is the softmax: $\partial_{x_i} g = \dfrac{e^{x_i}}{\sum_k e^{x_k}} = p_i$.
- **Hessian** is $\nabla^2 g = \mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$,
  the covariance matrix of the class indicator under $\mathbf{p}$. Covariance
  matrices are positive semidefinite ⇒ $g$ is **convex**.
- **Translation invariance:**
  $g(\mathbf{x} + b) = \log\!\big(e^b \sum_i e^{x_i}\big) = b + g(\mathbf{x})$.
- **Stability:** invariance lets us pick $b = -\max_i x_i$, giving
  $g(\mathbf{x}) = \max_i x_i + \log\sum_i e^{x_i - \max_i x_i}$. Now every
  exponent is $\le 0$, so nothing overflows — this *is* the LogSumExp trick.

### 3. Compute the second derivative of the cross-entropy loss and interpret it.

From $l = \log\sum_k e^{o_k} - \sum_j y_j o_j$ we had the first derivative
$\partial_{o_j} l = p_j - y_j$ (where $\mathbf{p} = \mathrm{softmax}(\mathbf{o})$).
Differentiating again:

$$\partial_{o_j}^2\, l = p_j(1 - p_j), \qquad
\partial_{o_i}\partial_{o_j}\, l = -\,p_i p_j \ \ (i \ne j)$$

The Hessian $\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ is the
**covariance of the predicted class distribution** — PSD, so the loss is
**convex in the logits** (a global optimum exists). This is the general
exponential-family fact: the variance of the sufficient statistic equals the
second derivative of the log-partition function.

### 4. Temperature: let $Q(i) \propto P(i)^{1/T}$. What happens as $T \to 0$ and $T \to \infty$?

Writing $P(i) \propto e^{o_i}$ gives $Q(i) \propto e^{o_i / T}$, so $T$ rescales
the logits:

- $T \to 0$ — logits blow up in contrast; $Q$ collapses onto the single most
  probable class → a **one-hot / argmax** (maximally confident).
- $T \to \infty$ — logits are flattened to $0$; $Q$ becomes the **uniform**
  distribution (maximally uncertain).

Low temperature *sharpens* predictions, high temperature *smooths* them — the
knob behind techniques like distillation and calibrated sampling.
