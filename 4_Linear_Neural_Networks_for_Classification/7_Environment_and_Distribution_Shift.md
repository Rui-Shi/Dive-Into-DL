# 4.7 Environment and Distribution Shift

Everything so far assumed that training and test data were drawn **iid** from
the same distribution. That assumption is doing an enormous amount of work. A
model is not deployed into the dataset it was fit on; it is deployed into a
world that keeps moving — new users, new cameras, new seasons, new spammers.

Worse, a good score on a held-out test set can be a *false comfort*: the test
set was carved out of the same snapshot as the training set, so it certifies
only that the model generalizes to more data **of that snapshot**. It says
nothing about tomorrow's data, or about the hospital across town.

This section is about naming the ways the world can differ from the training
snapshot, recognizing them in practice, and — where possible — correcting for
them.

---

## 4.7.1 Types of Distribution Shift

Fix the setup. Training data is drawn from a **source** distribution
$p_S(\mathbf{x}, y)$, but the model is evaluated on — and must perform under —
a **target** distribution $p_T(\mathbf{x}, y)$:

$$\text{train on } (\mathbf{x}_i, y_i) \sim p_S(\mathbf{x}, y),
\qquad \text{deploy on } (\mathbf{x}, y) \sim p_T(\mathbf{x}, y)$$

If $p_S = p_T$ we are in the familiar iid world. If they differ arbitrarily,
nothing can be learned — the training data would simply be irrelevant. Useful
theory lives in between: we assume the two distributions differ in a
**restricted, structured way**, and each restriction gets a name. Any joint
distribution factors two ways,

$$p(\mathbf{x}, y) = p(\mathbf{x})\, p(y \mid \mathbf{x})
= p(y)\, p(\mathbf{x} \mid y)$$

and the three classical shift types are exactly the choices of *which factor is
allowed to move and which is held fixed*.

### 4.7.1.1 Covariate Shift

**Covariate shift**: the input marginal changes but the labeling rule does not.

$$p_S(\mathbf{x}) \ne p_T(\mathbf{x}),
\qquad p_S(y \mid \mathbf{x}) = p_T(y \mid \mathbf{x})$$

The canonical example: train a cat-vs-dog classifier on photographs, then test
it on cartoon drawings. The set of images you see has changed dramatically, but
a picture of a cat — photo or cartoon — is still labeled *cat*. The rule
$p(y \mid \mathbf{x})$ is stable; only the pool of $\mathbf{x}$'s moved.

Covariate shift is the natural assumption when we believe **$\mathbf{x}$ causes
$y$**: the image determines the label, so the mechanism producing labels from
images doesn't care how the images were sampled.

### 4.7.1.2 Label Shift

**Label shift** is the mirror image: the label marginal changes but the
class-conditional appearance of each class does not.

$$p_S(y) \ne p_T(y),
\qquad p_S(\mathbf{x} \mid y) = p_T(\mathbf{x} \mid y)$$

Think of diagnosing disease from symptoms. During an outbreak the *prevalence*
$p(y)$ of the disease spikes, but the disease still produces the same symptoms
it always did — $p(\mathbf{x} \mid y)$ is unchanged. Note that
$p(y \mid \mathbf{x})$ *does* change here (a cough means something different
during an epidemic), which is precisely why an uncorrected classifier
misbehaves.

Label shift is the natural assumption when **$y$ causes $\mathbf{x}$**:
diseases cause symptoms, not the other way around.

Only one of covariate shift and label shift can be assumed at a time, and the
choice should follow the causal direction of the problem. Happily, label shift
is also the easier one to correct, since we can often estimate the target label
frequencies without any target labels at all (Section 4.7.3.3).

### 4.7.1.3 Concept Shift

**Concept shift** is the awkward one: the labeling function itself drifts.

$$p_S(y \mid \mathbf{x}) \ne p_T(y \mid \mathbf{x})$$

The *meaning* of a label changes with time or place. Diagnostic criteria in
psychiatry have been rewritten repeatedly, so identical patient records get
different labels in different decades. What counts as *fashionable* clothing
turns over every year. Ask for a "soft drink" in the United States and the word
you hear back is *soda*, *pop*, or *coke* depending on which state you're
standing in — same referent, different label.

Concept shift is usually **gradual**, which is both the difficulty (there is no
clean before/after to correct between) and the saving grace (small, continual
updates can track it).

---

## 4.7.2 Examples of Distribution Shift

The theory is easy to nod along to. The point of these vignettes is that in
each case the failure was invisible on the test set.

### 4.7.2.1 Medical Diagnostics

A company sets out to build a blood test for a disease. Healthy blood samples
are easy to get from university student volunteers; sick samples come from
hospital patients. The classifier separates the two groups almost perfectly.

Of course it does. The two groups differ in age, hormones, diet, activity, time
of day of the draw — dozens of signals that have nothing to do with the
disease. The model learned *student vs. patient*, not *healthy vs. sick*. This
is **covariate shift** introduced by the sampling procedure itself, and no
amount of held-out accuracy on the same two cohorts will reveal it.

### 4.7.2.2 Self-Driving Cars

A team trains a roadside detector on cheaply generated data rendered by a game
engine. Test performance on rendered scenes is excellent; performance on the
actual car is catastrophic. The renderer had textured every roadside with the
same repeating pattern, so "roadside" became "that texture" — a spurious
feature that simply does not exist in reality.

The older folk version of the same story: a tank detector that turned out to
have learned *morning shadows vs. midday sun*, because the tank photographs and
the empty-forest photographs happened to be taken at different times of day.

### 4.7.2.3 Nonstationary Distributions

Sometimes the shift is not between two places but between now and later — the
distribution is **nonstationary** and the model quietly goes stale:

- A **spam filter** works until spammers read its outputs and rewrite their
  messages. The adversary is adapting to your model (concept shift with a
  motive).
- An **ad ranking model** trained before a product category existed keeps
  serving the world as it was; new products never get their shot.
- A **recommender** trained through the winter holidays keeps pushing gift
  wrap and pine wreaths into February.

### 4.7.2.4 More Anecdotes

- A **face detector** trained on ordinary snapshots fails on extreme close-ups
  where a single face fills the frame — a scale never present in training.
- A **web search engine** built for one country is launched in another, where
  the distribution of queries, languages and intents is unrecognizable.
- An **image classifier** trained on a benchmark with tidy, near-uniform class
  frequencies meets a real deployment where classes follow a brutal long tail.

---

## 4.7.3 Correction of Distribution Shift

### 4.7.3.1 Empirical Risk and Risk

What we actually minimize during training is the **empirical risk**, the
average loss over the $n$ training examples:

$$\min_f \ \frac{1}{n} \sum_{i=1}^{n} \ell\big(f(\mathbf{x}_i), y_i\big)$$

What we *want* to minimize is the **true risk** (the population risk), the
expected loss over the distribution we will actually face:

$$R_T(f) = \mathbb{E}_{(\mathbf{x}, y) \sim p_T}\big[\ell(f(\mathbf{x}), y)\big]
= \iint \ell\big(f(\mathbf{x}), y\big)\, p_T(\mathbf{x}, y)\ d\mathbf{x}\, dy$$

**Empirical risk minimization** is the bet that the first is a good proxy for
the second. Under iid sampling from $p_T$ that bet is sound. Under distribution
shift the training average estimates the risk under $p_S$ — the wrong
integral — and the whole justification for ERM evaporates.

The fix, when the shift is structured, is **importance weighting**: reweight
the source samples so their weighted average estimates the target expectation.
The identity is just multiply-and-divide:

$$\mathbb{E}_{(\mathbf{x}, y) \sim p_T}\big[\ell(f(\mathbf{x}), y)\big]
= \mathbb{E}_{(\mathbf{x}, y) \sim p_S}
\left[\frac{p_T(\mathbf{x}, y)}{p_S(\mathbf{x}, y)}\,
\ell\big(f(\mathbf{x}), y\big)\right]$$

which holds as long as $p_S(\mathbf{x}, y) > 0$ wherever
$p_T(\mathbf{x}, y) > 0$ — you cannot reweight your way onto data you never
sampled. The ratio $p_T / p_S$ is unknowable in general; the shift assumptions
are exactly what make it estimable.

### 4.7.3.2 Covariate Shift Correction

Assume covariate shift, so $p_S(y \mid \mathbf{x}) = p_T(y \mid \mathbf{x})$.
Factor both joints as $p(\mathbf{x})\,p(y \mid \mathbf{x})$ and the conditional
cancels:

$$\frac{p_T(\mathbf{x}, y)}{p_S(\mathbf{x}, y)}
= \frac{p_T(\mathbf{x})\, p(y \mid \mathbf{x})}{p_S(\mathbf{x})\, p(y \mid \mathbf{x})}
= \frac{p_T(\mathbf{x})}{p_S(\mathbf{x})}$$

So the target risk is a reweighted source expectation, with weights depending
on $\mathbf{x}$ alone:

$$R_T(f) = \mathbb{E}_{(\mathbf{x}, y) \sim p_S}
\left[\frac{p_T(\mathbf{x})}{p_S(\mathbf{x})}\, \ell\big(f(\mathbf{x}), y\big)\right],
\qquad \beta_i = \frac{p_T(\mathbf{x}_i)}{p_S(\mathbf{x}_i)}$$

Training then becomes **weighted empirical risk minimization**:

$$\min_f \ \frac{1}{n} \sum_{i=1}^{n} \beta_i\, \ell\big(f(\mathbf{x}_i), y_i\big)$$

Each training example is counted $\beta_i$ times: examples typical of the
target domain are amplified, examples that only occur in the source domain are
damped.

**Estimating $\beta_i$ without knowing either density.** We never need
$p_T(\mathbf{x})$ or $p_S(\mathbf{x})$ separately — only their ratio, and a
ratio of densities is what a classifier estimates. Assume we have unlabeled
data from the target domain (usually cheap: we have the inputs, we just lack
the labels). Pool the two samples, taking $m$ points from each, and attach an
auxiliary label

$$z = 1 \ \text{ if } \mathbf{x} \text{ came from the target sample},
\qquad z = -1 \ \text{ if from the source sample}$$

With equal sample sizes the pooled distribution is the 50/50 mixture, so

$$P(z = 1 \mid \mathbf{x})
= \frac{p_T(\mathbf{x})}{p_T(\mathbf{x}) + p_S(\mathbf{x})},
\qquad
\frac{P(z = 1 \mid \mathbf{x})}{P(z = -1 \mid \mathbf{x})}
= \frac{p_T(\mathbf{x})}{p_S(\mathbf{x})}$$

The density ratio is exactly the **odds** of the binary discriminator. Fit that
discriminator with logistic regression, parameterized as
$P(z = 1 \mid \mathbf{x}) = \dfrac{1}{1 + \exp(-h(\mathbf{x}))}$; then the odds
are $\exp(h(\mathbf{x}))$ and

$$\beta_i = \exp\big(h(\mathbf{x}_i)\big)$$

The recipe: (1) train a **domain classifier** to tell source data from target
data; (2) read off $\beta_i = \exp(h(\mathbf{x}_i))$ for each training example;
(3) retrain the real model with weighted ERM.

Two cautions. If the discriminator can separate the two domains *perfectly*,
the ratio is unbounded, a handful of examples soak up all the weight and the
effective sample size collapses — in practice weights are **clipped** to some
$[0, c]$. And if the domains are disjoint, no reweighting can help: the target
region simply has no source data to reweight. (With unequal sample sizes the
odds must be rescaled by the sampling ratio $n_S/n_T$.)

### 4.7.3.3 Label Shift Correction

Assume label shift, so $p_S(\mathbf{x} \mid y) = p_T(\mathbf{x} \mid y)$. Now
factor the joints the other way, $p(y)\, p(\mathbf{x} \mid y)$, and the
class-conditional cancels:

$$\frac{p_T(\mathbf{x}, y)}{p_S(\mathbf{x}, y)}
= \frac{p_T(y)\, p(\mathbf{x} \mid y)}{p_S(y)\, p(\mathbf{x} \mid y)}
= \frac{p_T(y)}{p_S(y)},
\qquad \beta_i = \frac{p_T(y_i)}{p_S(y_i)}$$

giving the same weighted objective as before,
$\min_f \frac{1}{n} \sum_i \beta_i\, \ell(f(\mathbf{x}_i), y_i)$, but with
weights that depend only on the **label**. For $q$ classes there are only $q$
distinct weights to estimate, which is why label shift is the more tractable
case: $p_S(y)$ is just the training class frequencies, and $p_T(y)$ is a
$q$-vector.

**The mean-prediction trick.** Estimate $p_T(y)$ from *unlabeled* target data,
using the classifier we already have.

1. On a held-out **source validation set**, build the $q \times q$ **confusion
   matrix** $\mathbf{C}$, with
   $C_{ij} = P\big(\hat{y} = i \mid y = j\big)$ — the fraction of true-class-$j$
   examples that the model predicts as class $i$. Its columns sum to $1$.
   (Hard $\arg\max$ predictions or averaged soft outputs both work, as long as
   the same convention is used in step 2.) Crucially, because
   $p(\mathbf{x} \mid y)$ is unchanged under label shift and $f$ is fixed,
   **this matrix transfers to the target domain unchanged**.

2. Draw unlabeled target inputs $\mathbf{x}_1, \dots, \mathbf{x}_m$ from
   $p_T(\mathbf{x})$ and average the model's predicted distributions:

   $$\mu(\hat{\mathbf{y}}) = \frac{1}{m} \sum_{i=1}^{m} f(\mathbf{x}_i)
   \ \in \mathbb{R}^q$$

   whose $i$-th entry estimates $P_T(\hat{y} = i)$, how often the model *says*
   class $i$ on target data.

3. Marginalizing over the true label,
   $P_T(\hat{y} = i) = \sum_j P(\hat{y} = i \mid y = j)\, p_T(y = j)$, which in
   matrix form is a linear system in the unknown label frequencies:

   $$\mathbf{C}\, p_T(\mathbf{y}) = \mu(\hat{\mathbf{y}})
   \qquad \Longrightarrow \qquad
   p_T(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$$

Solve, then set $\beta_i = p_T(y_i) / p_S(y_i)$ and retrain with weighted ERM.
(Equivalently, if $\mathbf{C}$ is built as the *joint* confusion matrix — the
fraction of all validation examples with true label $j$ and prediction $i$ —
then $\mathbf{C}^{-1}\mu(\hat{\mathbf{y}})$ returns the weight vector
$\boldsymbol{\beta}$ directly.)

This works only if $\mathbf{C}$ is **invertible**, which requires a classifier
that is reasonably accurate on the source domain to begin with. A model that
confuses two classes symmetrically leaves the system degenerate: its
predictions carry no information about how those two frequencies split.

### 4.7.3.4 Concept Shift Correction

There is no principled general correction for concept shift — if the labeling
rule can change arbitrarily, the past tells you nothing about the present.

What saves us in practice is that concept shift is normally **slow**. So rather
than retraining from scratch, keep the current weights and take a few update
steps on the freshest data: **continual, online adaptation** that lets the model
drift along with the concept. Sudden concept shift ("the company redefined what
counts as a churned customer") is different in kind and generally requires new
labels and a genuine retrain.

---

## 4.7.4 A Taxonomy of Learning Problems

Distribution shift is really a question about *what kind of environment* the
learner lives in. Ordered by how much the environment answers back:

- **Batch learning** — a fixed pile of labeled data in, a frozen model out; the
  model is deployed and never updated. The vending machine that recognizes
  cats and refuses entry to dogs. Fine only while the world stays put.
- **Online learning** — examples arrive one at a time. Observe
  $\mathbf{x}_t$, predict $f(\mathbf{x}_t)$, see the true $y_t$, pay a loss,
  update, repeat. Tomorrow's stock prices, today's model.
- **Bandits** — online learning where the action set is a *finite* list of arms
  rather than a continuously parameterized function. Weaker models, but much
  stronger guarantees; the exploration/exploitation trade-off becomes explicit.
- **Control** — the environment has **memory** and reacts to what we did. A
  thermostat's PID controller, a user who saw yesterday's price and is now
  anticipating today's. Our own past actions have shaped the current state.
- **Reinforcement learning** — the environment has memory *and* an agenda,
  cooperative or adversarial: chess, Go, driving in traffic.
- **Considering the environment** — the practical takeaway. A strategy that is
  optimal against a static environment can be exploited by an adaptive one
  (spammers, competitors, users gaming a recommender). Knowing *how fast* and
  *in what way* the environment changes is what tells you which class of
  algorithm you're allowed to use.

---

## 4.7.5 Fairness, Accountability, and Transparency in Machine Learning

Once a model makes decisions about people, distribution shift stops being a
purely statistical concern.

- **Aggregate accuracy hides subgroup failure.** A diagnostic model can be
  excellent overall and useless for a population under-represented in the
  training data — a covariate shift affecting only some of the people it will
  be applied to.
- **Accuracy is the wrong objective anyway.** Different errors carry wildly
  different costs. A false negative on a treatable illness and a false positive
  on the same illness are not interchangeable, however symmetric your loss
  function is.
- **Feedback loops are the deepest problem.** A deployed model can change the
  distribution it will later be trained on. **Predictive policing** is the
  standard cautionary tale: send more patrols where the model predicts crime,
  and more crime is *recorded* there simply because more officers are looking.
  Those records become the next round of training data, which reinforces the
  prediction. The loop runs away, and the model looks increasingly "accurate"
  the whole time, because it is validated against data it helped generate.
- **The mechanism is usually unmodeled.** Almost none of the standard machinery
  accounts for the model's own effect on the world. When it matters, it has to
  be reasoned about explicitly — the math will not surface it for you.

Deciding what a system optimizes, who it is measured on, and what it does to
the environment it sits inside are engineering choices, not afterthoughts.

---

## 4.7.6 Summary

- **Distribution shift** is the ordinary case, not the exception: training and
  deployment data usually come from different distributions
  $p_S(\mathbf{x}, y) \ne p_T(\mathbf{x}, y)$. An iid test score certifies
  generalization only within the training snapshot.
- Progress requires assuming the shift is **structured**, and the structure is
  a choice of which factor of $p(\mathbf{x}, y)$ stays fixed:
  - **Covariate shift** — $p(\mathbf{x})$ moves, $p(y \mid \mathbf{x})$ fixed.
    Natural when $\mathbf{x}$ causes $y$ (photos → cartoons).
  - **Label shift** — $p(y)$ moves, $p(\mathbf{x} \mid y)$ fixed. Natural when
    $y$ causes $\mathbf{x}$ (disease prevalence rises, symptoms don't change).
  - **Concept shift** — $p(y \mid \mathbf{x})$ itself drifts; the meaning of
    the labels changes over time or place.
- Empirical risk minimization estimates the risk under the distribution you
  **sampled**, not the one you will **face**. The bridge is importance
  weighting:
  $\mathbb{E}_{p_T}[\ell] = \mathbb{E}_{p_S}\!\left[\frac{p_T(\mathbf{x}, y)}{p_S(\mathbf{x}, y)} \ell\right]$,
  valid only where the source has support.
- The ratio simplifies to $p_T(\mathbf{x})/p_S(\mathbf{x})$ under covariate
  shift and $p_T(y)/p_S(y)$ under label shift; either way you then run
  **weighted ERM**, $\min_f \frac{1}{n}\sum_i \beta_i \ell(f(\mathbf{x}_i), y_i)$.
- **Covariate shift correction**: train a logistic **domain classifier**
  separating source from target inputs; its odds are the density ratio, so
  $\beta_i = \exp(h(\mathbf{x}_i))$. Clip extreme weights — they destroy the
  effective sample size.
- **Label shift correction**: the source confusion matrix $\mathbf{C}$ survives
  the shift, so the average prediction on unlabeled target data gives a linear
  system $\mathbf{C}\, p_T(\mathbf{y}) = \mu(\hat{\mathbf{y}})$, solved as
  $p_T(\mathbf{y}) = \mathbf{C}^{-1}\mu(\hat{\mathbf{y}})$. Needs an invertible
  $\mathbf{C}$, i.e. a decent classifier.
- **Concept shift** has no general fix; because it is usually gradual, adapt
  **online** with small continual updates rather than retraining from scratch.
- Learning problems form a spectrum by how much the environment reacts: batch,
  online, bandits, control, reinforcement learning. The more it reacts, the
  less a static model can be trusted.
- Deployed models **change the world they are measured in**. Feedback loops
  (predictive policing), subgroup failures, and asymmetric error costs mean
  that how data is generated — and what the model does to that process — is
  part of the modeling problem.
