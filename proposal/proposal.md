# A Synthetic Identifiability Benchmark for Sequence-Based versus Attractor-Based Evidence Accumulation in Calcium-Imaging-Like Population Data

**Aarush Kandukoori** (Carnegie Mellon University, Lead Student Author)  
**Aditya Kandukoori** (Poolesville High School, Co-Author)

> **Status: early-stage research proposal.** No experimental results
> are claimed. We are actively seeking expert feedback.

## Abstract

Neural circuit models of evidence accumulation range from integrator-like attractor dynamics to choice-selective sequences of time-localized activity. Both families can produce similar trial-averaged activity, and two-photon calcium imaging adds slow indicator kinetics and noise that further blur the difference. We propose an early-stage framework, and an accompanying synthetic-data benchmark, for asking when the two mechanisms can be told apart from calcium-imaging-like recordings. The framework has three parts: a one-parameter family of generative models that interpolates between a ramping (attractor-like) code and a sequence-like code, a forward model from latent spikes to fluorescence, and a cross-temporal decoder-generalization statistic as a candidate discriminator. We describe the protocol and planned baselines but report no results, and we use only clearly labeled synthetic data. We are seeking expert feedback on whether the question and the generative assumptions are sensible. AI tools (Claude) were used for literature synthesis, ideation, and draft preparation, and an AI review checked the draft before release, and the authors take responsibility for all content.

## Introduction & Motivation

Evidence accumulation is often modeled with attractor dynamics, in which a (near-)line attractor integrates inputs and a fixed readout recovers the accumulated quantity (e.g., [mante2013]). A different account holds that accumulation can be carried by choice-selective sequences: neurons fire one after another rather than persistently, and the subset of neurons in the sequence depends on the animal's choice. Recent circuit models make this account concrete [brown2026]. In these models, sequentially active neurons pass evidence on to the next active population, and evidence is encoded either in the relative firing of two competing chains of neurons or in the network location of a bump-like pattern of activity.

Both accounts can yield similar trial-averaged tuning, and trial averages can hide differences in population structure. For example, when three cortical areas were imaged while mice switched between a working-memory task and a simpler task, trial-averaged single-neuron activity looked similar across the two tasks while pairwise correlations differed [diamanti2025]. Activity in such experiments is often measured with two-photon calcium imaging, which low-pass filters spikes, adds noise, and subsamples neurons. A practical question follows: **under what recording conditions (number of neurons, trial count, indicator kinetics, noise) can an analysis distinguish the two mechanisms?**

This is an early-stage proposal, and we are seeking feedback from researchers who know the experimental constraints. We make no claim of having answered the question; we propose a framework and a plan.

Our contributions, to our knowledge and as plans only, are: (1) a one-parameter generative family interpolating between ramp-like and sequence-like population codes; (2) a calcium-imaging forward model; (3) a candidate discriminator statistic based on cross-temporal decoder generalization; (4) an open synthetic-data benchmark and baseline plan.

## Related Work

**Attractor and integrator accounts.** Prefrontal population activity in macaques that select and integrate noisy inputs depending on context has been described as a single dynamical process and reproduced by a trained recurrent network, in which integration proceeds along an approximate line attractor [mante2013]. Head-direction circuits provide a canonical continuous-attractor example: mammalian head-direction population states lie on a one-dimensional ring manifold that is preserved across waking and REM sleep [chaudhuri2019], and in the fly ellipsoid body a population heading signal tracks visual landmarks, relies on self-motion cues in darkness, and shows features suggestive of a ring attractor [seelig2015].

**Sequence accounts.** In mouse posterior parietal cortex during evidence accumulation in a virtual navigation task, population activity moved rapidly between different sets of active neurons, and the authors argue that accumulation could emerge from general dynamics that support short-term memory, without explicit competition between groups of neurons [morcos2016]. Circuit models in which choice-selective sequences pass evidence from one active population to the next were developed in [brown2026], in two variants: one encodes evidence in the relative firing of two competing chains, and the other in the network location of a bump-like activity pattern. The same work compared the two models with recordings from four brain regions during an evidence accumulation task, and different regions showed evidence tuning consistent with different models. In a task-switching imaging study, firing fields were more disjoint during a navigation-based working-memory task than during a simpler task with matched sensorimotor demands, and neurons in association areas formed sequences along the memory dimension [diamanti2025].

**Latent-dynamics tools.** Sequential autoencoders such as LFADS infer single-trial latent dynamics from population recordings [pandarinath2018]. Recurrent switching linear dynamical systems model latent dynamics as switching between linear regimes, with Bayesian inference [linderman2017]. CEBRA learns joint behavioral-neural embeddings by contrastive learning [schneider2023].

**Calcium imaging observation models.** Fluorescence is commonly modeled as a filtered spike train with additive noise, and fast deconvolution methods invert this model [friedrich2017].

We are not aware of a benchmark that parameterizes the ramp-to-sequence continuum and measures identifiability under calcium-imaging observation noise, but our literature review is limited and we welcome corrections.

## Proposed Framework

### 1. Task and latent evidence
A trial has $T$ time bins, indexed $t = 1, \dots, T$. In each bin a signed evidence pulse $u_t \in \{-1, 0, +1\}$ arrives ($-1$ for a left pulse, $+1$ for a right pulse, $0$ for none): a pulse occurs with probability $p$, and its sign is $-1$ or $+1$ with equal probability ($p = 0.25$ in the reference code). The ideal accumulated evidence is $x_t = \sum_{t'=1}^{t} u_{t'}$.

### 2. Interpolating population code
For neuron $i = 1, \dots, N$, the firing rate (in spikes per second) at bin $t$ is
$$\lambda_i(t) = \text{softplus}\Big( b_i + (1-\rho)\, w_i\, x_t + \rho\, \eta\, a_i\, g\big(t - \tau_i\big)\, \mathrm{sgn}(x_t)\, \big|x_t\big|^{\kappa} \Big),$$
where:
- $\text{softplus}(z) = \log(1 + e^{z})$, which keeps rates positive;
- $b_i$ is a baseline, so $\text{softplus}(b_i)$ is the rate when the other two terms are zero;
- $\rho \in [0,1]$ is the **sequence index**;
- $w_i$ is a fixed ramp (encoding) weight, drawn once per simulated population;
- $\eta > 0$ is the amplitude of the sequence term;
- $a_i \in \{-1, +1\}$ is a choice-selectivity sign that assigns neuron $i$ to one of two chains;
- $g(z) = \exp\big(-z^2 / (2\sigma^2)\big)$ is a Gaussian bump, and $\sigma > 0$ is the firing-field width in bins;
- $\tau_i$ is the preferred time of neuron $i$, with the $\tau_i$ evenly spaced across the trial;
- $\mathrm{sgn}(\cdot)$ is the sign function, with $\mathrm{sgn}(0) = 0$;
- $\kappa \in (0, 1]$ is a compression exponent: for $\kappa < 1$ the sequence term grows sublinearly in $|x_t|$.

At $\rho = 0$ the population is a pure ramp code (attractor-like): every neuron's rate is a monotonic function of $x_t$ at every time. At $\rho = 1$ each neuron departs from baseline only near its preferred time $\tau_i$ (sequence-like). Near that time, neurons with $a_i = +1$ fire above baseline when $x_t > 0$ and below it when $x_t < 0$, and neurons with $a_i = -1$ do the opposite, so the evidence is carried by the relative firing of the two chains. Intermediate $\rho$ gives mixtures. The field width $\sigma$, relative to the trial length $T$, sets how much the firing fields of neurons with different preferred times overlap.

### 3. Why a decoder-generalization statistic can discriminate (checkable argument)
Ignore the nonlinearity and noise, and take the pure ramp code ($\rho = 0$) with linear rates $r(t) = b + w\, x_t$, where $r(t), b, w \in \mathbb{R}^N$ stack $r_i(t)$, $b_i$ and $w_i$ over neurons, and $w \neq 0$. Consider an affine decoder $\hat{x} = v^\top r + v_0$ with weights $v \in \mathbb{R}^N$ and offset $v_0 \in \mathbb{R}$, fit without error at a time $t_0$ at which $x_{t_0}$ takes at least two distinct values across trials. Then $(v^\top w)\, x_{t_0} + v^\top b + v_0 = x_{t_0}$ holds for at least two values of $x_{t_0}$, which forces $v^\top w = 1$ and $v_0 = -v^\top b$. The same decoder therefore returns $\hat{x} = x_t$ at every other time $t$. **Cross-temporal generalization is perfect.**

Now take the pure sequence code ($\rho = 1$), still without noise. Because $g(t - \tau_i) \approx 0$ when $|t - \tau_i| \gg \sigma$, neuron $i$ stays at its baseline rate $\text{softplus}(b_i)$ except when $|t - \tau_i| \lesssim \sigma$. At time $t_0$, neurons with $\tau_i$ far from $t_0$ have (nearly) the same rate on every trial, so a ridge decoder fit at $t_0$ gives them (nearly) zero weight and relies on the neurons with $\tau_i$ near $t_0$. At a time $t$ with $|t - t_0| \gg \sigma$, those neurons are back at baseline, so the decoder output carries almost no evidence signal and accuracy falls toward chance. The cross-temporal accuracy matrix is then concentrated in a band of width on the order of $\sigma$ around its diagonal. A larger $\sigma$ widens this band, so a sequence code with broad, overlapping firing fields is expected to generalize further from the diagonal. This is a second route, besides slow calcium decay (Section 4), by which a sequence code could look integrator-like.

This argument is heuristic, and it uses idealized tuning. Noise, partial overlap of bumps, and mixed $\rho$ will blur it, which is what the benchmark is meant to quantify.

### 4. Calcium forward model
Spike counts are $s_{i,t} \sim \text{Poisson}(\lambda_i(t)\,\Delta)$, where $\Delta$ is the bin width in seconds. Calcium follows the autoregressive process $c_{i,t} = \gamma\, c_{i,t-1} + s_{i,t}$, with $c_{i,1} = s_{i,1}$ and $\gamma = e^{-\Delta/\tau_{\text{ca}}}$, where $\tau_{\text{ca}}$ is the indicator decay time constant in seconds. Fluorescence is
$$F_{i,t} = \alpha_i\, c_{i,t} + \beta_i + \epsilon_{i,t}, \quad \epsilon_{i,t} \sim \mathcal{N}(0, \sigma_\epsilon^2),$$
where $\alpha_i > 0$ is a per-neuron gain, $\beta_i$ is a per-neuron baseline, and $\epsilon_{i,t}$ is independent Gaussian noise with standard deviation $\sigma_\epsilon$. This is the standard first-order model also used in deconvolution work [friedrich2017]. Because $\gamma$ close to 1 smears sequence bumps over time, slow indicator decay is expected to make sequences look more integrator-like, which is a confound the benchmark should expose.

### 5. Discriminator statistic
Let $A(t_{\text{train}}, t_{\text{test}})$ be the cross-validated $R^2$ of a ridge decoder (linear regression with an $\ell_2$ penalty) of $x$ that is fit to population activity at bin $t_{\text{train}}$ and applied to activity at bin $t_{\text{test}}$. Here $R^2$ is one minus the ratio of the squared prediction error to the squared deviation of $x_{t_{\text{test}}}$ from its mean, summed over held-out trials, and it is clipped at 0, so the chance level is $A_{\text{chance}} = 0$. Bins at which $x$ has no variance across trials are excluded. For a minimum lag $L$ ($L = \lfloor T/4 \rfloor$ in the reference code), define the generalization index
$$G = \frac{\overline{A}_{\text{off}} - A_{\text{chance}}}{\overline{A}_{\text{diag}} - A_{\text{chance}}},$$
where $\overline{A}_{\text{diag}}$ is the mean of $A$ over pairs with $t_{\text{train}} = t_{\text{test}}$, and $\overline{A}_{\text{off}}$ is the mean over pairs with $|t_{\text{train}} - t_{\text{test}}| \ge L$. $G$ is defined only when $\overline{A}_{\text{diag}} > A_{\text{chance}}$. In the idealized ramp case of Section 3, an exact affine fit gives $A = 1$ for every pair, so $G = 1$. We hypothesize that $G$ is close to 1 for $\rho = 0$ and decreases as $\rho$ grows. This is a hypothesis to be tested, not a finding. We will also examine whether $G$ separates the two regimes as $\gamma$, $\sigma$, $N$, trial count, and $\sigma_\epsilon$ vary, and where it fails.

## Methodology

### Synthetic data generation
All data in the proposed repository is **synthetic and clearly labeled as such**. No real neural recordings are used. The generator draws trials from the model above over a grid of $\rho$, $\sigma$, $N$, trials per condition, $\tau_{\text{ca}}$, and $\sigma_\epsilon$, with fixed random seeds. The field width $\sigma$ is on the grid because field overlap can differ between conditions in real recordings: firing fields were more disjoint during a working-memory task than during a simpler task [diamanti2025], and in our family $\sigma$ is the parameter that sets how much fields overlap.

### Analysis pipeline
1. Simulate latent rates, spikes, calcium, and fluorescence.
2. Optionally apply a deconvolution step (a first-order autoregressive inversion in the spirit of [friedrich2017]) versus analyzing the fluorescence $F$ directly.
3. Compute $A(t_{\text{train}}, t_{\text{test}})$ with cross-validated ridge decoders, then $G$.
4. Fit baseline model-based discriminators (see Planned Evaluation & Baselines).
5. Measure how well each method recovers the ground-truth regime label, with confidence intervals over simulation seeds.

### Identifiability question
For each configuration, we ask whether the two generative families (ramp-dominated vs. sequence-dominated) can be classified above chance, and how this depends on recording parameters. We plan to report this as a phase diagram over parameters, but only after the experiments are run.

### AI Assistance Disclosure
AI tools (Claude) were used for literature synthesis, ideation, and draft preparation of this proposal. An AI review checked the draft before release, and the authors take responsibility for it, including any errors.

## Planned Evaluation & Baselines

**Protocol (no results yet).** For each cell of the parameter grid we will simulate many independent datasets, apply each method, and score regime classification accuracy (ramp-dominated $\rho<0.5$ vs. sequence-dominated $\rho \ge 0.5$) and, separately, error in estimating $\rho$. We will report mean and spread across seeds, and check sensitivity to model misspecification (e.g., simulate with one tuning shape and analyze assuming another).

**Planned baselines.**
- **Single-neuron temporal-localization index:** for each neuron, the peak-to-mean ratio over time of its absolute choice difference (mean activity on trials with positive final evidence $x_T$ minus mean activity on trials with negative $x_T$), averaged over neurons.
- **Cross-temporal decoder generalization $G$:** the proposed statistic.
- **Dimensionality of population activity:** the participation ratio $\text{PR} = \big(\sum_k \mu_k\big)^2 / \sum_k \mu_k^2$, where the $\mu_k$ are the eigenvalues of the neuron-by-neuron covariance of activity pooled over trials and time, as a coarse sequence-vs-integrator signature.
- **Latent-dynamics model comparison:** fit a sequential autoencoder in the style of LFADS [pandarinath2018] and a recurrent switching linear dynamical system [linderman2017], and test whether fit quality or inferred dynamics separate regimes.
- **Learned embedding baseline:** a CEBRA-style embedding [schneider2023] followed by a decoding check.

**Ablations.** Remove the calcium forward model (decode directly from spike counts) to isolate the effect of indicator kinetics; vary the number of observed neurons to mimic imaging field-of-view limits; vary $\sigma$ at fixed $\rho$ to separate field width from the sequence index.

**Sanity checks.** On noiseless rates, the statistic should give $G$ close to 1 for $\rho = 0$ and a diagonal-dominated matrix for $\rho = 1$, as the Section 3 argument implies. A failure of these checks would point to an implementation bug or a flaw in the argument.

## Limitations & Open Questions

- **Model simplicity.** The interpolating family is our own construction and is far simpler than real circuits. Real choice-selective sequences, and the circuit models in [brown2026], may have structure (recurrent connectivity, heterogeneity, mixed selectivity) that this family omits. We would value guidance on which features matter most.
- **One sequence mechanism.** Our sequence term is closest in spirit to the two-chain variant in [brown2026], with $a_i$ labeling the chain. It does not include the variant in which evidence is carried by the network location of a bump of activity. A planned extension adds a family member in which each neuron also has a preferred evidence value, so that which neurons are active encodes $x_t$.
- **Identifiability is relative to the family.** A benchmark on synthetic data shows only what is distinguishable within our assumptions. It cannot show what the real brain does.
- **Observation model.** First-order calcium dynamics with Gaussian noise ignores nonlinear indicator response, neuropil contamination, motion artifacts, and spike-inference errors.
- **Decoder choice.** Linear decoders could miss nonlinear readouts that a downstream circuit might use.
- **Single task.** The pulsed-evidence task is a simplification. Extending it to navigation-based tasks in virtual reality, as used in [morcos2016] and [diamanti2025], is an open direction.
- **Open questions for experts:** What range of $\tau_{\text{ca}}$, $N$, and trial counts is realistic? Are there existing datasets and ground-truth conventions we should align with? Is the cross-temporal generalization analysis already standard here, and are there known pitfalls?

## References

- [mante2013] Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013). *Context-dependent computation by recurrent dynamics in prefrontal cortex*. Nature, 503, 78-84. https://doi.org/10.1038/nature12742
- [chaudhuri2019] Chaudhuri, R., Gerçek, B., Pandey, B., Peyrache, A., & Fiete, I. (2019). *The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep*. Nature Neuroscience, 22, 1512-1520. https://doi.org/10.1038/s41593-019-0460-x
- [seelig2015] Seelig, J. D., & Jayaraman, V. (2015). *Neural dynamics for landmark orientation and angular path integration*. Nature, 521, 186-191. https://doi.org/10.1038/nature14446
- [morcos2016] Morcos, A. S., & Harvey, C. D. (2016). *History-dependent variability in population dynamics during evidence accumulation in cortex*. Nature Neuroscience, 19, 1672-1681. https://doi.org/10.1038/nn.4403
- [brown2026] Brown, L. S., Cho, J. R., Bolkan, S. S., Nieh, E. H., Schottdorf, M., Tank, D. W., Brody, C. D., Witten, I. B., & Goldman, M. S. (2026). *Neural circuit models for evidence accumulation through choice-selective sequences*. Nature Communications, 17, 4055. https://doi.org/10.1038/s41467-026-70267-9
- [diamanti2025] Diamanti, E. M., Pinto, L., Schottdorf, M., Russo, A. A., Thiberge, S. Y., Brody, C. D., & Tank, D. W. (2025). *Working memory expands shared task representations in cortex*. bioRxiv preprint. https://doi.org/10.1101/2025.09.29.679345
- [pandarinath2018] Pandarinath, C., O'Shea, D. J., Collins, J., Jozefowicz, R., Stavisky, S. D., Kao, J. C., Trautmann, E. M., Kaufman, M. T., Ryu, S. I., Hochberg, L. R., Henderson, J. M., Shenoy, K. V., Abbott, L. F., & Sussillo, D. (2018). *Inferring single-trial neural population dynamics using sequential auto-encoders*. Nature Methods, 15, 805-815. https://doi.org/10.1038/s41592-018-0109-9
- [linderman2017] Linderman, S. W., Miller, A. C., Adams, R. P., Blei, D. M., Paninski, L., & Johnson, M. J. (2016). *Recurrent switching linear dynamical systems*. arXiv:1610.08466. Published in the AISTATS 2017 proceedings (PMLR 54, 914-922) as *Bayesian learning and inference in recurrent switching linear dynamical systems*. https://arxiv.org/abs/1610.08466
- [schneider2023] Schneider, S., Lee, J. H., & Mathis, M. W. (2023). *Learnable latent embeddings for joint behavioural and neural analysis*. Nature, 617, 360-368. https://doi.org/10.1038/s41586-023-06031-6
- [friedrich2017] Friedrich, J., Zhou, P., & Paninski, L. (2017). *Fast online deconvolution of calcium imaging data*. PLOS Computational Biology, 13(3), e1005423. https://doi.org/10.1371/journal.pcbi.1005423
