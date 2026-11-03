# A Synthetic Identifiability Benchmark for Sequence-Based versus Attractor-Based Evidence Accumulation in Calcium-Imaging-Like Population Data

**Aarush Kandukoori** (Carnegie Mellon University) — Lead Student Author  
**Aditya Kandukoori** (Poolesville High School) — Co-Author

> **Status: early-stage research proposal.** No experimental results
> are claimed. We are actively seeking expert feedback.

## Abstract

Neural circuit models of evidence accumulation range from integrator-like attractor dynamics to choice-selective sequences of time-localized activity. Both families can produce similar trial-averaged activity, and two-photon calcium imaging adds slow indicator kinetics and noise that further blur the difference. We propose an early-stage framework, and an accompanying synthetic-data benchmark, for asking when the two mechanisms can be told apart from calcium-imaging-like recordings. The framework has three parts: a one-parameter family of generative models that interpolates between a ramping (attractor-like) code and a sequence-like code, a forward model from latent spikes to fluorescence, and a cross-temporal decoder-generalization statistic as a candidate discriminator. We describe the protocol and planned baselines but report no results, and we use only clearly labeled synthetic data. We are seeking expert feedback on whether the question and the generative assumptions are sensible. AI tools (Claude) were used for literature synthesis, ideation, and draft preparation, and the authors reviewed and take responsibility for all content.

## Introduction & Motivation

Evidence accumulation is often modeled with attractor dynamics, where a (near-)line attractor integrates inputs and a fixed readout recovers the accumulated quantity. A different account, supported by recent circuit-modeling work from the Tank lab (listed in the profile as 'Neural circuit models for evidence accumulation through choice-selective sequences'), proposes that accumulation is carried by choice-selective sequences, where *which* neurons are active at *which* time encodes the running evidence.

Both accounts can yield similar trial-averaged tuning, and in practice neural activity is measured through two-photon calcium imaging, which low-pass filters spikes, adds noise, and subsamples neurons. A practical question follows: **under what recording conditions (number of neurons, trial count, indicator kinetics, noise) can an analysis distinguish the two mechanisms?**

This is an early-stage proposal. We are students, not experts in this circuit, and we are seeking feedback from researchers who know the experimental constraints. We make no claim of having answered the question; we propose a framework and a plan.

Our contributions, to our knowledge and as plans only, are: (1) a one-parameter generative family interpolating between ramp-like and sequence-like population codes; (2) a calcium-imaging forward model; (3) a candidate discriminator statistic based on cross-temporal decoder generalization; (4) an open synthetic-data benchmark and baseline plan.

## Related Work

**Attractor and integrator accounts.** Context-dependent recurrent computation in prefrontal cortex has been analyzed as low-dimensional dynamics with line-attractor-like integration of inputs [mante2013]. Head-direction circuits provide a canonical continuous-attractor example, with population activity confined to a low-dimensional manifold [chaudhuri2019] and with bump dynamics that integrate angular velocity [seelig2015].

**Sequence accounts.** Posterior parietal cortex activity during a navigation-based decision task shows history-dependent variability and sequence-like, time-localized activity in the population [morcos2016].

**Latent-dynamics tools.** Sequential autoencoders such as LFADS infer single-trial latent dynamics from population recordings [pandarinath2018]. Switching linear dynamical systems model piecewise-linear latent dynamics with Bayesian inference [linderman2017]. CEBRA learns joint behavioral-neural embeddings by contrastive learning [schneider2023].

**Calcium imaging observation models.** Fluorescence is commonly modeled as a filtered spike train with additive noise, and fast deconvolution methods invert this model [friedrich2017].

We are not aware of a benchmark that parameterizes the ramp-to-sequence continuum and measures identifiability under calcium-imaging observation noise, but our literature review is limited and we welcome corrections.

## Proposed Framework

### 1. Task and latent evidence
A trial has $T$ time bins. At bin $t$, a signed evidence pulse $u_t \in \{-1, 0, +1\}$ arrives (left/right/none). The ideal accumulated evidence is $x_t = \sum_{s \le t} u_s$.

### 2. Interpolating population code
For neuron $i = 1..N$, the firing rate at time $t$ is
$$\lambda_i(t) = \text{softplus}\Big( b_i + (1-\rho)\, w_i\, x_t + \rho\, a_i\, g\big(t - \tau_i\big)\, \mathrm{sgn}(x_t)\, \big|x_t\big|^{\kappa} \Big),$$
where $\rho \in [0,1]$ is the **sequence index**, $w_i$ are fixed readout-aligned weights, $g$ is a Gaussian bump of width $\sigma$, $\tau_i$ is a preferred time spread across the trial, $a_i$ is a choice-selectivity sign, and $\kappa \in (0, 1]$ is a saturation exponent (used only to keep the sequence term bounded in magnitude).

At $\rho = 0$ the population is a pure linear ramp code (attractor-like). At $\rho = 1$ each neuron is active only near $\tau_i$ (sequence-like). Intermediate $\rho$ gives mixtures.

### 3. Why a decoder-generalization statistic can discriminate (checkable argument)
Ignore the nonlinearity and noise (pure ramp code, $\rho = 0$, linear rates $r_i(t) = b_i + w_i x_t$). A single fixed linear readout $v$ with $v^\top w \neq 0$ gives $v^\top (r(t) - b) = (v^\top w)\, x_t$ for every $t$. So one decoder, trained at any time $t_0$, recovers $x_t$ at every other time up to a constant scale. **Cross-temporal generalization is therefore perfect.**

In a pure sequence code, neuron $i$ carries information about $x_t$ only for $|t - \tau_i| \lesssim \sigma$. A decoder trained at $t_0$ puts weight only on the neurons with $\tau_i$ near $t_0$. At a time $t$ with $|t - t_0| \gg \sigma$, those neurons are near baseline, so their contribution carries almost no evidence signal, and generalization degrades toward chance. The cross-temporal accuracy matrix is then concentrated near its diagonal.

This argument is heuristic, and it uses idealized tuning. Noise, partial overlap of bumps, and mixed $\rho$ will blur it, which is what the benchmark is meant to quantify.

### 4. Calcium forward model
Spike counts $s_{i,t} \sim \text{Poisson}(\lambda_i(t)\,\Delta)$ with bin width $\Delta$. Calcium follows an autoregressive process $c_{i,t} = \gamma\, c_{i,t-1} + s_{i,t}$ with $\gamma = e^{-\Delta/\tau_{\text{ca}}}$, and fluorescence is
$$F_{i,t} = \alpha_i\, c_{i,t} + \beta_i + \epsilon_{i,t}, \quad \epsilon_{i,t} \sim \mathcal{N}(0, \sigma_\epsilon^2).$$
This is the standard first-order model also used in deconvolution work [friedrich2017]. Because $\gamma$ close to 1 smears sequence bumps over time, slow indicator decay is expected to make sequences look more integrator-like, which is a confound the benchmark should expose.

### 5. Discriminator statistic
Let $A(t_{\text{train}}, t_{\text{test}})$ be the cross-validated accuracy (or $R^2$) of a linear decoder of $x$ trained at $t_{\text{train}}$ and tested at $t_{\text{test}}$. Define the generalization index
$$G = \frac{\overline{A_{\text{off-diag}}} - A_{\text{chance}}}{\overline{A_{\text{diag}}} - A_{\text{chance}}}.$$
We hypothesize that $G$ is close to 1 for $\rho = 0$ and decreases as $\rho$ grows. This is a hypothesis to be tested, not a finding. We will also examine whether $G$ separates the two regimes as $\gamma$, $N$, trial count, and $\sigma_\epsilon$ vary, and where it fails.

## Methodology

### Synthetic data generation
All data in the proposed repository is **synthetic and clearly labeled as such**. No real neural recordings are used. The generator draws trials from the model above over a grid of $\rho$, $N$, trials per condition, $\tau_{\text{ca}}$, and $\sigma_\epsilon$, with fixed random seeds.

### Analysis pipeline
1. Simulate latent rates, spikes, calcium, and fluorescence.
2. Optionally apply a deconvolution step (a first-order AR inversion in the spirit of [friedrich2017]) versus analyzing raw $\Delta F/F$-like traces.
3. Compute $A(t_{\text{train}}, t_{\text{test}})$ with cross-validated ridge decoders, then $G$.
4. Fit baseline model-based discriminators (see Evaluation).
5. Measure how well each method recovers the ground-truth regime label, with confidence intervals over simulation seeds.

### Identifiability question
For each configuration, we ask whether the two generative families (ramp-dominated vs. sequence-dominated) can be classified above chance, and how this depends on recording parameters. We plan to report this as a phase diagram over parameters, but only after the experiments are run.

### AI Assistance Disclosure
AI tools (Claude) were used for literature synthesis, ideation, and draft preparation of this proposal. The authors reviewed all content and take responsibility for it, including any errors.

## Planned Evaluation & Baselines

**Protocol (no results yet).** For each cell of the parameter grid we will simulate many independent datasets, apply each method, and score regime classification accuracy (ramp-dominated $\rho<0.5$ vs. sequence-dominated $\rho \ge 0.5$) and, separately, error in estimating $\rho$. We will report mean and spread across seeds, and check sensitivity to model misspecification (e.g., simulate with one tuning shape and analyze assuming another).

**Planned baselines.**
- **Single-neuron temporal-localization index:** peak-to-mean ratio of each neuron's trial-averaged activity over time, aggregated across neurons.
- **Cross-temporal decoder generalization $G$:** the proposed statistic.
- **Dimensionality of population activity over time:** e.g., participation ratio of the covariance, as a coarse sequence-vs-integrator signature.
- **Latent-dynamics model comparison:** fit a sequential autoencoder in the style of LFADS [pandarinath2018] and a switching linear dynamical model [linderman2017], and test whether fit quality or inferred dynamics separate regimes.
- **Learned embedding baseline:** a CEBRA-style embedding [schneider2023] followed by a decoding check.

**Ablations.** Remove the calcium forward model (decode directly from spike counts) to isolate the effect of indicator kinetics; vary the number of observed neurons to mimic imaging field-of-view limits.

**Sanity checks.** The statistic must give $G \approx 1$ on noiseless $\rho = 0$ data and a diagonal-dominated matrix on noiseless $\rho = 1$ data, as the Section 3 argument implies. Failure of these checks indicates an implementation bug.

## Limitations & Open Questions

- **Model simplicity.** The interpolating family is our own construction and is far simpler than real circuits. Real choice-selective sequences, and the circuit models in the Tank lab's recent work, may have structure (recurrent connectivity, heterogeneity, mixed selectivity) that this family omits. We would value guidance on which features matter most.
- **Identifiability is relative to the family.** A benchmark on synthetic data shows only what is distinguishable within our assumptions. It cannot show what the real brain does.
- **Observation model.** First-order calcium dynamics with Gaussian noise ignores nonlinear indicator response, neuropil contamination, motion artifacts, and spike-inference errors.
- **Decoder choice.** Linear decoders could miss nonlinear readouts that a downstream circuit might use.
- **Single task.** The pulsed-evidence task is a simplification. Extending to spatially embedded navigation-based decisions, as in other work from the lab, is an open direction.
- **Open questions for experts:** What range of $\tau_{\text{ca}}$, $N$, and trial counts is realistic? Are there existing datasets and ground-truth conventions we should align with? Is the cross-temporal generalization analysis already standard here, and are there known pitfalls?

## References

- [mante2013] Mante et al., *Context-dependent computation by recurrent dynamics in prefrontal cortex*, Nature 2013. https://doi.org/10.1038/nature12742
- [chaudhuri2019] Chaudhuri et al., *The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep*, Nature Neuroscience 2019. https://doi.org/10.1038/s41593-019-0460-x
- [seelig2015] Seelig et al., *Neural dynamics for landmark orientation and angular path integration*, Nature 2015. https://doi.org/10.1038/nature14446
- [morcos2016] Morcos et al., *History-dependent variability in population dynamics during evidence accumulation in cortex*, Nature Neuroscience 2016. https://doi.org/10.1038/nn.4403
- [pandarinath2018] Pandarinath et al., *Inferring single-trial neural population dynamics using sequential auto-encoders*, Nature Methods 2018. https://doi.org/10.1038/s41592-018-0109-9
- [linderman2017] Linderman et al., *Recurrent switching linear dynamical systems*, AISTATS 2017. https://arxiv.org/abs/1610.08466
- [schneider2023] Schneider et al., *Learnable latent embeddings for joint behavioral and neural analysis*, Nature 2023. https://arxiv.org/abs/2204.00673
- [friedrich2017] Friedrich et al., *Fast Online Deconvolution of Calcium Imaging Data*, PLOS Computational Biology 2017. https://arxiv.org/abs/1609.00639
