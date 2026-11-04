# The 2-level model is fitted by Laplace-EM, not by a joint MAP

In the 2-level model, each session's parameters vary around the subject's typical values,
θ_s ~ N(μ, σ²) on the unconstrained scale, under study 13's fixed hyperprior (μ ~ N(0, 1),
log σ ~ N(−1, 1)). MEvolve fits μ and σ by Laplace-EM, the hierarchical EM of Huys et al. (2011).
The E-step finds each session's MAP under N(μ, σ²) and the diagonal of its Laplace covariance. The
M-step updates μ in closed form and each log σ by Newton steps on its one-dimensional stationarity
condition. A held-out session is then scored by its session marginal over 2,048 draws from
N(μ, σ²) (decided 2026-11-01, during build step 2). This is **2-level EM**; study 13's NUTS fit of
the same model is **2-level HB** (labels from 2026-11-02; this ADR first said "session level").

The first implementation, a joint MAP of (μ, log σ, z_s) in non-centred form (θ_s = μ + σ·z_s),
failed on study 13's 149 test subjects. Nothing in that objective resists σ growing: each offset's
prior cost z_s² = (θ_s − μ)²/σ² falls as σ rises, and only the weak prior on log σ pushes back.
The spreads came out about 15 times study 13's posterior medians, the session-marginal weight ESS
had a median of 1.1, and HB-Hattori's comparison fell from 0.7213 to 0.7066, below the per-subject
fit. The centred form fails the other way: σ → 0. Laplace-EM integrates the session parameters
out instead of maximizing over them, and its covariance term keeps short, noisy sessions from
inflating σ.

| Hattori2019 on the 149 test subjects | pooled held-out likelihood |
|---|---|
| study 13 HB-Hattori (NUTS posterior, session marginal) | 0.72129 |
| MEvolve scorer with study 13's posterior medians of μ and σ, 2,048 draws (6 subjects) | within 0.0002 per subject |
| MEvolve joint MAP, 256 draws | 0.70656 |
| MEvolve Laplace-EM, 2,048 draws | 0.72111 |
| MEvolve Laplace-EM with β capped at 10, as study 13 | 0.72112 |

With 256 draws the scorer read up to 0.0009 low, even with study 13's own μ and σ: the log of an
average over few draws is biased low. Hence 2,048 draws, with the weight ESS reported per session.

**Iterations.** EM converges linearly, but the held-out score settles long before the spreads of
poorly identified parameters do. The search caps EM at 20 iterations so that one evaluation of
the generation and validation subjects (96) takes about 3 minutes on a 112-core node (gate 2,
literature models tried: 103 to 145 s with the first fitter; 179 to 204 s once bound-face starts and
4 differential-evolution restarts were added, MEvolve #16 and #18; corrected 2026-11-04). On the
149 test subjects the cap costs little:

| 2-level EM, 149 test subjects | 100 iterations | 20 iterations | 2-level HB (study 13) |
|---|---|---|---|
| CompareToThreshold | 0.72234 | 0.72233 | 0.72246 |
| Hattori2019 | 0.72111 | 0.72093 | 0.72129 |

Both pass gate 1c either way. Scores that are reported, not ranked, such as the pre-registered
programs on the test subjects, run EM to convergence (``max_em=100``).

## Considered options

- **The full posterior (NUTS), as study 13.** The reference, but about 650 s per subject: too slow
  for a search that evaluates thousands of programs.
- **Variational inference (a Gaussian per session).** Also integrates the sessions out, with more
  code and tuning; kept in reserve if Laplace-EM fails on programs whose likelihoods are far from
  Gaussian.
