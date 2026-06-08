# Adaptive ODE Event-Time Generation (`_inverse_cdf_ode`)

## Overview

PDMP samplers (Bouncy Particle, ZigZag) advance by drawing a sequence of *event
times*. Between events the state moves deterministically along a straight ray,
`x(t) = x₀ + t·v`, and an event (a velocity bounce / coordinate flip) fires at a
random time `τ` whose hazard is the position-dependent **event rate**
`λ(t) = λ(x₀ + t·v)`. Sampling `τ` is the inner loop of every PDMP step, and its
cost (number of rate evaluations) dominates runtime when the rate is expensive —
e.g. when each evaluation calls a finite-element forward model or a surrogate
gradient.

The classic way to sample `τ` is the **inverse-CDF method**: draw a unit
exponential threshold `S ~ Exp(1)` and solve

```
∫₀^τ λ(s) ds = S
```

for `τ`. The integral is the *cumulative hazard*; because the survival function
of the first event is `P(τ > t) = exp(−∫₀ᵗ λ)`, the variable `S = ∫₀^τ λ` is
exactly unit-exponential, which is the inverse-CDF (a.k.a. inversion / time
re-scaling) sampling identity.

The baseline implementation `_inverse_cdf` solves this with a **fixed-step
trapezoidal march** (step `int_dt`). `_inverse_cdf_ode` solves the *same*
equation but recasts it as an **ODE with a terminal event** and hands it to an
adaptive Runge–Kutta integrator (`scipy.integrate.solve_ivp`, `RK45`). The
adaptive step size and the built-in root-finding for the event replace the fixed
grid, controlling the integration error through tolerances `ode_rtol` /
`ode_atol` while typically spending far fewer rate evaluations.

Both methods are exposed through the same dispatch slot, selected by the
constructor argument `integrator`:

```python
self._general_inverse_cdf = (self._inverse_cdf_ode if integrator == 'ode'
                             else self._inverse_cdf)
```

This *general* generator is used whenever no analytic shortcut applies. (With a
`LaplaceSurrogate` the rate is linear and `_inverse_cdf_linear` solves a
quadratic in closed form; with a constant surrogate `_inverse_cdf_constant` is
purely analytic. The ODE path is the fallback for genuinely nonlinear rates —
the true target, or a GP / neural-network surrogate.)

---

## From cumulative hazard to an ODE

Define the running cumulative hazard

```
I(t) = ∫₀ᵗ λ(s) ds.
```

Differentiating turns the integral equation into an **initial value problem**:

```
I'(t) = λ(t) = λ(x₀ + t·v),     I(0) = 0.
```

The event time `τ` is the first `t` at which `I(t)` crosses the sampled
threshold `S`:

```
I(τ) = S.
```

This is precisely a **terminal-event ODE**: integrate `I` forward and stop the
integration the instant the event function `g(t) = I(t) − S` hits zero.
`solve_ivp` supports exactly this through its `events` mechanism, which detects
sign changes of `g` along the solution and refines the crossing time by a
bracketed root-find (Brent) on the dense RK interpolant — so the returned `τ` is
accurate to the integrator tolerance, not to the step size.

Because `λ ≥ 0`, `I` is **monotone non-decreasing**, so the crossing is unique
and we tell the solver to only look for it in the increasing direction
(`direction = 1`).

---

## Bouncy Particle Sampler

In the BPS the canonical event is a *bounce* (reflection of `v` off the level
sets of the potential). The rate used for the inversion is

```
λ(x) = max(0, −⟨∇U(x), v⟩ + offset) + γ
```

(`_cdf_rates`), where `γ` is a small floor (`gamma`) ensuring ergodicity and
`offset` is the thinning/bound offset. Velocity **refreshment** is handled by a
*separate* exponential clock, not by the rate above: a refresh time is drawn
independently as `refresh_time ~ Exp(refresh_rate)`, and the actual next event is
whichever of {bounce, refresh} comes first.

```python
def _inverse_cdf_ode(self) -> tuple[float, int]:
    s = -np.log(self._rng.uniform())                       # S ~ Exp(1)
    refresh_time = self._rng.exponential(scale=1.0 / self._refresh_rate)

    x0 = self.positions[self._iter]
    v  = self.velocities[self._iter]

    def fun(t, _integral):           # I'(t) = λ(x₀ + t v)
        return self._cdf_rates(x0 + t * v)

    def hit_threshold(t, integral):  # event: I(t) − S = 0
        return integral[0] - s
    hit_threshold.terminal  = True
    hit_threshold.direction = 1

    sol = solve_ivp(fun, (0.0, refresh_time), [0.0],
                    method='RK45', events=hit_threshold,
                    rtol=self._ode_rtol, atol=self._ode_atol)

    if sol.t_events[0].size > 0:
        return float(sol.t_events[0][0]), 0   # bounce
    return refresh_time, 1                     # refresh
```

Key points:

- **Integration is capped at `refresh_time`.** The bounce time is only ever
  *used* when it precedes the refresh; integrating past `refresh_time` would be
  wasted work. By passing `(0.0, refresh_time)` as the integration span we get
  the `min(bounce, refresh)` selection for free: if the terminal event fires
  inside the span we return the bounce (`event_type = 0`); if the integrator
  reaches `refresh_time` without `I` ever reaching `S`, no bounce occurred in
  that window and we return the refresh (`event_type = 1`). This also guarantees
  the loop terminates even when the canonical rate sits near zero for a long
  stretch (where the old fixed-step march could stall).
- **Return value** is `(time, type)` with `type ∈ {0 = bounce, 1 = refresh}`,
  matching the baseline so the downstream `_step` logic is unchanged.

---

## ZigZag Sampler

The ZigZag runs **one independent clock per coordinate**: each dimension `d` has
its own rate `λ_d(t)` and its own exponential threshold `s_d`, and the next
event flips the velocity of whichever coordinate reaches its threshold first.
The ODE is therefore *vector-valued*, `I_d'(t) = λ_d(t)` with `I_d(0) = 0`, and
there is **one terminal event per coordinate**:

```python
def _inverse_cdf_ode(self) -> tuple[np.floating, np.integer]:
    if self._s is None:                                    # persist across
        self._s = -np.log(self._rng.uniform(0, 1, self._dim))   # rejections
    s = self._s

    x0 = self.positions[self._iter]
    v  = self.velocities[self._iter]

    def fun(t, _integral):                # I_d'(t) = λ_d(x₀ + t v)
        return self._cdf_rates(x0 + t * v)

    events = []
    for d in range(self._dim):
        def hit_threshold(t, integral, d=d):   # I_d(t) − s_d = 0
            return integral[d] - s[d]
        hit_threshold.terminal  = True
        hit_threshold.direction = 1
        events.append(hit_threshold)

    t_cap = float(np.min(s) / self._gamma)     # see below
    sol = solve_ivp(fun, (0.0, t_cap), np.zeros(self._dim),
                    method='RK45', events=events,
                    rtol=self._ode_rtol, atol=self._ode_atol)

    tau = np.inf
    j = int(np.argmin(s))
    for d in range(self._dim):
        if sol.t_events[d].size > 0 and sol.t_events[d][0] < tau:
            tau = float(sol.t_events[d][0])
            j = d
    if not np.isfinite(tau):
        tau = t_cap
    return tau, j
```

Two ZigZag-specific details:

- **Threshold persistence.** `self._s` is drawn lazily and *kept* across calls.
  If an event is rejected (e.g. a forward-model failure, or thinning rejection),
  the same thresholds are reused on the next attempt so the random stream stays
  consistent — exactly as in the fixed-step `_inverse_cdf`. It is reset to
  `None` after an accepted event.

- **The integration cap `t_cap = min_d s_d / γ`.** Every rate is floored by `γ`
  (`λ_d ≥ γ`), so `I_d(t) ≥ γ·t`, which means coordinate `d` is *guaranteed* to
  cross `s_d` no later than `t = s_d / γ`. The earliest possible crossing is thus
  at most `min_d s_d / γ`. Capping the integration span there bounds the work
  and **guarantees at least one event fires** even in the worst case where every
  rate sits exactly on the `γ` floor — the defensive `tau = t_cap` fallback
  covers the (numerically negligible) case where no event is recorded.

After integration the routine picks the **earliest** triggered event across
coordinates and returns `(τ, j)`, where `j` is the dimension that flipped.

---

## Why this is cheaper than the fixed-step march

The baseline `_inverse_cdf` evaluates the rate on a uniform grid of width
`int_dt` and accumulates a trapezoidal sum until the running integral exceeds
`S`, then linearly interpolates the last step. Its cost and accuracy are both
dictated by a single hand-tuned `int_dt`:

| | Fixed step (`_inverse_cdf`) | Adaptive ODE (`_inverse_cdf_ode`) |
|---|---|---|
| Integrator | Trapezoid, step `int_dt` | RK45, adaptive step |
| Error control | Implicit in `int_dt` | Explicit via `ode_rtol`, `ode_atol` |
| Event location | Linear interpolation on last step | Brent root-find on dense interpolant |
| Cost | ∝ `τ / int_dt` rate evals | Few RK stages between adaptive steps |
| Failure mode | Stalls if rate ≈ 0 for long | Bounded by integration cap |

Where the rate varies smoothly the adaptive integrator takes large steps and
spends a handful of evaluations per event; where it varies sharply it
automatically refines. The error is then governed by *tolerances you set* rather
than by a global step you must guess. The companion study
`examples/bps_event_study/integration_scheme_study.py` quantifies the
accuracy/cost trade-off on a mildly non-Gaussian target where the integration
scheme genuinely matters (a single Gaussian has a piecewise-linear rate that the
trapezoid already integrates almost exactly, so it is *not* a useful test case).

---

## Instrumentation

Both generators count the rate evaluations they spend so the integration cost can
be profiled:

- `self._n_rate_evals` — cumulative rate evaluations across the whole run.
- `self._rate_evals_per_event` — list with one entry per generated event.

In `_inverse_cdf_ode` the count is incremented inside the `fun` closure (via a
`nonlocal`/closure counter in BPS; the ZigZag version counts per call), so it
reflects the *actual* number of RHS evaluations the adaptive integrator
requested — the honest cost proxy for comparing schemes.

---

## Configuration

| Parameter | Meaning | Used by |
|---|---|---|
| `integrator` | `'ode'` selects `_inverse_cdf_ode`; anything else selects the fixed-step `_inverse_cdf`. | constructor dispatch |
| `ode_rtol` | Relative tolerance passed to `solve_ivp`. | `_inverse_cdf_ode` |
| `ode_atol` | Absolute tolerance passed to `solve_ivp`. | `_inverse_cdf_ode` |
| `int_dt` | Fixed step for the trapezoidal baseline (ignored by the ODE path). | `_inverse_cdf` |
| `gamma` | Rate floor `γ`; also sets the ZigZag integration cap. | both |
| `refresh_rate` | Rate of the independent BPS velocity-refresh clock. | BPS only |

Selecting the ODE integrator (BPS shown; ZigZag is identical):

```python
sampler = BouncyParticleSampler(
    target=target,
    integrator='ode',
    ode_rtol=1e-6,
    ode_atol=1e-9,
    gamma=1e-3,
    refresh_rate=0.1,
)
```

---

## References in the code

- `pdmp/bouncy_particle.py` — `BouncyParticleSampler._inverse_cdf_ode`
  (BPS, scalar ODE with refresh cap).
- `pdmp/zigzag.py` — `ZigZagSampler._inverse_cdf_ode`
  (ZigZag, vector ODE with one event per coordinate).
- `pdmp/bouncy_particle.py` / `pdmp/zigzag.py` — `_inverse_cdf`
  (fixed-step trapezoidal baseline solving the same equation).
- `tests/test_event_integration.py` — verifies the ODE generator is selected and
  agrees with the baseline.
- `examples/bps_event_study/integration_scheme_study.py` — accuracy/cost study.
