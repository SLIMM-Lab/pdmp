# RVE 2D — Mathematical Notes

## 1. Problem formulation

The total displacement field is split into a linear (macroscopic) part and a periodic fluctuation:

$$
\mathbf{u}(\mathbf{x}) = \pmb{\varepsilon}^{\text{mac}} \cdot \mathbf{x} + \tilde{\mathbf{u}}(\mathbf{x})
$$

- $\pmb{\varepsilon}^{\text{mac}}$ — prescribed macroscopic strain tensor (symmetric, 2×2)
- $\tilde{\mathbf{u}}$ — unknown periodic fluctuation (zero average over the RVE)

The total strain is:

$$
\pmb{\varepsilon}(\mathbf{u}) = \pmb{\varepsilon}^{\text{mac}} + \pmb{\varepsilon}(\tilde{\mathbf{u}}), \qquad
\pmb{\varepsilon}(\tilde{\mathbf{u}}) = \tfrac{1}{2}\!\left(\nabla\tilde{\mathbf{u}} + (\nabla\tilde{\mathbf{u}})^T\right)
$$

The plane-strain constitutive law (isotropic, linear elastic) is:

$$
\pmb{\sigma} = \lambda \,\mathrm{tr}(\pmb{\varepsilon})\,\mathbf{I} + 2\mu\,\pmb{\varepsilon},
\qquad
\mu = \frac{E}{2(1+\nu)}, \quad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
$$

The weak form solved by jax-fem is:

$$
\int_\Omega \pmb{\sigma}\bigl(\pmb{\varepsilon}^{\text{mac}} + \pmb{\varepsilon}(\tilde{\mathbf{u}})\bigr) : \pmb{\varepsilon}(\mathbf{v})\,d\Omega = 0
\quad \forall\, \mathbf{v} \text{ periodic}
$$

$\pmb{\varepsilon}^{\text{mac}}$ acts as a **body-force-like source term** baked into the constitutive law; the solver finds $\tilde{\mathbf{u}}$ only.

---

## 2. How the macroscopic strain is enforced

`eps_macro_voigt = [eps_xx, eps_yy, gamma_xy]` is converted to tensor form:

$$
\pmb{\varepsilon}^{\text{mac}} = \begin{pmatrix} \varepsilon_{xx} & \tfrac{1}{2}\gamma_{xy} \\ \tfrac{1}{2}\gamma_{xy} & \varepsilon_{yy} \end{pmatrix}
$$

Inside `get_tensor_map`, every quadrature-point stress evaluation adds $\pmb{\varepsilon}^{\text{mac}}$ to the fluctuation gradient before computing $\pmb{\sigma}$:

```python
eps_fluct = 0.5 * (u_grad + u_grad.T)   # sym(∇ũ) at this quad point
eps_total = eps_fluct + eps_macro        # ε_mac captured in closure
sigma     = λ tr(eps_total) I + 2μ eps_total
```

This is the closure trick: `eps_macro` is a Python-level constant captured by the inner function, so no extra unknowns or boundary integrals are needed.

---

## 3. Periodic boundary conditions

Periodicity of $\tilde{\mathbf{u}}$ means that opposite boundary faces carry the **same** fluctuation:

$$
\tilde{\mathbf{u}}(L, y) = \tilde{\mathbf{u}}(0, y) \quad \forall\, y \in [0, L]
$$
$$
\tilde{\mathbf{u}}(x, L) = \tilde{\mathbf{u}}(x, 0) \quad \forall\, x \in [0, L]
$$

These are **multipoint constraints** (MPCs): a node on the right edge is *slaved* to the matching node on the left edge (same $y$-coordinate), and a node on the top edge is slaved to its bottom counterpart (same $x$-coordinate).

gmsh's `setPeriodic` guarantees that for every interior node on the right edge there is a node on the left edge with the same $y$-coordinate, and likewise for top/bottom. Without this the MPC matching step would fail.

---

## 4. What is P_mat?

jax-fem (and many FEM codes) supports linear constraints of the form

$$
\mathbf{u}_{\text{full}} = P\,\mathbf{u}_{\text{reduced}}
$$

where:

| Symbol | Meaning |
|---|---|
| $\mathbf{u}_{\text{full}} \in \mathbb{R}^N$ | all DOFs, $N = 2 \times n_{\text{nodes}}$ |
| $\mathbf{u}_{\text{reduced}} \in \mathbb{R}^M$ | **independent** DOFs only, $M < N$ |
| $P \in \mathbb{R}^{N \times M}$ | sparse Boolean map (0s and 1s here) |

$P$ is called the **constraint matrix** or **prolongation matrix**.

The assembled stiffness problem $K\mathbf{u} = \mathbf{f}$ is reduced to:

$$
K_{\text{red}}\,\mathbf{u}_{\text{reduced}} = \mathbf{f}_{\text{red}},
\qquad
K_{\text{red}} = P^T K P \in \mathbb{R}^{M \times M}, \quad \mathbf{f}_{\text{red}} = P^T \mathbf{f}
$$

which is a smaller, non-singular system. After solving, the full solution is reconstructed as $\mathbf{u}_{\text{full}} = P\,\mathbf{u}_{\text{reduced}}$.

---

## 5. Building P_mat step by step

There are three categories of DOFs:

| Category | Description | Row in P |
|---|---|---|
| **Free** interior DOFs | neither slave nor pinned | one entry of value 1 in its own unique column |
| **Slave** DOFs | right/top boundary, matched to a master | one entry of value 1 in the master's column |
| **Pinned** DOFs | all 4 corners | all zeros (DOF forced to 0) |

### Step 1 — identify node sets

```
left   : x ≈ 0        right  : x ≈ L
bottom : y ≈ 0        top    : y ≈ L
corners: {BL, BR, TL, TR}  (intersection of edge sets)
interior edges: edges minus corners
```

### Step 2 — build slave→master pairs

For each right-interior node $r$, find the left-interior node $\ell$ with $|y_r - y_\ell| < \epsilon$:

$$
u_r = u_\ell, \quad v_r = v_\ell
$$

Likewise for top→bottom pairs. This enforces $\tilde{\mathbf{u}}_\text{right} = \tilde{\mathbf{u}}_\text{left}$ and $\tilde{\mathbf{u}}_\text{top} = \tilde{\mathbf{u}}_\text{bottom}$ node by node.

### Step 3 — expand nodes to DOFs

Each node $i$ contributes two DOFs: $2i$ (x-displacement) and $2i+1$ (y-displacement). The same constraint applies to each component independently.

### Step 4 — corner pinning

All 4 corners are pinned to zero by setting their rows of $P$ to zero. This:
1. satisfies corner periodicity (all corners share the same value, which is 0)
2. eliminates the **rigid-body translation** mode (the only zero-energy mode of a fully periodic problem)

> **Why not use a Dirichlet BC instead?**
> jax-fem applies Dirichlet BCs *after* $P$ is assembled by zeroing the corresponding row *and column* of the reduced system. If a corner node is also a master node (it has slaves pointing to it), zeroing its column would destroy those slave constraints. Pinning via a zero row in $P$ is safe because the master column in the reduced system is simply never assembled.

### Step 5 — assemble the sparse matrix

```
M = number of free DOFs (interior + left/bottom interior + left corners — actually just interior and left/bottom-non-corner)

For each full DOF i:
  if pinned:   skip (zero row)
  if slave:    P[i, reduced_idx[master_of[i]]] = 1
  if free:     P[i, reduced_idx[i]]            = 1
```

The result is a $N \times M$ sparse matrix with exactly one non-zero per row (for non-pinned DOFs).

---

## 6. Summary diagram

| Full DOF type | Action in P | Effect after back-substitution |
|---|---|---|
| **Corner** (pinned) | zero row — no column assigned | $u_i = 0$ |
| **Slave** (right/top edge) | row points to master's column | $u_i = u_{\text{master}}$ |
| **Free** (interior + left/bottom edge) | row points to its own column | $u_i$ solved directly |

After solving the reduced system, $\mathbf{u}_{\text{full}} = P\,\mathbf{u}_{\text{reduced}}$ automatically propagates each master's solution to all its slaves and leaves corners at zero.

---

## 7. Recovering the full displacement field

The solver returns $\tilde{\mathbf{u}}$ only. The full displacement is:

$$
\mathbf{u}(\mathbf{x}) = \pmb{\varepsilon}^{\text{mac}} \cdot \mathbf{x} + \tilde{\mathbf{u}}(\mathbf{x})
$$

In components:

$$
u_x = \varepsilon_{xx}\,x + \tfrac{1}{2}\gamma_{xy}\,y + \tilde{u}_x, \qquad
u_y = \tfrac{1}{2}\gamma_{xy}\,x + \varepsilon_{yy}\,y + \tilde{u}_y
$$

```python
sol_tilde = np.array(sol_list[0])           # (num_nodes, 2)
x, y = mesh.points[:, 0], mesh.points[:, 1]

u_full = sol_tilde.copy()
u_full[:, 0] += eps_macro_voigt[0] * x + 0.5 * eps_macro_voigt[2] * y
u_full[:, 1] += eps_macro_voigt[1] * y + 0.5 * eps_macro_voigt[2] * x
```

Note: stresses and strains computed from `sol_tilde` alone are already correct, because
`get_tensor_map` adds $\pmb{\varepsilon}^{\text{mac}}$ back at every quadrature point.
The linear ramp only matters if you need the deformed geometry.

---

## 8. Extension to nonlinear materials and finite strain

### Nonlinear materials at small strain

The decomposition $\mathbf{u} = \pmb{\varepsilon}^{\text{mac}} \cdot \mathbf{x} + \tilde{\mathbf{u}}$ and the P_mat
structure are **unchanged**. Only `get_tensor_map` changes — instead of a linear
$\pmb{\sigma}(\pmb{\varepsilon})$, one provides a nonlinear constitutive law (e.g. damage,
hyperelasticity in small-strain form, or a return-mapping algorithm for plasticity).
jax-fem already uses Newton-Raphson internally.

For path-dependent materials (plasticity, damage) **incremental loading** is needed:
apply $\pmb{\varepsilon}^{\text{mac}}$ in steps and carry internal variables (plastic strain,
back-stress, damage variable) at each quadrature point across steps.

### Finite strain

The framework generalises, but the kinematic decomposition changes. Instead of a
strain tensor, one prescribes a macroscopic deformation gradient $\mathbf{F}^{\text{mac}}$:

$$
\mathbf{F}(\mathbf{X}) = \mathbf{F}^{\text{mac}} + \nabla_{\mathbf{X}}\tilde{\mathbf{u}}(\mathbf{X})
$$

The total displacement is then:

$$
\mathbf{u}(\mathbf{X}) = \underbrace{(\mathbf{F}^{\text{mac}} - \mathbf{I}) \cdot \mathbf{X}}_{\text{linear ramp}} + \tilde{\mathbf{u}}(\mathbf{X})
$$

**What stays the same:**

- P_mat construction and corner pinning — periodicity is enforced on the reference mesh, identical to the small-strain case.
- The closure trick — $\mathbf{F}^{\text{mac}}$ is captured in `get_tensor_map` exactly as `eps_macro` is today.

**What changes:**

- The constitutive law computes a 1st Piola-Kirchhoff stress $\mathbf{P}(\mathbf{F})$ from a hyperelastic potential instead of Cauchy stress from strain.
- The prescribed loading is $\mathbf{F}^{\text{mac}}$ (4 independent components in 2D) rather than $\pmb{\varepsilon}^{\text{mac}}$ (3 independent components).
- Incremental loading is almost always necessary.

A Neo-Hookean `get_tensor_map` would look like:

```python
def get_tensor_map(self):
    def stress(u_grad, E, nu):
        F = F_mac + jnp.eye(2) + u_grad        # total deformation gradient
        J = jnp.linalg.det(F)
        mu    = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        Finv  = jnp.linalg.inv(F)
        P     = mu * F + (lmbda * jnp.log(J) - mu) * Finv.T  # 1st PK stress
        return P
    return stress
```

### Summary

| | Small strain, linear | Small strain, nonlinear | Finite strain |
|---|---|---|---|
| Kinematic decomposition | additive strain | additive strain | additive in $\mathbf{F}$ |
| P_mat / periodic BC | unchanged | unchanged | unchanged |
| Closure trick | yes | yes | yes |
| Constitutive law | linear | nonlinear, same interface | hyperelastic / finite-strain plasticity |
| Incremental loading | no | sometimes | almost always |
| Newton-Raphson | not needed | needed | needed |
