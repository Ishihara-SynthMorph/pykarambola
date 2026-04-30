# Minkowski Tensor Reference

This file documents every quantity returned by pykarambola: standard mathematical
notation, analytical integral definition, exact discrete formula on a triangulated
mesh, pykarambola key, and physical interpretation.

**Mesh notation used throughout:**

| Symbol | Meaning |
|--------|---------|
| $K$ | Body (3D region) |
| $\partial K$ | Surface (triangulated mesh) |
| $f$ | Triangular face with vertices $v_1, v_2, v_3$ |
| $e$ | Mesh edge shared by two adjacent faces, with endpoints $e_1, e_2$ and midpoint $c_e = (e_1+e_2)/2$ |
| $v$ | Mesh vertex |
| $A_f$ | Area of face $f$: $\tfrac{1}{2}\lVert(v_2-v_1)\times(v_3-v_1)\rVert$ |
| $c_f$ | Centroid of face $f$: $(v_1 + v_2 + v_3)/3$ |
| $\hat{n}_f$ | Outward unit normal of face $f$ |
| $\ell_e$ | Length of edge $e$ |
| $\alpha_e$ | Exterior dihedral angle at edge $e$ (angle between outward normals of the two adjacent faces) |
| $\delta_v$ | Angle deficit at vertex $v$: $2\pi - \sum_{f \ni v} \theta_{fv}$ |
| $\theta_{fv}$ | Interior angle of face $f$ at vertex $v$ |
| $\chi$ | Euler characteristic: $\chi = 2 - 2g$, where $g$ is the genus (number of handles) |

---

## Computational structure: face / edge / vertex decomposition

Every Minkowski tensor $W_\nu^{r,s}$ on a triangulated surface follows a unified
discretization pattern. The index $\nu$ determines which mesh element type carries
the contribution:

| $\nu$ | Element | Weight per element | Position factor ($r$) | Normal factor ($s$) |
|---|---|---|---|---|
| 0 | tetrahedra (body) | signed tet volume | $x_i x_j \ldots$ via divergence theorem | — |
| 1 | **faces** | $\tfrac{1}{3}A_f$ | moment of $c_f$ over face | $\hat{n}_f^{\otimes s}$ |
| 2 | **edges** | $\tfrac{1}{6}\ell_e\,\alpha_e$ | moment of position along edge | edge-normal tensor |
| 3 | **vertices** | $\tfrac{1}{3}\delta_v$ | $r_v^{\otimes r}$ | — (Gauss–Bonnet: no normal) |

The $1/3$ and $1/6$ prefactors are the binomial normalizations $1/\binom{3,\nu}$ from the Steiner
formula (see demo notebook §1). Using this pattern, the individual formulas below can
be read as: *"sum the appropriate weight over the appropriate element type, accumulating
position and/or normal factors at the appropriate rank."*

For $\nu=1$ (face sums), the position moment over a triangle of rank $r=1$ is $c_f$ and
of rank $r=2$ is the surface second moment $\frac{1}{A_f}\int_f x_i x_j\, dA$.
For $\nu=2$ (edge sums), the position moment of rank $r=1$ is the edge midpoint $c_e$
and of rank $r=2$ is $\int_0^1 x(t)\otimes x(t)\, dt = \tfrac{1}{3}(e_1{\otimes}e_1 + \tfrac{1}{2}(e_1{\otimes}e_2+e_2{\otimes}e_1) + e_2{\otimes}e_2)$.
For the normal factor at edges ($s=2$, $\nu=2$), the curvature tensor splits into a
parallel component $\bar{n}_e \otimes \bar{n}_e$ (average face normal) and a
perpendicular component $n_{\perp,e} \otimes n_{\perp,e}$ with coefficients
$(\alpha_e \pm \sin\alpha_e)/4$ (see w202).

---

## Scalars — rank-0 tensors $W_\nu^{0,0}$

### w000 — $W_0^{0,0}$ — Volume

| | |
|---|---|
| **pykarambola key** | `w000` |
| **Analytical** | $W_0 = \int_K dV$ |
| **Computational** | $w_{000} = \dfrac{1}{3}\displaystyle\sum_f (\hat{n}_f \cdot c_f)\, A_f$ (divergence theorem: $\nabla\cdot(\mathbf{x}/3) = 1$) |
| **Interpretation** | Volume of $K$. Negative value indicates inverted face winding; true volume $= \lvert w_{000}\rvert$. |

### w100 — $W_1^{0,0}$ — Surface Area / 3

| | |
|---|---|
| **pykarambola key** | `w100` |
| **Analytical** | $W_1 = \dfrac{1}{3}\int_{\partial K} dA$ |
| **Computational** | $w_{100} = \dfrac{1}{3}\displaystyle\sum_f A_f$ |
| **Interpretation** | One-third of the total surface area $A$. Recover area as $A = 3\,w_{100}$. |

### w200 — $W_2^{0,0}$ — Integrated Mean Curvature / 3

| | |
|---|---|
| **pykarambola key** | `w200` |
| **Analytical** | $W_2 = \dfrac{1}{3}\int_{\partial K} H\, dA$, where $H = (\kappa_1 + \kappa_2)/2$ is mean curvature |
| **Computational** | $w_{200} = \dfrac{1}{6}\displaystyle\sum_e \alpha_e\, \ell_e$ (each edge counted once; the code accumulates $\alpha_e \ell_e / 12$ per triangle per edge-slot, and each edge appears in exactly 2 triangles) |
| **Interpretation** | One-third of the integrated mean curvature $M$. Recover $M = 3\,w_{200}$. Sensitive to surface bending; more curved surfaces yield larger values. |

### w300 — $W_3^{0,0}$ — Euler Characteristic $\times\,2\pi/3$

| | |
|---|---|
| **pykarambola key** | `w300` |
| **Analytical** | $W_3 = \dfrac{1}{3}\displaystyle\int_{\partial K} K_G\, dA = \dfrac{2\pi}{3}\chi$, by the Gauss–Bonnet theorem $\bigl(\int_{\partial K} K_G\, dA = 2\pi\chi\bigr)$ |
| **Computational** | $w_{300} = \dfrac{1}{3}\displaystyle\sum_v \delta_v$ (each vertex's angle deficit, distributed to triangles by angle fraction) |
| **Interpretation** | Encodes topology via $\chi = 2-2g$: sphere ($g=0$) $\Rightarrow w_{300}=4\pi/3$; torus ($g=1$) $\Rightarrow w_{300}=0$; each additional handle subtracts $4\pi/3$. Recover $\chi = 3\,w_{300}/(2\pi)$. Particularly informative in biological imaging where topology (vesicle vs. ring organelle) is a meaningful descriptor. |

---

## Vectors — rank-1 tensors $W_\nu^{1,0}$

Position-weighted integrals. Each vector yields a physically meaningful centroid only
when divided by the corresponding scalar.

### w010 — $W_0^{1,0}$ — Volume moment

| | |
|---|---|
| **pykarambola key** | `w010` |
| **Analytical** | $(W_0^{1,0})_i = \int_K x_i\, dV$ |
| **Computational** | Divergence theorem: $(w_{010})_i = \tfrac{1}{2}\int_{\partial K} x_i^2\, n_i\, dA$ (no Einstein sum on $i$), evaluated face-by-face per triangle. |
| **Interpretation** | $w_{010} / w_{000}$ = centroid (center of mass) of the solid body. Zero for a body centered at the origin. |

Discrete triangle formula (from `pykarambola/minkowski.py`), with $\mathbf{v}=c_2-c_1$, $\mathbf{w}=c_3-c_1$, indices cyclic mod 3:

$$
(w_{010})_i = \frac{1}{24}\sum_f (\mathbf{v}\times\mathbf{w})_{i\oplus 1} \cdot P_i, \quad
P_i = 2v_i v_{i\oplus 1} + v_{i\oplus 1}w_i + v_i w_{i\oplus 1} + 2w_i w_{i\oplus 1}
      + 4c_{1,i\oplus 1}(v_i+w_i) + 4c_{1,i}(3c_{1,i\oplus 1}+v_{i\oplus 1}+w_{i\oplus 1})
$$

### w110 — $W_1^{1,0}$ — Surface moment

| | |
|---|---|
| **pykarambola key** | `w110` |
| **Analytical** | $(W_1^{1,0})_i = \dfrac{1}{3}\int_{\partial K} x_i\, dA$ |
| **Computational** | $w_{110} = \dfrac{1}{3}\displaystyle\sum_f A_f\, c_f$ (face weight $\tfrac{1}{3}A_f$, position factor = face centroid $c_f$) |
| **Interpretation** | $w_{110} / w_{100}$ = area-weighted centroid of the surface. Differs from $w_{010}/w_{000}$ for hollow or non-uniform shells. |

### w210 — $W_2^{1,0}$ — Mean-curvature-weighted moment

| | |
|---|---|
| **pykarambola key** | `w210` |
| **Analytical** | $(W_2^{1,0})_i = \dfrac{1}{3}\int_{\partial K} H\, x_i\, dA$ |
| **Computational** | $w_{210} = \dfrac{1}{6}\displaystyle\sum_e \alpha_e\, \ell_e\, c_e$ (edge weight $\tfrac{1}{6}\ell_e\alpha_e$, position factor = edge midpoint $c_e$; code accumulates as $\tfrac{\alpha\ell}{24}(e_1+e_2)$ per edge-slot, doubled by the two adjacent triangles) |
| **Interpretation** | $w_{210} / w_{200}$ = mean-curvature-weighted centroid; highlights regions of high bending. |

### w310 — $W_3^{1,0}$ — Gaussian-curvature-weighted moment

| | |
|---|---|
| **pykarambola key** | `w310` |
| **Analytical** | $(W_3^{1,0})_i = \dfrac{1}{3}\int_{\partial K} K_G\, x_i\, dA$ |
| **Computational** | $w_{310} = \dfrac{1}{3}\displaystyle\sum_v \delta_v\, r_v$ (vertex weight $\tfrac{1}{3}\delta_v$, position factor = vertex position $r_v$) |
| **Interpretation** | $w_{310} / w_{300}$ = Gaussian-curvature-weighted centroid; concentrates weight at topological features (corners, saddle points). |

---

## Rank-2 tensors $W_\nu^{2,0}$ — position-weighted

Symmetric $3\times3$ matrices from integrating $x_i x_j$ with the same geometric
weights as the scalars. All return `{key}_eigvals`, `{key}_eigvecs`, and (with
`compute='all'`) `{key}_beta`, `{key}_trace`, `{key}_trace_ratio`.

### w020 — $W_0^{2,0}$ — Solid moment tensor

| | |
|---|---|
| **pykarambola key** | `w020` |
| **Analytical** | $(W_0^{2,0})_{ij} = \int_K x_i\, x_j\, dV$ |
| **Computational** | Divergence theorem: $(w_{020})_{ij} = \tfrac{1}{3}\int_{\partial K} x_i\, x_j\, x_k\, n_k\, dA$ (sum over $k$), evaluated face-by-face using the closed-form triangle integral. Diagonal entries use prefactor $1/60$, off-diagonal $1/120$, times $2A_f n_{f,k}$ for the appropriate normal component $k$. |
| **Interpretation** | Inertia-like tensor of the filled solid. Eigenvalues give principal shape extents. $\text{Tr}(w_{020})/w_{000}$ = mean squared distance from the reference point. `w020_beta` = anisotropy index ($0$ = rod-like, $1$ = isotropic). |

### w120 — $W_1^{2,0}$ — Hollow (surface) moment tensor

| | |
|---|---|
| **pykarambola key** | `w120` |
| **Analytical** | $(W_1^{2,0})_{ij} = \dfrac{1}{3}\int_{\partial K} x_i\, x_j\, dA$ |
| **Computational** | $w_{120,ij} = \displaystyle\sum_f \frac{A_f}{18}\Bigl(v_{1i}v_{1j} + v_{2i}v_{2j} + v_{3i}v_{3j} + \tfrac{1}{2}(v_{1i}v_{2j} + v_{2i}v_{3j} + v_{3i}v_{1j}) + \tfrac{1}{2}(v_{1j}v_{2i} + v_{2j}v_{3i} + v_{3j}v_{1i})\Bigr)$ (face weight $\tfrac{1}{3}A_f$, position factor = second moment of the triangle $= \tfrac{1}{6A_f}\int_f x_i x_j\, dA$) |
| **Interpretation** | Inertia tensor of the surface shell. `w120_beta` = anisotropy of the surface shape. $\text{Tr}(w_{120})/w_{100}$ = mean squared distance of the surface from the reference point. |

### w220 — $W_2^{2,0}$ — Wire (curvature-weighted) moment tensor

| | |
|---|---|
| **pykarambola key** | `w220` |
| **Analytical** | $(W_2^{2,0})_{ij} = \dfrac{1}{3}\int_{\partial K} H\, x_i\, x_j\, dA$ |
| **Computational** | $w_{220,ij} = \dfrac{1}{6}\displaystyle\sum_e \alpha_e\,\ell_e \cdot \tfrac{1}{3}\Bigl(e_{1i}e_{1j} + \tfrac{1}{2}(e_{1i}e_{2j}+e_{1j}e_{2i}) + e_{2i}e_{2j}\Bigr)$ (edge weight $\tfrac{1}{6}\ell_e\alpha_e$, position factor = second moment of the edge $\int_0^1 x(t)_i x(t)_j\, dt$; code accumulates as $\tfrac{\alpha\ell}{36}(\cdot)$ per edge-slot, doubled by two adjacent triangles) |
| **Interpretation** | Mean-curvature-weighted moment tensor. `w220_beta` = anisotropy of where bending is concentrated spatially. |

### w320 — $W_3^{2,0}$ — Vertex moment tensor

| | |
|---|---|
| **pykarambola key** | `w320` |
| **Analytical** | $(W_3^{2,0})_{ij} = \dfrac{1}{3}\int_{\partial K} K_G\, x_i\, x_j\, dA$ |
| **Computational** | $w_{320,ij} = \dfrac{1}{3}\displaystyle\sum_v \delta_v\, r_{vi}\, r_{vj}$ (vertex weight $\tfrac{1}{3}\delta_v$, position factor = $r_v \otimes r_v$) |
| **Interpretation** | Gaussian-curvature-weighted moment tensor; mass at topological vertices. `w320_beta` = anisotropy of topological structure. $\text{Tr}(w_{320})/w_{300}$ = mean squared distance of topological features from the reference point. |

---

## Rank-2 tensors $W_\nu^{0,2}$ — normal-weighted

Symmetric $3\times3$ matrices from integrating $n_i n_j$. Encode the distribution
of surface normal orientations. All return `{key}_eigvals`, `{key}_eigvecs`, and
(with `compute='all'`) `{key}_beta` and `{key}_trace`.
Note: `{key}_trace_ratio` is **not** defined for normal-weighted tensors (no natural paired scalar).

### w102 — $W_1^{0,2}$ — Normal distribution tensor

| | |
|---|---|
| **pykarambola key** | `w102` |
| **Analytical** | $(W_1^{0,2})_{ij} = \dfrac{1}{3}\int_{\partial K} n_i\, n_j\, dA$ |
| **Computational** | $w_{102,ij} = \dfrac{1}{3}\displaystyle\sum_f A_f\, \hat{n}_{f,i}\, \hat{n}_{f,j}$ (face weight $\tfrac{1}{3}A_f$, normal factor $\hat{n}_f \otimes \hat{n}_f$) |
| **Interpretation** | Distribution of surface normal orientations. For a sphere: $w_{102} = (A/9)\,I$ (isotropic). $\text{Tr}(w_{102}) = w_{100}$ exactly (since $\lvert\hat{n}\rvert^2=1$). `w102_beta` = anisotropy of normals ($0$ = all normals aligned, $1$ = isotropic). |

### w202 — $W_2^{0,2}$ — Curvature-weighted normal tensor

| | |
|---|---|
| **pykarambola key** | `w202` |
| **Analytical** | $(W_2^{0,2})_{ij} = \dfrac{1}{3}\int_{\partial K} H\, n_i\, n_j\, dA$ |
| **Computational** | $w_{202,ij} = \displaystyle\sum_e \left[\frac{\ell_e(\alpha_e + \sin\alpha_e)}{24}\,\bar{n}_{e,i}\bar{n}_{e,j} + \frac{\ell_e(\alpha_e - \sin\alpha_e)}{24}\,n_{\perp,e,i}\,n_{\perp,e,j}\right]$, where $\bar{n}_e = (n_{f_1}+n_{f_2})/\lvert n_{f_1}+n_{f_2}\rvert$ is the average unit normal at edge $e$ and $n_{\perp,e} = \hat{e}\times\bar{n}_e$. (The $\alpha \pm \sin\alpha$ split follows from the exact integral of $n \otimes n$ over the cylindrical patch at the edge.) |
| **Interpretation** | Curvature-weighted normal distribution; sensitive to both orientation and bending magnitude. `w202_beta` = anisotropy of curvature-weighted normals. |

---

## Higher-rank tensors (`compute='all'`)

### w103 — $W_1^{0,3}$ — Rank-3 normal tensor

| | |
|---|---|
| **pykarambola key** | `w103` |
| **Analytical** | $(W_1^{0,3})_{ijk} = \dfrac{1}{3}\int_{\partial K} n_i\, n_j\, n_k\, dA$ |
| **Computational** | $w_{103,ijk} = \dfrac{1}{3}\displaystyle\sum_f A_f\, \hat{n}_{f,i}\, \hat{n}_{f,j}\, \hat{n}_{f,k}$ (face weight $\tfrac{1}{3}A_f$, triple normal product; stored as $3\times3\times3$ array) |
| **Interpretation** | Third-order symmetric tensor of normal orientations. Vanishes for centrosymmetric shapes (every $\hat{n}$ paired with $-\hat{n}$). Non-zero components indicate a preferred normal handedness. |

### w104 — $W_1^{0,4}$ — Rank-4 normal tensor

| | |
|---|---|
| **pykarambola key** | `w104` |
| **Analytical** | $(W_1^{0,4})_{ijkl} = \dfrac{1}{3}\int_{\partial K} n_i\, n_j\, n_k\, n_l\, dA$ |
| **Computational** | $w_{104} = \dfrac{1}{3}\displaystyle\sum_f A_f\, \mathbf{t}_f \mathbf{t}_f^\top$ (face weight $\tfrac{1}{3}A_f$; stored as symmetric $6\times6$ matrix in Voigt notation with $\mathbf{t}_f = [n_x^2,\, n_y^2,\, n_z^2,\, \sqrt{2}n_yn_z,\, \sqrt{2}n_xn_z,\, \sqrt{2}n_xn_y]^\top$) |
| **Interpretation** | Fourth-order symmetric tensor of normal orientations; related to the Minkowski structure metrics (MSM) used in materials science to characterise crystallographic texture and fabric. |

---

## Derived scalars (`compute='all'`)

For every rank-2 tensor $W$ above, pykarambola additionally computes:

| Key suffix | Formula | Interpretation |
|---|---|---|
| `{name}_eigvals` | Eigenvalues $\lambda_1 \le \lambda_2 \le \lambda_3$ of $W$ | Principal magnitudes |
| `{name}_eigvecs` | Eigenvectors (columns) of $W$ | Principal axes |
| `{name}_beta` | $\lvert\lambda_1\rvert / \lvert\lambda_3\rvert$ | Anisotropy index: $0$ = maximally anisotropic, $1$ = isotropic |
| `{name}_trace` | $\text{Tr}(W) = \lambda_1 + \lambda_2 + \lambda_3$ | Additive under disjoint union (with shared reference frame `center=None`) |
| `{name}_trace_ratio` | $\text{Tr}(W) /$ scalar counterpart | e.g. $\text{Tr}(w_{020})/w_{000}$; defined only for the $W_\nu^{2,0}$ family |

> **Additivity.** The tensor matrix $W$, its trace, and all scalar functionals are additive
> for disjoint objects computed with a shared reference frame (`center=None`).
> Derived quantities — `_beta`, `_eigvals`, `_eigvecs`, `_trace_ratio` — are **not** additive
> and must be recomputed from the merged tensor.

---

## References

- Schröder-Turk, G. E. et al. *Minkowski Tensors of Anisotropic Spatial Structure.*
  New J. Phys. **15**, 083028 (2013). [doi:10.1088/1367-2630/15/8/083028](https://doi.org/10.1088/1367-2630/15/8/083028)
- Schaller, F. M., Kapfer, S. C., & Schröder-Turk, G. E.
  *karambola — 3D Minkowski Tensor Package* (v2.0).
  <https://github.com/morphometry/karambola>
