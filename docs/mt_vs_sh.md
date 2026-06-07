# Minkowski Tensors vs. Spherical Harmonics: A Pros/Cons Discussion

This note compares Minkowski tensors (MT) and spherical harmonic (SH) expansions
as sources of rotation-equivariant shape descriptors for 3D biological objects.
The aim is to identify where the two approaches genuinely differ and where they
are the same construction in different clothing.

## Shared substrate: the CG machinery is downstream of either

Both pipelines produce streams of SO(3) irreducible representations, and both
feed those irreps through identical Clebsch-Gordan contraction to produce
rotation-invariant scalars.
The symmetry framework, the irrep structure, and the contraction rules are
shared.
The interesting question is therefore not "which symmetry framework" but
"what gets fed in upstream of the contraction."

## Where the real divergence lives

Spherical harmonics are a *basis* on $S^2$.
The shape literature uses this basis in three distinct ways, and the choice
matters for what follows:

- **Radial function expansion.**
  $r(\theta,\phi) = \sum_{l,m} f_{l,m} Y_l^m$, where $r$ is the distance from a
  chosen center (typically the centroid) to the surface along the direction
  $(\theta,\phi)$.
  This is the default in `aics-shparam`: the package computes
  $r = \sqrt{x^2 + y^2 + z^2}$ at each mesh vertex relative to the centroid,
  interpolates onto a regular angular grid, and expands the resulting single
  scalar field with `pyshtools.expand.SHExpandDH`
  (see `shparam.py:_get_shcoeffs_from_mesh_coords`).
  Defined only for star-shaped surfaces about the chosen center.
  This is what the rest of this note means by "SH" unless stated otherwise.
- **Gauss map / extended Gaussian image.**
  Expand the area-weighted outward-normal density on $S^2$.
  Topology-independent but discards position information entirely.
- **Coordinate parametrization.**
  Continuously map the surface to $S^2$ via a non-trivial parametrization
  (Brechbühler-style, CALD, heat-equation-based), then expand each Cartesian
  coordinate as a separate scalar field on the sphere:
  $x(\theta,\phi) = \sum_{l,m} a^x_{l,m} Y_l^m$, and similarly for $y$ and $z$.
  Reportedly used in SPHARM-PDM and SPHARM-MAT, but this has not been verified
  against either package's source in writing this note and should be checked
  before being cited.
  Requires genus zero and a single connected component but tolerates
  non-star-shaped (e.g.\ invaginated) genus-zero surfaces.

In every variant, each $l$-band of coefficients is an irrep of SO(3) by
construction.

Minkowski tensors are a *family of integral-geometric tensors*: a canonical set
of surface integrals (position-weighted, area-weighted, mean-curvature-weighted,
Gaussian-curvature-weighted) that package geometric content into
mixed-rank irreducible components.

Both pipelines yield the same downstream object, a stream of SO(3) irreps.
They differ in what those irreps *describe*.

## What Minkowski tensors do well

**Canonical multi-field structure by construction.**

The $W^0$, $W^1$, $W^2$, $W^3$ families arise automatically from the integral
definition.
You do not choose which fields to project; you compute a fixed canonical set.

**Completeness theorem (on convex bodies).**

The classical Hadwiger-Alesker results identify the Minkowski tensors as
*the* continuous, motion-equivariant tensor valuations on the space of convex
bodies, with a known basis (Hadwiger 1957 for the scalar intrinsic volumes;
Alesker 1999a for the rotation-invariant scalar generalization and 1999b for
the tensor-valued extension; Hug, Schneider & Schuster 2008 for the
integral-geometric framework specific to Minkowski tensors).
There is no analogous a priori completeness statement for SH applied to "the
right set of projected fields"; one chooses fields ad hoc.

The convexity restriction is real and worth flagging: a generic non-convex
biological mesh (deep invaginations, handles, lumens) lies outside the cone of
convex bodies, outside the convex ring (Schneider 2013), and typically outside
Federer's class of sets of positive reach (Federer 1959).
The Minkowski tensors remain *computable* on such a mesh as surface integrals,
and they are used in this regime as a matter of integral-geometric tradition,
but no theorem of comparable strength to Hadwiger / Alesker guarantees that
they form a complete basis for continuous motion-equivariant valuations on
non-convex closed orientable surfaces.
The "MT is the canonical field set" advantage is therefore strict on convex
bodies and weaker but still defensible on non-convex ones.

**Defensible default for non-convex shapes.**

The lack of a formal extension does not change which framework is the right
starting point.
No competing equivariant-valuation framework has comparable theoretical
backing on either convex or non-convex bodies, and the MT construction
extends to non-convex closed orientable surfaces by continuity of the same
surface-integral definition.
The pragmatic case rests on three legs:
(i) principled extension from the convex case via the same definition,
inheriting the construction even where the uniqueness proof does not;
(ii) the absence in practice of an obviously-missing equivariant valuation
that the existing MT family fails to capture;
(iii) empirical performance in shape-classification and morphometry
benchmarks on biological data.
The third leg is where this line of work contributes: showing that the MT
framework, with its full cross-tensor invariant family, describes and
discriminates biological shapes at least as well as competing descriptors
where direct comparison is possible.

**One surface integral, no per-field projection.**

The whole tensor family is derived from a single surface integral with different
integrands.
SH equivalents require evaluating each field of interest pointwise and
performing a separate projection per field.

**Topology-agnostic for closed orientable meshes.**

Surface integrals are defined for any closed orientable triangulated surface
regardless of genus, convexity, star-shapedness, or connected-component count.
The SH variants each impose preconditions the integral framework does not:
the radial variant (used by `aics-shparam`) requires star-shapedness about
the chosen center, so a single invagination, budding event, or concavity
breaks the $r(\theta,\phi)$ representation; the coordinate parametrization
relaxes this to genus zero (no handles or internal lumens) and a single
connected component (the parametrization is a bijection to $S^2$, so
multi-object scenes break it).
MT integrals are additive over connected components, so a clustered organoid
or a field of cells can be described in one pass without per-object
segmentation and parametrization.
(Open meshes and non-manifold edges from segmentation noise remain a separate
practical concern that the framework does not magically solve.)

## What spherical harmonics do well

**Cheap angular resolution.**

Raising $L$ adds basis functions linearly.
Matching $l=12$ angular detail with Minkowski tensors requires rank-12 tensors
and increasingly fiddly trace removal during harmonic projection.

**Clean, continuously refinable scale parameter.**

$L$ is a single dial.
The MT analog (which ranks and which $W^\alpha$ families to include) is more
discrete and less obviously orderable.

**Compactness when one scalar field is the right descriptor.**

For nearly spherical cells where deviations from a sphere are the physical
signal, the radial SH expansion at moderate $L$ is hard to beat: one scalar
field, one truncation parameter, complete reconstruction of the radial profile.
The coordinate-parametrization variant scales the same way for non-star-shaped
genus-zero surfaces, at the cost of an explicit sphere parametrization step.

## A common over-claim, and the right version of it

A natural-sounding claim is that cross-field couplings (invariants that mix
position, normal, and curvature information) are uniquely available to MT and
have no SH analog.
For the radial-function SH used by `aics-shparam`, this is correct as stated:
the descriptor is a single scalar field $r(\theta,\phi)$, and all of its
invariants (power spectrum, bispectrum) live entirely within that one field's
frequency content.
Cross-channel CG contractions have nothing else to couple to.

Representation-theoretically, however, nothing forces SH to stay single-field.
One could project *additional* geometric quantities (outward normal density,
mean curvature density, Gaussian curvature density) onto $Y_l^m$ alongside the
radial expansion and contract across those channels via the same CG machinery.
The expressive class would then match MT.
What such a multi-field SH descriptor would lack is not representational power
but integral-geometric grounding: it would commit ad hoc to a list of
hand-picked fields with no completeness guarantee.

MT's field set, by contrast, is canonical: position, normal, mean curvature,
and Gaussian curvature emerge from a single surface integral with different
integrands, and the resulting tensor family is backed by a completeness
theorem on convex bodies (Hadwiger 1957; Alesker 1999b).
The advantage MT has over the default `aics-shparam` workflow is therefore
twofold: at the level of practice, the multi-field structure is already there
rather than something the user has to build; at the level of theory, the
field set is the right one rather than a chosen one.

## Two claims worth resisting in writing

**Interpretability beyond degree 1.**

Degree-1 invariants (traces: volume, area, mean curvature, Euler characteristic)
have direct geometric readings in either framework.
Degree-2 and degree-3 contractions in MT (bilinear and trilinear cross-tensor
invariants) are about as opaque as SH bispectrum components.
The "MT is interpretable" advantage applies to traces and named rank-2 irreps,
not to higher-degree contractions.

**Specific feature-count comparisons.**

Statements of the form "MT at rank 0-4 gives $N$ features, matching that with
SH requires $L \approx L_*$" are easy to write and hard to derive defensibly.
The "comparable" criterion is rarely the same on both sides (count? rank? signal
energy?), and the resulting cross-walk numbers tend to be wrong.
Better to compare at the level of structure than at the level of integer
feature counts.

## How to choose between them

The choice is not "which symmetry framework" but "which upstream geometric
content do you want to commit to representing."
MT commits to the canonical integral-geometric family at modest angular
resolution.
SH commits to a single (or deliberately chosen) field at arbitrary angular
resolution.

For non-star-shaped morphologies (invaginations, buds, concavities),
genus-$> 0$ surfaces (lumens, handles), multi-object scenes, or shapes where
multiple geometric channels (curvature alongside extent, for example) are
jointly diagnostic, MT is the natural starting point.
For star-shaped single-object morphologies where fine angular texture in the
radial profile is the discriminator, the `aics-shparam` radial expansion is
the natural starting point.
For non-star-shaped genus-zero single-object surfaces, a coordinate
parametrization SH workflow (SPHARM-PDM, SPHARM-MAT) is an intermediate
option that relaxes the star-shape requirement at the cost of a non-trivial
parametrization step.
For a hybrid pipeline, the SH coefficients of a chosen field can be converted
into the irrep components of the corresponding rank-$l$ Minkowski tensor and
fed into the same downstream invariant construction, but this unifies only one
of MT's four canonical channels and is therefore a supplement to MT rather than
a replacement.

## References

Alesker, S. (1999a).
Continuous rotation invariant valuations on convex sets.
*Annals of Mathematics* 149(3), 977-1005.
doi:[10.2307/121078](https://doi.org/10.2307/121078)

Alesker, S. (1999b).
Description of continuous isometry covariant valuations on convex sets.
*Geometriae Dedicata* 74(3), 241-248.
doi:[10.1023/A:1005035232264](https://doi.org/10.1023/A:1005035232264)

Federer, H. (1959).
Curvature measures.
*Transactions of the American Mathematical Society* 93(3), 418-491.
doi:[10.1090/S0002-9947-1959-0110078-1](https://doi.org/10.1090/S0002-9947-1959-0110078-1)

Hadwiger, H. (1957).
*Vorlesungen über Inhalt, Oberfläche und Isoperimetrie*.
Springer.
doi:[10.1007/978-3-642-94702-5](https://doi.org/10.1007/978-3-642-94702-5)

Hug, D., Schneider, R., & Schuster, R. (2008).
Integral geometry of tensor valuations.
*Advances in Applied Mathematics* 41(4), 482-509.
doi:[10.1016/j.aam.2008.04.001](https://doi.org/10.1016/j.aam.2008.04.001)

Schneider, R. (2013).
*Convex Bodies: The Brunn-Minkowski Theory* (2nd expanded ed.).
Cambridge University Press.
doi:[10.1017/CBO9781139003858](https://doi.org/10.1017/CBO9781139003858)
