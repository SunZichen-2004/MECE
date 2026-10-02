
# BondNet: edge-frame SO(2) cluster expansion

## 1. Indices, frames, and shapes

Atoms are `i`, `j`, and `j_prime`. Every directed edge is `i -> j`; its reverse
is also stored. An edge pair is `(i -> j, i -> j_prime)` with `j_prime != j`.

`edge_rotation_matrices[edge]` has shape `[3, 3]`. Its rows are the local x, y,
z axes in the global frame, so it maps global Cartesian components to local
edge-frame components. The z axis is the directed bond. The x axis is the
projected first moment of the endpoint environments; y completes a right-handed
frame.

```text
states[l]                     [number_of_edges, channels, 2*l+1]
relative_rotation            [number_of_edge_pairs, 3, 3]
real_D_l                     [number_of_edge_pairs, 2*l+1, 2*l+1]
neighbor_geometry[l]         [number_of_edge_pairs, channels, 2*l+1]
```

The real harmonic order is `[m=0, cos(1), sin(1), cos(2), sin(2), ...]`.

## 2. Initial atom-centered expansion and edge state

The global atom-centered coefficients are

$$
A_{i,n\ell m}^{g}
=\sum_{j\in\mathcal N(i)}R_n(r_{ij})Y_{\ell m}(\widehat{\mathbf r}_{ij})c(Z_i,Z_j).
$$

For edge `i -> j`, let `F_ij` map global Cartesian components to its edge frame:

$$
h_{ij,n\ell m}^{(0)}
=\sum_{m'}D_{mm'}^\ell(F_{ij})A_{i,n\ell m'}^g.
$$

The code evaluates the exactly equivalent direct expression

$$
h_{ij,n\ell m}^{(0)}
=\sum_{j'\in\mathcal N(i)}R_n(r_{ij'})
Y_{\ell m}(F_{ij}\widehat{\mathbf r}_{ij'})c(Z_i,Z_{j'}).
$$

This avoids materializing global coefficients and a separate initial Wigner
rotation. `own_geometry` supplies `j'=j`; line-graph pairs supply `j'!=j`.

## 3. Relative-frame transport

For `(i -> j, i -> j_prime)`, Cartesian components transform by

$$
Q_{ij\leftarrow ij'}=F_{ij}F_{ij'}^T,
$$

and the neighbor state is transported by

$$
\widetilde h_{ij'}^{(t),\ell}
=D^\ell(Q_{ij\leftarrow ij'})h_{ij'}^{(t),\ell}.
$$

If the two bond axes differ, `Q` is a general SO(3), or O(3) under reflection,
frame change—not an SO(2) rotation. A multilayer state must therefore retain its
`l` label. An unlabeled collection of SO(2) orders cannot be transported between
arbitrary axes. BondNet uses analytic real `D^l` matrices for `l<=2`; this is the
small unavoidable full-frame operation. Products after transport occur in one
common edge frame and use SO(2) algebra.

## 4. SO(2) message product

The neighboring edge geometry, already in the center-edge frame, is

$$
\phi_{ij'}^{(ij),\ell m}=R_n(r_{ij'})
Y_{\ell m}(F_{ij}\widehat{\mathbf r}_{ij'})c(Z_i,Z_{j'}).
$$

The one-particle message is

$$
M_{ij}^{(t)}
=\frac{1}{|\mathcal N(i)\setminus j|}
\sum_{j'\ne j}
\left(T_{ij\leftarrow ij'}h_{ij'}^{(t)}\right)
\otimes_{SO(2)}\phi_{ij'}^{(ij)}.
$$

A positive order m is the real block `[a,b]`, representing `a+ib`. Orders p and
q produce both irreducible outputs

$$z_pz_q\in V_{p+q},\qquad z_p\overline{z_q}\in V_{|p-q|}.$$

The sum-order map is `(a,b) tensor (c,d) -> (ac-bd, ad+bc)`. The difference
cosine is `ac+bd`; its sine sign depends on whether `p>=q`. An `m=0` factor is
ordinary scalar multiplication. Learned channel mixing is shared by cosine and
sine components.

## 5. Explicit edge cluster expansion and body order

Define

$$B_{ij}^{(1)}=M_{ij},$$

and recursively

$$
B_{ij}^{(\nu+1)}=B_{ij}^{(\nu)}\otimes_{SO(2)}M_{ij},
\qquad 1\le\nu<\nu_{\max}.
$$

The update is

$$
h_{ij}^{(t+1)}=w_{\rm res}h_{ij}^{(t)}
+w_{\rm msg}\sum_{\nu=1}^{\nu_{\max}}W_\nu B_{ij}^{(\nu)}.
$$

Each factor introduces another summed neighboring edge, so `B^(nu)` is an
explicit order-nu density correlation around the center bond. Repeated neighbor
indices occur, as in density-trick ACE/MACE bases; correlation order is not a
claim that every term contains distinct atoms. The implementation truncates
orders above `maximum_angular_momentum` and uses channelwise products before
learned output mixing, which is compact but not the most general SO(2) basis.

## 6. Readout and edge reversal

`m=0` blocks are invariant. A positive-m block contributes

$$I_{ij,\ell m}=\sqrt{a_{ij,\ell m}^2+b_{ij,\ell m}^2+\epsilon}.$$

An MLP maps all invariants to a directed proposal `epsilon_tilde_ij`. The
physical bond energy is explicitly reversal symmetric:

$$
\epsilon_{\{i,j\}}=\frac12
(\widetilde\epsilon_{ij}+\widetilde\epsilon_{ji}),
\qquad
E=\frac12\sum_{i\to j}\epsilon_{\{i,j\}}.
$$

Thus `edge_energy` stores half the shared bond energy on each directed record.
Forces are exact derivatives, `F_i=-partial E/partial r_i`.

## 7. Current scope and caveats

The prototype is non-periodic, uses dense neighbor construction, supports
`l<=2`, and predicts energies and forces but not stress. The Cartesian fallback
for exactly degenerate edge gauges is stable but not globally continuous. A
production version needs periodic shifts, scalable neighbor lists, a multi-frame
or gauge-equivariant degeneracy treatment, batching, and fused SO(2) products.
