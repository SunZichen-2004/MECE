# BondNet

BondNet is a compact, readable prototype of an edge-centered (line-graph) equivariant
interatomic model. It follows the broad separation used by MACE projects—data handling,
geometric basis modules, interaction modules, models, scripts, and tests—while its code
and implementation are original. It deliberately avoids high-level equivariant network
layers. The only required runtime dependency is PyTorch.

## Mathematical map

For every directed edge from atom `i` to atom `j`, the code constructs

```text
displacement = position_j - position_i
distance = length(displacement)
direction = displacement / distance
```

Atomic state starts with a learned element embedding. The initial geometric state is

```text
radial_basis(distance) * real_spherical_harmonic(direction) * initial_element_pair_state
```

which is the requested `R(r_ij) Y_lm(r_ij) h_0` construction. The radial basis uses the
standard sinusoidal Bessel family with a smooth polynomial envelope, matching the default
design choice commonly used by MACE without copying its source.

Each edge receives a local frame. Its third axis is the directed bond; its first axis is
the projected first moment of the endpoint environments; its second axis completes a
right-handed frame. Under a proper rotation the frame rotates with the atoms. Under a
reflection, the second axis acquires the expected parity, so positive and negative
magnetic orders become real two-component O(2) blocks.

The neighbor directions around center atom `i` are evaluated in the `i -> j` edge frame and
summed. This is mathematically the same basis transformation as constructing global
atom-centered spherical coefficients and applying the matching real Wigner rotation;
direct evaluation is shorter, easier to audit, and avoids a Wigner-D dependency. The
neighbors of edge `i -> j` are the directed edges `i -> j_prime`, with `j_prime != j`,
shorter than `edge_cutoff_radius` (default: the atomic cutoff). `AtomicData` builds this
line-graph neighbor list once, before the model runs.

The real harmonic component order is

```text
m = 0, cosine(1), sine(1), cosine(2), sine(2), ...
```

Every nonzero absolute magnetic order is therefore a two-component block. Neighbor
states are transported into the center-edge frame with analytic real `D^l` matrices.
In that common frame, real SO(2) tensor products generate sum and difference orders;
repeated self products form explicit correlation orders `B^(1), ..., B^(nu_max)`.
The readout uses scalar blocks and norms of two-component blocks. Directed predictions
are averaged with their reverse edge and counted by one half, so every physical bond
contributes once. See `bondnet/note/bondnet_architecture.md` for formulas and shapes.

Forces are computed as the negative position gradient of total energy.

## Layout

- `bondnet/data`: directed neighborhood construction
- `bondnet/modules`: radial basis, real spherical harmonics, edge frames, scatter, and O(2) blocks
- `bondnet/models`: complete edge-centered model
- `scripts/train_toy.py`: minimal optimization example
- `scripts/download_3bpa.py`: reproducible public 3BPA download
- `scripts/train_3bpa.py`: energy/force smoke experiment and resource metrics
- `tests`: construction, cutoff, rotation, reflection, force, and overfit checks

## Run

The existing local environment can be used directly:

```bash
conda activate bondnet
cd /Users/zichen/Desktop/phd_first_year/bondnet
python -m pip install -e .
pytest -q
python scripts/train_toy.py
python -m pip install -e '.[experiment]'
python scripts/download_3bpa.py
python scripts/train_3bpa.py --train-size 16 --valid-size 8 --epochs 5
```

## Recorded 3BPA smoke result

`runs/3bpa_smoke/results.json` records a pipeline smoke test, not a converged
literature benchmark. Seed 7, CPU, 16 training structures, 8 validation
structures, 8 test-300K structures, 5 epochs, and 2037 parameters produced:

- validation: 16.01 meV/atom energy MAE and 588.59 meV/angstrom force MAE;
- test 300 K: 12.04 meV/atom energy MAE and 624.38 meV/angstrom force MAE;
- 1.03 seconds training, about 5.2 ms/configuration evaluation;
- 353.23 MiB peak process resident memory.

Apple MPS was available but CUDA was not; the recorded run used CPU. A meaningful
accuracy study needs more data, channels, epochs, and convergence monitoring.

## Scope and deliberate limitations

This is a research prototype rather than a production potential. It supports angular
momentum zero through two, dense all-pairs neighbor construction, and non-periodic
structures. The Cartesian fallback used for exactly degenerate local frames is stable but
cannot define a globally continuous equivariant gauge; symmetric environments require a
multi-frame or gauge-equivariant treatment in a production implementation. Periodic
boundaries, scalable neighbor lists, configuration files, and distributed training are
natural next steps. General edge-to-edge transport remains a small O(3) `D^l`
operation; the message and cluster products in the common target frame are SO(2).
