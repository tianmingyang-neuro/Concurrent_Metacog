# Concurrent uncertainty response project: standalone package

Author: LI Xiaodong @CEBSIT 

Organized date: 20261106 with the Kimi K3-256k help.

This package accompanies the paper on uncertainty coding in a recurrent neural network trained with proximal policy optimization (PPO). It contains a vanilla RNN actor-critic, trained with PPO, that performs the CUE and CUEP variants of a two-stage random-dot-motion task:

- **CUE** (concurrent uncertainty estimation): in stage 1 the agent fixates, commits to a motion direction (perceptual decision, PD), or emits an uncertainty response (UR) that triggers stage 2 — a high-coherence stimulus in which the direction choice is easy.
- **CUEP**: CUE with a passive stage 1. The UR timing is imposed externally (by the CUE-trained network acting as a passive agent) and signaled to the CUEP-trained network through a dedicated observation channel; the network only decides how to react to it.

The package is inference only: no training is run, and the trained stage-4 weights are already included under `saved_params/` (stage 4 is the final stage of the training curriculum; earlier stages are not needed for reproduction). Running the three notebooks regenerates the simulated data reported in the paper and reproduces both the behavioral figures and the single-neuron figures.

| Notebook | Produces | Paper figure |
|---|---|---|
| `1_generate_data.ipynb` | Simulated datasets in `data/` | (input to all figures below) |
| `2_behavior_analysis.ipynb` | Psychometric / RT / UR-proportion panels | Supplementary Fig. 4 (CUE: a-c, CUEP: d-f) |
| `3_neuron_ur_all_analysis.ipynb` | Single-neuron PSTH grids | Extended Fig. 5a (CUE), 5b (CUEP) |

## Quick start

Two steps:

1. **Generate the simulated data.** Run `1_generate_data.ipynb`. It loads the trained stage-4 weights from `saved_params/`, runs the networks on the CUE and CUEP tasks, and writes the simulated datasets to `data/` (`CUE_part*.pkl`, `CUEP_part*.pkl`, `CUEP_part*_never_ur.pkl` — the last is the never-UR control, labeled "No Ext." in the figures). This step is required — `data/` starts empty.
2. **View the results.** Run `2_behavior_analysis.ipynb` for the behavioral figures and `3_neuron_ur_all_analysis.ipynb` for the single-neuron figures. Both read the datasets from `data/` and render the figures inline. The two analysis notebooks are independent of each other; either can be run right after step 1.

For headless execution, run all three notebooks in sequence from the package root:

```bash
jupyter nbconvert --to notebook --execute *.ipynb
```

## Environment setup

Python 3.10 or newer is required. From the package root:

**With uv (recommended):**

```bash
uv venv .venv --python 3.13
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**With pip:**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`neurogym==2.2.0` is pinned because the tasks were developed against that version. The remaining floors (`numpy>=1.26`, `pandas>=2.0`, etc.) match the versions this package was tested with (torch 2.10, numpy 2.4, pandas 2.3, matplotlib 3.9, seaborn 0.13, statsmodels 0.14, tqdm 4.67).

## Runtime expectations

Measured on an Apple M3 Ultra (256 GB RAM, macOS 27, CPU-only): the default scale (`N_PARTS=2`, `N_TRIALS=10000`) takes about **9 minutes** for the full data-generation sequence. CUE runs at ~330 trials/s; the CUEP variants run at ~90-110 trials/s because of the additional passive-agent forward pass.

The paper generated 30 parts per condition and used the first 8 for the main analyses. Set `N_PARTS=8` in `1_generate_data.ipynb` to match that analysis scale (~**35 minutes**); the default `N_PARTS=2` is a quick end-to-end check. Notebooks 2 and 3 then run over the generated dataset, and their own runtime grows accordingly.

## Reproducibility

Network-side randomness (RNN noise, action sampling) is seeded per part (`SEED_BASE + i`). The task environment's randomness — trial conditions, period timing, and stimulus observation noise — is drawn from neurogym's unseeded RNG, so rerunning `1_generate_data.ipynb` produces a new, statistically equivalent dataset rather than a bitwise-identical one. All reported results are aggregate statistics over tens of thousands of trials and are unaffected by this run-to-run variability; likewise, exact numeric agreement is not guaranteed across library versions, but the qualitative and aggregate results should hold.

## Package layout

- `1_generate_data.ipynb`, `2_behavior_analysis.ipynb`, `3_neuron_ur_all_analysis.ipynb` — the three reproduction notebooks (see table above).
- `data_load.py` — rollout drivers (`DataCUE`, `DataCUEP`) that load the trained weights and run the networks, returning one per-trial DataFrame per condition.
- `envs/envs_uncertainty.py` — the CUE and CUEP neurogym environments.
- `formal_model/` — the RNN actor-critic (`ActorCritic`), the RNN cell (`RNNcells`), and the inference-only PPO agent (`PPO`).
- `config/hyperparams.py` — network hyperparameters (the PPO training hyperparameters included there are unused at inference).
- `saved_params/` — trained stage-4 weights (`4_trained_model.pth`) and param files. The `4_param.pkl` files also contain training-time metadata (e.g. learning rate, stopping criteria); only `dt`, `LR` and `rewards` are read at inference.
- `neurons_dict.npy` + `utils_type_neuron.py` — frozen neuron selections (see below).
- `utils_neuron_analysis.py` — PSTH alignment/averaging utilities used by notebook 3.

## Note on `neurons_dict.npy`

`neurons_dict.npy` is a frozen neuron-selection file, loaded by `utils_type_neuron.py` and used by `3_neuron_ur_all_analysis.ipynb` to choose which neurons to plot. It was produced by an upstream change-point (dp/dx) analysis on the analysis-scale simulated data (8 parts × 10000 trials per condition): `CUE_UR_NEURONS` / `CUEP_UR_NEURONS` are the neurons whose mean dp/dx along the uncertainty-response dimension is positive; the `PDL` / `PDR` entries are the analogous selections for the two perceptual-decision dimensions; and the `*_SE` entries come from an independent stimulus-excitability screen (units with significantly elevated firing after stimulus onset). That selection analysis is not part of this package, so the file ships as a fixed input — like the trained weights — rather than being regenerated here.

## License

This package is released under the Creative Commons Attribution 4.0 International license (CC-BY-4.0). See the `LICENSE` file for the full legal text.

## Figures and outputs

Figures render inline in the notebooks. They are not written to disk, so view or export them from the notebook interface if you need image files. The only files written by the notebooks are the simulated datasets under `data/`.


