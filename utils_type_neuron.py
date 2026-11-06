"""Neuron selections used by 3_neuron_ur_all_analysis.ipynb.

The lists are loaded from the frozen ``neurons_dict.npy`` selection file;
see the README ("Note on neurons_dict.npy") for how they were produced.
"""
from pathlib import Path

import numpy as np

_selections = np.load(Path(__file__).resolve().parent / "neurons_dict.npy", allow_pickle=True).item()
CUE_UR_NEURONS = _selections["CUE_UR_NEURONS"]
CUEP_UR_NEURONS = _selections["CUEP_UR_NEURONS"]
CUE_PDL_NEURONS = _selections["CUE_PDL_NEURONS"]
CUEP_PDL_NEURONS = _selections["CUEP_PDL_NEURONS"]
CUE_PDR_NEURONS = _selections["CUE_PDR_NEURONS"]
CUEP_PDR_NEURONS = _selections["CUEP_PDR_NEURONS"]
CUE_UR_NEURONS_SE = _selections["CUE_UR_NEURONS_SE"]
CUEP_UR_NEURONS_SE = _selections["CUEP_UR_NEURONS_SE"]
