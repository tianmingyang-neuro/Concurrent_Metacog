import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

# Truncated 'copper' colormap (skip the near-black tail) used for UR-trial traces
COPPER_TRUNCATED = LinearSegmentedColormap.from_list(
    'copper_truncated',
    colormaps['copper'](np.linspace(0.3, 1.0, 256))
)


class FixedOrderFormatter(ScalarFormatter):
    """ScalarFormatter pinned to a fixed order of magnitude."""

    def __init__(self, order_of_magnitude=0, useMathText=True):
        super().__init__(useMathText=useMathText)
        self._order_of_magnitude = order_of_magnitude

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._order_of_magnitude


def getAlignedMeanFr(tmpdf, frn_col, align_col, term='dircohc'):
    """Align per-trial firing-rate traces on an event and average within groups.

    Parameters
    ----------
    tmpdf : pd.DataFrame
        Trial table with a firing-rate column and a grouping column.
    frn_col : str
        Column of per-trial 1D firing-rate arrays (one neuron, NaN-padded).
    align_col : str
        Column with the per-trial alignment index (e.g. 'stim_idx').
    term : str
        Grouping column; one trace per group value is produced.

    Returns
    -------
    dict
        Maps each group value to {'time_axis', 'mean_fr', 'sem_fr'}.
        time_axis is in ms (bins are env steps; dt = 50 ms in the shipped data),
        with 0 at the alignment event.
    """
    result = {}
    for group_val in np.sort(tmpdf[term].unique()):
        group_data = tmpdf[tmpdf[term] == group_val]
        if len(group_data) == 0:
            continue

        all_fr = np.stack(group_data[frn_col].values)
        align_idxs = group_data[align_col].values.astype(float)

        if align_col == 'ch_idx':
            valid_mask = ~np.isnan(align_idxs)
        else:
            valid_mask = ~np.isnan(align_idxs) & (align_idxs < all_fr.shape[1])
        if not np.any(valid_mask):
            continue

        valid_fr = all_fr[valid_mask]
        valid_align_idxs = align_idxs[valid_mask].astype(int)

        # Place each trial in a common NaN-padded matrix aligned on the event
        max_pre_fix = np.max(valid_align_idxs)
        max_post_fix = np.max([len(fr) - idx for fr, idx in zip(valid_fr, valid_align_idxs)])
        aligned_fr_matrix = np.full((len(valid_fr), max_pre_fix + max_post_fix), np.nan)

        for k, (fr, fix_idx) in enumerate(zip(valid_fr, valid_align_idxs)):
            pre_idx_start = max_pre_fix - fix_idx
            aligned_fr_matrix[k, pre_idx_start:pre_idx_start + len(fr)] = fr

        mean_fr = np.nanmean(aligned_fr_matrix, axis=0)
        sem_fr = np.nanstd(aligned_fr_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(aligned_fr_matrix), axis=0))

        time_axis = np.arange(-max_pre_fix + 1, max_post_fix + 1) * 50
        result[group_val] = {
            'time_axis': time_axis,
            'mean_fr': mean_fr,
            'sem_fr': sem_fr,
        }
    return result


def selectStimExcitedUnits(df, fr_col='fr', stim_idx_col='stim_idx',
                           min_pre_window=1, min_post_window=1,
                           return_means=False):
    """
    Select units whose mean firing rate after stimulus onset is larger than
    before stimulus onset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing trial-level data. Must include columns for firing
        rates (`fr_col`) and stimulus onset indices (`stim_idx_col`).
    fr_col : str, default 'fr'
        Column name for firing rate arrays. Each entry should have shape
        (time_steps, n_units).
    stim_idx_col : str, default 'stim_idx'
        Column name for stimulus onset index per trial.
    min_pre_window : int, default 1
        Minimum number of time steps before stim_idx to include a trial.
    min_post_window : int, default 1
        Minimum number of time steps after stim_idx to include a trial.
    return_means : bool, default False
        If True, also return the before/after mean arrays.

    Returns
    -------
    excited_units : np.ndarray
        Array of unit indices where mean(post-stim) > mean(pre-stim).
    pre_means : np.ndarray, optional
        Mean firing rate before stim_idx for each unit (returned if
        `return_means=True`).
    post_means : np.ndarray, optional
        Mean firing rate after stim_idx for each unit (returned if
        `return_means=True`).
    """
    pre_means_all = []
    post_means_all = []

    for _, row in df.iterrows():
        fr = row[fr_col]
        stim_idx = row[stim_idx_col]

        # Skip trials with invalid stim_idx
        if stim_idx <= 0 or stim_idx >= fr.shape[0]:
            continue

        n_pre = stim_idx
        n_post = fr.shape[0] - stim_idx

        # Skip trials that don't meet minimum window requirements
        if n_pre < min_pre_window or n_post < min_post_window:
            continue

        pre_fr = fr[:stim_idx, :]
        post_fr = fr[stim_idx:, :]

        pre_means_all.append(np.nanmean(pre_fr, axis=0))
        post_means_all.append(np.nanmean(post_fr, axis=0))

    if len(pre_means_all) == 0:
        raise ValueError("No valid trials found for analysis.")

    pre_means = np.nanmean(np.stack(pre_means_all), axis=0)
    post_means = np.nanmean(np.stack(post_means_all), axis=0)

    excited_units = np.where(post_means > pre_means)[0]

    if return_means:
        return excited_units, pre_means, post_means
    return excited_units
