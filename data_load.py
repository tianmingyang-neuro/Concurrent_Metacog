"""Rollout drivers for the trained CUE / CUEP networks.

Inference only: this module loads the trained weights from ``saved_params/`` and
runs the networks in eval mode on the two tasks, returning one per-trial
DataFrame per condition. No training happens in this package.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from formal_model import PPOAgent, ActorCritic, Transition_activity
from envs.envs_uncertainty import CUEExpTwoStage, CUEExpTwoStagePassive

_PACKAGE_ROOT = Path(__file__).resolve().parent

DEFAULT_COHS = [2, 4, 6, 8, 13, 26, 52, 78]

# Termination signals emitted by the environments' ``_step`` (info['trig']).
_CORRECT_TRIGS = ('correct', 'ur_correct')
_WRONG_TRIGS = ('wrong', 'ur_wrong')
_ABORT_TRIGS = ('abort', 'ur_abort', 'time_expired', 'ur_time_expired')
_ALL_TRIGS = _CORRECT_TRIGS + _WRONG_TRIGS + _ABORT_TRIGS


class _DataBase:
    """Shared weight loading and DataFrame assembly for CUE and CUEP."""

    def __init__(self, hp, celltype='vanilla', actype='wang', ppotype='classic'):
        self.celltype = celltype
        self.actype = actype
        self.ppotype = ppotype

        self.model = ActorCritic(
            hp['INPUT_DIM'], hp['HIDDEN_DIM'], hp['ACTION_DIM'],
            hp['tau'], hp['delta_t'], hp['sigma'],
            celltype=self.celltype, actype=self.actype,
        )
        self.EPS_CLIP, self.GAMMA, self.EPOCHS = hp['EPS_CLIP'], hp['GAMMA'], hp['EPOCHS']

    def _load_agent(self, stage, mark, postfix, disp):
        """Load trained weights and return (agent, dt, rewards).

        ``mark``/``postfix`` select the weights folder/file:
        ``saved_params/{mark}_{celltype}_{ppotype}_{actype}/{stage}_trained_model{postfix}.pth``.
        The ``{stage}_param.pkl`` file next to the weights also stores training-time
        metadata; only ``dt``, ``LR`` and ``rewards`` are read here.
        """
        folder = _PACKAGE_ROOT / "saved_params" / f"{mark}_{self.celltype}_{self.ppotype}_{self.actype}"
        if disp:
            print("folder name:", folder)
        with open(folder / f'{stage}_param.pkl', 'rb') as f:
            param = pickle.load(f)
        dt, lr, rewards = param['dt'], param['LR'], param['rewards']

        agent = PPOAgent(self.model, self.EPS_CLIP, self.GAMMA, lr, self.EPOCHS, ppotype=self.ppotype)
        agent.model.eval()
        self.model.load_state_dict(torch.load(folder / f'{stage}_trained_model{postfix}.pth'))
        return agent, dt, rewards

    @staticmethod
    def _current_period(env, periods):
        for period in periods:
            if env.in_period(period):
                return period
        return None

    @staticmethod
    def _record_outcome(info, episode, correct, abort):
        assert info['trig'] in _ALL_TRIGS
        if info['trig'] in _CORRECT_TRIGS:
            correct[episode] = 1.0
        elif info['trig'] in _WRONG_TRIGS:
            correct[episode] = 0.0
        abort[episode] = 1.0 if info['trig'] in _ABORT_TRIGS else 0.0

    def _logs_to_dataframe(self, logs, trials, episode_rewards, intoStage2,
                           correct, abort, dt, extra_columns=None, disp=True,
                           keep_abort=False):
        """Assemble the per-trial DataFrame shared by CUE and CUEP rollouts."""
        # Chosen direction as 0/1: ground_truth - 1 on correct trials; on error
        # trials the other direction was chosen. NaN on aborts (dropped below).
        choice = [t['ground_truth'] - 1 if c == 1 else 2 - t['ground_truth']
                  for t, c in zip(trials, correct)]

        columns = {
            'episode_reward': episode_rewards,
            'coh': np.array([t['coh'] for t in trials]),
            'dircoh': np.array([t['signed_coh'] for t in trials]),
            'intoStage2': intoStage2,
            'correct': correct,
            'choice': choice,
            'abort': abort,
            'state': [np.array(log['state']) for log in logs],
            'action': [np.array(log['action']) for log in logs],
        }
        if extra_columns and 'action_passive' in extra_columns:
            columns['action_passive'] = extra_columns['action_passive']
        columns['period'] = [np.array(log['period']) for log in logs]
        columns['chprob'] = [np.array(log['chprob']) for log in logs]
        if extra_columns and 'chprob_passive' in extra_columns:
            columns['chprob_passive'] = extra_columns['chprob_passive']
        columns.update({
            # RT = time spent in the stage-1 ('stimulus') / stage-2 ('stimulus2') period
            'rt': np.array([np.sum(np.array(log['period']) == 'stimulus') * dt for log in logs]),
            'rt2': np.array([np.sum(np.array(log['period']) == 'stimulus2') * dt for log in logs]),
            'fr': [np.array(log['activities']) for log in logs],
        })
        df = pd.DataFrame(columns)

        # Indices of the blank/fixation -> stimulus -> stimulus2 period transitions
        # within each trial's time course (0 means "absent" and is fixed up below).
        period_changes = df['period'].apply(lambda x: np.where(np.array(x[1:]) != np.array(x[:-1]))[0])
        df['fix_idx'] = period_changes.apply(lambda x: 0 if len(x) < 1 else x[0] + 1)
        df['stim_idx'] = period_changes.apply(lambda x: 0 if len(x) < 2 else x[1] + 1)
        df['ch_idx'] = period_changes.apply(lambda x: 0 if len(x) < 3 else x[2] + 1)
        # Trials that never entered stage 2 have no stage-2 choice; mark the end
        # of the trial as the choice index instead.
        df['ch_idx'] = df.apply(
            lambda x: x['fr'].shape[0] - 1 if x['ch_idx'] == 0 else x['ch_idx'], axis=1)

        df['raw_rtg'] = pd.qcut(df['rt'], q=5, labels=False, duplicates='drop')
        df['dir'] = (df['dircoh'] > 0).astype(int)
        # RT quintiles, computed separately within PD and UR trials
        for expr in ['intoStage2==False', 'intoStage2==True']:
            idx = df.index[df.eval(expr)]
            if len(idx) > 0:
                df.loc[idx, 'rtg'] = pd.qcut(df.loc[idx, 'rt'], q=5, labels=False, duplicates='drop')

        if disp:
            print("abort proportion:", df['abort'].sum() / len(df['abort']))
        if not keep_abort:
            df = df.query('abort!=1')
            if disp:
                print("abort dropped")
        return df


class DataCUE(_DataBase):
    """Rollouts of the CUE-trained network on the CUE task."""

    stype = 'cue'

    def simulate(self, stage, sim_num_trial, mark='new', postfix='', disp=True,
                 cohs=DEFAULT_COHS, keep_abort=False):
        """Run the trained network for ``sim_num_trial`` trials and return the data.

        Returns
        -------
        df : pd.DataFrame
            One row per trial; see ``1_generate_data.ipynb`` for the column layout.
        """
        self.mark = mark
        self.stage = stage

        agent, dt, rewards = self._load_agent(stage, mark, postfix, disp)
        env = CUEExpTwoStage(dt=dt, cohs=cohs, rewards=rewards, disp=disp)

        episode_rewards = np.zeros(sim_num_trial)
        intoStage2 = np.zeros(sim_num_trial, dtype=bool)
        correct = np.full(sim_num_trial, np.nan)
        abort = np.zeros(sim_num_trial)
        trials = []
        logs = []

        periods = env.timing.keys()

        with tqdm(range(sim_num_trial)) as pbar:
            for episode in pbar:
                hidden = agent.model.init_hidden(1)  # fresh hidden state every trial
                state, _ = env.reset()
                state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

                terminated = False
                episode_reward = 0
                log = {"state": [], "action": [], "reward": [], "period": [], "chprob": []}
                trials.append(env.trial)
                while not terminated:
                    log["state"].append(state.squeeze().numpy())
                    log["period"].append(self._current_period(env, periods))
                    with torch.no_grad():
                        policy, _, hidden = agent.model(state, hidden)
                        dist = torch.distributions.Categorical(policy)
                        action = dist.sample()
                        log["chprob"].append(policy.squeeze().numpy())

                    ob, reward, terminated, truncated, info = env.step(action)
                    agent.store_transition(Transition_activity(state, action, reward, ob, terminated, hidden))
                    ob = torch.FloatTensor(ob).unsqueeze(0).unsqueeze(0)

                    state = ob
                    episode_reward += reward

                    log['action'].append(action.squeeze().numpy())
                    log['reward'].append(float(reward))

                batch = Transition_activity(*zip(*agent.memory))
                agent.memory.clear()
                # RNN activities, shape (time steps, hidden units)
                log['activities'] = torch.cat(batch.activity, dim=0).numpy()
                log["state"] = np.array(log["state"])
                log["action"] = np.array(log["action"])
                log["reward"] = np.array(log["reward"])
                log["period"] = np.array(log["period"])
                log["chprob"] = np.array(log["chprob"])

                episode_rewards[episode] = episode_reward
                intoStage2[episode] = info['stage']
                logs.append(log)
                self._record_outcome(info, episode, correct, abort)

        df = self._logs_to_dataframe(logs, trials, episode_rewards, intoStage2,
                                     correct, abort, dt, disp=disp, keep_abort=keep_abort)
        return df

    def load_agent_from_weights(self, stage, mark='new', postfix='', disp=True):
        """Load the PPOAgent from saved weights without running any trials."""
        agent, _, _ = self._load_agent(stage, mark, postfix, disp)
        return agent


class DataCUEP(_DataBase):
    """Rollouts of the CUEP-trained network on the CUEP (passive stage 1) task."""

    stype = 'cuep'

    def __init__(self, hp, celltype='vanilla', actype='wang', ppotype='classic'):
        super().__init__(hp, celltype=celltype, actype=actype, ppotype=ppotype)
        self.hp = hp
        self.device = hp.get('device', 'cpu')

    def _build_passive_agent(self, passive_cfg, postfix):
        """Build the CUE-trained agent that drives stage 1 of CUEP.

        Recognized ``passive_cfg`` keys: ``hp``, ``stage``, ``mark``, ``postfix``,
        ``disp`` and ``hidden_inherit`` (whether the passive agent's hidden state
        persists across trials; default True).
        """
        passive_hp = passive_cfg.get('hp', self.hp)
        passive_stage = passive_cfg.get('stage', self.stage)
        passive_mark = passive_cfg.get('mark', self.mark)
        passive_postfix = passive_cfg.get('postfix', postfix)
        passive_disp = passive_cfg.get('disp', False)
        passive_hidden_inherit = passive_cfg.get('hidden_inherit', True)

        passive_loader = DataCUE(passive_hp, celltype=self.celltype,
                                 actype=self.actype, ppotype=self.ppotype)
        passive_agent = passive_loader.load_agent_from_weights(
            stage=passive_stage, mark=passive_mark,
            postfix=passive_postfix, disp=passive_disp,
        )
        return passive_agent, passive_hp, passive_hidden_inherit

    def simulate(self, stage, sim_num_trial, mark='cuep', postfix='', disp=True,
                 cohs=DEFAULT_COHS, keep_abort=False, passive_agent_cfg=None,
                 passive_toldur=200, force_choice=True, disable_tqdm=False,
                 never_ur=False):
        """Run the trained network for ``sim_num_trial`` trials and return the data.

        Parameters
        ----------
        passive_agent_cfg : dict
            Weight selection for the CUE-trained passive agent (see
            ``_build_passive_agent``).
        passive_toldur : int
            Tolerance window (ms) after the passive UR within which the active
            agent may still take the UR option.
        never_ur : bool
            Control variant: the passive agent never emits the UR, so the
            ``ur_choice`` observation stays off (the "No Ext." condition).
        """
        self.mark = mark
        self.stage = stage

        agent, dt, rewards = self._load_agent(stage, mark, postfix, disp)

        passive_cfg = passive_agent_cfg or {}
        passive_agent, passive_hp, passive_hidden_inherit = self._build_passive_agent(passive_cfg, postfix)
        passive_agent.model.eval()

        # Share one noise generator between the active and passive RNNs so that
        # both experience identical recurrent noise within a step: each forward
        # pass is bracketed by get_state()/set_state() so the two models draw
        # the same noise values.
        shared_seed = torch.randint(0, 2**31 - 1, (1,), device=torch.device(self.device)).item()
        shared_noise_generator = torch.Generator(device=self.device)
        shared_noise_generator.manual_seed(shared_seed)
        for mdl in [agent.model, passive_agent.model]:
            if hasattr(mdl, "shared_rnn") and hasattr(mdl.shared_rnn, "set_noise_generator"):
                mdl.shared_rnn.set_noise_generator(shared_noise_generator)

        env = CUEExpTwoStagePassive(dt=dt, cohs=cohs, rewards=rewards, disp=disp,
                                    passive_toldur=passive_toldur, force_choice=force_choice)

        episode_rewards = np.zeros(sim_num_trial)
        intoStage2 = np.zeros(sim_num_trial, dtype=bool)
        correct = np.full(sim_num_trial, np.nan)
        abort = np.zeros(sim_num_trial)
        trials = []
        logs = []

        periods = env.timing.keys()
        passive_hidden = passive_agent.model.init_hidden(1)

        with tqdm(range(sim_num_trial), disable=disable_tqdm) as pbar:
            for episode in pbar:
                hidden = agent.model.init_hidden(1)  # fresh hidden state every trial
                if not passive_hidden_inherit:
                    passive_hidden = passive_agent.model.init_hidden(1)
                state, _ = env.reset()
                state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

                terminated = False
                episode_reward = 0
                log = {"state": [], "action": [], "action_passive": [], "reward": [],
                       "period": [], "chprob": [], "chprob_passive": []}
                trials.append(env.trial)
                while not terminated:
                    log["state"].append(state.squeeze().numpy())
                    log["period"].append(self._current_period(env, periods))
                    with torch.no_grad():
                        passive_state = state[:, :, :passive_hp['INPUT_DIM']]
                        gen_state = shared_noise_generator.get_state()
                        # Log the passive policy with identical noise...
                        policy_passive, _, _ = passive_agent.model(passive_state, passive_hidden)
                        log["chprob_passive"].append(policy_passive.squeeze().numpy())
                        # ...then rewind the generator so the action-sampling
                        # forward pass sees the same noise draw.
                        shared_noise_generator.set_state(gen_state)
                        action_passive, passive_hidden = passive_agent.select_action(passive_state, passive_hidden)
                        if never_ur:
                            action_passive = torch.tensor([0])

                        # Same replay for the active agent.
                        shared_noise_generator.set_state(gen_state)
                        policy, _, _ = agent.model(state, hidden)
                        log["chprob"].append(policy.squeeze().numpy())
                        shared_noise_generator.set_state(gen_state)
                        action, hidden = agent.select_action(state, hidden)

                    ob, reward, terminated, truncated, info = env.step((action, action_passive))
                    agent.store_transition(Transition_activity(state, action, reward, ob, terminated, hidden))
                    ob = torch.FloatTensor(ob).unsqueeze(0).unsqueeze(0)

                    state = ob
                    episode_reward += reward

                    log['action'].append(action.squeeze().numpy())
                    log['action_passive'].append(action_passive.squeeze().numpy())
                    log['reward'].append(float(reward))

                batch = Transition_activity(*zip(*agent.memory))
                agent.memory.clear()
                log['activities'] = torch.cat(batch.activity, dim=0).numpy()
                log["state"] = np.array(log["state"])
                log["action"] = np.array(log["action"])
                log["action_passive"] = np.array(log["action_passive"])
                log["reward"] = np.array(log["reward"])
                log["period"] = np.array(log["period"])
                log["chprob"] = np.array(log["chprob"])
                log["chprob_passive"] = np.array(log["chprob_passive"])

                episode_rewards[episode] = episode_reward
                intoStage2[episode] = info['stage']
                logs.append(log)
                self._record_outcome(info, episode, correct, abort)

        extra = {
            'action_passive': [np.array(log['action_passive']) for log in logs],
            'chprob_passive': [np.array(log['chprob_passive']) for log in logs],
        }
        df = self._logs_to_dataframe(logs, trials, episode_rewards, intoStage2,
                                     correct, abort, dt, extra_columns=extra,
                                     disp=disp, keep_abort=keep_abort)
        return df
