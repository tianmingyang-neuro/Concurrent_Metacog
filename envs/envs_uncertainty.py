"""Neurogym environments for the CUE and CUEP two-stage random-dot-motion tasks.

CUE (concurrent uncertainty estimation): in stage 1 the agent fixates, commits
to a direction (perceptual decision, PD), or emits an uncertainty response (UR)
that triggers stage 2, a high-coherence stimulus in which the direction choice
is easy. CUEP is the passive variant: the UR timing in stage 1 is imposed by an
external (passive) agent and signaled through a dedicated observation channel.

One environment step is ``dt`` ms (50 ms in the shipped simulations; the RNN
integrates with its own smaller time step, see ``config/hyperparams.py``).
"""
import numpy as np
from neurogym import spaces

import neurogym as ngym

DEFAULT_COHS = [2, 4, 6, 8, 13, 26, 52, 78]


class CUEExpTwoStage(ngym.core.TrialEnv):
    """CUE task: active two-stage RDM with an uncertainty response (UR) option.

    Observation channels: fixation, fixation2 (stage indicator), and two motion
    evidence channels (stim1_mod1, stim2_mod1). Actions: 0 = fixate/wait,
    1/2 = direction choices, 3 = UR (enter stage 2).
    """

    def __init__(self, dt=50, tau=100, rewards=None, timing=None, cohs=DEFAULT_COHS, sigma=0.01, disp=True) -> None:
        super().__init__(dt=dt)

        # trial conditions
        self.choices = [1, 2]  # left, right choice
        self.cohs = cohs
        # stimulus observation noise (independent of the RNN's internal noise)
        self.sigma = np.sqrt(2 * tau) * sigma

        # Rewards
        self.rewards = {"abort": -1.0, "correct": 1.0, 'wrong': 0.0, "time_cost": -0.01, "switching_cost": -0.2,
                        'ur_correct': 1.0, 'ur_wrong': 0, 'ur_abort': -1.0}
        if rewards:
            self.rewards.update(rewards)
        if disp:
            print("reward: ", rewards)

        # 'stimulus' (stage 1) and 'stimulus2' (stage 2) have a fixed 10 s budget;
        # exceeding it aborts the trial ('time_expired' / 'ur_time_expired').
        self.timing = {
            "blank": lambda: self.rng.uniform(500, 1500),
            "fixation": lambda: self.rng.uniform(300, 700),
            "stimulus": 10000,
            "stimulus2": 10000
        }
        if timing:
            self.timing.update(timing)

        self.scale = 1.0

        # set action and observation space
        names = [
            "fixation", "fixation2",
            "stim1_mod1", "stim2_mod1",
        ]
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(4,),
            dtype=np.float32,
            name=name,
        )

        name = {"fixation": 0, "choice1": 1, "choice2": 2, 'uncertainty_choice': 3}
        self.action_space = spaces.Discrete(4, name=name)

    def _new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "coh": self.rng.choice(self.cohs),

        }
        trial.update(kwargs)

        choice = trial["ground_truth"]
        coh = trial["coh"]
        signed_coh = coh if choice == 1 else -coh
        trial.update(
            {"signed_coh": signed_coh}
        )

        # Stage 2 uses the maximum coherence (an easy, high-certainty stimulus)
        max_coh = max(self.cohs)
        self.signed_max_coh = max_coh if choice == 1 else -max_coh

        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        self.periods = ["blank", "fixation", "stimulus", "stimulus2"]
        self.add_period(self.periods)

        # Add observations for first stage
        self.add_ob(1, period='blank', where='fixation')
        for period in self.periods[1:3]:
            self.add_ob(0, period=period, where="fixation")

        for period in self.periods[:3]:
            self.add_ob(1, period=period, where='fixation2')
        self.add_ob(0, period='stimulus2', where="fixation2")

        # First stage stimulus
        self.add_ob((1 + self.scale * signed_coh / 100) / 2, period="stimulus", where="stim1_mod1")
        self.add_ob((1 - self.scale * signed_coh / 100) / 2, period="stimulus", where="stim2_mod1")
        self.add_randn(0, self.sigma, "stimulus", where=[2, 3])
        # Second stage stimulus (high coherence)
        self.add_ob((1 + self.scale * self.signed_max_coh / 100) / 2, period="stimulus2", where="stim1_mod1")
        self.add_ob((1 - self.scale * self.signed_max_coh / 100) / 2, period="stimulus2", where="stim2_mod1")
        self.add_randn(0, self.sigma, "stimulus2", where=[2, 3])

        self.set_groundtruth(trial["ground_truth"], period=["stimulus", "stimulus2"])
        self.into_stage2 = False  # Reset stage to 1 for new trial

        return trial

    def _step(self, action):
        ob = self.ob_now
        gt = self.gt_now

        terminated = False
        truncated = False
        trig = None
        reward = 0

        if self.in_period('blank'):
            trig = 'blank'
        elif self.in_period("fixation"):
            if action != 0:
                reward = self.rewards["abort"]
                trig = 'abort'
                terminated = True
            else:
                trig = 'fix'
        elif self.in_period("stimulus") and not self.into_stage2:
            if action != 0:  # broke fixation
                if action == gt:
                    reward = self.rewards["correct"]
                    self.performance = 1
                    trig = 'correct'
                    terminated = True
                elif action == 3:  # Uncertainty choice - jump to the stage-2 period
                    self.into_stage2 = True
                    trig = 'uncertainty'
                    reward = self.rewards['time_cost'] + self.rewards['switching_cost']
                    self.t = self.start_t['stimulus2']
                    self.t_ind = int(self.t // self.dt)
                else:
                    reward = self.rewards['wrong']
                    self.performance = 0
                    trig = 'wrong'
                    terminated = True
            else:
                self.into_stage2 = False
                reward = self.rewards['time_cost']
                trig = 'watch'
                # If stimulus time is exhausted, end trial with abort
                if self.t >= self.end_t['stimulus'] - self.dt:
                    reward = self.rewards["abort"]
                    trig = 'time_expired'
                    terminated = True
        elif self.in_period("stimulus2") and self.into_stage2:
            if action in [0]:
                reward = self.rewards['time_cost']
                trig = 'ur_watch'
                # If second stage time is exhausted, end trial with abort
                if self.t >= self.end_t['stimulus2'] - self.dt:
                    reward = self.rewards["abort"]
                    trig = 'ur_time_expired'
                    terminated = True
            elif action in [1, 2]:
                if action == gt:
                    reward = self.rewards["ur_correct"]
                    self.performance = 1
                    trig = 'ur_correct'
                else:
                    reward = self.rewards['ur_wrong']
                    self.performance = 0
                    trig = 'ur_wrong'
                terminated = True
            else:
                trig = 'ur_abort'
                reward = self.rewards['ur_abort']
                terminated = True

        elif self.in_period("stimulus"):
            raise Exception("never reach stimulus with false into_stage2")
        elif self.in_period("stimulus2"):
            raise Exception("never reach stimulus with true into_stage2")
        else:
            raise Exception("unknown")

        info = {"new_trial": terminated, "gt": gt, 'trig': trig, 'stage': self.into_stage2}
        assert trig is not None

        return ob, reward, terminated, truncated, info


class CUEExpTwoStagePassive(ngym.core.TrialEnv):
    """CUEP task: CUE with a passive stage 1.

    The UR timing in stage 1 is imposed externally: once the passive agent emits
    the UR (action_passive == 3), the ``ur_choice`` observation channel flips to
    0 and the active agent may take the UR option (action 3) within
    ``passive_toldur`` ms. Direction choices are forbidden while the passive UR
    is pending; with ``force_choice=True`` a late UR aborts the trial, otherwise
    it is scored as a stage-2 choice.
    """

    def __init__(self, dt=50, tau=100,
                 rewards=None, timing=None, cohs=DEFAULT_COHS, sigma=0.01, disp=True,
                 passive_toldur=200, force_choice=True) -> None:
        super().__init__(dt=dt)

        # trial conditions
        self.choices = [1, 2]  # left, right choice
        self.cohs = cohs
        self.sigma = np.sqrt(2 * tau) * sigma
        self.passive_toldur = passive_toldur
        self.force_choice = force_choice

        # Rewards
        self.rewards = {"abort": -1.0, "correct": 1.0, 'wrong': 0.0, "time_cost": -0.01, "switching_cost": -0.2,
                        'ur_correct': 1.0, 'ur_wrong': 0, 'ur_abort': -1.0, 'ur_early': -1.0}
        if rewards:
            self.rewards.update(rewards)
        if disp:
            print("reward: ", rewards)

        self.timing = {
            "blank": lambda: self.rng.uniform(500, 1500),
            "fixation": lambda: self.rng.uniform(300, 700),
            "stimulus": 10000,
            "stimulus2": 10000
        }
        if timing:
            self.timing.update(timing)

        self.scale = 1.0

        # set action and observation space
        names = [
            "fixation", "fixation2",
            "stim1_mod1", "stim2_mod1", "ur_choice"
        ]
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(5,),
            dtype=np.float32,
            name=name,
        )

        name = {"fixation": 0, "choice1": 1, "choice2": 2, 'uncertainty_choice': 3}
        # (active agent action, passive agent action)
        self.action_space = spaces.MultiDiscrete([4, 4])

    def _new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "coh": self.rng.choice(self.cohs)
        }

        trial.update(kwargs)

        choice = trial["ground_truth"]
        coh = trial["coh"]
        signed_coh = coh if choice == 1 else -coh
        trial.update(
            {"signed_coh": signed_coh}
        )

        # Stage 2 uses the maximum coherence (an easy, high-certainty stimulus)
        max_coh = max(self.cohs)
        self.signed_max_coh = max_coh if choice == 1 else -max_coh

        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        self.periods = ["blank", "fixation", "stimulus", "stimulus2"]
        self.add_period(self.periods)

        # Add observations for first stage
        self.add_ob(1, period='blank', where='fixation')
        for period in self.periods[1:3]:
            self.add_ob(0, period=period, where="fixation")

        for period in self.periods[:3]:
            self.add_ob(1, period=period, where='fixation2')
        self.add_ob(0, period='stimulus2', where="fixation2")

        # First stage stimulus
        self.add_ob((1 + self.scale * signed_coh / 100) / 2, period="stimulus", where="stim1_mod1")
        self.add_ob((1 - self.scale * signed_coh / 100) / 2, period="stimulus", where="stim2_mod1")
        self.add_randn(0, self.sigma, "stimulus", where=[2, 3])
        # Second stage stimulus (high coherence)
        self.add_ob((1 + self.scale * self.signed_max_coh / 100) / 2, period="stimulus2", where="stim1_mod1")
        self.add_ob((1 - self.scale * self.signed_max_coh / 100) / 2, period="stimulus2", where="stim2_mod1")
        self.add_randn(0, self.sigma, "stimulus2", where=[2, 3])
        for period in self.periods:
            self.add_ob(1, period=period, where="ur_choice")

        self.set_groundtruth([trial["ground_truth"], 0], period=["stimulus", "stimulus2"])
        self.into_stage2 = False  # Reset stage to 1 for new trial

        # Passive-UR state for the new trial
        self.passive_into_stage2 = False
        self.passive_dur = 0

        return trial

    def _step(self, action_tuple):

        action, action_passive = action_tuple

        ob = self.ob_now
        gt = self.gt_now[0]

        terminated = False
        truncated = False
        trig = None
        reward = 0

        if self.in_period('blank'):
            trig = 'blank'
        elif self.in_period("fixation"):
            if action != 0:
                reward = self.rewards["abort"]
                trig = 'abort'
                terminated = True
            else:
                trig = 'fix'
        elif self.in_period("stimulus") and not self.into_stage2:

            if action_passive == 3:  # passive agent requests stage 2
                self.passive_into_stage2 = True
            if self.passive_into_stage2:
                # Signal the passive UR on the ur_choice channel of the NEXT
                # observation (index 4 matches the observation_space ordering);
                # the agent sees it from the next step onward.
                self.ob[self.t_ind + 1, 4] = 0
                self.passive_dur += self.dt

            # --- Allowed actions while a passive UR is pending ---
            if self.passive_into_stage2:
                if action == 0:
                    reward = self.rewards['time_cost']
                    trig = 'watch'
                    # If stimulus time is exhausted, end trial with abort
                    if self.t >= self.end_t['stimulus'] - self.dt:
                        reward = self.rewards["abort"]
                        trig = 'time_expired'
                        terminated = True
                elif action == 3:
                    # UR allowed only within passive_toldur of the passive UR
                    if self.passive_dur <= self.passive_toldur:
                        self.into_stage2 = True
                        trig = 'uncertainty'
                        reward = self.rewards['time_cost'] + self.rewards['switching_cost']
                        self.t = self.start_t['stimulus2']
                        self.t_ind = int(self.t // self.dt)
                    else:
                        if self.force_choice:
                            reward = self.rewards["abort"]
                            trig = 'abort'
                            terminated = True
                        else:
                            if action == gt:
                                reward = self.rewards["ur_correct"]
                                self.performance = 1
                                trig = 'ur_correct'
                            else:
                                reward = self.rewards['ur_wrong']
                                self.performance = 0
                                trig = 'ur_wrong'
                            terminated = True

                    # Once action 3 is chosen under passive, reset passive ur flag
                    self.passive_into_stage2 = False
                else:
                    # Direction choices are forbidden while a passive UR is pending
                    reward = self.rewards["abort"]
                    trig = 'abort'
                    terminated = True
            else:
                if action == 3:
                    # UR before any passive UR request
                    reward = self.rewards["ur_early"]
                    trig = 'abort'
                    terminated = True
                elif action in [1, 2]:
                    if action == gt:
                        reward = self.rewards["correct"]
                        self.performance = 1
                        trig = 'correct'
                        terminated = True
                    else:
                        reward = self.rewards['wrong']
                        self.performance = 0
                        trig = 'wrong'
                        terminated = True
                elif action == 0:
                    reward = self.rewards['time_cost']
                    trig = 'watch'
                    # If stimulus time is exhausted, end trial with abort
                    if self.t >= self.end_t['stimulus'] - self.dt:
                        reward = self.rewards["abort"]
                        trig = 'time_expired'
                        terminated = True
        elif self.in_period("stimulus2") and self.into_stage2:
            if action == 0:
                reward = self.rewards['time_cost']
                trig = 'ur_watch'
                # If second stage time is exhausted, end trial with abort
                if self.t >= self.end_t['stimulus2'] - self.dt:
                    reward = self.rewards["abort"]
                    trig = 'ur_time_expired'
                    terminated = True
            elif action in [1, 2]:
                if action == gt:
                    reward = self.rewards["ur_correct"]
                    self.performance = 1
                    trig = 'ur_correct'
                else:
                    reward = self.rewards['ur_wrong']
                    self.performance = 0
                    trig = 'ur_wrong'
                terminated = True
            else:
                trig = 'ur_abort'
                reward = self.rewards['ur_abort']
                terminated = True
        elif self.in_period("stimulus"):
            raise Exception("never reach stimulus with false into_stage2")
        elif self.in_period("stimulus2"):
            raise Exception("never reach stimulus with true into_stage2")
        else:
            raise Exception("unknown")

        info = {"new_trial": terminated, "gt": gt, 'trig': trig, 'stage': self.into_stage2}
        assert trig is not None

        return ob, reward, terminated, truncated, info
