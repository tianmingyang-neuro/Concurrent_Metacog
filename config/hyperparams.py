# Hyperparameters of the shipped networks.
#
# GAMMA / EPS_CLIP / EPOCHS are PPO training hyperparameters: this package is
# inference-only, so they are never used (they are kept to document the training
# configuration and to satisfy the PPOAgent constructor signature).
# Note the two time scales: the environment steps in dt = 50 ms bins (stored in
# saved_params/*/4_param.pkl), while the RNN integrates with delta_t = 10 ms.

# CUE: two-stage random-dot-motion task with an uncertainty response (UR)
hy_params = dict(
    INPUT_DIM=4,  # fixation, fixation2, stimulus x, stimulus y
    HIDDEN_DIM=48,  # Number of RNN units
    ACTION_DIM=4,  # fixation, choice1, choice2, ur
    GAMMA=0.99,  # Discount factor (training only)
    EPS_CLIP=0.3,  # PPO clipping parameter (training only)
    EPOCHS=3,  # PPO update epochs (training only)
    tau=100,
    delta_t=10,
    sigma=0.1,
    device='cpu'
)

# CUEP: CUE with a passive stage 1 (UR timing imposed externally)
hy_params_cuep = dict(
    INPUT_DIM=5,  # fixation, fixation2, stimulus x, stimulus y, ur_choice
    HIDDEN_DIM=48,  # Number of RNN units
    ACTION_DIM=4,  # fixation, choice1, choice2, ur
    GAMMA=0.99,  # Discount factor (training only)
    EPS_CLIP=0.3,  # PPO clipping parameter (training only)
    EPOCHS=3,  # PPO update epochs (training only)
    tau=100,
    delta_t=10,
    sigma=0.1,
    device='cpu'
)
