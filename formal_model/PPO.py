import torch


# PPO agent, inference only: select_action / store_transition are used to run
# the trained network; the PPO update step is not part of this package.
class PPOAgentClassical:
    def __init__(self, model, EPS_CLIP, GAMMA, LR, EPOCHS):
        self.EPS_CLIP, self.GAMMA, self.LR, self.EPOCHS = \
            EPS_CLIP, GAMMA, LR, EPOCHS
        self.model = model
        self.memory = []

    def select_action(self, state, hidden):
        with torch.no_grad():
            policy, _, hidden = self.model(state, hidden)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
        return action, hidden

    def store_transition(self, transition):
        self.memory.append(transition)


def PPOAgent(model, EPS_CLIP, GAMMA, LR, EPOCHS, ppotype='classic', **kwargs):
    if ppotype in ['classic', '']:
        return PPOAgentClassical(model, EPS_CLIP, GAMMA, LR, EPOCHS)
    raise NotImplementedError(f"PPO type '{ppotype}' is not implemented. Only 'classic' is supported.")
