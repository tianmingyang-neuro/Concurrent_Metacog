import numpy as np
import torch
import torch.nn as nn
from .RNNcells import VanillaRNNCell


# RNN-based Actor-Critic Model.
# 'wang' / 'vanillazero' are variant labels kept for compatibility with the
# saved_params folder names; only this one architecture ships with the package.
class ActorCriticWang(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, tau, delta_t, sigma, celltype='vanilla', device='cpu', **cell_kwargs):
        super(ActorCriticWang, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.delta_t = delta_t
        self.action_dim = action_dim

        # Policy head (actor) with vanilla RNN structure
        if celltype in ['vanilla', 'vanillazero']:
            self.shared_rnn = VanillaRNNCell(hidden_dim, input_dim, tau, delta_t, sigma, device=device, **cell_kwargs)
        else:
            raise Exception(f"{celltype} not included")
        self._actor_layer()
        self._value_layer()

    def _value_layer(self):
        # Initialization is a placeholder: trained weights are loaded over it.
        critic_linear = nn.Linear(self.hidden_dim, 1)
        critic_bound = 0.4 / np.sqrt(self.hidden_dim)
        nn.init.uniform_(critic_linear.weight, 0, critic_bound)
        nn.init.zeros_(critic_linear.bias)
        self.critic = critic_linear

    def _actor_layer(self):
        # Initialization is a placeholder: trained weights are loaded over it.
        actor_linear = nn.Linear(self.hidden_dim, self.action_dim)
        actor_bound = 0.4 / np.sqrt(self.hidden_dim)
        nn.init.uniform_(actor_linear.weight, 0, actor_bound)
        nn.init.zeros_(actor_linear.bias)
        self.actor = nn.Sequential(
            actor_linear,
            nn.Softmax(dim=-1)
        )

    def forward(self, x, r=None):
        # x: (batch_size, seq_len, input_dim)
        # r: hidden state, (batch_size, hidden_dim)
        frs, r = self.shared_rnn(x, r)
        policy = self.actor(frs)
        value = self.critic(frs)
        return policy, value, r

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)


def ActorCritic(input_dim, hidden_dim, action_dim, tau, delta_t, sigma, actype='wang', celltype='vanilla', device='cpu', **cell_kwargs):
    if actype == 'wang':
        return ActorCriticWang(input_dim, hidden_dim, action_dim, tau, delta_t, sigma, celltype=celltype, device=device, **cell_kwargs)
    raise NotImplementedError(f"Actor-critic type '{actype}' is not implemented. Only 'wang' is supported.")
