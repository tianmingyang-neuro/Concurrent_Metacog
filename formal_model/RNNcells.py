import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNNCell(nn.Module):
    """Leaky vanilla RNN with recurrent noise.

    One Euler step per call:  r <- (1 - dt/tau) * r + (dt/tau) * relu(J r + B x + c + noise).
    ``delta_t`` is the RNN integration step (10 ms) and is intentionally
    independent of the environment time step (dt = 50 ms, set in the param file).
    """

    def __init__(self, hidden_size, input_size, tau, delta_t, sigma=0.1, device='cpu'):
        super(VanillaRNNCell, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.tau = tau
        self.sigma = torch.sqrt(torch.tensor(2 * self.tau)) * sigma
        self.delta_t = delta_t
        self.noise_generator = None  # optional shared generator for reproducible noise across models
        self.J = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        # Initialize weights diagonally
        nn.init.zeros_(self.J.weight)
        self.B = nn.Linear(input_size, hidden_size, bias=False).to(device)
        nn.init.uniform_(self.B.weight, 0, 1 / np.sqrt(input_size))
        self.c = nn.Parameter(torch.zeros(hidden_size, device=device))

    def set_noise_generator(self, generator):
        """Allow external code to share a torch.Generator for noise sampling."""
        self.noise_generator = generator

    def _sample_noise(self, like):
        if self.noise_generator is not None:
            noise = torch.randn(like.shape, device=like.device, generator=self.noise_generator)
        else:
            noise = torch.randn_like(like)
        return noise * self.sigma

    def recurrence(self, input, r_prev):
        noise = self._sample_noise(r_prev)
        i_x = F.relu(self.J(r_prev) + self.B(input) + self.c + noise)

        r = (1 - self.delta_t / self.tau) * r_prev + (self.delta_t / self.tau) * i_x
        return r

    def forward(self, input, hidden=None):

        if hidden is None:
            r = self.init_hidden(input.shape[0]).to(input.device)
        else:
            r = hidden
        frs = []
        for i in range(input.shape[1]):
            r = self.recurrence(input[:, i, :], r)
            frs.append(r)
        frs = torch.stack(frs, dim=1)
        return frs, r

    def init_hidden(self, batch_size):
        return torch.zeros([batch_size, self.hidden_size])
