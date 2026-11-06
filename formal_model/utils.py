# Experience tuple for rollouts
from collections import namedtuple

Transition_activity = namedtuple('Transition_activity',
                                 ('state', 'action', 'reward', 'ob', 'terminated', 'activity'))