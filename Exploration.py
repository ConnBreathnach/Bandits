from jax import numpy as jnp
from jax import random

key = random.PRNGKey()
class GreedyExplorer:
    def select_action(self, model, state):
        q_values = model(state)
        action = jnp.argmax(q_values)
        return action

class RandomExplorer:
    def select_action(self, model, state):
        q_values = model(state)
        return random.randint(key, q_values)

class EpsilonGreedyExplorer:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select_action(self, model, state):
        q_values = model(state)
        if random.uniform(key, 0, 1) < self.epsilon:
            action = random.randint(key, q_values)
        else:
            action = jnp.argmax(q_values)
        return action

class LinearDecayEpsilonGreedyExplorer:
        def __init__(self, epsilon=1.00, min_epsilon=0.01, decay=0.01):
            self.epsilon = epsilon
            self.min_epsilon = epsilon
            self.decay = decay

        def select_action(self, model, state):
            q_values = model(state)
            if random.uniform(key, 0, 1) < self.epsilon:
                action = random.randint(key, q_values)
            else:
                action = jnp.argmax(q_values)
            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay)
            return action

class ExponentialDecayEpsilonGreedyExplorer:
    def __init__(self, epsilon=1.00, min_epsilon=0.01, decay=0.99):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def select_action(self, model, state):
        q_values = model(state)
        if random.uniform(key, 0, 1) < self.epsilon:
            action = random.randint(key, q_values)
        else:
            action = jnp.argmax(q_values)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        return action

class OptimisticStartExplorer:
    def __init__(self, start_values = 10, Q = None, N = None):
        self.start_values = start_values
        self.Q = Q
        self.N = N

    def select_action(self, env):
        if self.Q is None:
            self.Q = jnp.full(shape=env.action_space.n, fill_value=self.start_values)
        if self.N is None:
            self.N = jnp.full(shape=env.action_space.n, fill_value=1)
        q_values = self.Q / (1 + self.N)
        action = jnp.argmax(q_values)
        self.N[action] += 1
        reward = env.step(action)[0]
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        return action

class UpperConfidenceBoundExplorer:
    def __init__(self, c=1.0):
        self.c = c
        self.N = None
        self.Q = None
        self.t = 0
        self.times = None

    def select_action(self, env):
        if self.N is None:
            self.N = jnp.full(shape=env.action_space.n, fill_value=1)
        if self.Q is None:
            self.Q = jnp.full(shape=env.action_space.n, fill_value=0)
        if self.times is None:
            self.times = jnp.full(shape=env.action_space.n, fill_value=0)
        confidences = [self.Q[action] / self.N[action] + self.c * jnp.sqrt(jnp.log(self.t) / self.N[action]) for action in range(env.action_space.n)]
        action = jnp.argmax(confidences)
        reward = env.step(action)[0]
        self.times[action] += 1
        self.N[action] == 0
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        self.t += 1
        for item in range(self.N.shape[0]):
            self.N[item] += 1
        return action



