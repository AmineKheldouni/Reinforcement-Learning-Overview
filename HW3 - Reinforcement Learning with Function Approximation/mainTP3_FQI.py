import numpy as np
import lqg1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import collect_episodes, estimate_performance
from tqdm import tqdm

lbd = 0.1
discount = 0.9
horizon = T = 50
N = 100

env = lqg1d.LQG1D(initial_state_type='random')

actions = discrete_actions = np.linspace(-8, 8, 20)

class PolicyBEH:
    def __init__(self, discrete_actions):
        self.discrete_actions = discrete_actions

    def draw_action(self, state):
        return self.discrete_actions[np.random.choice(range(20))]

class FQI:
    def __init__(self, discrete_actions, theta):
        self.discrete_actions = discrete_actions
        self.theta = theta

    def draw_action(self, s):
        idx_action = np.argmax(np.array([empirical_Qfunction(self.theta, s, a) for a in discrete_actions]))
        return self.discrete_actions[idx_action]

#################################################################
# Show the optimal Q-function
#################################################################
def make_grid(x, y):
    m = np.meshgrid(x, y, copy=False, indexing='ij')
    return np.vstack(m).reshape(2, -1).T

states = discrete_states = np.linspace(-10, 10, 20)
SA = make_grid(states, actions)
S, A = SA[:, 0], SA[:, 1]

K, cov = env.computeOptimalK(discount), 0.001
print('Optimal K: {} Covariance S: {}'.format(K, cov))

Q_fun_ = np.vectorize(lambda s, a: env.computeQFunction(s, a, K, cov, discount, 1))
Q_fun = lambda X: Q_fun_(X[:, 0], X[:, 1])

Q_opt = Q_fun(SA)


def empirical_Qfunction(theta, state, action):
    phi = np.array([action, state*action, state**2 + action**2]).reshape((1,3))
    Q = np.sum(phi.dot(theta))
    return Q


def compute_Bellman(discount, next_states, rewards, theta):
    Y = []
    for i in range(len(rewards)):
        next_state = next_states[i]
        reward = rewards[i]

        maxQ = max([empirical_Qfunction(theta, next_state, a) for a in discrete_actions])
        Y.append(reward + discount * maxQ)

    return np.array(Y).reshape(-1)

# FQI algorithm
beh_policy = PolicyBEH(discrete_actions)
dataset = collect_episodes(env, n_episodes=N, policy=beh_policy, horizon=horizon)
theta_vector = []
performance_vector = []

theta = np.zeros((3,1))

for t in range(T):
    dataset_states = np.array([path['states'][t] for path in dataset])
    dataset_next_states = np.array([path['next_states'][t] for path in dataset])
    dataset_rewards = np.array([path['rewards'][t] for path in dataset])
    dataset_actions = np.array([path['actions'][t] for path in dataset])

    y = compute_Bellman(discount, dataset_next_states, dataset_rewards, theta)
    Z = np.array([[np.sum(a), np.sum(s*a), np.sum(s**2 + a**2)] for s,a in zip(dataset_states, dataset_actions)])

    theta = np.linalg.inv(Z.T.dot(Z) + lbd*np.eye(3)).dot(Z.T).dot(y).reshape((3,1))
    fqi = FQI(discrete_actions, theta)
    Jt = estimate_performance(env, policy=fqi, horizon=50, n_episodes=50, gamma=discount)
    print(t, theta, Jt)
    performance_vector.append(Jt)
    theta_vector.append(theta)

# plot obtained Q-function against the true one
print(theta)

estimated_Qfunction = lambda X: empirical_Qfunction(X[:,0], X[:,1], theta)
Q_function_estimated = np.vectorize(lambda s, a: empirical_Qfunction(theta, s, a))
Q_estimated = lambda X: Q_function_estimated(X[:, 0], X[:, 1])

Q_opt = Q_fun(SA)
Q = Q_estimated(SA)


J = estimate_performance(env, policy=fqi, horizon=100, n_episodes=500, gamma=discount)
print('Policy performance: {}'.format(J))
fig, ax = plt.subplots(1)
plt.plot(performance_vector)
plt.ylabel('Performance', fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.show()

fig, ax = plt.subplots(1)
theta_vector=np.array(theta_vector)
plt.plot(range(T), theta_vector[:,0], color='blue')
plt.plot(range(T), theta_vector[:,1], color='red')
plt.plot(range(T), theta_vector[:,2], color='green')
plt.ylabel('Performance', fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S, A, Q_opt, color='blue')
ax.scatter(S, A, Q.flatten(), color='red')

plt.show()
