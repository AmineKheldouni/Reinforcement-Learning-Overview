import numpy as np
import matplotlib.pyplot as plt

gamma = 0.95

dictTransition = dict({
    0: np.array([[0.55,0.45,0],[1,0,0],[0,1,0]]),
    1: np.array([[0.3,0.7,0],[0,0.4,0.6],[0,0.6,0.4]]),
    2: np.eye(3)
})
rewardMatrix = np.array([[0,0,0],[0,0,1],[0.05,0,0.9]])

class MDP:
    def __init__(self, gamma=0.95):
        self.gamma = gamma

    def R(self, state, action):
        global rewardMatrix
        return rewardMatrix[action,state]

    def T(self, state, action):
        global dictTransition
        return dictTransition[action][state,:]

    def actions(self):
        return [0, 1, 2]

    def states(self):
        return [0, 1, 2]

def value_iteration(mdp, epsilon=0.01):
    """This function computes the Value Iteration algorithm (VI).
    It returns the 0.01-optimal value and policy using Bellman's equation.
    """
    V1 = dict([(s, 0) for s in mdp.states()])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    delta = []
    deltas = 1
    counter = 0
    pi = dict([(s, 0) for s in mdp.states()])
    while deltas >= epsilon:
        V = V1.copy()
        deltas = 0
        for s in mdp.states():
            V1[s] = max([R(s, a) + gamma * sum([p * V[sp] for sp, p in enumerate(T(s, a))])
                                        for a in mdp.actions()])
            pi[s] = np.argmax([R(s, a) + gamma * sum([p * V[sp] for sp, p in enumerate(T(s, a))])
                                        for a in mdp.actions()])
            deltas = max(deltas, abs(V1[s] - V[s]))
        delta.append(deltas)
        counter += 1
    print("Number of iterations until convergence: ", counter)
    return V, pi, delta

def policy_evaluation(mdp, policy, k=1000):
    V = dict([(s, 0) for s in mdp.states()])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states():
            V[s] = R(s, policy[s]) + gamma * sum([p * V[s1] for (s1, p) in enumerate(T(s, policy[s]))])
    return V

def policy_iteration(mdp):
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    policy = dict({0:0, 1:0, 2:1})
    V = dict([(s, 0) for s in mdp.states()])
    counter = 0
    pi = dict([(s, 0) for s in mdp.states()])
    policyConv = False
    while not policyConv:
        V = policy_evaluation(mdp, policy)
        policyConv = True
        for s in mdp.states():
            a = np.argmax([sum([p * V[sp] for (sp, p) in enumerate(T(s, a))]) for a in range(3)])
            if a!= policy[s]:
                policy[s] = a
                policyConv = False
        counter += 1
    print("Number of iterations: ", counter)
    return V, policy

S = MDP()
V, pi, delta = value_iteration(S)
print("Optimal Value: ", V)
print("Best policy: ", pi)
vstar = policy_evaluation(S, pi)
print("Vstar: ", vstar)

print("###################")
print("Policy Iteration")
print("###################")

S = MDP()
V2, pi2 = policy_iteration(S)
print("Optimal Value: ", V2)
print("Best policy: ", pi2)

plt.plot(range(len(delta)), delta, color='r', label="Value Iteration Algorithm")
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Value differences", fontsize=14)
plt.title("Value Iteration Congervence", fontsize=16)
plt.draw()
plt.show()
