from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import matplotlib.pyplot as plt
import time

env = GridWorld1

################################################################################
# investigate the structure of the environment
# - env.n_states: the number of states
# - env.state2coord: converts state number to coordinates (row, col)
# - env.coord2state: converts coordinates (row, col) into state number
# - env.action_names: converts action number [0,3] into a named action
# - env.state_actions: for each state stores the action availables
#   For example
#       print(env.state_actions[4]) -> [1,3]
#       print(env.action_names[env.state_actions[4]]) -> ['down' 'up']
# - env.gamma: discount factor
################################################################################
print(env.state2coord)
print(env.coord2state)
print(env.state_actions)
for i, el in enumerate(env.state_actions):
        print("s{}: {}".format(i, env.action_names[el]))

################################################################################
# Policy definition
# If you want to represent deterministic action you can just use the number of
# the action. Recall that in the terminal states only action 0 (right) is
# defined.
# In this case, you can use gui.renderpol to visualize the policy
################################################################################
# pol = [1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3]
# gui.render_policy(env, pol)
#
# ################################################################################
# # Try to simulate a trajectory
# # you can use env.step(s,a, render=True) to visualize the transition
# ################################################################################
#
# env.render = True
# state = 0
# fps = 1
# for i in range(5):
#         action = np.random.choice(env.state_actions[state])
#         nexts, reward, term = env.step(state,action)
#         state = nexts
#         time.sleep(1./fps)
#
# ################################################################################
# # You can also visualize the q-function using render_q
# ################################################################################
# # first get the maximum number of actions available
# q = np.random.rand(env.n_states, 4)
# print(q)
# gui.render_q(env, q)

################################################################################
# Work to do: Q4
################################################################################
# here the v-function and q-function to be used for question 4
v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]

def policy(env,state):
        """
        Implementation of the policy:
        Right if possible, Up otherwise.
        """
        actions = env.state_actions[state]
        if 0 in actions:
                return 0
        elif 3 in actions:
                return 3
        else:
                return 1

def startingStateDistribution(env, N=100000):
        """
        This function samples initial states for the environment and computes
        an empirical estimator for the starting distribution mu_0
        """
        rdInit = []
        sample = {}
        # Computing the starting state distribution
        mu_0 = np.zeros((env.n_states,1))
        for i in range(N):
                rdInit.append(env.reset())
        for i in range(0, env.n_states):
                sample[i] = rdInit.count(i)
                mu_0[i] = sample[i]/N
        return mu_0

def empiricalValue(env, Tmax=20, N=10000, gamma=0.95):
        """
        This function returns the empirical value function using Monte Carlo
        simulations
        """
        Ns = [0 for i in range(env.n_states)]
        V = np.zeros(env.n_states)
        for i in range(N):
                s = env.reset()
                Ns[s] += 1
                t = 1
                term = False
                states = [s]
                while (t < Tmax and not term):
                        at = policy(env, states[t-1])
                        sNext, rt, term = env.step(states[t-1], at)
                        states.append(sNext)
                        V[s] += gamma**(t-1) * rt
                        t += 1
        V = V.reshape(-1,1) / (np.array(Ns).reshape(-1,1)+10e-20)
        return V

def exactValueFunction(env, mu0):
        """
        This function returns the expected value function towards which
        the empirical estimator of value function should converge
        """
        Vpi = np.array([0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855, -0.93358351, -0.99447514])
        return (np.sum(mu0.reshape(-1,1)*Vpi.reshape(-1,1)))

def empiricalValueFunction(env, mu0, n):
        """
        This function returns the empirical value function computed with n
        simulations
        """
        Vn = np.array(empiricalValue(env,N=n))
        return (np.sum(mu0.reshape(-1,1)*Vn.reshape(-1,1)))

mu0 = startingStateDistribution(env)
print("Starting state distribution: ", mu0.T)
lJn = []
I = range(0, 10000, 100)
for i in I:
        lJn.append(abs(empiricalValueFunction(env,mu0,i)-exactValueFunction(env, mu0)))
plt.plot(I, lJn)
plt.xlabel("Number of samples", fontsize=16)
plt.ylabel("Value function differences", fontsize=16)
plt.show()

################################################################################
# Work to do: Q5
################################################################################
v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]

def explorationPolicy(env, state, epsilon, Q):
        actions = env.state_actions[state]
        if np.random.uniform() > epsilon:
                possibleBestActions = []
                for a in actions:
                        value = Q[state][a]
                        if value == max(Q[state][actions]):
                                possibleBestActions.append(a)
                next_action = np.random.choice(possibleBestActions)
        else :
             next_action = np.random.choice(actions)
        return next_action


def QLearning(env, N=10000, gamma=0.95, Tmax=20, epsilon = 0.001):
        Q = dict()
        episodes = []
        cumulatedReward = []
        for i in range(env.n_states):
                Q[i] = np.array([0.,0.,0.,0.])
        Nxa = np.zeros((env.n_states, 4))
        # Looping over episodes
        for i in range(N):
                t = 0
                s = [env.reset()]
                term = False
                cumReward = 0
                # For each time step until Tmax or terminal state
                while (not term and t < Tmax):
                        at = explorationPolicy(env, s[t], epsilon, Q)
                        Nxa[s[t],at] += 1
                        sNext, rt, term = env.step(s[t], at)
                        s.append(sNext)
                        alpha = 1/(Nxa[s[t],at]+0.01)**0.8
                        tmp = rt + gamma * np.max(Q[sNext])
                        Q[s[t]][at] *= (1-alpha)
                        Q[s[t]][at] += alpha * tmp
                        cumReward += rt
                        t += 1

                Vpi = np.zeros(env.n_states)
                for s in range(env.n_states):
                        Vpi[s] = np.max(Q[s])
                episodes.append(np.max(abs(Vpi - np.array(v_opt))))
                cumulatedReward.append(cumReward)

        # Plotting performance over all states (convergence of Q-Learning)
        plt.figure()
        plt.plot(range(N), episodes)
        plt.xlabel("Episodes", fontsize=16)
        plt.ylabel("Performance over states", fontsize=16)
        plt.show()

        # Plotting cumulated reward for each episode
        plt.figure()
        plt.scatter(range(N), cumulatedReward)
        plt.xlabel("Episodes", fontsize=16)
        plt.ylabel("Cumulated reward for each episode ", fontsize=16)
        plt.show()

        # Plotting cumulated reward over all past episodes
        plt.figure()
        plt.plot(range(N), np.cumsum(cumulatedReward))
        plt.xlabel("Episodes", fontsize=16)
        plt.ylabel("Cumulated reward with episodes", fontsize=16)
        plt.show()

        return Q

def Vpn(env,n):
        Q = QLearning(env, N=n, gamma=0.95, Tmax=20)
        Vpi = np.zeros(env.n_states)
        for s in range(env.n_states):
                Vpi[s] = np.max(Q[s])
        return Vpi


Qdict = QLearning(env, N=20000, Tmax=20)

# Rendering Q-Value in Grid World
Q = np.zeros((env.n_states, 4))
for i in range(env.n_states):
        Q[i,:] = Qdict[i]
gui.render_q(env, Q)
