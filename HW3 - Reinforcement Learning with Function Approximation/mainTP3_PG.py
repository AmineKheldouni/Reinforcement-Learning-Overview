import numpy as np
import lqg1d
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
import copy
import pylab

# Steppers

class ConstantStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gt):
        return self.learning_rate * gt

class StochasticStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update(self, gt, niter):
        return self.learning_rate/(1+niter)**(0.6) * gt

class AdagradStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update(self, gt, gsquare, eps=10e-10):
        return self.learning_rate / np.sqrt(gsquare + eps) * gt


#####################################################
# Define the environment and the policy
#####################################################
env = lqg1d.LQG1D(initial_state_type='random')
class Policy:
    def __init__(self, theta):
        self.sigma = 0.4
        self.theta = theta

    def value(self, s, a):
        return 1./(self.sigma*np.sqrt(2*np.pi)) * np.exp(-(a-s*self.theta)**2/(2*self.sigma**2))

    def draw_action(self, state):
        return np.clip(np.random.normal(self.theta*state, self.sigma**2), -40, 40)

def grad_log_policy(state, action, theta, sigma=0.4):
    return (action-state*theta) * state / (sigma*sigma)

#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 50
# Each trajectory will have at most T time steps
T = 106
# Number of policy parameters updates
n_itr = 500
# Number of simulations
N_simu = 5
# Set the discount factor for the problem
discount = 0.9

beta = 1.
bins = 100

constantStepper = ConstantStep(0.0001)
stochasticStepper = StochasticStep(0.001)
adagradStepper = AdagradStep(0.001)

def compute_bonus(states, actions, N):
    global beta, bins
    grid_bins = utils.discretization_2d(states.reshape(-1), actions.reshape(-1), binx=bins, biny=bins)
    for b in grid_bins:
        N[b] += 1
    return beta/np.sqrt(N[grid_bins])

def REINFORCE(hasBonus, stepper='constant'):
    parameters_simu = []
    returns_simu = []

    low_parameters = []
    up_parameters = []

    for n_simu in tqdm(range(N_simu), desc="Simulating REINFORCE algorithm"):
        N_tot = np.array([1 for i in range((bins+2) ** 2)])
        policy = Policy(-0.4)
        mean_parameters = []
        avg_return = []
        for ni in range(n_itr):
            paths = utils.collect_episodes(env, policy=policy, horizon=T, n_episodes=N)
            grad = 0.
            Gsquare = 0
            R = 0
            for path in paths:
                if hasBonus:
                    bonus = compute_bonus(path['states'], path['actions'], N_tot)
                for t in range(0, T):
                    if hasBonus:
                        vt = np.sum(np.array([discount**(k-1) for k in range(1,T-t+1)]) * (path['rewards'][t:]+bonus[t:]))
                    else:
                        vt = np.sum(np.array([discount**(k-1) for k in range(1,T-t+1)]) * path['rewards'][t:])
                    G = grad_log_policy(path['states'][t][0], path['actions'][t], policy.theta)
                    Gsquare += np.square(G)
                    grad += G * vt
                    R += vt

            # Choosing stepper
            if stepper == 'constant':
                update = constantStepper.update(grad / N)
            elif stepper == 'adagrad':
                update = adagradStepper.update(grad / N, Gsquare / N**2)
            else:
                update = stochasticStepper.update(grad / N, ni)
            # Performing iteration update on parameters
            policy.theta = policy.theta + update
            # print(policy.theta)
            avg_return.append(R/N)
            mean_parameters.append(policy.theta)
        parameters_simu.append(np.array(mean_parameters))
        returns_simu.append(np.array(avg_return))

    return np.array(parameters_simu), np.array(returns_simu)



def plot_performance(parameters_simu, returns_simu):
    upper_parameters,  lower_parameters = np.percentile(parameters_simu, [5, 95], axis=0)
    mean_parameters = np.mean(parameters_simu, axis=0)
    avg_return = np.mean(returns_simu, axis=0)

    fig, ax = pylab.subplots(1)
    ax.plot(range(n_itr), avg_return, lw=2, label='Average returns')
    plt.ylabel('Returns', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

    fig, ax = plt.subplots(1)
    ax.plot(range(n_itr), mean_parameters, lw=2, label='Mean parameters')
    ax.fill_between(range(n_itr), lower_parameters, upper_parameters, facecolor='blue', alpha=0.4, label = "Confidence interval (95%)")
    plt.ylabel('Parameters', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

parameters_simu, returns_simu = REINFORCE(hasBonus = True)

plot_performance(parameters_simu, returns_simu)
