import matplotlib.pyplot as plt
import numpy as np
import sys

from collections import defaultdict

def mc_prediction(env,agent, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state,_ = env.reset() ############
        for t in range(100):
            action = agent.policy(state) ############
            obs, reward, done, _, info = env.step(action) ############
            episode.append((obs, reward, done)) ############
            if done:
                break
            state = obs

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([x[0] for x in episode]) ############

        for state in states_in_episode:
            # Find the first occurence of the state in the episode
            first_occurence_idx = np.where(np.array([state == x[0] for x in episode]) == True)[0][0] # YOUR CODE HERE #
            # Sum up all rewards since the first occurance
            G = sum([ x[1]*(discount_factor)**i for i,x in enumerate(episode[0:first_occurence_idx+1])]) # YOUR CODE HERE #
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G # YOUR CODE HERE #
            returns_count[state] += 1 # YOUR CODE HERE #
            V[state] = returns_sum[state] /returns_count[state] # YOUR CODE HERE #

    return V

def td_prediction(env, agent, ep, gamma, alpha):
    """TD Prediction

    Params:
        env - environment
        ep - number of episodes to run
        policy - function in form: policy(state) -> action
        gamma - discount factor [0..1]
        alpha - step size (0..1]
    """
    assert 0 < alpha <= 1
    V = defaultdict(float)    # default value 0 for all states

    for _ in range(ep):
        S,_ = env.reset() 
        score =0
        while score<1000:
            A =  agent.policy(S)# YOUR CODE HERE #
            S_, R, done , _, info=  env.step(A)# YOUR CODE HERE #
            V[S] =  V[S] + alpha * (R + gamma*V[S_] - V[S])# YOUR CODE HERE #
            S =  S_# YOUR CODE HERE #
            score = info['score']
            if done: break

    return V


def plot_V(V):
    """Param V is dictionary int[0..7]->float"""
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')

    _x,_y,_z = [],[],[]
    for x,y in V.keys():
        _x.append(x)
        _y.append(y)
        _z.append(V[(x,y)])

    N= len(_x)
    dx = np.ones(N)
    dy = np.ones(N)
    dz = np.arange(N)

    ax1.bar3d(_x, _y, _z, dx, dy, dz, shade=True)
    ax1.set_title('Shaded')
    plt.show()
