import random as rd 
import numpy as np
from collections import defaultdict

class Agent1():
    def __init__(self):
        pass
    
    def policy(self,observation):
        if observation[1] <=0:
            return 0
        else :
            return 1

class Agent2():
    def __init__(self,p):
        self.p = p
    
    def policy(self,observation):
        if observation[1] <0:
            return 0
        elif observation[1] >0:
            return 1
        else:
            x = rd.random()
            return 1 if x >= self.p else 0
        

class GreedyAgent():
    def __init__(self,ValueFunction, height = 15, width = 20):
        self.V = ValueFunction
        self.height = height
        self.width = width
    
    def dummy_policy(self,obs):
        if obs[1] <=0:
            return 0
        else :
            return 1
    
    def get_possible_state(self,V):
        states = V.keys()
        dx,dy = self.state
        up_state = [] 
        down_state = []
        for y in range(-self.height,self.height +1):
                if (dx+1,y) in states:
                    if y < dy:
                        up_state.append(V[(dx+1,y)])
                    elif y > dy:
                        down_state.append(V[(dx+1,y)])
        return np.mean(up_state), np.mean(down_state)

    def policy(self,observation):
        self.state = observation
        if -1<=self.state[1]<=1:
        #print('state',self.state)
            up,down = self.get_possible_state(self.V)
        #print(up,down)
            if up > down:
                return 1
            else:
                return 0
        else:
            return self.dummy_policy(self.state)
            


class QLearningAgent():
    def __init__(self,env,epsilon,alpha,gamma,height = 15, width = 20):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = 2
        self.num_states = height*width
        self.epsilon = epsilon
        self.step_size = alpha
        self.discount = gamma
        self.rand_generator = np.random.RandomState(12)
        
        # Create an array for action-value estimates and initialize it to zero.
        """
        self.q = {}
        for h in range(-height,height+1):
            for w in range(width +1):
                self.q[(w,h)] = np.zeros(2)# The array of action-value estimates."""
        self.q = defaultdict(lambda: np.zeros(env.action_space.n))

        
    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        current_q = self.q[state]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            action = self.argmax(current_q) # greedy action selection
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        # Choose action using epsilon greedy.
        current_q = self.q[state]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        # Perform an update (1 line)
        ### START CODE HERE ###
        self.q[self.prev_state][self.prev_action] += self.step_size*(reward + self.discount*np.max(self.q[state]) - self.q[self.prev_state][ self.prev_action])
        
        ### END CODE HERE ###
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode (1 line)
        ### START CODE HERE ###
        self.q[self.prev_state][self.prev_action] += self.step_size*(reward - self.q[self.prev_state][self.prev_action])
        ### END CODE HERE ###
        
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(q_values.shape[0]):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
    
    def training(self, num_episodes,env):
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")

            obs = env.reset()
            obs = obs[0]

            action = self.agent_start(obs)
            
            done = False
            while not done:
                obs, reward, done, _, info = env.step(action)
                action = self.agent_step(reward,obs)
            
            self.agent_end(reward)


    def policy(self,obs):
        return np.argmax(self.q[obs])
    
    def get_policy(self):
        self.p = dict((k,np.argmax(v)) for k, v in self.q.items())

    def get_valuefunction(self):
        return dict((k,np.max(v)) for k, v in self.q.items())
    

class SarsaAgent():
    def __init__(self,env,epsilon,alpha,gamma,height = 15, width = 20):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = 2
        self.num_states = height*width
        self.epsilon = epsilon
        self.step_size = alpha
        self.discount = gamma
        self.rand_generator = np.random.RandomState(12)
        
        # Create an array for action-value estimates and initialize it to zero.
        """
        self.q = {}
        for h in range(-height,height+1):
            for w in range(width +1):
                self.q[(w,h)] = np.zeros(2)# The array of action-value estimates."""
        self.q = defaultdict(lambda: np.zeros(env.action_space.n))

        
    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        current_q = self.q[state]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            action = self.argmax(current_q) # greedy action selection
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        # Choose action using epsilon greedy.
        current_q = self.q[state]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        # Perform an update (1 line)
        ### START CODE HERE ###
        self.q[self.prev_state][self.prev_action] += self.step_size*(reward + self.discount*self.q[state][action] - self.q[self.prev_state][ self.prev_action])
        
        ### END CODE HERE ###
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode (1 line)
        ### START CODE HERE ###
        self.q[self.prev_state][self.prev_action] += self.step_size*(reward - self.q[self.prev_state][self.prev_action])
        ### END CODE HERE ###
        
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(q_values.shape[0]):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
    
    def training(self, num_episodes,env):
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")

            obs = env.reset()
            obs = obs[0]

            action = self.agent_start(obs)
            
            done = False
            while not done:
                obs, reward, done, _, info = env.step(action)
                action = self.agent_step(reward,obs)
            
            self.agent_end(reward)


    def policy(self,obs):
        return np.argmax(self.q[obs])
    
    def get_policy(self):
        self.p = dict((k,np.argmax(v)) for k, v in self.q.items())

    def get_valuefunction(self):
        return dict((k,np.max(v)) for k, v in self.q.items())

class MCcontrolAgent():
    def __init__(self, env,height = 15, width = 20):
        self.height = height
        self.width = width

    
    @staticmethod
    def generate_episode_from_Q(env, Q, epsilon, nA):
        """ generates an episode from following the epsilon-greedy policy """
        def get_probs(Q_s, epsilon, nA):
            """ obtains the action probabilities corresponding to epsilon-greedy policy """
            policy_s = (epsilon/nA)*np.ones(nA) ## YOUR CODE HERE
            best_a = np.argmax(Q_s)## YOUR CODE HERE
            policy_s[best_a] = epsilon/nA + 1-epsilon ## YOUR CODE HERE
            return policy_s
        episode = []
        state,_ = env.reset()
        while True:
            action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                        if state in Q else env.action_space.sample()
            # take a step in the environement 
            next_state, reward, done, info,_ = env.step(action)## YOUR CODE HERE 
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    
    @staticmethod
    def update_Q(env, episode, Q, alpha, gamma):
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            old_Q = Q ## YOUR CODE HERE
            G = sum([ r*(gamma)**i for i,r in enumerate(rewards[i:])])
            Q[state][actions[i]] = old_Q[state][actions[i]] + alpha*(G -old_Q[state][actions[i]]) ## YOUR CODE HERE
        return Q
    
    def mc_control(self,env, num_episodes, alpha=0.02, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
        nA = env.action_space.n
        # initialize empty dictionary of arrays
        self.Q = defaultdict(lambda: np.zeros(nA))
        epsilon = eps_start
        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            # set the value of epsilon
            epsilon = max(epsilon*eps_decay, eps_min)
            # generate an episode by following epsilon-greedy policy
            episode = self.generate_episode_from_Q(env, self.Q, epsilon, nA)## YOUR CODE HERE -- call the appropriate function
            # update the action-value function estimate using the episode
            self.Q = self.update_Q(env, episode, self.Q, alpha, gamma)## YOUR CODE HERE -- call the appropriate function
        # determine the policy corresponding to the final action-value function estimate
        self.p = dict((k,np.argmax(v)) for k, v in self.Q.items())

    
    def policy(self,obs):
        if obs in self.p.keys():
            return self.p[obs]
        else:
            if obs[1] <=0:
                return 0
            else :
                return 1
