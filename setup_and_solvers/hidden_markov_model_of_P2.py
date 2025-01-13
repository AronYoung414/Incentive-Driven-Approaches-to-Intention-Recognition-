from setup_and_solvers.markov_decision_process import *
from Examples.sensors import *
import itertools
from collections import defaultdict
import random


class HiddenMarkovModelP2:
    def __init__(self, agent_mdp, sensors, state_obs2=dict([]), value_dict=dict([]), secret_goal_states=list([])):
        if not isinstance(agent_mdp, MDP):
            raise TypeError("Expected agent_mdp to be an instance of MDP.")

        if not isinstance(sensors, Sensor):
            raise TypeError("Expected sensors to be an instance of Sensor.")

        self.agent_mdp = agent_mdp
        self.sensors = sensors

        self.augmented_states = list()  # The augmented states space S x \Sigma # TODO: Check if this should be a set
        self.augmented_states = self.agent_mdp.states
        # instead? I have made it a list here to ensure that we can have a transition matrix defined appropriately.
        # self.get_augmented_states()

        self.augmented_states_indx_dict = dict()
        indx_num = 0
        for aug_st in self.augmented_states:
            self.augmented_states_indx_dict[aug_st] = indx_num
            indx_num += 1

        self.actions = self.agent_mdp.actlist  # The actions of the agent MDP.
        self.actions_indx_dict = dict()
        indx_num = 0
        for act in self.actions:
            self.actions_indx_dict[act] = indx_num
            indx_num += 1

        self.act_indx_reverse_dict = {v: k for k, v in self.actions_indx_dict.items()}

        # self.transition_dict = defaultdict(
        #     lambda: defaultdict(dict))  # The transition dictionary for the augmented state-space.

        self.transition_dict = self.agent_mdp.trans
        # self.get_transition_dict()

        # self.transition_mat = {a: np.zeros((len(self.augmented_states), len(self.augmented_states))) for a in
        # self.masking_acts}
        self.transition_mat = np.zeros(
            (len(self.augmented_states), len(self.augmented_states), len(self.agent_mdp.actlist)))

        self.get_transition_mat()

        self.state_obs2 = state_obs2  # state_obs2 is the state observation dict. state_obs2[aug_states]={sensors
        # that cover state}
        self.get_state_obs2()
        self.observations = set(
            self.sensors.sensors)  # TODO: This needs to be changed for generalization if the observation function
        # TODO: changes!!!
        self.observations.add('0')  # '0' represents the null observation.
        self.observations_indx_dict = dict()  # Defining a dictionary with [obs]=indx
        indx_num = 0
        for ob in self.observations:
            self.observations_indx_dict[ob] = indx_num
            indx_num += 1

        self.obs_noise = self.sensors.sensor_noise

        self.emission_prob = defaultdict(
            lambda: defaultdict(dict))  # The emission probability for the observations. emission_prob[
        # aug_state][obs]=probability
        self.get_emission_prob()

        self.initial_dist = dict([])  # The initial distribution of the augmented state-space. initial_dist[augstate]=probability
        # initial distribution array.
        self.mu_0 = np.zeros(len(self.augmented_states))

        self.get_initial_dist()

        # # sampled initial state
        # self.initial_state = None
        # self.get_initial_state()

        # set of initial states.
        self.initial_states = set()
        self.get_initial_states()

        # set the value dictionary.
        self.value_dict_input = value_dict # The value dictionary for the augmented state-space i.e., only state dependednt.
        self.value_dict = defaultdict(lambda: defaultdict(dict))  # The format is [aug_st_indx][mask_act_indx]=value.
        self.get_value_dict()

        self.secret_goal_states = secret_goal_states  # The secret goal states.
        # self.get_secret_goal_states(secret_goal_states)

    def get_value_dict(self):
        # Assign cost/reward/value.
        for state in self.augmented_states:
            for act in self.actions:
                self.value_dict[self.augmented_states_indx_dict[state]][self.actions_indx_dict[act]] = self.value_dict_input[state]

        return

    # def get_secret_goal_states(self, secret_goal_states):
    #     # Construct a list of augmented secret states.
    #     for state in self.augmented_states:
    #         if state[0] in secret_goal_states:
    #             self.secret_goal_states.append(state)
    #     return

    # def get_transition_dict(self): # The transition dict is the transition function such that transition_dict[
    # state][mask][next_state]=probability. for state, mask in itertools.product(self.augmented_states,
    # self.masking_acts): # TODO: Consider only the post states to populate the trans dict.

    # def get_transition_dict(self):
    #     # The transition_dict is the transition function such that transition_dict[state][mask][next_state]=probability.
    #     for state, act, next_state in itertools.product(self.augmented_states, self.augmented_states):
    #         if next_state[1] == mask:
    #             self.transition_dict[state][mask][next_state] = self.get_transition_probability(state[0], next_state[
    #                 0])
    #         else:
    #             self.transition_dict[state][mask][next_state] = 0.0
    #     return



    # def get_transition_dict(self):
    #     # The transition_dict is the transition function such that transition_dict[state][mask][next_state]=probability.
    #     for state, act, next_state in itertools.product(self.augmented_states, self.augmented_states):
    #         if next_state[1] == mask:
    #             self.transition_dict[state][mask][next_state] = self.get_transition_probability(state[0], next_state[
    #                 0])
    #         else:
    #             self.transition_dict[state][mask][next_state] = 0.0
    #     return

    def get_transition_mat(self):
        # # The matrix representation of the transition function. transition_mat[i, j, action] = probability.
        # for state, next_state in itertools.product(self.augmented_states, self.augmented_states):
        #     self.transition_mat[self.augmented_states_indx_dict[state], self.augmented_states_indx_dict[next_state]] = \
        #     self.transition_dict[state][next_state]

        # The matrix representation of the transition function. transition_mat[i, j, action] = probability.
        for state, next_state, action in itertools.product(self.augmented_states, self.augmented_states,
                                                           self.agent_mdp.actlist):
            self.transition_mat[
                self.augmented_states_indx_dict[state], self.augmented_states_indx_dict[next_state], self.agent_mdp.actlist.index(action)] = \
                self.transition_dict[state][action][next_state]

        return

    # def get_transition_probability(self, state, act, next_state):
    #     # This is computed as \sum_{a\in A} P(s,a,s') \pi_G(a\mid s)
    #     # TODO: Check if the transition probability is correct.
    #     probability = self.agent_mdp.P(state, act, next_state)
    #     # for a in self.agent_mdp.actlist:
    #     #     # probability = probability + (self.agent_mdp.P(state, a, next_state) * self.goal_policy[(state, a)])
    #
    #     return float(probability)

    def get_emission_prob(self):
        # In the emission function for each state, and observation pairs.
        for state in self.augmented_states:
            for obs in self.observations:
                self.emission_prob[state][obs] = self.get_emission_probability(state, obs)
        return

    def get_emission_probability(self, state,
                                 obs):  # Check if the following is correct! In the sense, what happens to the
        # probabilities on masking?!
        # Here, I'm considering that when masked, null observation is received with probability 1.
        if state in self.sensors.coverage['NO']:
            if obs == '0':
                return 1
            else:
                return 0
        else:
            if obs == '0':
                return self.obs_noise
            elif obs in self.state_obs2[state]:
                return 1 - self.obs_noise
            else:
                return 0

    def get_initial_dist(self):
        # Each augmented state, have an initial distribution. Consider initial mask to be the first sensor. # TODO: Change this to no masking action.
        for state in self.augmented_states:
            self.initial_dist[state] = self.agent_mdp.initial_distribution[state]
            self.mu_0[self.augmented_states_indx_dict[state]] = self.initial_dist[state]

        return

    def get_state_obs2(self):
        for state in self.augmented_states:
            obs = set([])
            for sensors in self.sensors.sensors:
                if state in self.sensors.coverage[sensors]:
                    obs.add(sensors)
            self.state_obs2[state] = obs

        return

    def get_initial_state(self):
        self.initial_state = random.choices(list(self.initial_dist.keys()), weights=list(self.initial_dist.values()))[0]
        return

    def get_initial_states(self):
        # Obtain the set of initial states.
        for state in self.augmented_states:
            if self.initial_dist[state] > 0:
                self.initial_states.add(state)
        return

    # the following is the sample_observation for different 'NO' and 'Null'.
    def sample_observation(self, state):
        # Given an augmented state it gives a sample observation - true observation or null observation. TODO: Check
        #  if this is correct-- I am considering that whenever the robot is not under a sensor, it only gets the true
        #  information that it is not under any sensor. TODO: Check if that has to be changed to include null
        #   observations for it as well.
        obs_list = list(self.state_obs2[state])
        if len(obs_list) == 0:  # To return null observation with prob. 1 when masked.
            obs_list.append('0')
            return random.choices(obs_list)[0]
        elif state not in self.sensors.coverage[
            'NO']:  # When not masked and under a sensor, probabilistic observation.
            obs_list.append('0')
            return random.choices(obs_list, weights=[1 - self.obs_noise, self.obs_noise])[0]
        else:  # When not under a sensor, return 'NO' with prob. 1.
            return random.choices(obs_list)[0]

    # The following is the sample_observation for SAME 'NO' and 'Null'.
    def sample_observation_same_NO_Null(self, state):
        # Given an augmented state it gives a sample observation - true observation or null observation. TODO: Check
        #  if this is correct-- I am considering that whenever the robot is not under a sensor, it only gets null.

        obs_list = list(self.state_obs2[state])
        if len(obs_list) == 0:  # To return null observation with prob. 1 when masked.
            obs_list.append('0')
            return random.choices(obs_list)[0]
        elif state not in self.sensors.coverage[
            'NO']:  # When not masked and under a sensor, probabilistic observation.
            obs_list.append('0')
            return random.choices(obs_list, weights=[1 - self.obs_noise, self.obs_noise])[0]
        else:  # When not under a sensor, return '0' with prob. 1.
            obs_new_list = list()
            obs_new_list.append('0')
            return random.choices(obs_new_list)[0]

    def sample_next_state(self, state, act):
        # Given an augmented state, a action, the function returns a sampled next state.
        # next_states_supp = list(self.transition_dict[state][act].keys())
        next_states_supp = list(self.transition_dict[state][self.act_indx_reverse_dict[act]].keys())
        next_states_prob = [self.transition_dict[state][self.act_indx_reverse_dict[act]][next_state] for next_state in next_states_supp]
        return random.choices(next_states_supp, weights=next_states_prob)[0]
