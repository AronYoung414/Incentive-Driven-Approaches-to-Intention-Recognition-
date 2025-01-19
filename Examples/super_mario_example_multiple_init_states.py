import math

from setup_and_solvers.gridworld_env_multi_init_states import *
# from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_entropy_calculations import *

logger.add("logs_for_examples/log_file_mario_example_information_theoretic_opacity.log")

logger.info("This is the log file for the 6X6 gridworld with goal states 9, 20, 23 test case.")

# Initial set-up for a 6x6 gridworld.
ncols = 6
nrows = 6
target = [9, 16, 23]
# target for testing.
# target = [23]

secret_goal_states = [2, 20, 34]
reward_states = [14, 22]

obstacles = [17, 19]
unsafe_u = [7, 15, 25]
non_init_states = [1, 25, 9, 14, 15, 17, 19, 23, 35]
initial = {30, 24, 18, 12, 6, 0}
# initial = {30, 12}
initial_dist = dict([])
# considering a single initial state.
for state in range(36):
    if state in initial:
        initial_dist[state] = 1/len(initial)
    else:
        initial_dist[state] = 0

robot_ts = read_from_file_MDP_old('robotmdp_1.txt')

# sensor setup
sensors = {'A', 'B', 'C', 'D', 'E', 'NO'}

setA = {9}
setB = {21, 22, 27, 28}
setC = {23, 29}
setD = {7, 8, 13, 14}
setE = {20}
setNO = {0, 1, 2, 3, 4, 6, 10, 12, 15, 16, 17, 18, 19, 24, 25, 30, 31, 32, 5, 11, 26, 33, 34, 35}

# sensor noise
sensor_noise = 0.3

sensor_net = Sensor()
sensor_net.sensors = sensors

sensor_net.set_coverage('A', setA)
sensor_net.set_coverage('B', setB)
sensor_net.set_coverage('C', setC)
sensor_net.set_coverage('D', setD)
sensor_net.set_coverage('E', setE)
sensor_net.set_coverage('NO', setNO)

# sensor_net.jamming_actions = masking_action
sensor_net.sensor_noise = sensor_noise
# sensor_net.sensor_cost_dict = sensor_cost

agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts, target, obstacles, unsafe_u, initial_dist)
agent_gw_1.mdp.get_supp()
agent_gw_1.mdp.gettrans()
agent_gw_1.mdp.get_reward()
agent_gw_1.draw_state_labels()

# reward/ value matrix for the agent.
value_dict = dict()
for state in agent_gw_1.mdp.states:
    if state in reward_states:
        value_dict[state] = 1
    else:
        value_dict[state] = 0


# TODO: The augmented states still consider the gridcells with obstacles. Try by omitting the obstacle filled states
#  -> reduces computation.

hmm_p2 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, value_dict=value_dict, secret_goal_states=secret_goal_states)

# masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1, epsilon=threshold)
# masking_policy_gradient.solver()

masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, ex_num=7, iter_num=5000, batch_size=100, V=2000, T=12,
                                                       eta=0.8,
                                                       kappa=0.05,
                                                       epsilon=0.3)
# ex3: iter_num=3000, batch_size=100, V=1000, T=12, eta=2, kappa=0.5, epsilon=0.3
# ex5: iter_num=3000, batch_size=100, V=2000, T=12, eta=1, kappa=0.2, epsilon=0.7
masking_policy_gradient.solver()
