import math

from setup_and_solvers.gridworld_env_multi_init_states import *
# from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.initial_opacity_gradient_calculation import *

# logger.add("logs_for_examples/log_file_mario_example_information_theoretic_opacity.log")
#
# logger.info("This is the log file for the 6X6 gridworld with goal states 9, 20, 23 test case.")

# Initial set-up for a 6x6 gridworld.
ncols = 6
nrows = 6
target = [5, 35]
# target for testing.
# target = [23]

# secret_goal_states = [2, 20, 34]
reward_states = target
penalty_states = []
obstacles = [13, 17, 21, 29, 33]
unsafe_u = []
# non_init_states = [1, 25, 9, 14, 15, 17, 19, 23]
initial = {0}
initial_dist = dict([])
# considering a single initial state.
for state in range(36):
    if state in initial:
        initial_dist[state] = 1 / len(initial)
    else:
        initial_dist[state] = 0

robot_ts_1 = read_from_file_MDP_old('robotmdp_1.txt')
robot_ts_2 = read_from_file_MDP_old('robotmdp_2.txt')

# sensor setup
sensors = {'A', 'B', 'C', 'D', 'NO'}

setA = {6, 7, 8, 12, 13, 14}
setB = {19, 20, 25, 26, 31, 32}
setC = {3, 4, 9, 10, 15, 16}
setD = {21, 22, 23, 27, 28, 29}
setNO = {0, 1, 2, 5, 11, 17, 18, 24, 30, 33, 34, 35}

# sensor noise
sensor_noise = 0.1

sensor_net = Sensor()
sensor_net.sensors = sensors

sensor_net.set_coverage('A', setA)
sensor_net.set_coverage('B', setB)
sensor_net.set_coverage('C', setC)
sensor_net.set_coverage('D', setD)
# sensor_net.set_coverage('E', setE)
sensor_net.set_coverage('NO', setNO)

# sensor_net.jamming_actions = masking_action
sensor_net.sensor_noise = sensor_noise
# sensor_net.sensor_cost_dict = sensor_cost

agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts_1, target, obstacles, unsafe_u, initial_dist)
agent_gw_1.mdp.get_supp()
agent_gw_1.mdp.gettrans()
agent_gw_1.mdp.get_reward()
agent_gw_1.draw_state_labels()
trans_1 = agent_gw_1.mdp.trans

agent_gw_2 = GridworldGui(initial, nrows, ncols, robot_ts_2, target, obstacles, unsafe_u, initial_dist)
agent_gw_2.mdp.get_supp()
agent_gw_2.mdp.gettrans()
agent_gw_2.mdp.get_reward()
agent_gw_2.draw_state_labels()
trans_2 = agent_gw_1.mdp.trans

# reward/ value matrix for each agent.
value_dict_1 = dict()
for state in agent_gw_1.mdp.states:
    if state == 5:
        value_dict_1[state] = 0.1
    elif state == 35:
        value_dict_1[state] = 0.1
    elif state in penalty_states:
        value_dict_1[state] = -0.1
    else:
        value_dict_1[state] = -0.01

value_dict_2 = dict()
for state in agent_gw_2.mdp.states:
    if state == 5:
        value_dict_2[state] = 0.1
    elif state == 35:
        value_dict_2[state] = 0.1
    elif state in penalty_states:
        value_dict_2[state] = -20
    else:
        value_dict_2[state] = -0.01

side_payment = {}
for state in agent_gw_1.mdp.states:
    s_idx = agent_gw_1.mdp.states.index(state)
    side_payment[state] = {}
    for action in agent_gw_1.mdp.actlist:
        a_idx = agent_gw_1.mdp.actlist.index(action)
        side_payment[s_idx][a_idx] = 0

# E_idx = agent_gw_1.actlist.index('E')
# N_idx = agent_gw_1.actlist.index('N')
# s1_idx = agent_gw_1.states.index('')
# idx1 = 4*len(agent_gw_1.actlist) + E_idx
# idx2 = 11*len(agent_gw_1.actlist) + N_idx
modify_list = [140]


# # TODO: The augmented states still consider the gridcells with obstacles. Try by omitting the obstacle filled states
# #  -> reduces computation.
#
hmm_1 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_1)
hmm_2 = HiddenMarkovModelP2(agent_gw_2.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_2)
hmm_list = [hmm_1, hmm_2]


masking_policy_gradient = InitialOpacityPolicyGradient(hmm_list=hmm_list, ex_num=8, true_type_num=1, iter_num=2000, batch_size=100, V=2000,
                                                       T=12,
                                                       eta=0.5)

iteration_list = range(masking_policy_gradient.iter_num)

with open(f'../Data/x_list_{masking_policy_gradient.ex_num}', 'rb') as file:
    x_list = pickle.load(file)

# print(x_list)

side_payment_list = []
for i in iteration_list:
    side_payment_list.append(x_list[i][modify_list[0]].item())

posterior_collection = []
for type_num in range(masking_policy_gradient.num_of_types):
    posterior_collection.append([])

x_need = x_list[0:2000]
side_payment_need = side_payment_list[0:2000]

for x in x_need[0::10]:
    torch.cuda.empty_cache()
    masking_policy_gradient.x = x
    masking_policy_gradient.update_HMMs()
    # print(masking_policy_gradient.x)
    # print(masking_policy_gradient.theta_torch_list)
    with torch.no_grad():
        # Start with sampling the trajectories.
        state_data, action_data, y_obs_data = masking_policy_gradient.sample_trajectories(masking_policy_gradient.true_type_num)
    # A_matrices_list = []
    # for type_num in range(masking_policy_gradient.num_of_types):
    #     # construct the A matrices.
    #     A_matrices_list.append(
    #         masking_policy_gradient.compute_A_matrices(type_num, y_obs_data[2]))  # Compute for each y_v.
    # for type_num in range(masking_policy_gradient.num_of_types):
    #     P_T_y, gradient_P_T_y, prob_P_y, gradient_P_y = masking_policy_gradient.P_T_g_Y(type_num, A_matrices_list)
    #     # print(P_T_y)
    P_T_y_list = masking_policy_gradient.approximate_posterior(y_obs_data)
    for type_num in range(masking_policy_gradient.num_of_types):
        posterior_collection[type_num].append(P_T_y_list[type_num])
    # P_T_y, nabla_H = masking_policy_gradient.approximate_conditional_entropy_and_gradient_S0_given_Y(y_obs_data)
    print("One iteration Done.")

print(posterior_collection[0][0])
print(posterior_collection[0][-1])
print(posterior_collection[1][0])
print(posterior_collection[1][-1])


plt.scatter(side_payment_need[0::10], posterior_collection[0], color='blue', linestyle='-', label='type 1')
plt.scatter(side_payment_need[0::10], posterior_collection[1], color='red', linestyle='-', label='type 2')
plt.xlabel("Side payment")  # Set xlabel for the second subplot
plt.ylabel(r"The posterior $P_\theta(T|y)$")  # Set ylabel for the second subplot
plt.legend()  # Add legend to the second subplot
plt.grid(True)
plt.savefig(f'../Data/posterior_{masking_policy_gradient.ex_num}.png')
plt.show()
