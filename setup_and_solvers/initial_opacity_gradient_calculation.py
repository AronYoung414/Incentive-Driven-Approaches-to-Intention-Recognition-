# import itertools
# import os

import matplotlib.pyplot as plt

from setup_and_solvers.hidden_markov_model_of_P2 import *
import numpy as np
import torch
import time
import torch.nn.functional as F
import itertools
# import gc
import pickle

# from loguru import logger

# torch.manual_seed(0)  # set random seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


class InitialOpacityPolicyGradient:
    def __init__(self, hmm_list, ex_num, iter_num=1000, batch_size=1, V=100, T=10, eta=1):
        for hmm in hmm_list:
            if not isinstance(hmm, HiddenMarkovModelP2):
                raise TypeError("Expected hmm to be an instance of HiddenMarkovModelP2.")

        self.num_of_types = len(hmm_list)
        self.true_type_num = 1
        self.modify_list = hmm_list[0].modify_list
        self.prior = np.ones(self.num_of_types) / self.num_of_types
        self.prior = torch.from_numpy(self.prior).type(dtype=torch.float32)
        self.hmm_list = hmm_list  # Hidden markov model of type 1.
        self.iter_num = iter_num  # number of iterations for gradient ascent
        self.ex_num = ex_num
        self.V = V  # number of sampled trajectories.
        self.batch_size = batch_size  # number of trajectories processed in each batch.
        self.T = T  # length of the sampled trajectory.
        self.eta = eta  # step size for theta.
        # self.kappa = kappa  # step size for lambda.
        # self.epsilon = epsilon  # value threshold.

        # The states and actions of original MDP
        self.states = self.hmm_list[0].states
        self.actions = self.hmm_list[0].actions
        self.num_of_states = len(self.states)
        self.num_of_actions = len(self.actions)
        # About side payment
        self.x_size = self.num_of_states * self.num_of_actions
        self.x = torch.nn.Parameter(
            torch.zeros(self.x_size, dtype=torch.float32, device=device,
                        requires_grad=False))
        self.weight = 0.1

        # Defining optimal theta in pyTorch ways.
        self.theta_torch_list = []
        # Define the list of optimal policies
        # self.policy_list = []
        # Define transition matrix for each type
        self.transition_mat_torch_list = []
        # Define initial distribution for each type
        self.mu_0_torch_list = []
        # Format: [observation_indx, aug_state_indx] = probability
        self.B_torch_list = []
        # Define the transition matrix for each type
        self.T_theta_list = []
        # Get all the lists we need
        self.get_all_lists()

        self.entropy_list = list([])
        self.threshold_list = list([])
        self.iteration_list = list([])
        self.theta_torch_collection = list([])
        self.x_list = list([])

        # Format: [observation_indx, aug_state_indx] = probability
        # self.B_torch_1 = self.construct_B_matrix_torch(self.hmm_1)
        # self.B_torch_2 = self.construct_B_matrix_torch(self.hmm_2)
        # self.B_torch = self.B_torch.to(device)

        # Construct the cost matrix -> Format: [state_indx, masking_act] = cost ## TODO: Change the cost matrix to value matrix.
        # self.value_matrix_1 = self.construct_value_matrix(hmm_1)
        # self.value_matrix_2 = self.construct_value_matrix(hmm_2)

    def get_all_lists(self):
        for type_num in range(self.num_of_types):
            hmm = self.hmm_list[type_num]
            # Construct theta in pyTorch ways.
            self.theta_torch_list.append(torch.from_numpy(hmm.optimal_theta).type(dtype=torch.float32))
            self.theta_torch_list[type_num] = self.theta_torch_list[type_num].to(device)
            self.theta_torch_list[type_num].requires_grad_(True)
            # Construct the list of optimal policy for each type of agent
            # self.policy_list.append(hmm.policy)
            # Construct transition matrix for each type
            self.transition_mat_torch_list.append(torch.from_numpy(hmm.transition_mat).type(dtype=torch.float32))
            self.transition_mat_torch_list[type_num] = self.transition_mat_torch_list[type_num].to(device)
            # Construct initial distribution for each type
            self.mu_0_torch_list.append(torch.from_numpy(hmm.mu_0).type(dtype=torch.float32))
            self.mu_0_torch_list[type_num] = self.mu_0_torch_list[type_num].to(device)
            # Construct the transition matrices
            self.T_theta_list.append(self.construct_transition_matrix_T_theta_torch(type_num))
            # Construct observation matrices
            self.B_torch_list.append(self.construct_B_matrix_torch(type_num))

    def update_the_lists(self):
        self.theta_torch_list = []
        self.T_theta_list = []
        for type_num in range(self.num_of_types):
            hmm = self.hmm_list[type_num]
            # Update the theta list for the future computation
            self.theta_torch_list.append(torch.from_numpy(hmm.optimal_theta).type(dtype=torch.float32))
            self.theta_torch_list[type_num] = self.theta_torch_list[type_num].to(device)
            self.theta_torch_list[type_num].requires_grad_(True)
            # Update the transition matrices which depends on theta
            self.T_theta_list.append(self.construct_transition_matrix_T_theta_torch(type_num))

    def convert_policy(self, policy):
        policy_m = np.zeros(self.x_size)
        i = 0
        for st in self.states:
            for act in self.actions:
                policy_m[i] = policy[st][act]
                i += 1
        return policy_m

    def get_x(self, side_payment):
        x = []
        for s_idx in range(self.num_of_states):
            for a_idx in range(self.num_of_actions):
                x.append(side_payment[s_idx][a_idx])
        x = np.array(x)
        return x

    def construct_value_matrix(self, type_num):
        hmm = self.hmm_list[type_num]
        value_matrix = torch.zeros(len(hmm.states), len(hmm.actions), device=device)
        for s in hmm.value_dict:
            for a in hmm.value_dict[s]:
                value_matrix[s, a] = hmm.value_dict[s][a]
        return value_matrix

    def sample_action_torch(self, state, type_num):
        hmm = self.hmm_list[type_num]
        theta = self.theta_torch_list[type_num]
        # sample's actions given state and theta, following softmax policy.
        state_indx = hmm.states_indx_dict[state]
        # extract logits corresponding to the given state.
        logits = theta[state_indx]
        logits = logits - logits.max()  # logit regularization.

        # compute the softmax probabilities for the actions.
        action_probs = F.softmax(logits, dim=0)

        # sample an action based on the computed probabilities.
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def sample_trajectories(self, type_num):
        hmm = self.hmm_list[type_num]
        # theta = self.theta_torch_list[type_num]
        state_data = np.zeros([self.batch_size, self.T], dtype=np.int32)
        action_data = np.zeros([self.batch_size, self.T], dtype=np.int32)
        y_obs_data = []

        for v in range(self.batch_size):
            y = []
            # # starting from the initial state.
            # state = self.hmm.initial_state

            # starting from the initial state. Choose an initial state from a set of initial states.
            state = random.choice(list(hmm.initial_states))

            # # observation for the initial state. y.append(self.hmm.sample_observation(state))

            act = self.sample_action_torch(state, type_num)
            for t in range(self.T):
                # Obtain the observation and add it to observation data.
                # y.append(self.hmm.sample_observation(state))
                # Use the above when 'Null' and 'NO' are the same. Else use the following.
                y.append(hmm.sample_observation_same_NO_Null(state))
                # Add the corresponding state and action values to state_data and action_data.
                s = hmm.states_indx_dict[state]
                state_data[v, t] = s
                # a = self.hmm.mask_act_indx_dict[act]
                # action_data[v, t] = a
                # Use the above two lines when the action sampler returns the actions itself and not its index.
                # Use the below with self.sample_action_torch as it directly outputs the index.
                action_data[v, t] = act
                # next state sampling given the state and action.
                state = hmm.sample_next_state(state, act)
                # # Obtain the observation.
                # y.append(self.hmm.sample_observation(state))
                # next action sampling given the new state.
                act = self.sample_action_torch(state, type_num)
            y_obs_data.append(y)
        return state_data, action_data, y_obs_data

    def construct_transition_matrix_T_theta_torch(self, type_num):
        transition_mat_torch = self.transition_mat_torch_list[type_num]
        theta_torch = self.theta_torch_list[type_num]
        # Constructing the transtion matrix given the policy pi_\theta.
        # That T_\theta where P_\theta(p, q) = \sum_{\sigma' \in \Sigma} P(q|p, \sigma').pi_\theta(\sigma'|p).
        # T_\theta(i, j) --> from j to i.

        # Apply softmax to logits to obtain the policy probabilities pi_theta.
        logits = theta_torch.clone()
        logits = logits - logits.max()  # logits regularization.

        pi_theta = F.softmax(logits, dim=1)

        # Multiplication and sum over actions for each element of T_theta.
        T_theta = torch.einsum('sa, sna->ns', pi_theta, transition_mat_torch)

        # # Compute T_theta manually for comparison.
        # T_theta_compare = self.T_theta_for_comparison(pi_theta)
        return T_theta

    def construct_B_matrix_torch(self, type_num):
        hmm = self.hmm_list[type_num]
        # Populate the B matrix with emission probabilities.
        # B(i\mid j) = Obs_2(o=i|z_j).
        # Format-- [observation_indx, aug_state_indx] = probability
        B_torch = torch.zeros(len(hmm.observations), len(hmm.states), device=device)
        for state, obs in itertools.product(hmm.states, hmm.observations):
            B_torch[hmm.observations_indx_dict[obs], hmm.states_indx_dict[state]] = \
                hmm.emission_prob[state][obs]
        return B_torch

    def construct_A_matrix_torch(self, type_num, o_t):
        hmm = self.hmm_list[type_num]
        B_torch = self.B_torch_list[type_num]
        T_theta = self.T_theta_list[type_num]
        # Construct the A matrix. A^\theta_{o_t} = T_theta.diag(B_{o_t, 1},...., B_{o_t, N}).
        # o_t is the particular observation.
        # TODO: see if you can save computation by not repeating the computations of A_o_t by saving them!!!!!!!!!!!!!!!

        o_t_index = hmm.observations_indx_dict[o_t]
        B_diag = torch.diag(B_torch[o_t_index, :])

        # Compute A^\theta_{o_t}.
        # A_o_t = torch.matmul(T_theta, B_diag)

        # return A_o_t
        return T_theta @ B_diag

    def compute_A_matrices(self, type_num, y_v):
        # hmm = self.hmm_list[type_num]
        # B_torch = self.B_torch_list[type_num]
        # T_theta = self.T_theta_list[type_num]
        # Construct all of the A_o_t.
        # Outputs a list of all of the A matrices given an observation sequence.
        A_matrices = []  # sequece -> Ao1, Ao2, ..., AoT.
        for o_t in y_v:
            A_o_t = self.construct_A_matrix_torch(type_num, o_t)
            A_matrices.append(A_o_t)
        return A_matrices

    def compute_probability_of_observations_given_type(self, type_num, A_matrices):
        # Computes P_\theta(y|T) = P(o_{1:T}) = 1^T.A^\theta_{o_{T:1}}.\mu_{0,T}
        # Also computes A^\theta_{o_{T-1:1}}.\mu_0 -->  Required in later calculations.
        theta_torch = self.theta_torch_list[type_num]
        # A_matrices is a list of A matrices computed given T_theta and a sequence of observations.

        # # Define one hot vector
        # one_hot_vec = np.zeros(len(hmm.states))  # The vector 1_s0
        # one_hot_vec[s_0] = 1
        # one_hot_vec = torch.from_numpy(one_hot_vec).type(dtype=torch.float32)
        # one_hot_vec = one_hot_vec.to(device)

        result_prob = self.mu_0_torch_list[type_num]  # For P_\theta(y) = P(o_{1:T}) = 1^T.A^\theta_{o_{T:1}}.\mu_0
        # p_y_s0 = one_hot_vec  # For P_\theta(y|s_0) = 1^T.A^\theta_{o_{T:1}}.1_s0
        # resultant_matrix = self.mu_0_torch  # For A^\theta_{o_{T-1:1}}.\mu_0 -->  Required in later calculations.

        # Define a counter to stop the multiplication at T-1 for one of the results and T for the other.
        # counter = len(A_matrices)
        # sequentially multiply with A matrices.
        for A in A_matrices:
            result_prob = torch.matmul(A, result_prob)
            # p_y_s0 = torch.matmul(A, p_y_s0)

        # Multiplying with 1^T is nothing but summing up. Hence, we do the following.
        result_prob_P_y_g_T = result_prob.sum()
        # result_prob_P_y_s0 = p_y_s0.sum()

        # resultant_matrix_prob_y_one_less = resultant_matrix.sum()
        # Compute the gradient later by simply using result_prob_to_return.backward() --> This uses autograd to
        # compute gradient.
        result_prob_P_y_g_T.backward(retain_graph=True)  # Gradient of P_\theta(y).
        gradient_P_y_g_T = theta_torch.grad.clone()

        # result_prob_P_y_s0.backward(retain_graph=True)  # Gradient of P_\theta(y|s0).
        # gradient_P_y_s0 = self.theta_torch.grad.clone()

        # resultant_matrix_prob_y_one_less.backward(retain_graph=True)  # Gradient of P_\theta(O_{1:T-1}).
        # gradient_P_y_one_less = self.theta_torch.grad.clone()

        # clearing .grad for the next gradient computation.
        theta_torch.grad.zero_()

        return result_prob_P_y_g_T, gradient_P_y_g_T
        # return resultant_matrix_prob_y_one_less, resultant_matrix, gradient_P_y_one_less

    def compute_probability_of_observations(self, A_matrices_list):
        """
        A_matrices_list: The list of A matrix for each type of agents
        """
        result_prob_P_y = 0
        gradient_P_y = torch.zeros([self.num_of_states, self.num_of_actions],
                                   device=device)
        for type_num in range(self.num_of_types):
            result_prob_P_y_g_T, gradient_P_y_g_T = self.compute_probability_of_observations_given_type(
                type_num, A_matrices_list[type_num])
            result_prob_P_y = result_prob_P_y + result_prob_P_y_g_T * self.prior[type_num]
            gradient_P_y = gradient_P_y_g_T + self.prior[type_num] * gradient_P_y_g_T
        return result_prob_P_y, gradient_P_y

    def P_T_g_Y(self, type_num, A_matrices_list):
        # Computes P_\theta(s_0|y) = P_\theta(y|s_0) \mu_0(s_0) / P_\theta(y)
        prob_P_y_T, gradient_P_y_T = self.compute_probability_of_observations_given_type(type_num,
                                                                                         A_matrices_list[type_num])
        prob_P_y, gradient_P_y = self.compute_probability_of_observations(A_matrices_list)
        P_T_y = prob_P_y_T * self.prior[type_num] / prob_P_y
        gradient_P_T_y = ((self.prior[type_num] / prob_P_y) * gradient_P_y_T -
                          (self.prior[type_num] * prob_P_y_T / prob_P_y ** 2) * gradient_P_y)
        return P_T_y, gradient_P_T_y, prob_P_y, gradient_P_y
        # return resultant_matrix_prob_y_one_less, resultant_matrix, gradient_P_y_one_less

    def approximate_conditional_entropy_and_gradient_S0_given_Y(self, y_obs_data):
        # Computes the conditional entropy H(S_0 | Y; \theta); AND the gradient of conditional entropy \nabla_theta
        # H(S_0|Y; \theta).

        H = torch.tensor(0, dtype=torch.float32, device=device)
        nabla_H = torch.zeros([self.num_of_types, self.num_of_states, self.num_of_actions],
                              device=device)

        for v in range(self.batch_size):
            y_v = y_obs_data[v]

            A_matrices_list = []
            for type_num in range(self.num_of_types):
                # construct the A matrices.
                A_matrices_list.append(
                    self.compute_A_matrices(type_num, y_v))  # Compute for each y_v.

            for type_num in range(self.num_of_types):
                # values for the term w_T = 1.
                P_T_y, gradient_P_T_y, result_P_y, gradient_P_y = self.P_T_g_Y(type_num, A_matrices_list)

                # to prevent numerical issues, clamp the values of p_theta_w_t_g_yv_1 between 0 and 1.
                P_T_y = torch.clamp(P_T_y, min=0.0, max=1.0)

                if P_T_y != 0:
                    log2_P_T_y = torch.log2(P_T_y)
                else:
                    log2_P_T_y = torch.zeros_like(P_T_y, device=device)

                # Calculate the term P_\theta(s_0|y) * \log P_\theta(s_0|y).
                term_p_logp = P_T_y * log2_P_T_y

                # Computing the gradient for w_T = 1. term for gradient term w_T = 1. Computed as [log_2 P_\theta(
                # w_T|y_v) \nabla_\theta P_\theta(w_T|y_v) + P_\theta(w_T|y_v) log_2 P_\theta(w_T|y_v) (\nabla_\theta
                # P_\theta(y))/P_\theta(y) + (\nabla_\theta P_\theta(w_T|y_v))/log2]
                gradient_term = (log2_P_T_y * gradient_P_T_y) + (
                        P_T_y * log2_P_T_y * gradient_P_y / result_P_y) + (
                                        gradient_P_T_y / 0.301029995664)  # 0.301029995664 = log2

                H = H + term_p_logp

                nabla_H[type_num, :, :] = nabla_H[type_num, :, :] + gradient_term

        H = H / self.batch_size
        # H.backward()
        # test_nabla_H = self.theta_torch.grad.clone()
        nabla_H = nabla_H / self.batch_size

        return -H, -nabla_H

    def dtheta_T_dx(self, type_num):
        # returns a NM X NM matrix, (i, j) is dtheta_i/dx_j
        grad = np.zeros((self.x_size, self.x_size))
        for m in self.modify_list:
            grad_l = self.dtheta_T_dx_line(m, type_num)
            grad[:, m] = grad_l
        return grad

    def dtheta_T_dx_line(self, index, type_num, epsilon=0.0001):
        hmm = self.hmm_list[type_num]
        policy_m = self.convert_policy(hmm.policy)
        # dtheta_dx(s, a) = dtheta_dr(s, a) * dr(s, a)_dx(s, a), dr(s, a)_dx(s, a) = 1
        # what we realize is dtheta_dr(s, a) here
        # dtheta_dx_line returns one column in the dtheta_dx matrix
        dtheta = np.zeros(self.x_size)
        r_indicator = np.zeros(self.x_size)
        r_indicator[index] = 1
        dtheta_old = dtheta.copy()
        delta = np.inf
        itcount_d = 0
        while delta > epsilon:
            # print(f"{itcount_d} iterations")
            # print(self.policy_m)
            # print(self.policy_m * dtheta)
            dtheta = r_indicator + hmm.agent_mdp.disc_factor * self.construct_P(type_num).dot(policy_m * dtheta)
            delta = np.max(abs(dtheta - dtheta_old))
            dtheta_old = dtheta
            itcount_d += 1
        # dtheta_ = self.mdp.theta_evaluation(r_indicator, self.policy)
        # print("x is", self.x)
        # print("Matrix_result:", dtheta)
        # print("Evaluation result:", dtheta_)
        return dtheta

    def construct_P(self, type_num):
        hmm = self.hmm_list[type_num]
        P = np.zeros((self.x_size, self.x_size))
        for i in range(self.num_of_states):
            for j in range(self.num_of_actions):
                for next_index, pro in hmm.transition_dict[i][hmm.actions[j]].items():
                    if hmm.states[next_index] != 'Sink':
                        # next_index = self.mdp.states.index(next_st)
                        P[i * self.num_of_actions + j][
                        next_index * self.num_of_actions: (next_index + 1) * self.num_of_actions] = pro
        return P

    def dtheta_dx(self):
        grads = []
        for type_num in range(0, self.num_of_types):
            grad_T = self.dtheta_T_dx(type_num)
            for m in self.modify_list:
                grad_T[m] = 0
            grads.append(grad_T)
        return np.vstack(grads)

    def dh_dx(self):
        # Initialize gradient array
        grad = np.zeros(self.x_size)

        # Update gradient for each index in modify_list
        for m in self.modify_list:
            if self.x[m] >= 0:
                grad[m] = self.weight  # Positive values, update gradient
            else:
                grad[m] = -self.weight  # Negative values, update gradient

        # Convert to torch tensor and move to the correct device
        grad_tensor = torch.from_numpy(grad).type(dtype=torch.float32)  # No in-place modification
        grad_tensor = grad_tensor.to(device)  # Move to the specified device

        return grad_tensor

    def total_derivative(self, y_obs_data):
        H, nabla_H = self.approximate_conditional_entropy_and_gradient_S0_given_Y(y_obs_data)
        nabla_H = nabla_H.reshape(-1)
        nabla_H = nabla_H.unsqueeze(0)
        nabla_Q = self.dtheta_dx()
        nabla_Q = torch.from_numpy(nabla_Q).type(dtype=torch.float32)
        nabla_Q = nabla_Q.to(device)
        # nonzero_indices = nabla_Q.nonzero(as_tuple=False)
        # print("Nonzero elements:", nabla_Q[nonzero_indices].tolist())
        product = torch.matmul(nabla_H, nabla_Q)
        product = product + self.dh_dx()
        # nonzero_indices = product.nonzero(as_tuple=False)
        # print("Nonzero elements:", product[nonzero_indices].tolist())
        return H, product

    def get_side_payment(self, x):
        side_payment = {}
        # Get the device of x
        device = x.device
        # Create the mask on the same device
        mask = ~torch.isin(
            torch.arange(len(x), device=device),
            torch.tensor(self.modify_list, device=device)
        )
        # Use where with tensors on the same device
        x = torch.where(mask, torch.zeros_like(x), x)
        idx = 0
        for state in self.states:
            s_idx = self.states.index(state)
            side_payment[state] = {}
            for action in self.actions:
                a_idx = self.actions.index(action)
                side_payment[s_idx][a_idx] = x[idx].item()
                idx += 1
        return side_payment

    def update_HMMs(self):
        for type_num in range(self.num_of_types):
            # Update the side payment based on updated x
            self.hmm_list[type_num].side_payment = self.get_side_payment(self.x)
            # Update the reward function
            self.hmm_list[type_num].get_value_dict()
            # Update the optimal value function and the optimal policy
            self.hmm_list[type_num].optimal_V, self.hmm_list[type_num].policy = self.hmm_list[
                type_num].get_policy_entropy(tau=0.1)
            policy_1 = self.hmm_list[0].policy
            policy_2 = self.hmm_list[1].policy
            # Update the
            self.hmm_list[type_num].optimal_theta = self.hmm_list[type_num].get_optimal_theta(
                self.hmm_list[type_num].optimal_V)
            self.update_the_lists()

    def solver(self):
        # Solve using policy gradient for initial-state opacity enforcement.
        for i in range(self.iter_num):
            start = time.time()
            torch.cuda.empty_cache()

            approximate_cond_entropy = 0
            grad = 0
            # grad_V_comparison_total = 0
            # approximate_value_total = 0

            trajectory_iter = int(self.V / self.batch_size)
            # self.kappa = self.kappa / (i + 1)
            # self.eta = self.eta / (i + 1)
            self.update_HMMs()
            self.theta_torch_collection.append(self.theta_torch_list)
            self.x_list.append(self.x)

            for j in range(trajectory_iter):
                torch.cuda.empty_cache()

                with torch.no_grad():
                    # Start with sampling the trajectories.
                    state_data, action_data, y_obs_data = self.sample_trajectories(self.true_type_num)

                # Gradient ascent algorithm.

                # # Construct the matrix T_theta.
                # T_theta = self.construct_transition_matrix_T_theta_torch(type_num)
                # Compute approximate conditional entropy and approximate gradient of entropy.
                approximate_cond_entropy_new, grad_new = self.total_derivative(y_obs_data)
                approximate_cond_entropy = approximate_cond_entropy + approximate_cond_entropy_new.item()

                # self.theta_torch = torch.nn.Parameter(self.theta_torch.detach().clone(), requires_grad=True)

                grad = grad + grad_new
                # SGD gradients.
                # grad_V = self.compute_policy_gradient_for_value_function(state_data, action_data, 1)

                # Compare the above value with traditional function. #TODO: comment the next line if you only want entropy term.
                # grad_V_comparison, approximate_value = self.nabla_value_function(state_data, action_data, 1)

                # approximate_value_total = approximate_value_total + approximate_value
                # grad_V_comparison_total = grad_V_comparison_total + grad_V_comparison

                # self.theta_torch = torch.nn.Parameter(self.theta_torch.detach().clone(), requires_grad=True)

                # Computing gradient of Lagrangian with grad_H and grad_V.
                # grad_L = grad_H + self.lambda_mul * grad_V
            print("The approximate entropy is", approximate_cond_entropy / trajectory_iter)
            self.entropy_list.append(approximate_cond_entropy / trajectory_iter)

            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

            # grad_L = (grad_H / trajectory_iter)
            # Use the above line for only the entropy term.
            grad = (grad / trajectory_iter)
            # print("The gradient of entropy", grad / trajectory_iter)
            # print("The gradient of value", grad_V_comparison_total / trajectory_iter)

            # print("The approximate value is", approximate_value_total / trajectory_iter)
            # self.threshold_list.append(approximate_value_total / trajectory_iter)

            # SGD updates.
            # Update theta_torch under the no_grad() to ensure that it remains as the 'leaf node.'
            with torch.no_grad():
                self.x = self.x - self.eta * grad
                # print(grad)

            # self.lambda_mul = (self.lambda_mul - self.kappa *
            #                    ((approximate_value_total / trajectory_iter) - self.epsilon))
            #
            # self.lambda_mul = torch.clamp(self.lambda_mul,
            #                               min=0.0)  # Clamping lambda values to be greater than or equal to 0.

            # re-initialize self.x to ensure it tracks the new set of computations.
            self.x = torch.nn.Parameter(self.x[0].detach().clone(), requires_grad=False)
            # print('The side payment is', self.x)

            end = time.time()
            print("Time for the iteration", i, ":", end - start, "s.")
            print("#" * 100)

        self.iteration_list = range(self.iter_num)

        # Saving the results for plotting later.
        with open(f'../Data/entropy_values_{self.ex_num}.pkl', 'wb') as file:
            pickle.dump(self.entropy_list, file)

        with open(f'../Data/value_function_list_{self.ex_num}', 'wb') as file:
            pickle.dump(self.threshold_list, file)

        with open(f'../Data/x_list_{self.ex_num}', 'wb') as file:
            pickle.dump(self.x_list, file)

        with open(f'../Data/theta_collection_{self.ex_num}', 'wb') as file:
            pickle.dump(self.theta_torch_collection, file)

        figure, axis = plt.subplots(2, 1)

        axis[0].plot(self.iteration_list, self.entropy_list, label='Entropy')
        axis[1].plot(self.iteration_list, self.threshold_list, label='Estimated Cost')
        plt.xlabel("Iteration number")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../Data/graph_{self.ex_num}.png')
        plt.show()

        return
