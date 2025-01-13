import itertools

import matplotlib.pyplot as plt

from hidden_markov_model_of_P2 import *
import numpy as np
import torch
import time
import torch.nn.functional as F
import itertools
import torch.multiprocessing as mp

import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


class PrimalDualPolicyGradientTest:
    def __init__(self, hmm, iter_num=1000, batch_size=1, V=100, T=10, eta=1, kappa=0.1, epsilon=0):
        if not isinstance(hmm, HiddenMarkovModelP2):
            raise TypeError("Expected hmm to be an instance of HiddenMarkovModelP2.")

        self.hmm = hmm  # Hidden markov model of P2.
        self.iter_num = iter_num  # number of iterations for gradient ascent
        self.V = V  # number of sampled trajectories.
        self.batch_size = batch_size  # number of trajectories processed in each batch.
        self.T = T  # length of the sampled trajectory.
        self.eta = eta  # step size for theta.
        self.kappa = kappa  # step size for lambda.
        self.epsilon = epsilon  # cost threshold for masking.

        self.num_of_aug_states = len(self.hmm.augmented_states)
        self.num_of_maskin_actions = len(self.hmm.masking_acts)

        # Defining theta in pyTorch ways.
        self.theta_torch = torch.nn.Parameter(
            torch.randn(self.num_of_aug_states, self.num_of_maskin_actions, dtype=torch.float32, device=device,
                        requires_grad=True))

        self.transition_mat_torch = torch.from_numpy(self.hmm.transition_mat).type(dtype=torch.float32)
        self.transition_mat_torch = self.transition_mat_torch.to(device)

        self.mu_0_torch = torch.from_numpy(self.hmm.mu_0).type(dtype=torch.float32)
        self.mu_0_torch = self.mu_0_torch.to(device)

        # Initialize the Lagrangian multiplier.
        # self.lambda_mul = np.random.uniform(0, 1)
        self.lambda_mul = torch.rand(1, device=device)
        # self.lambda_mul = self.lambda_mul.to(device)
        # Lists for entropy and threshold.
        self.entropy_list = list([])
        self.threshold_list = list([])
        self.iteration_list = list([])

        # Format: [observation_indx, aug_state_indx] = probability
        self.B_torch = torch.zeros(len(self.hmm.observations), len(self.hmm.augmented_states), device=device)
        # self.B_torch = self.B_torch.to(device)
        self.construct_B_matrix_torch()

        # Construct the cost matrix -> Format: [state_indx, masking_act] = cost
        self.cost_matrix = torch.zeros(len(self.hmm.augmented_states), len(self.hmm.masking_acts), device=device)
        self.construct_cost_matrix()

    def construct_cost_matrix(self):
        for s in self.hmm.cost_dict:
            for a in self.hmm.cost_dict[s]:
                self.cost_matrix[s, a] = self.hmm.cost_dict[s][a]
        return

    def sample_action_torch(self, state):
        # sample's actions given state and theta, following softmax policy.
        state_indx = self.hmm.augmented_states_indx_dict[state]
        # extract logits corresponding to the given state.
        logits = self.theta_torch[state_indx]
        logits = logits - logits.max()  # logit regularization.

        # if torch.isnan(logits).any() or torch.isinf(logits).any():
        #     print("The logits are:", logits, "for", state_indx)
        #     print("The state is: ", self.hmm.augmented_states[state_indx])
        #     raise ValueError("Logits contain Nan or inf values.")

        # compute the softmax probabilities for the actions.
        action_probs = F.softmax(logits, dim=0)

        # sample an action based on the computed probabilities.
        action = torch.multinomial(action_probs, num_samples=1).item()

        # trying to save space.
        del logits

        return action

    def sample_trajectories(self):

        state_data = np.zeros([self.batch_size, self.T], dtype=np.int32)
        action_data = np.zeros([self.batch_size, self.T], dtype=np.int32)
        y_obs_data = []

        for v in range(self.batch_size):
            y = []
            # starting from the initial state.
            state = self.hmm.initial_state

            act = self.sample_action_torch(state)
            for t in range(self.T):
                # Obtain the observation and add it to observation data.
                y.append(self.hmm.sample_observation(state))
                # Add the corresponding state and action values to state_data and action_data.
                s = self.hmm.augmented_states_indx_dict[state]
                state_data[v, t] = s
                a = self.hmm.mask_act_indx_dict[act]
                action_data[v, t] = a
                # next state sampling given the state and action.
                state = self.hmm.sample_next_state(state, act)
                # # Obtain the observation.
                # y.append(self.hmm.sample_observation(state))
                # next action sampling given the new state.
                act = self.sample_action_torch(state)
            y_obs_data.append(y)
        return state_data, action_data, y_obs_data

    def construct_transition_matrix_T_theta_torch(self):
        # Constructing the transtion matrix given the policy pi_\theta.
        # That T_\theta where P_\theta(p, q) = \sum_{\sigma' \in \Sigma} P(q|p, \sigma').pi_\theta(\sigma'|p).
        # T_\theta(i, j) --> from j to i.

        # Apply softmax to logits to obtain the policy probabilities pi_theta.
        logits = self.theta_torch.clone()
        logits = logits - logits.max()  # logits regularization.

        pi_theta = F.softmax(logits, dim=1)

        # Multiplication and sum over actions for each element of T_theta.
        T_theta = torch.einsum('sa, sna->ns', pi_theta, self.transition_mat_torch)  # TODO: Check for correctness.

        return T_theta

    def construct_B_matrix_torch(self):
        # Populate the B matrix with emission probabilities.
        # B(i\mid j) = Obs_2(o=i|z_j).
        # Format-- [observation_indx, aug_state_indx] = probability

        for state, obs in itertools.product(self.hmm.augmented_states, self.hmm.observations):
            self.B_torch[self.hmm.observations_indx_dict[obs], self.hmm.augmented_states_indx_dict[state]] = \
                self.hmm.emission_prob[state][obs]
        return

    def construct_A_matrix_torch(self, T_theta, o_t):
        # Construct the A matrix. A^\theta_{o_t} = T_theta.diag(B_{o_t, 1},...., B_{o_t, N}).
        # o_t is the particular observation.
        # TODO: see if you can save computation by not repeating the computations of A_o_t by saving them!!!!!!!!!!!!!!!

        o_t_index = self.hmm.observations_indx_dict[o_t]
        B_diag = torch.diag(self.B_torch[o_t_index, :])
        B_diag.to_sparse()

        # Compute A^\theta_{o_t}.
        A_o_t = torch.matmul(T_theta, B_diag)

        return A_o_t

    def compute_A_matrices(self, T_theta, y_v):
        # Construct all of the A_o_t.
        # Outputs a list of all of the A matrices given an observation sequence.
        A_matrices = []  # sequece -> Ao1, Ao2, ..., AoT.
        for o_t in y_v:
            A_o_t = self.construct_A_matrix_torch(T_theta, o_t)
            A_matrices.append(A_o_t)

        return A_matrices

    def compute_probability_of_observations(self, A_matrices):
        # Computes P_\theta(y) = P(o_{1:T}) = 1^T.A^\theta_{o_{T:1}}.\mu_0
        # Also computes A^\theta_{o_{T-1:1}}.\mu_0 -->  Required in later calculations.

        # A_matrices is a list of A matrices computed given T_theta and a sequence of observations.

        result_prob = self.mu_0_torch  # For P_\theta(y) = P(o_{1:T}) = 1^T.A^\theta_{o_{T:1}}.\mu_0
        resultant_matrix = self.mu_0_torch  # For A^\theta_{o_{T-1:1}}.\mu_0 -->  Required in later calculations.

        # Define a counter to stop the multiplication at T-1 for one of the results and T for the other.
        counter = len(A_matrices)
        # sequentially multiply with A matrices.
        for A in A_matrices:  # TODO: Check if the order of multiplication of the matrices correct!!!
            if counter > 1:  # TODO: Check if this should be changed.
                result_prob = torch.matmul(A, result_prob)
                resultant_matrix = torch.matmul(A, resultant_matrix)
                counter -= 1
            else:
                result_prob = torch.matmul(A, result_prob)
                counter -= 1

        # Multiplying with 1^T is nothing but summing up. Hence, we do the following.
        result_prob_P_y = result_prob.sum()
        # Compute the gradient later by simply using result_prob_to_return.backward() --> This uses autograd to
        # compute gradient.

        result_prob_P_y.backward(retain_graph=True)  # Gradient of P_\theta(y).
        gradient_P_y = self.theta_torch.grad.clone()
        # clearing .grad for the next gradient computation.
        self.theta_torch.grad.zero_()

        return result_prob_P_y, resultant_matrix, gradient_P_y

    def compute_joint_dist_of_zT_and_obs_less_than_T(self, resultant_matrix, g):
        # Computes P_\theta(Z_T, o_{1:T-1})
        # The resultant_matrix --> A^\theta_{o_{T-1:1}}.\mu_0
        # g -> the secret state
        # Outputs: 1^T_g.A^\theta_{o_{T-1:1}}.\mu_0

        ones_g = torch.zeros(self.num_of_aug_states, device=device)
        ones_g[self.hmm.augmented_states_indx_dict[g]] = 1

        # joint_dist_zT_and_obs_less_T = torch.matmul(ones_g, resultant_matrix)

        # return joint_dist_zT_and_obs_less_T
        return torch.dot(ones_g, resultant_matrix)

    def P_W_g_Y(self, y_v, A_matrices):
        # Computes the probability of P_\theta(w_T=1|y).
        # Outputs \sum_{g\in G} P(o_T|g).(1^T_g A^\theta_{o_{T-1:1}} \mu_0)/P_\theta(y).

        # TODO: Check if you can obtain the gradient to P_W_g_Y using .backward()?

        # result_P_W_g_Y = 0
        flag = 0
        o_T = y_v[-1]
        result_P_y, resultant_matrix, gradient_P_y = self.compute_probability_of_observations(A_matrices)

        for g in self.hmm.secret_goal_states:
            joint_dist_zT_and_obs_less_T = self.compute_joint_dist_of_zT_and_obs_less_than_T(resultant_matrix, g)
            if flag == 0:
                result_P_W_g_Y = (self.hmm.emission_prob[g][o_T] * joint_dist_zT_and_obs_less_T) / result_P_y
                flag = 1
            else:
                result_P_W_g_Y += (self.hmm.emission_prob[g][o_T] * joint_dist_zT_and_obs_less_T) / result_P_y

        # Compute the gradient of P_\theta(w_T=1|y).

        result_P_W_g_Y.backward(retain_graph=True)
        # result_P_W_g_Y.backward()
        gradient_P_W_g_Y = self.theta_torch.grad.clone()

        # Clearing .grad for next gradient computation.
        self.theta_torch.grad.zero_()

        return result_P_W_g_Y, gradient_P_W_g_Y, result_P_y, gradient_P_y

    def approximate_conditional_entropy_and_gradient_W_given_Y(self, T_theta, y_obs_data):
        # Computes the conditional entropy H(W_T | Y; \theta); AND the gradient of conditional entropy \nabla_theta
        # H(W_T|Y; \theta).

        # H = 0
        H = torch.tensor(0, dtype=torch.float32, device=device)
        nabla_H = torch.zeros([self.num_of_aug_states, self.num_of_maskin_actions],
                              device=device)  # TODO:Check if this is defined correctly!!!!!
        # V = len(y_obs_data)  # TODO: Check if this is needed. We can simply have it to be self.V
        for v in range(self.batch_size):
            y_v = y_obs_data[v]

            # construct the A matrices.
            A_matrices = self.compute_A_matrices(T_theta, y_v)  # Compute for each y_v.
            # result_prob_P_y, resultant_matrix = self.compute_probability_of_observations(A_matrices)  # TODO: Check
            #  TODO: if this is better?! currently, it is done differently to be able to use autograd.

            # values for the term w_T = 1.
            p_theta_w_t_g_yv_1, gradient_p_theta_w_t_g_yv_1, result_P_y, gradient_P_y = self.P_W_g_Y(y_v, A_matrices)

            # to prevent numerical issues, clamp the values of p_theta_w_t_g_yv_1 between 0 and 1.
            p_theta_w_t_g_yv_1 = torch.clamp(p_theta_w_t_g_yv_1, min=0.0, max=1.0)

            if p_theta_w_t_g_yv_1 != 0:
                log2_p_w_t_g_yv_1 = torch.log2(p_theta_w_t_g_yv_1)
            else:
                log2_p_w_t_g_yv_1 = torch.zeros_like(p_theta_w_t_g_yv_1, device=device)

            # Calculate the term when w_T = 1.
            term_w_T_1 = p_theta_w_t_g_yv_1 * log2_p_w_t_g_yv_1

            # Computing the gradient for w_T = 1. term for gradient term w_T = 1. Computed as [log_2 P_\theta(
            # w_T|y_v) \nabla_\theta P_\theta(w_T|y_v) + P_\theta(w_T|y_v) log_2 P_\theta(w_T|y_v) (\nabla_\theta
            # P_\theta(y))/P_\theta(y) + (\nabla_\theta P_\theta(w_T|y_v))/log2]
            gradient_term_w_T_1 = (log2_p_w_t_g_yv_1 * gradient_p_theta_w_t_g_yv_1) + (
                    p_theta_w_t_g_yv_1 * log2_p_w_t_g_yv_1 * gradient_P_y / result_P_y) + (
                                          gradient_p_theta_w_t_g_yv_1 / 0.301029995664)

            # Values for term w_T = 0.
            p_theta_w_t_g_yv_0 = 1 - p_theta_w_t_g_yv_1

            if p_theta_w_t_g_yv_0 != 0:
                log2_p_w_t_g_yv_0 = torch.log2(p_theta_w_t_g_yv_0)
            else:
                log2_p_w_t_g_yv_0 = torch.zeros_like(p_theta_w_t_g_yv_0, device=device)

            # Calculate the term when w_T = 0.
            term_w_T_0 = p_theta_w_t_g_yv_0 * log2_p_w_t_g_yv_0

            # Computing the gradient for w_T = 0. term for gradient term w_T = 0. Computed as [log_2 P_\theta(
            # w_T|y_v) \nabla_\theta P_\theta(w_T|y_v) + P_\theta(w_T|y_v) log_2 P_\theta(w_T|y_v) (\nabla_\theta
            # P_\theta(y))/P_\theta(y) + (\nabla_\theta P_\theta(w_T|y_v))/log2]

            # Gradient of P_\theta(w_T|y_v) when w_T = 0.
            gradient_p_theta_w_t_g_yv_0 = -gradient_p_theta_w_t_g_yv_1

            gradient_term_w_T_0 = (log2_p_w_t_g_yv_0 * gradient_p_theta_w_t_g_yv_0) + (
                    p_theta_w_t_g_yv_0 * log2_p_w_t_g_yv_0 * gradient_P_y / result_P_y) + (
                                          gradient_p_theta_w_t_g_yv_0 / 0.301029995664)

            H += term_w_T_1 + term_w_T_0

            nabla_H += gradient_term_w_T_1 + gradient_term_w_T_0
            # test_flag = 0

        H = H / self.batch_size

        nabla_H = nabla_H / self.batch_size

        return -H, -nabla_H

    def log_policy_gradient(self, state, act):
        # gradient = torch.zeros([len(self.hmm.augmented_states), len(self.hmm.masking_acts)], dtype=torch.float32,
        #                        device=device)

        # test_gradient = torch.zeros([len(self.hmm.augmented_states), len(self.hmm.masking_acts)], dtype=torch.float32,
        #                             device=device)

        gradient = torch.zeros_like(self.theta_torch, dtype=torch.float32, device=device)
        # test_a_indicators = torch.zeros_like(self.theta_torch, dtype=torch.float32, device=device)
        # test_action_probs_a_prime = torch.zeros_like(self.theta_torch, dtype=torch.float32, device=device)

        # for s_prime in self.hmm.augmented_states:
        #     for a_prime in self.hmm.mask_act_indx_dict.values():
        for s_prime, a_prime in itertools.product(self.hmm.augmented_states,
                                                  self.hmm.mask_act_indx_dict.keys()):

            # state_p = env.states[s_prime]
            # act_p = env.actions[a_prime]
            indicator_s = 0
            indicator_a = 0
            if state == self.hmm.augmented_states_indx_dict[s_prime]:
                indicator_s = 1
            if act == self.hmm.mask_act_indx_dict[a_prime]:
                indicator_a = 1

            # Debugging tensors.
            # test_a_indicators[self.hmm.augmented_states_indx_dict[s_prime], a_prime] = indicator_a

            logits = self.theta_torch[self.hmm.augmented_states_indx_dict[s_prime]]
            logits = logits - logits.max()

            actions_probs = F.softmax(logits, dim=0)
            actions_probs_a_prime = actions_probs[self.hmm.mask_act_indx_dict[a_prime]]

            # test_action_probs_a_prime[self.hmm.augmented_states_indx_dict[s_prime], a_prime] = actions_probs_a_prime

            # partial_pi_theta = indicator_s * (indicator_a - torch.softmax(
            #     self.theta_torch[self.hmm.augmented_states_indx_dict[s_prime]], dim=0)[
            #     self.hmm.mask_act_indx_dict[a_prime]])
            # testing if the above is correct or below .
            partial_pi_theta = indicator_s * (indicator_a - actions_probs_a_prime)
            # gradient[self.hmm.augmented_states_indx_dict[s_prime], self.hmm.mask_act_indx_dict[
            #     a_prime]] = partial_pi_theta
            # Trying if the following is correct or the one in the previous line.
            gradient[s_prime, a_prime] = partial_pi_theta
            # test_gradient[
            #     self.hmm.augmented_states_indx_dict[s_prime], self.hmm.mask_act_indx_dict[a_prime]] = partial_pi_theta

        return gradient

    def nabla_value_function(self, state_data, action_data, gamma=1):
        value_function_gradient = torch.zeros([len(self.hmm.augmented_states), len(self.hmm.masking_acts)],
                                              dtype=torch.float32, device=device)

        # value_function_gradient = torch.zeros_like(self.theta_torch, dtype=torch.float32, device=device)
        value_function = 0
        # batch_size = state_data.shape[0]  #TODO: Check if this can be eliminated as we know the batch size before
        #  hand.
        batch_size = self.V
        # Debugging test tensor.
        log_policy_gradient_all = torch.zeros((self.V, len(self.hmm.augmented_states), len(self.hmm.masking_acts)),
                                              dtype=torch.float32, device=device)
        total_returns_all = torch.zeros(self.V, dtype=torch.float32, device=device)

        for i in range(batch_size):
            log_P_x_i_gradient = torch.zeros([len(self.hmm.augmented_states), len(self.hmm.masking_acts)],
                                             dtype=torch.float32, device=device)

            # log_P_x_i_gradient = torch.zeros_like(self.theta_torch, dtype=torch.float32, device=device)

            total_return = 0
            for t in range(self.T):
                s = state_data[i, t]
                a = action_data[i, t]
                # state = env.states[s]
                # act = env.actions[a]
                log_P_x_i_gradient += self.log_policy_gradient(s, a)
                # cost = self.hmm.cost_dict[s][a]
                # total_return += gamma ** t * cost
                total_return += self.cost_matrix[s, a]

            log_policy_gradient_all[i] = log_P_x_i_gradient
            total_returns_all[i] = total_return

            value_function_gradient += total_return * log_P_x_i_gradient
            value_function += total_return

        value_function_gradient /= batch_size
        value_function /= batch_size

        return value_function_gradient, value_function

    # def parallel_worker(self, i, return_dict):
    #     # Solve each trajectory separately and then add the values to update the gradient value.
    #     approximate_cond_entropy = 0
    #     grad_H = 0
    #     trajectory

    def solver(self):

        num_cores = 8

        # Create a pool of workers.
        with mp.Pool(processes=num_cores) as pool:
            for i in range(self.iter_num):
                start = time.time()
                torch.cuda.empty_cache()

                # Initialize accumulators for gradients and entropy
                approximate_cond_entropy = 0
                grad_H = torch.zeros_like(self.theta_torch)

                trajectory_iter = int(self.V/self.batch_size)
                batch_idxs = list(range(trajectory_iter))

                # Function to process batches of 8 trajectories at a time.
                for batch_start in range(0, trajectory_iter, num_cores):
                    # Get the next group of batches (8 at a time)
                    batch_idxs_for_core = batch_idxs[batch_start:batch_start + num_cores]

                    # Collect trajectory data for the next group.
                    batch_inputs = []
                    for batch_idx in batch_idxs_for_core:
                        state_data, action_data, y_obs_data = self.sample_trajectories()
                        batch_inputs.append((se))




        # # Solve using policy gradient for optimal masking policy.
        # for i in range(self.iter_num):
        #     start = time.time()
        #     torch.cuda.empty_cache()
        #
        #     # Solve for each trajectory separately and then add the values to update the gradient value.
        #     # This should help in preventing the memory explosion.
        #     approximate_cond_entropy = 0
        #     grad_H = 0
        #
        #     trajectory_iter = int(self.V / self.batch_size)
        #
        #     # # Construct the matrix T_theta.
        #     # T_theta = self.construct_transition_matrix_T_theta_torch()
        #
        #     for j in range(trajectory_iter):
        #         torch.cuda.empty_cache()
        #
        #         with torch.no_grad():
        #             # Start with sampling the trajectories.
        #             state_data, action_data, y_obs_data = self.sample_trajectories()
        #
        #         # Gradient ascent algorithm.
        #
        #         # # Construct the matrix T_theta.
        #         T_theta = self.construct_transition_matrix_T_theta_torch()
        #         # Compute approximate conditional entropy and approximate gradient of entropy.
        #         approximate_cond_entropy_new, grad_H_new = self.approximate_conditional_entropy_and_gradient_W_given_Y(
        #             T_theta,
        #             y_obs_data)
        #         approximate_cond_entropy = approximate_cond_entropy + approximate_cond_entropy_new.item()
        #
        #         self.theta_torch = torch.nn.Parameter(self.theta_torch.detach().clone(), requires_grad=True)
        #
        #         grad_H = grad_H + grad_H_new
        #         # SGD gradients.
        #         # grad_V = self.compute_policy_gradient_for_value_function(state_data, action_data, 1)
        #
        #         # Compare the above value with traditional function.
        #         # grad_V_comparison, approximate_value = self.nabla_value_function(state_data, action_data, 1)
        #
        #         # Computing gradient of Lagrangian with grad_H and grad_V.
        #         # grad_L = grad_H + self.lambda_mul * grad_V
        #
        #     self.entropy_list.append(approximate_cond_entropy / trajectory_iter)
        #
        #     grad_L = (grad_H / trajectory_iter)
        #
        #     # del T_theta, approximate_cond_entropy_new, grad_H_new
        #     # gc.collect()
        #     # approximate_value = self.compute_value_function(state_data, action_data, 1)
        #     # self.threshold_list.append(approximate_value)
        #
        #     # Gradient clipping to prevent exploding gradients.
        #     # max_norm = 1.0  # You can adjust the clipping norm if necessary.
        #     # torch.nn.utils.clip_grad_norm_([grad_L], max_norm=max_norm)
        #
        #     # SGD updates.
        #     # Update theta_torch under the no_grad() to ensure that it remains as the 'leaf node.'
        #     with torch.no_grad():
        #         self.theta_torch = self.theta_torch + self.eta * grad_L
        #
        #     # self.lambda_mul = self.lambda_mul - self.kappa * (self.epsilon - approximate_value)
        #
        #     # re-initialize self.theta_torch to ensure it tracks the new set of computations.
        #     # self.theta_torch = torch.nn.Parameter(self.theta_torch, requires_grad=True)
        #     self.theta_torch = torch.nn.Parameter(self.theta_torch.detach().clone(), requires_grad=True)
        #
        #     end = time.time()
        #     print("Time for the iteration", i, ":", end - start, "s.")
        #
        self.iteration_list = range(self.iter_num)

        # entropy_values = [tensor.detach().numpy() for tensor in self.entropy_list]

        figure, axis = plt.subplots(2, 1)

        axis[0].plot(self.iteration_list, self.entropy_list, label='Entropy')
        # axis[1].plot(self.iteration_list, self.threshold_list, label='Estimated Cost')
        plt.xlabel("Iteration number")
        plt.ylabel("Values")
        plt.legend()

        plt.show()

        return
