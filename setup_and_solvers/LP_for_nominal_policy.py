from mip import *
from setup_and_solvers.markov_decision_process import *
from loguru import logger


def LP(mdp, gamma):
    model = Model(solver_name=GRB)
    # gamma = 0.95 # For multi-init states
    # gamma = 0.999  # for single-init states

    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    init = np.zeros(st_len)

    # The below is the set-up for the multiple initial states.
    indx_of_init_states = list()
    for in_state in mdp.init:
        indx_of_init_states.append(mdp.statespace.index(in_state))

    for indx in range(st_len):
        if indx in indx_of_init_states:
            init[indx] = 1 / len(indx_of_init_states)  # this is for uniform distribution. Todo: Generalize this to
            # TODO: handle other possible initial dist. By taking the initial_distribution.

    # The below is the set-up for single initial state.
    # indx = mdp.statespace.index(mdp.init)
    # init[indx] = 1

    m = [model.add_var() for i in range(st_len * act_len)]

    R_1 = np.zeros(st_len * act_len)

    # # Reward for not ending in a bad state.
    # R_3 = np.zeros(st_len * act_len)

    # # The following is the Reward function
    # for i in range(st_len):
    #     R[i * act_len] = 1

    # The following is the Reward function -> R1 for our problem.
    # for i in range(st_len):
    #     for a in range(act_len):
    #         for i_dash in range(st_len):
    #             if (mdp.statespace[i], mdp.A[a], mdp.statespace[i_dash]) in mdp.R1:
    #                 R_1[(i * 3) + (a * 3) + i_dash] = mdp.R1[(mdp.statespace[i], mdp.A[a], mdp.statespace[i_dash])]
    #             else:
    #                 R_1[(i * 3) + (a * 3) + i_dash] = 0

    # R_1[5] = 1

    # ct = 0
    # for st in mdp.R1:
    #     R_1[ct] = mdp.R1[st]
    #     ct = ct + 1

    # The following is the Reward function -> R2 for our problem. As well as R1. Check the R1.
    for i in range(st_len):
        for a in range(act_len):
            # R_2[(i * 3) + a] = mdp.R2[(mdp.statespace[i], mdp.A[a])]
            #
            # R_1[(i * 3) + a] = mdp.R1[(mdp.statespace[i], mdp.A[a])]

            # R_2[(i * act_len) + a] = mdp.R2[(mdp.statespace[i], mdp.A[a])]  # Todo: Check if the rewards are
            # assigned appropriately?!

            R_1[(i * act_len) + a] = mdp.reward[mdp.statespace[i]][mdp.A[a]]

            # R_3[(i * act_len) + a] = mdp.R3[(mdp.statespace[i], mdp.A[a])]

    # # The objective function when we are maximizing the opacity.
    # model.objective = maximize(xsum(m[i] * R_2[i] for i in range(st_len * act_len)))

    # The objective function when we are maximizing the task specification.
    model.objective = maximize(xsum(m[i] * R_1[i] for i in range(st_len * act_len)))

    # # The objective function when we are minimizing the opacity or maximizing transparency.
    # model.objective = minimize(xsum(m[i] * R_2[i] for i in range(st_len * act_len)))

    E, F = generate_matrix(mdp)

    # The following is the constraint in the case for a weak opponent.
    for i in range(st_len * act_len):
        model += m[i] >= 0

    # # The following is the constraint in the case for a strong opponent.
    # for i in range(st_len * act_len):
    #     model += m[i] >= 0.01

    for i in range(st_len):
        model += xsum((E[i][j] * m[j] - gamma * F[i][j] * m[j]) for j in range(st_len * act_len)) - init[i] == 0

    # # This is the constraint when the constraint is task specification.
    # model += xsum(m[i] * R_1[i] for i in range(st_len * act_len)) >= beta

    # # This is the constraint when the constraint is opacity.
    # model += xsum(m[i] * R_2[i] for i in range(st_len * act_len)) >= beta

    # # This is the constraint to prevent going into the unsafe states. ----> Trying
    # model += xsum(m[i] * R_3[i] for i in range(st_len * act_len)) >= 0

    print("Start optimization")
    # model.max_gap = 0.05
    status = model.optimize()  # Set the maximal calculation time
    print("Finish optimization")
    print(status)
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:",
              model.objective_value)  # Todo: CHECK if the VALUE of MODEL OBJECTIVE is Correct!!!!!!!
        logger.debug(f"The model objective is: {model.objective_value}")

        # m_res = [m[i].x for i in range(st_len * act_len)]
        # print("m_res:", m_res)
        # logger.debug(f"m_res: {m_res}")

        # threshold_value = np.dot(m_res, R_1)
        # print("Threshold (beta) calculated from the result:", threshold_value)
        # logger.debug(f"Threshold (beta) calculated from the result: {threshold_value}")

    # Generate the policy from the output.
    # pol = dict()
    # for i in range(st_len):
    #     for a in range(act_len):
    #         denominator = sum(
    #             m[(i * 3) + a_dash].x for a_dash in range(act_len))
    #         if denominator != 0:
    #             pol[(mdp.statespace[i], mdp.A[a])] = m[(i * 3) + a].x / denominator
    #         else:
    #             pol[(mdp.statespace[i], mdp.A[a])] = 0

    pol = dict()
    for i in range(st_len):
        for a in range(act_len):
            denominator = sum(
                m[(i * act_len) + a_dash].x for a_dash in range(act_len))
            if denominator != 0:
                pol[(mdp.statespace[i], mdp.A[a])] = m[(i * act_len) + a].x / denominator
            else:
                pol[(mdp.statespace[i], mdp.A[a])] = 0

    # Printing the policy.
    # print(pol)
    # logger.debug(f"Policy generated.")
    # for st in pol:
    #     logger.debug(f"{st} : {pol[st]} \n")
    #
    # with open('policy_6_by_6_dynamic_sensor_0.4.pickle', 'wb') as file:
    #     pickle.dump(pol, file)
    return pol


def generate_matrix(mdp):
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)

    # sum over all action wrt one state, out visit
    E = np.zeros((st_len, st_len * act_len))
    for i in range(st_len):
        for j in range(act_len):
            E[i][i * act_len + j] = 1

    # in visit, corresponds to the upper one
    F = np.zeros((st_len, st_len * act_len))
    for st in mdp.stotrans.keys():
        for act in mdp.stotrans[st].keys():
            for st_, pro in mdp.stotrans[st][act].items():
                if st_ in mdp.statespace:
                    F[mdp.statespace.index(st_)][mdp.statespace.index(st) * act_len + mdp.A.index(act)] = pro
    return E, F

# if __name__ == "__main__":
#     mdp = MDP_manual.create_mdp()
#     LP(mdp, N=10, K=7.03)
