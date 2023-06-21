from Simulator2 import *
from Plotting_Rewards import plot
import pandas as pd


def evaluate_policy(model, policy, test):
    terminate_state = [model.t_state[item] for item in model.t_state]
    policy_score = 0
    gamma = 0.8
    max_iter = 50
    num_success = 0
    num_collision = 0
    for i in test:
        discount = 1
        state = next_state = i
        count = 0
        while next_state not in terminate_state:
            action = policy[state]
            next_state, reward = transition_model(index_to_state(state, model.num_distance), action, model.num_distance)
            policy_score += discount * reward
            discount *= gamma
            state = next_state
            count += 1

            if count > max_iter:
                break
        if next_state == model.t_state['success_merge']:
            num_success += 1
        elif next_state == model.t_state['collision']:
            num_collision += 1
        print('State index ' + str(i) + ' uses ' + str(count) + ' steps')

    return policy_score / len(test), num_success / len(test), num_collision / len(test)



def evaluate_policy_plot(model, policy, test, filename_plot, filename_plot_data):
    terminate_state = [model.t_state[item] for item in model.t_state]
    policy_score = 0
    gamma = 0.8
    max_iter = 50
    num_success = 0
    num_collision = 0
    rewardperep = []
    for i in test:
        discount = 1
        state = next_state = i
        count = 0
        totalreward = 0
        while next_state not in terminate_state:
            action = policy[state]
            next_state, reward = transition_model(index_to_state(state, model.num_distance), action, model.num_distance)
            policy_score += discount * reward
            totalreward += reward
            discount *= gamma
            state = next_state
            count += 1

            if count > max_iter:
                break
        if next_state == model.t_state['success_merge']:
            num_success += 1
        elif next_state == model.t_state['collision']:
            num_collision += 1
        rewardperep.append(totalreward)
        print('State index ' + str(i) + ' uses ' + str(count) + ' steps')
    
    plot(rewardperep, filename_plot)
    pd.DataFrame(rewardperep).to_csv(filename_plot_data)

    return policy_score / len(test), num_success / len(test), num_collision / len(test)
