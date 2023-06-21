import numpy as np
import sys
import pandas as pd
import time
from evaluate_policy import *

random.seed(1231)


class SARSA:
    def __init__(self, input_file, output_file):
        self.action_space = list(range(0, len(CONFIG['action_list'])))
        self.alpha = 0.1
        self.discount_factor = 0.95
        self.horizon = 1
        #self.horizon = 100
        self.eps = 0.8
        self.decay_factor = 0.9
        self.policy = []
        self.random_policy = []
        self.data = np.array(pd.read_csv(input_file).values)
        self.policy_file = output_file
        self.num_distance = CONFIG['max_distance'] - CONFIG['min_distance'] + 1
        self.num_vel = CONFIG['max_velocity'] - CONFIG['min_velocity'] + 1
        self.reward_dict = CONFIG['reward_dict']
        self.t_state = CONFIG['terminate_state']
        self.num_states = self.num_vel * self.num_distance ** 2
        self.Q = np.zeros((self.num_states, len(self.action_space)))

    def updateQ(self, q_s):
        cur_state, cur_action, reward, next_state = q_s
        cur_action -= 1
        # define update rule for terminate states
        if next_state == -1:
            next_u = self.reward_dict['out_of_bounds']
        elif next_state == 10000:
            next_u = self.reward_dict['success_merge']
        elif next_state == -10000:
            next_u = self.reward_dict['collision']
        else:
            chosen_action = self.explore(next_state)
            next_u = self.Q[next_state][chosen_action]

        self.Q[cur_state][cur_action] += self.alpha * (
                reward + self.discount_factor * next_u - self.Q[cur_state][cur_action])

    def explore(self, state):
        eps_greedy = random.uniform(0, 1)
        if eps_greedy > self.eps:
            action = np.argmax(self.Q[state, :])
        else:
            self.eps *= self.decay_factor
            action = random.sample(self.action_space, 1)
        return action

    def train(self):
        for _ in range(self.horizon):
            for q_s in self.data:
                self.updateQ(q_s)
        
        self.policy = list(np.argmax(self.Q, axis=1) + 1)
        self.policy_output(self.policy_file)


    def generate_random_policy(self):
        self.random_policy = [random.randint(1, 4) for _ in range(len(self. policy))]

    def policy_output(self, output_file):
        with open(output_file, 'w') as f:
            for action in self.policy:
                f.write("{}\n".format(action))


def main():
    #if len(sys.argv) != 3:
    #    raise Exception("usage: python q_learning.py <infile>.csv <outfile>.policy")

    #input_file = sys.argv[1]
    #output_file = sys.argv[2]

    input_file = "SimulationRollOuts.csv"
    output_file = "SARSA_Policy.py"
    start_time = time.time()

    # Train
    model = SARSA(input_file, output_file)

    model.train()

    Q_mat = "Q_matrix_SARSA.csv"
    with open(Q_mat, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(model.Q)

    # Random Policy
    model.generate_random_policy()

    # Evaluate the score of policy based on the mean discounted return of all possible initial states
    test = [i for i in range(model.num_states)]

    filename_plot = "Rewards_SARSA.png"
    filename_plot_data = "Rewards_SARSA.csv"

    policy_score, success_rate, collision_rate = evaluate_policy_plot(model, model.policy, test,filename_plot, filename_plot_data)

    random_policy_score, success_rate_random, collision_rate_random = evaluate_policy(model, model.random_policy, test)
    print("Relative score of the policy is: ", policy_score - random_policy_score)
    print("The rate of success merge in for the policy is: ", success_rate)
    print("The rate of collisions for the policy is: ", collision_rate)
    print("The rate of success merge in for the random policy is: ", success_rate_random)
    print("The rate of collisions for the random policy is: ", collision_rate_random)

    end_time = time.time()
    print("Time elapsed: ", end_time - start_time)


if __name__ == '__main__':
    main()
