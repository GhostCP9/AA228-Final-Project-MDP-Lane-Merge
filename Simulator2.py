import csv
import random
from config import CONFIG

random.seed(1231)


def state_to_index(state, num_distance):
    # state = [velocity, distance_1, distance_2]
    # convert state to linear index
    return state[0] * num_distance ** 2 + state[1] * num_distance + state[2]


def index_to_state(index, num_distance):
    # state = [velocity, distance_1, distance_2]
    v = index // (num_distance ** 2)
    index %= num_distance ** 2
    d1 = index // num_distance
    index %= num_distance
    d2 = index
    return v, d1, d2


def transition_model(state, action, num_distance):
    # state = [velocity, distance_1, distance_2]
    v = state[0] + 50
    reward = CONFIG['reward_dict']
    t_state = CONFIG['terminate_state']

    d1 = state[1]
    d2 = state[2]
    # safe following distance at this speed in terms of car length
    ds = v / 5
    p = random.uniform(0, 1)

    if action == 1:
        if d1 == 0 or d2 == 0:
            return t_state['collision'], reward['collision']
        else:
            eps = 0.7 ** (max(ds - d1, 0) + max(ds - d2, 0))
            if p <= eps:
                return t_state['success_merge'], reward['success_merge']
            else:
                return t_state['collision'], reward['collision']
    elif action == 2:
        v += 1
        if v > CONFIG['max_velocity']:
            return t_state['out_of_bounds'], reward['out_of_bounds']

        if d1 >= ds:
            d1 = d1 - 1 if p <= 0.9 else d1 if p <= 0.95 else d1 + 1
        else:
            d1 = d1 - 1 if p <= 0.6 else d1 if p <= 0.8 else d1 + 1

        if d2 >= ds:
            d2 = d2 + 1 if p <= 0.9 else d2 if p <= 0.95 else d2 - 1
        else:
            d2 = d2 + 1 if p <= 0.6 else d2 if p <= 0.8 else d2 - 1

        if d1 < 0 or d2 < 0:
            return t_state['out_of_bounds'], reward['out_of_bounds']
        d1 = min(d1, num_distance - 1)
        d2 = min(d2, num_distance - 1)
        return state_to_index([v - 50, d1, d2], num_distance), reward['not_merge']

    elif action == 3:
        v -= 1
        if v < CONFIG['min_velocity']:
            return t_state['out_of_bounds'], reward['out_of_bounds']

        if d1 >= ds:
            d1 = d1 + 1 if p <= 0.9 else d1 if p <= 0.95 else d1 - 1
        else:
            d1 = d1 + 1 if p <= 0.6 else d1 if p <= 0.8 else d1 - 1

        if d2 >= ds:
            d2 = d2 - 1 if p <= 0.9 else d2 if p <= 0.95 else d2 + 1
        else:
            d2 = d2 - 1 if p <= 0.6 else d2 if p <= 0.8 else d2 + 1

        if d1 < 0 or d2 < 0:
            return t_state['out_of_bounds'], reward['out_of_bounds']
        d1 = min(d1, num_distance - 1)
        d2 = min(d2, num_distance - 1)
        return state_to_index([v - 50, d1, d2], num_distance), reward['not_merge']
    else:
        if d1 >= ds:
            d1 = d1 if p <= 0.9 else d1 + 1 if p <= 0.95 else d1 - 1
        else:
            eps = 0.9 ** (ds - d1)
            d1 = d1 + 1 if p <= 1 - eps else d1 if p <= 1 - 0.1 * eps else d1 - 1

        if d2 >= ds:
            d2 = d2 if p <= 0.9 else d2 + 1 if p <= 0.95 else d2 - 1
        else:
            eps = 0.9 ** (ds - d2)
            d2 = d2 + 1 if p <= 1 - eps else d2 if p <= 1 - 0.1 * eps else d2 - 1

        if d1 < 0 or d2 < 0:
            return t_state['out_of_bounds'], reward['out_of_bounds']
        d1 = min(d1, num_distance - 1)
        d2 = min(d2, num_distance - 1)
        return state_to_index([v - 50, d1, d2], num_distance), reward['not_merge']


def simulation(config):
    num_distance = -(config['max_velocity'] // -5) + 1
    num_velocity = config['max_velocity'] - config['min_velocity'] + 1
    field = ['s', 'a', 'r', 'sp']
    # velocity[mph]: 50, 51, 52, ..., 70
    # action space: 0: merge in , 1: throttling, 2: braking, 3: remain constant speed
    action_space = [1, 2, 3, 4]
    dataset = []
    t_state = config['terminate_state']
    terminate_state = [t_state[item] for item in t_state]

    for i in range(num_velocity):
        for j in range(num_distance):
            for k in range(num_distance):
                for _ in range(30):
                    state = [i, j, k]
                    count = 0
                    next_state = 1
                    while next_state not in terminate_state:
                        action = random.randint(1, len(action_space))
                        next_state, reward = transition_model(state, action, num_distance)
                        dataset.append([state_to_index(state, num_distance), action, reward, next_state])
                        state = index_to_state(next_state, num_distance)
                        count += 1
                        if count > 20:
                            break

    # for _ in range(3):
    #    random.shuffle(dataset)

    filename = "SimulationRollOuts.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(field)
        csvwriter.writerows(dataset)


def main():
    simulation(CONFIG)


if __name__ == '__main__':
    main()




