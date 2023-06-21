import csv
# Load the Simrollouts dataset
simrollouts_file = "SimulationRollOuts.csv"
with open(simrollouts_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fieldnames = next(csvreader)
    simrollouts = [row for row in csvreader]

# Define the number of distance units
num_distance = 8

def index_to_state(index, num_distance):
    # state = [velocity, distance_1, distance_2]
    v = index // (num_distance ** 2)
    index %= num_distance ** 2
    d1 = index // num_distance
    index %= num_distance
    d2 = index
    return v, d1, d2


# Convert state indices to velocity and distance values
simrollouts_converted = []
for row in simrollouts:
    state = int(row[0])
    action = int(row[1])
    reward = float(row[2])
    next_state = int(row[3])
    v, d1, d2 = index_to_state(state, num_distance)
    v_next, d1_next, d2_next = index_to_state(next_state, num_distance)
    simrollouts_converted.append([v, d1, d2, action, reward, v_next, d1_next, d2_next])

# Plot the velocity and distance values
import matplotlib.pyplot as plt

v_values = [row[0] for row in simrollouts_converted]
d1_values = [row[1] for row in simrollouts_converted]
d2_values = [row[2] for row in simrollouts_converted]

plt.plot(v_values, d1_values, 'o', label='distance from front car')
plt.plot(v_values, d2_values, 'o', label='distance from back car')
plt.xlabel('Velocity (mph)')
plt.ylabel('Distance (units)')
plt.title('Movement of the car')
plt.legend()
plt.show()





