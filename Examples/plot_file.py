import matplotlib.pyplot as plt
import pickle

ex_num = 3
iter_num = 2000

iteration_list = range(iter_num)

with open(f'../Data/entropy_values_{ex_num}.pkl', 'rb') as file:
    entropy_list = pickle.load(file)

# with open(f'../Data/value_function_list_{ex_num}', 'rb') as file:
#     threshold_list = pickle.load(file)


print("The last entropy value is", entropy_list[-1])
# print("The policy value is", threshold_list[-5])

figure, axis = plt.subplots(1, 1)

# Plot data on the first subplot with a solid red line
axis.plot(iteration_list, entropy_list, color='red', linestyle='-', label='Entropy')
axis.set_ylabel("Estimated Entropy")  # Set ylabel for the first subplot
axis.legend()  # Add legend to the first subplot
axis.grid(True)

# Plot data on the second subplot with a dashed blue line
# axis[1].plot(iteration_list, threshold_list, color='blue', linestyle='-', label='Estimated Total Return')
# axis[1].set_xlabel("Iteration number")  # Set xlabel for the second subplot
# axis[1].set_ylabel("Estimated Policy Values")  # Set ylabel for the second subplot
# axis[1].legend()  # Add legend to the second subplot
# axis[1].grid(True)

# Save and display the plot
plt.savefig(f'../Data/graph_{ex_num}.png')
plt.show()


# with open(f'../Data/final_control_policy_{ex_num}.pkl', 'rb') as file:
#     control_policy = pickle.load(file)
#
# print(control_policy)