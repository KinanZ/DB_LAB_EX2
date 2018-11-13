import json
import os
import matplotlib.pyplot as plt

path = './results_filters/'
path2 = './results_learning_rates/'
files = os.listdir(path)

plt.figure(0)
files2 = os.listdir(path2)
lrs = [0.1, 0.01, 0.001, 0.0001]
for i in range(len(files2)):
    name = files2[i]
    json_data = open(path2 + name).read()
    data = json.loads(json_data)
    learning_curve = data['learning_curve']
    lr = lrs[i]
    plt.plot(range(12), learning_curve, label = "Learning Rate:" + str(lr))
    plt.xlabel('epochs')
    plt.ylabel('learning_curve')
    plt.title("Learning Curve for Learning Rate")
    plt.legend(loc='best')

plt.show()

plt.figure(1)
for i in range(len(files)):
    name = files[i]
filters = [1,3,5,7]
for i in range(len(files)):
    name = files[i]
    json_data = open(path + name).read()
    data = json.loads(json_data)
    learning_curve = data['learning_curve']
    filter_size = filters[i]
    plt.plot(range(12), learning_curve, label = "Filter size:" + str(filter_size))
    plt.xlabel('epochs')
    plt.ylabel('learning_curve')
    plt.title("Learning Curve for Filters")
    plt.legend(loc='best')

plt.show()
# for name in files2:
#     json_data = open(path2 + name).read()
#     data = json.loads(json_data)
#     learning_curve = data['learning_curve']
#
#     plot_curves(learning_curve, name, path2)
#
#

path3 = './results_best/'
json_data = open(path3 + 'results_run_0.json').read()
data = json.loads(json_data)
learning_curve = data['learning_curve']
plt.plot(range(12), learning_curve)
plt.xlabel('epochs')
plt.ylabel('learning_curve')
plt.title("Learning Curve for Best Configuration")
plt.show()
