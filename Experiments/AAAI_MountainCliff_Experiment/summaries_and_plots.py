import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from Experiments_Engine.Plots_and_Summaries import compute_tdist_confidence_interval, create_results_file

# Colours:
    # Dark Blue: #223355
    # Yellow: #FBB829
    # Green: #7FAF1B
    # Light Blue: #2A8FBD


def load_data(results_dir, method_name):
    with open(os.path.join(results_dir, method_name, 'results.p'), mode='rb') as results_file:
        method_data = pickle.load(results_file)
    return np.array(method_data)


def get_moving_average_data(method_data, average_window):
    averaged_data = np.average(method_data, axis=0)
    moving_average_data = np.zeros(len(averaged_data)-average_window+1, dtype=np.float64)
    index = 0
    for i in range(len(averaged_data)-average_window+1):
        moving_average_data[i] += np.average(averaged_data[index:index+average_window])
        index += 1
    return moving_average_data


if __name__ == "__main__":
    project_dir = os.getcwd()
    results_dir = os.path.join(project_dir, 'Results')
    methods_names = ['sarsa_n4_a1o4', 'treebackup_n10_a1o4','decayingsigma_n10_a1o4', 'qsigma0.5_n10_a1o6']

    method_data = []
    for name in methods_names:
        method_data.append(load_data(results_dir, name))

    """ Creating Table """
    evaluation_episodes = [50, 500]
    add_to_file = False
    ci_error = 0.95
    sample_size = 500
    column_names = ['Method_Name', 'Evaluation_Episodes', 'Mean', 'Standard_Deviation',
                    'Lower_CI', 'Upper_CI']
    for i in range(len(method_data)):
        print("Working on method " + methods_names[i] + "...")
        method_name = [methods_names[i]]
        for j in range(len(evaluation_episodes)):
            evaluation_data = method_data[i][:, 0:evaluation_episodes[j]]
            averaged_data = np.average(evaluation_data, axis=1)
            sample_std = np.std(averaged_data, ddof=1)
            sample_mean = np.average(averaged_data)
            upper_bound, lower_bound, error_margin = compute_tdist_confidence_interval(sample_mean, sample_std,
                                                                                       ci_error, sample_size)
            columns = [method_name, [evaluation_episodes[j]], [np.round(sample_mean, 2)], [np.round(sample_std,2)],
                       [np.round(lower_bound, 2)], [np.round(upper_bound, 2)]]
            create_results_file(pathname=results_dir, columns=columns, headers=column_names, addtofile=add_to_file,
                                results_name='average_results_at_50_and_500_episodes')
            add_to_file=True



    method_color = ['#7FAF1B',  # Green
                    '#FBB829',  # Yellow
                    '#223355',  # Dark Blue
                    '#2A8FBD']  # Light Blue

    lighter_method_color = ['#d5e4b3', # Lighter Green
                            '#ffe9b6', # Lighter Yellow
                            '#b3b9c3', # Lighter Dark Blue
                            '#b5d6e7'] # Lighter Light Blue

    moving_average_window = 40
    number_of_episodes = 500
    moving_average_data = []
    for i in range(len(method_data)):
        moving_average_data.append(get_moving_average_data(method_data[i], moving_average_window))

    averaged_data_xaxis = np.arange(number_of_episodes) + 1
    moving_average_data_xaxis = np.arange(moving_average_window, number_of_episodes + 1)
    for i in range(len(methods_names)):
        print('Method Name:', methods_names[i])
        print('Color:', method_color[i])
        averaged_data = np.average(method_data[i], axis=0)
        plt.plot(averaged_data_xaxis, averaged_data, color=lighter_method_color[i])
    for i in range(len(methods_names)):
        plt.plot(moving_average_data_xaxis, moving_average_data[i], color=method_color[i])

    plt.xlim([0,500])
    plt.ylim([-400, -160])
    plt.ylabel('Average Return per Episode')
    plt.xlabel('Episode Number')
    plt.savefig(fname='average_return', dpi=200)
    plt.show()

