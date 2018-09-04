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

NUMBER_OF_EPISODES = 500
SAMPLE_SIZE = 2000


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


def get_interval_average_data(method_data, interval_window, ci_error=0.05):
    interval_average_data = np.zeros(int(NUMBER_OF_EPISODES / interval_window), dtype=np.float64)
    standard_deviation_data = np.zeros(int(NUMBER_OF_EPISODES / interval_window), dtype=np.float64)
    upper_ci_data = np.zeros(int(NUMBER_OF_EPISODES / interval_window), dtype=np.float64)
    lower_ci_data = np.zeros(int(NUMBER_OF_EPISODES / interval_window), dtype=np.float64)
    error_margin_data = np.zeros(int(NUMBER_OF_EPISODES / interval_window), dtype=np.float64)
    index = 0
    while index < NUMBER_OF_EPISODES:
        interval_averages = np.average(method_data[:,index:index+interval_window], axis=1)
        sample_mean = np.average(interval_averages)
        sample_std = np.std(interval_averages, ddof=1)
        upper_ci, lower_ci, error_margin = \
            compute_tdist_confidence_interval(sample_mean, sample_std, ci_error, SAMPLE_SIZE)

        interval_average_data[index // interval_window] = sample_mean
        standard_deviation_data[index // interval_window] = sample_std
        upper_ci_data[index // interval_window] = upper_ci
        lower_ci_data[index // interval_window] = lower_ci
        error_margin_data[index // interval_window] = error_margin
        index += interval_window
    return interval_average_data, standard_deviation_data, upper_ci_data, lower_ci_data, error_margin_data


if __name__ == "__main__":
    project_dir = os.getcwd()
    results_dir = os.path.join(project_dir, 'Results')
    methods_names = ['sarsa_n6_a1o5', 'treebackup_n36_a1o5','decayingsigma_n21_a1o4', 'qsigma0.5_n32_a1o6']

    method_data = []
    for name in methods_names:
        method_data.append(load_data(results_dir, name))

    """ Summary Table """
    evaluation_episodes = [10, 25, 50, 100, 250, 500]
    add_to_file = False
    ci_error = 0.05
    sample_size = 2000
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
                                results_name='average_results_at_different_episodes')
            add_to_file=True

    method_color = ['#7FAF1B',  # Green --- Sarsa
                    '#FBB829',  # Yellow --- Tree Backup
                    '#223355',  # Dark Blue --- Decaying Sigma
                    '#2A8FBD']  # Light Blue --- Q(0.5)

    lighter_method_color = ['#d5e4b3', # Lighter Green
                            '#ffe9b6', # Lighter Yellow
                            '#b3b9c3', # Lighter Dark Blue
                            '#b5d6e7'] # Lighter Light Blue

    """ Moving Average Plot """
    print("Working on the moving average plot...")
    moving_average_window = 10
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
    plt.ylim([-350, -150])
    plt.ylabel('Average Return per Episode')
    plt.xlabel('Episode Number')
    plt.savefig(fname='average_return', dpi=200)
    plt.close()

    """ Interval Average Plot """
    print("Working on the interval average plot...")
    ci_error = 0.05
    interval_window = 50
    x_axis = np.arange(50,550,50, dtype=np.int32)

    add_to_file = False
    for i in range(len(method_data)):
        print("Method Name:", methods_names[i])
        print("Color:", method_color[i])

        average_interval_data, std_data, upper_ci_data, lower_ci_data, error_margin_data = \
            get_interval_average_data(method_data[i], interval_window, ci_error)

        column_names = ["Method_Name", "Episode_Number", "Mean in Previous 50 Episodes", "Standard Deviation",
                        "Upper CI", "Lower CI"]
        columns = [[methods_names[i]] * len(x_axis), x_axis, np.round(average_interval_data,2), np.round(std_data,2),
                   np.round(upper_ci_data,2), np.round(lower_ci_data,2)]
        create_results_file(pathname=results_dir, columns=columns, headers=column_names, addtofile=add_to_file,
                            results_name='average_results_previous_50_episodes')
        add_to_file = True

        plt.plot(x_axis, average_interval_data, color=method_color[i])
        plt.fill_between(x_axis, average_interval_data - error_margin_data, average_interval_data+error_margin_data,
                         color=lighter_method_color[i])

    plt.xlim([0,500])
    plt.ylim([-450, -100])
    plt.ylabel('Average Return of the Preceding 50 Episodes')
    plt.xlabel('Episode Number')
    plt.savefig(fname='average_return_previous_k_episodes', dpi=200)
    plt.close()
