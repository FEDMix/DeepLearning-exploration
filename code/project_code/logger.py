import matplotlib.pyplot as plt
import numpy as np
import os
import json


class Logger:
    def __init__(self, folder, parameters, description='', track_variables=[]):
        self.folder = folder
        if not os.path.isdir(folder):
            os.makedirs(folder)

        self.parameters = parameters
        self.description = description

        self.filename = ''
        for parameter in self.parameters:
            self.filename += ('%s%s_' % (parameter, parameters[parameter]))
        self.filename += '.json'

        self.variable_arrays = {}
        for variable_name in track_variables:
            self.variable_arrays[variable_name] = []

    def save(self):
        data = {}
        data['parameters'] = self.parameters
        data['description'] = self.description
        data['variable_arrays'] = self.variable_arrays
        json.dump(data, open(os.path.join(self.folder, self.filename), 'w'))

    def load(self, folder, filename):
        data = json.load(open(os.path.join(folder, filename), 'r'))
        self.parameters = data['parameters']
        self.description = data['description']
        self.variable_arrays = data['variable_arrays']

    def update(self, variable_name, variable_value):
        self.variable_arrays[variable_name].append(variable_value)

    def get_best_value(self, variable_name, type='max'):
        if type == 'max':
            if len(self.variable_arrays[variable_name]) == 0:
                return -1e9
            return np.max(self.variable_arrays[variable_name])
        elif type == 'min':
            if len(self.variable_arrays[variable_name]) == 0:
                return 1e9
            return np.min(self.variable_arrays[variable_name])

    def plot_one_variable(self, variable_name):
        plt.title(variable_name)
        plt.plot(self.variable_arrays[variable_name])
        plt.show()

    def plot_two_variables(self, variable_name_y, variable_name_x):
        plt.xlabel(variable_name_x)
        plt.ylabel(variable_name_y)
        plt.plot(self.variable_arrays[variable_name_x],
                 self.variable_arrays[variable_name_y])
        plt.show()

    def compare_experiments(self, folder):
        pass

    def plot_experiments(self, folder):
        all_experiments_files = ps.listdir(folder)
        all_variable_arrays = {}

        for experiment_file in all_experiments_files:

            data = self.load(folder, experiment_file)

            variable_arrays = data['variable_arrays']
            parameters = data['parameters']

            for variable in variable_arays:

                if variable not in all_variable_arrays:
                    all_variable_arrays[variable] = []

                all_variable_arrays[variable].append(
                    (data['variable_arrays'][variable], parameters))

        for variable in all_variable_arrays:

            all_plots, all_parameters = [], []

            for plot_data, parameters in all_variable_arrays[variable]:
                all_plots.append(plt.plot(plot_data))
                all_parameters.append(parameters)

            plt.title(variable)
            plt.legend(all_plots, all_parameters)
            plt.show()
