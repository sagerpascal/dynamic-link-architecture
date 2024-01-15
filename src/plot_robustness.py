import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def replace_square_list(sl):
    base_value = sl[0]
    delta = round(sl[1] - sl[0], 3)

    # check if list is linear
    for i in range(1, len(sl)):
        assert sl[i] == round(base_value + i * delta, 3), "List is not linear"

    return "{} + {}t".format(base_value, delta)


def get_data(file_path):
    results = []
    with open(str(file_path.absolute())) as f:
        filecontents = f.readlines()
        for entry in filecontents:
            data = json.loads(entry)
            result = {
                'noise': data['config']['noise'],
                'line_interrupt': data['config']['line_interrupt'],
                'act_bias': round(float(data['config']['lateral_model']['s2_params']['act_threshold'])-0.5, 2),
                'square_factor': replace_square_list(data['config']['lateral_model']['s2_params']['square_factor']),
                'noise_reduction': data['noise_reduction'],
                'avg_line_recon_accuracy_meter': data['avg_line_recon_accuracy_meter'],
                'avg_line_recon_accuracy_meter_2': (data['avg_line_recon_accuracy_meter'] - 0.75) / 0.25,
                'recon_accuracy': data['recon_accuracy'],
                'recon_recall': data['recon_recall'],
                'recon_precision': data['recon_precision'],
            }
            results.append(result)
    return pd.DataFrame.from_dict(results)


def feature_noise_to_location_noise(feature_noise, round_=False):
    # calculate probability of noise at each spatial location (can occur at each of the 4 feature channels)
    result = 1 - (1 - feature_noise) ** 4
    if round_:
        result = np.round(result, 2)
    return result


def plot_line(data, x_key, x_label, y_key, y_label, z_key, z_label, plot_key, plot_label, xmin, xmax, ymin, ymax,
              x2_func=None, x2_label=None, set_title=True, filename=None):

    # if problems with font, run it on local machine
    plt.rcParams["font.family"] = "Times New Roman"

    fig, axs = plt.subplots(ncols=len(data[plot_key].unique()), figsize=(13, 3), dpi=300)

    for ax, pk in zip(axs, sorted(data[plot_key].unique())):
        data_ = data[data[plot_key] == pk]

        z_values = list(data_[z_key].unique())
        z_values = sorted(z_values)

        for zv in z_values:
            z = data_[data_[z_key] == zv]
            ax.plot(z[x_key].values, z[y_key].values, label="{}{}".format(z_label, zv))

        if set_title:
            ax.set_title("{} = {}".format(plot_label, pk))

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_ylim(ymax=ymax, ymin=ymin)
        ax.set_xlim(xmin=xmin, xmax=xmax)

        if x2_func is not None:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(ax.get_xticks()[1:-1])
            ax2.set_xticklabels(x2_func(ax.get_xticks()[1:-1], round_=True))
            ax2.set_xlabel(x2_label)

        ax.legend()
        ax.grid()

    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"../tmp/{filename}")
    plt.show()


def plot(data, configname):

    def fname(x):
        return f"{configname}/{x}.png"


    # cleanup data
    data.loc[data.noise == 0, 'noise_reduction'] = 1.0

    # plot noise only
    data_1 = data[data['line_interrupt'] == 0]
    plot_line(data_1, x_key="noise", x_label="Feature Noise", y_key="noise_reduction", y_label="Noise Reduction Rate",
              z_key='act_bias', z_label='Act. Bias b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              x2_func=feature_noise_to_location_noise, x2_label="Spatial Noise", filename=fname("1_noise_reduction"),
              xmin=-0.005, xmax=0.205, ymin=0.795, ymax=1.005)

    plot_line(data_1, x_key="noise", x_label="Feature Noise", y_key="recon_recall", y_label="Recall",
              z_key='act_bias', z_label='Act. Bias b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              x2_func=feature_noise_to_location_noise, x2_label="Spatial Noise", set_title=False,
              filename=fname("2_recon_recall"), xmin=-0.005, xmax=0.205, ymin=0.08, ymax=1.02)

    plot_line(data_1, x_key="noise", x_label="Feature Noise", y_key="recon_precision", y_label="Precision",
              z_key='act_bias', z_label='Act. Bias b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              x2_func=feature_noise_to_location_noise, x2_label="Spatial Noise", set_title=False,
              filename=fname("3_recon_precision"), xmin=-0.005, xmax=0.205, ymin=0.08, ymax=1.02)

    # plot line interrupt only
    data_1 = data[data['noise'] == 0.0]
    # set accuracy to 1 where line is not interrupted for the plot
    data_1.loc[data_1.line_interrupt == 0, 'avg_line_recon_accuracy_meter'] = 1.0
    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="avg_line_recon_accuracy_meter",
              y_label="Feature Reconstruction Rate", z_key='act_bias', z_label='Act. Bias b = ',
              plot_key='square_factor', plot_label='Power Factor γ = ', filename=fname("4_avg_line_recon_accuracy"),
              xmin=-0.1, xmax=7.1, ymin=-0.02, ymax=1.02)
    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="recon_recall", y_label="Recall",
              z_key='act_bias', z_label='Act. Bias b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              set_title=False, filename=fname("5_recon_recall"), xmin=-0.1, xmax=7.1, ymin=0.49, ymax=1.01)
    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="recon_precision", y_label="Precision",
              z_key='act_bias', z_label='Act. Bias b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              set_title=False, filename=fname("6_recon_precision"), xmin=-0.1, xmax=7.1, ymin=0.49, ymax=1.01)


if __name__ == '__main__':
    for f in ['net-fragments']:
        file_path = Path(".").absolute().parent / "tmp" / f / "experiment_results.json"
        data = get_data(file_path)
        plot(data, f)
