

# read validation file(s)
# plot multiple runs

# plot costs against BLEU
# plot costs against time
# plot time against BLEU

# coding: utf-8
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import argparse
import numpy as np

plt.rcParams['axes.prop_cycle'] = ("cycler('color', 'rgbk') + cycler('linestyle', ['-', '--', ':', '-.'])")
                                   #" cycler('lw', [1, 2, 3, 4])")


def read_vfiles(vfiles):
    """
    Parse validation report files
    :param vfiles: list of files
    :return:
    """
    models = {}
    for vfile in vfiles:
        model_name = vfile.split("/")[-2] if "//" not in vfile \
            else vfile.split("/")[-3]
        with open(vfile, "r") as validf:
            steps = {}
            for line in validf:
                entries = line.strip().split()
                key = int(entries[1])
                steps[key] = {}
                for i in range(2, len(entries)-1, 2):
                    name = entries[i].strip(":")
                    value = float(entries[i+1])
                    steps[key][name] = value
        models[model_name] = steps
    return models


def plot_models(models, x_value, y_value, output_path):
    """
    Plot the learning curves for several models
    :param models:
    :param plot_values:
    :param output_path:
    :return:
    """
    # models is a dict: name -> ckpt values
    #f, axes = plt.subplots(len(plot_values), len(models),
    #                       sharex='col', sharey='row',
    #                       figsize=(3*len(models), 3*len(plot_values)))
    #axes = np.array(axes).reshape((len(plot_values), len(models)))

    print("model names", models.keys())
    print("X value: {}".format(x_value))
    print("Y value: {}".format(y_value))
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)

    # cut plot at shortest ys
    y_maxes = []
    x_maxes = []
    x_mins = []

    for col, model_name in enumerate(models):
        xs = []
        ys = []
        for step in sorted(models[model_name]):
            logged_values = models[model_name][step]
            #print(logged_values)
            if x_value == "time":
                xs.append(step)
            elif x_value == "Total_Cost":
                xs.append(logged_values["Total_Cost"])
            if y_value == "MT-bleu":
               ys.append(logged_values["MT-bleu"])
        print("XS", xs)
        print("YS", ys)
        assert len(xs) == len(ys)
        y_maxes.append(max(ys))
        x_maxes.append(max(xs))
        x_mins.append(min(xs))

        #f.plot(xs, ys)
        ax.plot(xs, ys, label=model_name)

    #ax.set_ylim()
    ax.set_xlim(min(x_mins), min(x_maxes))
    ax.set_ylabel(y_value)
    ax.set_xlabel(x_value)
    #f.show()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
   # plt.show()

        # now one plot instead of many
        #values = {}
        # get arrays for plotting
        #for step in sorted(models[model_name]):
        #    logged_values = models[model_name][step]
        #    for plot_value in plot_values:
        #        if plot_value not in logged_values:
        #            continue
        #        elif plot_value not in values:
        #            values[plot_value] = [[], []]
        #        values[plot_value][1].append(logged_values[plot_value])
        #        values[plot_value][0].append(step)

#        for row, plot_value in enumerate(plot_values):
#            axes[row][col].plot(values[plot_value][0], values[plot_value][1])
#            axes[row][0].set_ylabel(plot_value)
#            axes[0][col].set_title(model_name)
    plt.tight_layout()

    if output_path.endswith(".pdf"):
        pp = PdfPages(output_path)
        pp.savefig(f)
        pp.close()
    else:
        if not output_path.endswith(".png"):
            output_path += ".png"
        plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("JoeyNMT Validation plotting.")
    parser.add_argument("model_dirs", type=str, nargs="+",
                        help="Model directories.")
    parser.add_argument("--x_value", type=str, default="time")
    parser.add_argument("--y_value", type=str, default="MT-bleu")
    parser.add_argument("--output_path", type=str, default="plot.pdf",
                        help="Plot will be stored in this location.")
    args = parser.parse_args()

    vfiles = [m+"/validations.txt" for m in args.model_dirs]

    models = read_vfiles(vfiles)

    plot_models(models, x_value=args.x_value, y_value=args.y_value, output_path=args.output_path)