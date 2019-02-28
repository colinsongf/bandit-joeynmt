

# read validation file(s)
# plot multiple runs

# plot costs against BLEU
# plot costs against time
# plot time against BLEU

# coding: utf-8
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
import pandas as pd


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import argparse
import numpy as np
#print(plt.rcParams['axes.prop_cycle'])
#plt.rcParams['axes.prop_cycle'] = ("cycler('color', 'rgbkm') * cycler('linestyle', ['-', '--', ':', '-.'])")
                                   #" cycler('lw', [1, 2, 3, 4])")
#plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']) \
# + cycler('marker', ['o', 'v', '^', '<', '>', 's', 'D', 'h', '*', '+'])")

def read_vfiles(vfiles, labels, types):
    """
    Parse validation report files
    :param vfiles: list of files
    :return:
    """
    df = pd.DataFrame(columns=["steps", "model", "type", "MT-bleu", "Loss", "Total_Cost"])
    models = {}
    if labels is None:
        labels = vfiles
    for vfile, label, typ in zip(vfiles, labels, types):
        model_name = label
        #model_name = vfile.split("/")[-2] if "//" not in vfile \
        #    else vfile.split("/")[-3]
        with open(vfile, "r") as validf:
            steps = {}
            last_key = float('Inf')
            for line in reversed(list(validf)):
                row = {"model": model_name, "type": typ}
                entries = line.strip().split()
                key = int(entries[1])
                if key > last_key:
                    break
                steps[key] = {}
                row["time"] = key
                for i in range(2, len(entries)-1, 2):
                    name = entries[i].strip(":")
                    value = float(entries[i+1])
                    steps[key][name] = value
                    row[name] = value
                df = df.append(row, ignore_index=True)


                last_key = key
        models[model_name] = steps
    return models, df


def plot_models(models, df, x_value, y_value, output_path, plot_sup, lim_x):
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

    #print("model names", models.keys())
    #print("X value: {}".format(x_value))
    #print("Y value: {}".format(y_value))
    f = plt.figure()

    #if plot_sup:
    #    ax = f.add_subplot(2, 1, 1)
    #    ax_sup = f.add_subplot(2, 1, 2)
    #else:
    #    ax = f.add_subplot(1, 1, 1)

    # cut plot at shortest ys
    y_maxes = []
    x_maxes = []
    x_mins = []

    # TODO
    # find mins first
    # then plot only until min to make marker at the end visible
    #print(df)

    for col, model_name in enumerate(sorted(models)):
        xs = []
        ys = []
        sup_ys = {"weak": [], "full": [], "none": [], "self": []}
        for step in sorted(models[model_name]):
            logged_values = models[model_name][step]
            #print(logged_values)
            if x_value == "time":
                xs.append(step)
            elif x_value == "Total_Cost":
                cost = logged_values["Total_Cost"]
                xs.append(cost)
            if y_value == "MT-bleu":
                ys.append(logged_values["MT-bleu"])
                if plot_sup:
                    sup_ys["full"].append(logged_values["%full_sup"])
                    sup_ys["weak"].append(logged_values["%weak_sup"])
                    sup_ys["none"].append(logged_values["%no_sup"])
                    sup_ys["self"].append(logged_values["%self_sup"])
        #print("XS", xs)
        #print("YS", ys)
        #print("sUP YS", sup_ys)
        xs = [x_i - xs[0] for x_i in xs]
        assert len(xs) == len(ys)
        y_maxes.append(max(ys))
        xs_without_zeros = [i for i in xs if i > 0]
        if len(xs_without_zeros) > 1:
            x_mins.append(min(xs_without_zeros))
            x_maxes.append(max(xs_without_zeros))

        #fmri = sns.load_dataset("fmri")
        #print(fmri)
    if x_value == "Total_Cost":
        estimator = None
        lw = 1
        ci = None
        units = "model"
    else:
        estimator = 'mean'
        ci = 95
        units = None
    ax = sns.lineplot(x=x_value, y=y_value, data=df, hue="type", style="type", legend="brief", ci=ci, estimator=estimator, units=units)
        #ax = sns.lineplot(x=xs, y=ys, markers=None, labels=model_name)

        #f.plot(xs, ys)
        #ax.plot(xs, ys, label=model_name, markevery=[0, -1])
        #if plot_sup:
        #    bottom = None
        #    if sum(sup_ys["none"]) > 0:
        #        ax_sup.bar(xs, sup_ys["none"], label="none")
        #        if bottom is None:
        #            bottom = np.array(sup_ys["none"])
        #    if sum(sup_ys["self"]) > 0:
        #        ax_sup.bar(xs, sup_ys["self"], bottom=bottom, label="self")
        #        if bottom is None:
        #            bottom = np.array(sup_ys["self"])
        #        else:
        #            bottom += np.array(sup_ys["self"])
        #    if sum(sup_ys["weak"]) > 0:
        #        ax_sup.bar(xs, sup_ys["weak"], bottom=bottom, label="weak")
        #        if bottom is None:
        #            bottom = np.array(sup_ys["weak"])
        #        else:
        #            bottom += np.array(sup_ys["weak"])
        #    if sum(sup_ys["full"]) > 0:
#                ax_sup.bar(xs, sup_ys["full"], bottom=bottom,label="full")
#
    #ax.set_ylim()
    if lim_x:
        ax.set_xlim(min(x_mins), min(x_maxes))
    ax.set_ylabel("BLEU" if y_value=="MT-bleu" else y_value)
    ax.set_xlabel("Cumulative Cost" if x_value=="Total_Cost" else "Iterations")
    #f.show()
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels)
    #if plot_sup:
    #    handles, labels = ax_sup.get_legend_handles_labels()
    #    ax_sup.legend(handles, labels)
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
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--types", type=str, nargs="+", default=None,
                        help="Group plots by type.")
    parser.add_argument("--x_value", type=str, default="time")
    parser.add_argument("--y_value", type=str, default="MT-bleu")
    parser.add_argument("--plot_sup", action="store_true")
    parser.add_argument("--output_path", type=str, default="plot.pdf",
                        help="Plot will be stored in this location.")
    parser.add_argument("--lim_x", action="store_true")
    args = parser.parse_args()

    vfiles = [m+"/validations.txt" for m in args.model_dirs]

    models, df =  read_vfiles(vfiles, labels=args.labels, types=args.types)

    plot_models(models, df, x_value=args.x_value, y_value=args.y_value, output_path=args.output_path, plot_sup=args.plot_sup, lim_x=args.lim_x)