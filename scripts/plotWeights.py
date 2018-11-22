# coding: utf-8
import matplotlib.pyplot as plt
import argparse
import numpy as np

# plot the weights for each token in a colorful way

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_file", type=str, help="Source file")
    parser.add_argument("trg_file", type=str, help="Target file")
    parser.add_argument("weight_file", type=str, help="Weight file")
    parser.add_argument("examples", type=int, help="Index of examples to plot", nargs='+')
    args = parser.parse_args()
    print("Visualizing examples", args.examples)

    with open(args.src_file, "r") as sf, open(args.trg_file, "r") as tf, \
            open(args.weight_file, "r") as wf:

        num_examples = len(args.examples)
        examples = sorted(args.examples)
        plot_nums = 100*num_examples+10
        fig = plt.figure(figsize=(100, 10))

        # for now just take the first num_examples sentences
        i = 0
        plot_num = 1
        for src_line, trg_line, weights_line in zip(sf, tf, wf):
            if i in args.examples:
                # log weights
                weights = np.exp(
                    np.array([float(w) for w in weights_line.strip().split(" ")]))
                #print(weights)
                ax = fig.add_subplot(plot_nums+plot_num)
                cax = ax.imshow(np.expand_dims(weights,0), aspect="auto", cmap="viridis")
                # eos is included in weights, but not in targets
                trg = trg_line.strip().split(" ")+["</s>"]
                assert len(trg) == np.shape(weights)[0]
                ax.set_title(label=str(i)+": "+src_line.strip())
                ax.set_xticklabels(trg)
                ax.xaxis.set_ticks_position('bottom')
                ax.xaxis.set_label_position('bottom')
                ax.set_xticks(np.arange(0, len(trg), 1), minor=False)
                ax.get_yaxis().set_visible(False)
                plot_num += 1
            i += 1

        fig.subplots_adjust(hspace=1) #right=0.8,
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(cax, cax=cbar_ax)

        plt.show()

