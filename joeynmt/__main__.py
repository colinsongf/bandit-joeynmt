import argparse

from joeynmt.training import train
from joeynmt.prediction import test, lm

def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test", "lm"],
                    help="train a model or test")

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    ap.add_argument("--ckpt", type=str,
                    help="checkpoint for prediction")

    ap.add_argument("--output_path", type=str,
                    help="path for saving translation output")

    ap.add_argument("--save_attention", action="store_true",
                    help="save attention visualizations")

    ap.add_argument("--size", type=int, default=1000,
                    help="generate data set of this size")


    args = ap.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt,
             output_path=args.output_path, save_attention=args.save_attention)
    elif args.mode == "lm":
        lm(cfg_file=args.config_path, ckpt=args.ckpt,
           output_path=args.output_path, size=args.size)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
