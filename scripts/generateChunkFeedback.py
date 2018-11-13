# coding: utf-8
import sys

if __name__ == "__main__":
    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    out_file = hyp_file+".chunk.weights"

    with open(ref_file, "r") as ref_f, open(hyp_file, "r") as hyp_f, \
        open(out_file, "w") as out_f:
        for ref, hyp in zip(ref_f, hyp_f):
            for i, hyp_token in enumerate(hyp.strip().split(" ")):
                ref_tokens = ref.strip().split(" ")
                if i > len(ref_tokens):
                    f = 0
                elif ref_tokens[i] == hyp_token:
                    f = 1
                else:
                    f = 0
                out_f.write("{} ".format(f))
            out_f.write("\n")