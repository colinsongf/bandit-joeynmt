# coding: utf-8
import sys
import sacrebleu

if __name__ == "__main__":
    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    out_file = hyp_file+".sbleu.weights"

    with open(ref_file, "r") as ref_f, open(hyp_file, "r") as hyp_f, \
        open(out_file, "w") as out_f:
        for ref, hyp in zip(ref_f, hyp_f):
            sbleu = sacrebleu.raw_corpus_bleu(
                    sys_stream=[hyp],
                    ref_streams=[[ref]]).score/100
            out_f.write("{}\n".format(sbleu))