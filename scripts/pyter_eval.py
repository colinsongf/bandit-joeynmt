import pyter
import sys
import numpy as np

if __name__ == "__main__":
    refs = []
    with open(sys.argv[1], "r") as ref:
        for line in ref:
            refs.append(line.strip().split())
    hyps = []
    for line in sys.stdin:
        hyps.append(line.strip().split())

    assert len(refs) == len(hyps)
    print("Scoring {} refs/hyps with PyTER.".format(len(refs)))

    ters = []
    total_edits = 0
    total_ref_tokens = 0
    for h, r in zip(hyps, refs):
        if len(r) == 0:
            ters.append(0)
        else:
            edit_dist = pyter.edit_distance(h, r)
            total_edits += edit_dist
            total_ref_tokens += len(r)
            ters.append(edit_dist/len(r))
    avg_ter = np.mean(ters)
    corpus_ter = total_edits / total_ref_tokens

    print("avg TER: ", avg_ter)
    print("corpus TER: ", corpus_ter)
