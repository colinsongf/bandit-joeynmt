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
    for h, r in zip(hyps, refs):
        if len(r) == 0:
            ters.append(0)
        else:
            ters.append(pyter.ter(inputwords=h, refwords=r))
    ter = np.mean(ters)

    print("avg TER: ", ter)
