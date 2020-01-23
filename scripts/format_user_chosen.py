import sys

# Re-format targets and annotations from user choice session such that they have only 1/0s in the annotation file and translations (post-edited or model outputs) in the target file

# Usage: python format_user_chosen.py user.target user.annot user.format.target user.format.annot

if __name__ == "__main__":
    annots = sys.argv[2]
    targets = sys.argv[1]
    new_targets = sys.argv[3]
    new_annots = sys.argv[4]

    # create two new files:
    # one with targets (PEd or marked)
    # one with scores (1s or marked)

    def is_marking(line):
        try:
            words = [int(l) for l in line.strip().split()]
            tokens = set(words)
            if len(tokens) <= 2:
                return True
        except:
            return False

    with open(annots, "r") as afile, open(targets, "r") as tfile, open(new_annots, "w") as nafile, open(new_targets, "w") as ntfile:
        for aline, tline in zip(afile, tfile):
            if is_marking(aline):
                nafile.write(aline)
                ntfile.write(tline)
            else:
                nafile.write(" ".join(["1"]*len(aline.strip().split()))+"\n")
                ntfile.write(aline)

