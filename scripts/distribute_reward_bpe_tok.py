import sys

# Distribute rewards for post-processed targets over pre-processed targets.
# Background: rewards are obtained for natural-looking text (i.e. no tokenization), but MT model is trained on BPE-splits on tokenized text. We need a way to infer rewards for the MT-tokens from the annotations.
# Pre-processing covers tokenization and BPE-splits.

# Usage: python distribute_reward_bpe_tok.py target.preprocessed target.raw target.annotated target.annotated.preprocessed

if __name__ == "__main__":
    preprocessed_file = sys.argv[1]  # bped, tokenized
    raw_file = sys.argv[2]  # as annotated
    reward_file = sys.argv[3]  # annotations for raw_file
    new_reward_file = sys.argv[4]  # output file

    with open(preprocessed_file, "r") as pfile, open(raw_file, "r") as rafile, open(reward_file, "r") as refile, open(new_reward_file, "w") as nrefile:
        counter = 0
        for pline, raline, reline in zip(pfile, rafile, refile):

            ptokens = pline.strip().split()
            ratokens = raline.strip().split()
            retokens = reline.strip().split()

            assert len(ratokens) == len(retokens)

            # need to duplicate the right rewards so that all ptokens get a reward

            prewards = []
            j = 0  # index over raw tokens
            i = 0  # index over preprocessed tokens
            in_bpe = False
            inside_counter = 0
            while i < len(ptokens):  # loop over preprocessed tokens
                # perfect match
                if ratokens[j] == ptokens[i]:
                    prewards.append(retokens[j])
                    j += 1
                    i += 1
                # no match, either caused by BPE or tokenization or both
                else:
                    # check if there's a BPE marker
                    if "@@" in ptokens[i]:
                        # check if in raw token -> then it's a match and we can increase i
                        in_bpe = True
                        if ptokens[i].strip("@") in ratokens[j]:
                            prewards.append(retokens[j])
                            i += 1
                    # if there is no marker, but we know we're inside a BPE-split word
                    elif in_bpe and ptokens[i] in ratokens[j]: 
                        prewards.append(retokens[j])
                        # only increase j if end matches, we have covered the complete raw token
                        if ratokens[j][-len(ptokens[i]):] == ptokens[i]:
                            j += 1
                            in_bpe = False
                        i += 1
                    # we're not insde a BPE-split word, but there is an overlap
                    elif ptokens[i] in ratokens[j]:
                        # keep track how much of the raw token we have covered
                        inside_counter += len(ptokens[i])
                        prewards.append(retokens[j])
                        # check if suffix matches, then also j counter needs to get increased
                        if ratokens[j][-len(ptokens[i]):] == ptokens[i] and inside_counter >= len(ratokens[j]):
                            j += 1
                            inside_counter = 0
                        i += 1
                    else:
                        print("Unknown case.")
                        sys.exit(-1)
            assert len(prewards) == len(ptokens) 
            nrefile.write(" ".join(prewards)+"\n")
