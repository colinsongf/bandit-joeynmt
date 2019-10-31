import argparse
import sacrebleu
import pyter
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def sentence_bleu(h, r):
    return sacrebleu.sentence_bleu(hypothesis=h, reference=r)


def sentence_ter(h,r):
    edits = pyter.edit_distance(s=h.split(), t=r.split())
    return edits / len(r.split()) if len(r.split()) > 0 else 0


def sentence_random(h, r):
    # TODO tune stdev and mean
    return min(max(np.random.normal(0.5, 0.2), 0), 1)


def sentence_chrf(h,r):
    return sacrebleu.sentence_chrf(hypothesis=h, reference=r)


def sentence_simile(h, r):
    # get word embeddings, avg across sentence and compute cosine
    # requires subword splitting?
    pass


def sentence_exact_match(h, r):
    return float(h == r)


def token_exact_match(h, r):
    # 0 for edits, 1 for non-edits
    matches = [0.]*len(h.split())
    i = 0
    for hi, ri in zip(h.split(), r.split()):
        matches[i] = float(hi == ri)
        i += 1
    return matches


def token_random(h, r):
    # TODO tune edit rate
    edit_rate = 0.5
    probs = np.random.uniform(0, 1, size=(len(h.split())))
    return [float(p > edit_rate) for p in probs]


def token_lcs(h, r):
    pass


def token_lcs_all(h, r):
    pass


def _edit_matrix(s, t):
    """It's same as the Levenshtein distance"""
    l = pyter._gen_matrix(len(s) + 1, len(t) + 1, None)
    l[0] = [x for x, _ in enumerate(l[0])]
    for x, y in enumerate(l):
        y[0] = x
    for i, j in pyter.itrt.product(range(1, len(s) + 1), range(1, len(t) + 1)):
        l[i][j] = min(l[i - 1][j] + 1,  # 1) delete
                      l[i][j - 1] + 1,  # 2) insert
                      l[i - 1][j - 1] + (0 if s[i - 1] == t[j - 1] else 1))
                        # 3) replace or keep
    return l


def token_edit(h, r):
    # 0 for words that need edits, 1 for non-edit words
    # compute edit matrix
    h = h.split()
    r = r.split()
    m = _edit_matrix(h, r)
    # mark words that need to be replaced or deleted (no insertions here)
    # back-track through matrix
    rows = len(m)
    cols = len(m[0])
    current = (rows-1, cols-1)
    edits = []
    while current[0]>0:
        current_val = m[current[0]][current[1]]
        diagonal_val = m[current[0]-1][current[1]-1]
        up_val = m[current[0]-1][current[1]]
        #print(current)
        #print(h[current[0]-1], r[current[1]-1])
        if current_val == diagonal_val and h[current[0]-1] == r[current[1]-1]:
            # keep
            #print(current, "keep")
            current = (current[0]-1, current[1]-1)
            edits.append(1.)
        else:
            if current_val == diagonal_val+1:
                # replace
                #print(current, "replace")
                current = (current[0]-1, current[1]-1)
                edits.append(0.)
            elif current_val == up_val+1:
                # delete
                #print(current, "delete")
                current = (current[0]-1, current[1])
                edits.append(0.)
            else:
                # insert
                #print(current, "insert")
                current = (current[0], current[1]-1)
                # not captured in token edits for target
    # reverse since going backwards
    return reversed(edits)


def compute_token_reward(h, r, reward_type):
    reward_fun = {"lcs": token_lcs, "lcs_all": token_lcs_all,
                  "random": token_random,
                  "edit": token_edit, "match": token_exact_match}
    rewards = reward_fun[reward_type](h, r)
    return " ".join([str(r) for r in rewards])


def compute_sentence_reward(h, r, reward_type):
    reward_fun = {"ter": sentence_ter, "bleu": sentence_bleu,
                  "chrf": sentence_chrf, "simile": sentence_simile,
                  "match": sentence_exact_match, "random": sentence_random}
    reward = reward_fun[reward_type](h, r)
    return reward


def main(args):
    reward_suffix = ".{}.{}".format(args.reward_level, args.reward_type)
    all_rewards = []
    with open(args.target, "r") as tfile, open(args.reference, "r") as rfile, \
        open(args.target+reward_suffix, "w") as outfile:

        for hyp, ref in zip(tfile, rfile):
            # compute reward either token- or sentence-wise
            hyp = hyp.strip()
            ref = ref.strip()
            print("hyp", hyp)
            print("ref", ref)

            if args.reward_level == "token":
                # compute reward for each token
                rewards = compute_token_reward(hyp, ref, args.reward_type)
                print(len(rewards.split()), len(hyp.split()))
                assert len(rewards.split()) == len(hyp.split())
                print(*zip(rewards.split(), hyp.split()))
            else:
                # compute reward for the whole sentence
                rewards = compute_sentence_reward(hyp, ref, args.reward_type)
                print(rewards)

            all_rewards.extend([float(r) for r in rewards.split()])

            outfile.write("{}\n".format(rewards))

    print("Avg reward", np.mean(all_rewards))
    print("Std reward", np.std(all_rewards))

    plt.hist(all_rewards)
    plt.savefig(args.target+reward_suffix+".hist.pdf")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=str,
                    help="Path to MT output file (tokenized).")
    ap.add_argument("reference", type=str,
                    help="Path to reference output file (tokenized).")
    ap.add_argument("--reward_level", type=str, default="token",
                    help="Level on which rewards are computed.")
    ap.add_argument("--reward_type", type=str, default="exact_match",
                    help="Type of reward that is simulated.")
    args = ap.parse_args()

    main(args)
