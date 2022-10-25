"""
Borrowed from https://github.com/antonisa/inflection
"""
import argparse
import codecs
import os
import sys
from random import choice, random
from typing import Any, List

from collections import Counter, defaultdict
import math

sys.path.append("src")
import align  # noqa: E402

def get_bigram_probs():
    wordlist = []
    start = "<s>"
    stop = "</s>"
    alphabet = set()
    with open(LOW_PATH) as f:
        for line in f:
            lemma, form, tags = line.strip().split('\t')
            wordlist += [[start] + list(lemma) + [stop], [start] + list(form) + [stop]]
            alphabet.update(list(lemma))
            alphabet.update(list(form))

    alphabet.add(stop)
    bigram_probs = defaultdict(Counter)

    for word in wordlist:
        for i in range(len(word)-1):
            bigram_probs[word[i]][word[i+1]] += 1

    for token in bigram_probs:
        normalizer = sum(bigram_probs[token].values())
        for next_token in alphabet:
            bigram_probs[token][next_token] = (bigram_probs[token][next_token] + 1) / (normalizer + 1)

    return bigram_probs

def score(bigram_probs, word):
    logprob = math.log(bigram_probs["<s>"][word[0]])
    for i in range(len(word)-1):
        logprob += math.log(bigram_probs[word[i]][word[i+1]])
    logprob += math.log(bigram_probs[word[-1]]["</s>"])
    return logprob


def read_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for line in lines:
        line = line.strip().split("\t")
        if line:
            inputs.append(list(line[0].strip()))
            outputs.append(list(line[1].strip()))
            tags.append(line[2].strip().split(";"))
    return inputs, outputs, tags


def find_good_range(a, b):
    mask = [(a[i] == b[i] and a[i] != u" ") for i in range(len(a))]
    if sum(mask) == 0:
        return []
        # Some times the alignment is off-by-one
        b = " " + b
        mask = [(a[i] == b[i] and a[i] != u" ") for i in range(len(a))]
    ranges = []
    prev = False
    for i, k in enumerate(mask):
        if k and prev:
            prev = True
        elif k and not prev:
            start = i
            prev = True
        elif prev and not k:
            end = i
            ranges.append((start, end))
            prev = False
        elif not prev and not k:
            prev = False
    if prev:
        ranges.append((start, i + 1))
    ranges = [c for c in ranges if c[1] - c[0] > 2]
    return ranges


def augment(inputs, outputs, tags, characters):
    temp = [("".join(inputs[i]), "".join(outputs[i])) for i in range(len(outputs))]
    aligned = align.Aligner(temp, align_symbol=" ").alignedpairs

    bigram_probs = get_bigram_probs()

    vocab = list(characters)
    try:
        vocab.remove(u" ")
    except ValueError:
        pass

    new_inputs = []
    new_outputs = []
    new_tags = []
    for k, item in enumerate(aligned):
        # print(''.join(inputs[k]) + '\t' + ''.join(outputs[k]))
        i, o = item[0], item[1]
        good_range = find_good_range(i, o)
        # print(good_range)
        if good_range:
            best_score = -10000
            attempt = 0
            while attempt < 100:
                new_i, new_o = list(i), list(o)
                for r in good_range:
                    s = r[0]
                    e = r[1]
                    if e - s > 5:  # arbitrary value
                        s += 1
                        e -= 1
                    for j in range(s, e):
                        if random() > 0.5:  # arbitrary value
                            nc = choice(vocab)
                            new_i[j] = nc
                            new_o[j] = nc
                    new_i1 = [
                        c
                        for idx, c in enumerate(new_i)
                        if (c.strip() or (new_o[idx] == " " and new_i[idx] == " "))
                    ]
                    new_o1 = [
                        c
                        for idx, c in enumerate(new_o)
                        if (c.strip() or (new_i[idx] == " " and new_o[idx] == " "))
                    ]

                lemma = "".join(new_i1)
                form = "".join(new_o1)
                inp = [c for idx, c in enumerate(i)
                      if (c.strip() or (i[idx] == " " and o[idx] == " "))]
                outp = [c for idx, c in enumerate(o)
                      if (c.strip() or (i[idx] == " " and o[idx] == " "))]

                # print(attempt, lemma, form, "".join(inp), "".join(outp))
                if lemma != "".join(inp) or form != "".join(outp):
                    attempt += 1
                    new_score = (score(bigram_probs, lemma) + score(bigram_probs, form)) / (len(lemma) + len(form) + 4)
                    if new_score > best_score:
                        best_score = new_score
                        best_i1 = new_i1
                        best_o1 = new_o1

            new_inputs.append(best_i1)
            new_outputs.append(best_o1)
            new_tags.append(tags[k])
        else:
            new_inputs.append([])
            new_outputs.append([])
            new_tags.append([])

    return new_inputs, new_outputs, new_tags


def get_chars(words):
    flat_list = [char for word in words for char in word]
    return list(set(flat_list))


parser = argparse.ArgumentParser()
parser.add_argument("datapath", help="path to data", type=str)
parser.add_argument("language", help="language", type=str)
parser.add_argument(
    "--examples",
    help="number of hallucinated examples to create (def: 10000)",
    default=10000,
    type=int,
)
parser.add_argument(
    "--use_dev",
    help="whether to use the development set (def: False)",
    action="store_true",
)
args = parser.parse_args()

DATA_PATH = args.datapath
L2 = args.language
LOW_PATH = os.path.join(DATA_PATH, L2 + ".trn")
DEV_PATH = os.path.join(DATA_PATH, L2 + ".dev")

N = args.examples
usedev = args.use_dev

lowi, lowo, lowt = read_data(LOW_PATH)
devi, devo, devt = read_data(DEV_PATH)

train_items = []
with open(LOW_PATH) as f:
    for line in f:
        lemma, form, tags = line.strip().split('\t')
        train_items.append((lemma, tags))

if usedev:
    vocab = get_chars(lowi + lowo + devi + devo)
else:
    vocab = get_chars(lowi + lowo)

i: List[Any] = []
o: List[Any] = []
t: List[Any] = []
while len(i) < N:
    if usedev:
        # Do augmentation also using examples from dev
        ii, oo, tt = augment(devi + lowi, devo + lowo, devt + lowt, vocab)
    else:
        # Just augment the training set
        ii, oo, tt = augment(lowi, lowo, lowt, vocab)
    ii = [c for c in ii if c]
    oo = [c for c in oo if c]
    tt = [c for c in tt if c]
    i += ii
    o += oo
    t += tt
    if len(ii) == 0:
        break

# Wait is this needed?
i = [c for c in i if c]
o = [c for c in o if c]
t = [c for c in t if c]

# item_scores = Counter()

# for k in range(min(N, len(i))):
#     lemma = "".join(i[k])
#     form = "".join(o[k])
#     tags = ";".join(t[k])
#     if (lemma, tags) in train_items:
#         continue
#     item_scores[(lemma, form, tags)] = (score(bigram_probs, lemma) + score(bigram_probs, form)) / (len(lemma) + len(form) + 4)

# with codecs.open(os.path.join(DATA_PATH, L2 + ".hall.lm"), "w", "utf-8") as outp:
#     for instance, score in item_scores.most_common(args.examples):
#         lemma, form, tags = instance
#         outp.write(f"{lemma}\t{form}\t{tags}\n")

with codecs.open(os.path.join(DATA_PATH, L2 + ".hall.lm2"), "w", "utf-8") as outp:
    for k in range(min(N, len(i))):
        outp.write("".join(i[k]) + "\t" + "".join(o[k]) + "\t" + ";".join(t[k]) + "\n")

