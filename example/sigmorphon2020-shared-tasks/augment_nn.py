"""
Nearest neighbor augmentation script
Borrowed from https://github.com/antonisa/inflection
"""
import argparse
import codecs
import os
import sys
from random import choice, random, seed
from typing import Any, List
from collections import defaultdict
import editdistance

sys.path.append("src")
import align  # noqa: E402

seed(42) 

# A function from LemmaSplitting to work with the provided data format
# Borrowed from https://github.com/OnlpLab/LemmaSplitting/blob/master/lstm/utils.py
def get_langs_and_paths(data_dir, use_hall=False):
    train_paths = {}
    test_paths = {}

    for family in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, family)):
            lang, ext = os.path.splitext(filename)
            filename = os.path.join(data_dir,family,filename)
            hall_file = False
            if lang.endswith('.hall'):
                lang = lang[:-5]
                hall_file = True
            if ext=='.trn':
                if hall_file == use_hall:
                    train_paths[lang] = filename
            elif ext=='.tst':
                test_paths[lang] = filename
            family_name = os.path.split(family)[1]

    langs = train_paths.keys() # not all languages have hall data
    files_paths = {k:(train_paths[k],test_paths[k]) for k in langs}
    return files_paths


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

def levenshtein(candidate_tuple, lemma):
    return editdistance.eval(candidate_tuple[0], "".join(lemma))

def find_nearest_neighbor(lemma, candidates, similarity_function=levenshtein):
    if candidates:
        top_input, top_output = sorted(candidates.items(), key=lambda x: similarity_function(x, lemma))[0]
    else:
        # if this combination of tags has never appeared, COPY is our best guess
        top_input = top_output = "".join(lemma)
    return top_input, top_output

def augment(inputs, outputs, tags, characters, mode):
    new_inputs = []
    new_outputs = []
    new_tags = []
    
    if mode == "oracle0":
        # augment only with the gold answer
        for i, o, t in zip(inputs, outputs, tags):
            new_inputs.append(i + ['&'] + i + ['#'] + o)
            new_outputs.append(o)
            new_tags.append(t)

    elif mode == "oracle1": 
        # augment with a hallucination based on the gold answer
        temp = [("".join(inputs[i]), "".join(outputs[i])) for i in range(len(outputs))]
        aligned = align.Aligner(temp, align_symbol=" ").alignedpairs

        vocab = list(characters)
        try:
            vocab.remove(u" ")
        except ValueError:
            pass
        # TODO: investigate spaces before lemma for some langs
        for k, item in enumerate(aligned):
            i, o = item[0], item[1]
            i1 = [
                c
                for idx, c in enumerate(i)
                if (c.strip() or (i[idx] == " " and o[idx] == " "))
            ]
            o1 = [
                c
                for idx, c in enumerate(o)
                if (c.strip() or (i[idx] == " " and o[idx] == " "))
            ]

            good_range = find_good_range(i, o)
            # print(good_range)
            if good_range:
                new_i, new_o = list(i), list(o)
                for r in good_range:
                    s = r[0]
                    e = r[1]
                    if e - s > 5:  # arbitrary value
                        s += 1
                        e -= 1
                    for j in range(s, e):
                        if random() > 0.75:  # arbitrary value
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
            else:
                # if unable to hallucinate, augment with nearest neighbor
                new_i, new_o = find_nearest_neighbor("".join(i1), CELL_DICT[';'.join(tags[k])])
                new_i1 = list(new_i)
                new_o1 = list(new_o)

            new_inputs.append(i1 + ['&'] + new_i1 + ['#'] + new_o1)
            new_outputs.append(o1)
            new_tags.append(tags[k])

    elif mode == "baseline1":
        # augment with nearest neighbor found in CELL_DICT for the right tag combo
        for i, o, t in zip(inputs, outputs, tags):
            new_i, new_o = find_nearest_neighbor("".join(i), CELL_DICT[';'.join(t)])
            new_inputs.append(i + ['&'] + list(new_i) + ['#'] + list(new_o))
            new_outputs.append(o)
            new_tags.append(t)

    return new_inputs, new_outputs, new_tags


def get_chars(words):
    flat_list = [char for word in words for char in word]
    return list(set(flat_list))


parser = argparse.ArgumentParser()
parser.add_argument("datapath", help="path to data", type=str)
parser.add_argument("language", help="language", type=str)
parser.add_argument(
    "--mode",
    help="type of oracle or baseline",
    choices=["oracle0", "oracle1", "baseline1"],
    type=str,
)
args = parser.parse_args()

DATA_PATH = args.datapath
LANG = args.language

data_files_per_lang = get_langs_and_paths(DATA_PATH, use_hall=True)
TRN_PATH, TST_PATH = data_files_per_lang[LANG]

train_ins, train_outs, train_tags = read_data(TRN_PATH)
test_ins, test_outs, test_tags = read_data(TST_PATH)

# recording all lemma-form pairs per paradigm cell
CELL_DICT = defaultdict(dict)
for lemma, form, tags in zip(train_ins, train_outs, train_tags):
    CELL_DICT[';'.join(tags)]["".join(lemma)] = "".join(form)

train_augment_mode = "oracle1" if args.mode == "baseline1" else args.mode
test_augment_mode = args.mode

# augmenting the training file
vocab = get_chars(train_ins + train_outs)
ii, oo, tt = augment(train_ins, train_outs, train_tags, vocab, mode=train_augment_mode)
i = [c for c in ii if c]
o = [c for c in oo if c]
t = [c for c in tt if c]

# Outputs directly into TSV format for the lemma-splitting code
OUT_PATH = "/projects/tir4/users/mryskina/morphological-inflection/LemmaSplitting/data_NN_TSV"
with codecs.open(f"{OUT_PATH}/{LANG}.trn.tsv", "w", "utf-8") as outp:
    for k in range(len(i)):
        outp.write(",".join(i[k] + ['$'] + t[k]) + "\t" + ",".join(o[k]) + "\n")
        #TODO: allow adding hallucinations to CELL_DICT

# augmenting the test file
ii, oo, tt = augment(test_ins, test_outs, test_tags, vocab, mode=test_augment_mode)
i = [c for c in ii if c]
o = [c for c in oo if c]
t = [c for c in tt if c]

with codecs.open(f"{OUT_PATH}/{LANG}.tst.tsv", "w", "utf-8") as outp:
    for k in range(len(i)):
        outp.write(",".join(i[k] + ['$'] + t[k]) + "\t" + ",".join(o[k]) + "\n")
