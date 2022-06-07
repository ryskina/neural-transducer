"""
Borrowed from https://github.com/antonisa/inflection
"""
import argparse
import codecs
import os
import sys
from random import choice, random, seed
from typing import Any, List

sys.path.append("src")
import align  # noqa: E402

seed(42)

# A function from LemmaSplitting to work with the provided data format
# Borrowed from https://github.com/OnlpLab/LemmaSplitting/blob/master/lstm/utils.py
def get_langs_and_paths(data_dir):
    train_paths = {}
    dev_paths = {}

    for family in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, family)):
            lang, ext = os.path.splitext(filename)
            if len(lang) != 3:
                continue
            filename = os.path.join(data_dir,family,filename)
            if ext=='.trn':
                train_paths[lang] = filename
            elif ext=='.dev':
                dev_paths[lang] = filename

    langs = train_paths.keys()
    files_paths = {k:(train_paths[k],dev_paths[k]) for k in langs}
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


def augment(inputs, outputs, tags, characters):
    temp = [("".join(inputs[i]), "".join(outputs[i])) for i in range(len(outputs))]
    aligned = align.Aligner(temp, align_symbol=" ").alignedpairs

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
            # removing trailing whitespaces
            new_i = "".join(new_i).strip()
            new_o = "".join(new_o).strip()
            new_i1 = [
                c
                for idx, c in enumerate(new_i)
                if (c.strip() or (len(inputs[k]) > idx and inputs[k][idx] == " " and new_i[idx] == " "))
            ]
            new_o1 = [
                c
                for idx, c in enumerate(new_o)
                if (c.strip() or (len(outputs[k]) > idx and outputs[k][idx] == " " and new_o[idx] == " "))
            ]
            new_inputs.append(new_i1)
            new_outputs.append(new_o1)
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
LANG = args.language

data_files_per_lang = get_langs_and_paths(DATA_PATH)
TRN_PATH, DEV_PATH = data_files_per_lang[LANG]

N = args.examples
usedev = args.use_dev

trni, trno, trnt = read_data(TRN_PATH)
devi, devo, devt = read_data(DEV_PATH)

if usedev:
    vocab = get_chars(trni + trno + devi + devo)
else:
    vocab = get_chars(trni + trno)

i: List[Any] = []
o: List[Any] = []
t: List[Any] = []
while len(i) < N:
    if usedev:
        # Do augmentation also using examples from dev
        ii, oo, tt = augment(devi + trni, devo + trno, devt + trnt, vocab)
    else:
        # Just augment the training set
        ii, oo, tt = augment(trni, trno, trnt, vocab)
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

OUT_PATH = os.path.splitext(data_files_per_lang[LANG][0])[0]
with codecs.open(OUT_PATH + ".hall", "w", "utf-8") as outp:
    for k in range(min(N, len(i))):
        outp.write("".join(i[k]) + "\t" + "".join(o[k]) + "\t" + ";".join(t[k]) + "\n")

lemmas = []
with codecs.open(OUT_PATH + ".hall.trn", "w", "utf-8") as outp:
    with codecs.open(OUT_PATH + ".trn", "r", "utf-8") as inp:
        for line in inp:
            lemma, form, tags = line.strip().split('\t')
            if lemma not in lemmas:
                outp.write(f"{lemma}\t{lemma}\tCOPY\n")
                lemmas.append(lemma)
            outp.write(line)

    with codecs.open(OUT_PATH + ".hall", "r", "utf-8") as inp:
        for line in inp:
            lemma, form, tags = line.strip().split('\t')
            if lemma not in lemmas:
                outp.write(f"{lemma}\t{lemma}\tCOPY\n")
                lemmas.append(lemma)
            outp.write(line)

    outp.close()
