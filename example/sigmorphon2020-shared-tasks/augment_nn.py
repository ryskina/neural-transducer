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
import gensim
import epitran

sys.path.append("src")
import align  # noqa: E402

seed(42) 

fasttext_lang_codes = {
    'ben': 'bn',
    'ceb': 'ceb',
    'hin': 'hi',
    'kaz': 'kk',
    'kir': 'ky',
    'mlt': 'mt',
    'swa': 'sw',
    'tgk': 'tg',
    'tgl': 'tl',
    'elr': 'en'
}

epitran_lang_codes = {
    'ben': 'ben-Beng',
    'ceb': 'ceb-Latn',
    'hin': 'hin-Deva', 
    'kaz': 'kaz-Cyrl',
    'kir': 'kir-Cyrl',
    'mlt': 'mlt-Latn',
    'orm': 'orm-Latn',
    'sna': 'sna-Latn',
    'swa': 'swa-Latn',
    'tgk': 'tgk-Cyrl',
    'tgl': 'tgl-Latn',
    'zul': 'zul-Latn',
    'elr': 'eng-Latn'
}

# A function from LemmaSplitting to work with the provided data format
# Borrowed from https://github.com/OnlpLab/LemmaSplitting/blob/master/lstm/utils.py
def get_langs_and_paths(data_dir, use_hall=False):
    train_paths = {}
    test_paths = {}
    dev_paths = {}

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
            elif ext=='.dev':
                dev_paths[lang] = filename
            elif ext=='.tst':
                test_paths[lang] = filename
            family_name = os.path.split(family)[1]

    langs = train_paths.keys() # not all languages have hall data
    files_paths = {k:(train_paths[k],dev_paths[k],test_paths[k]) for k in langs}
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

def common_suffix(candidate_tuple, lemma):
    # candidate_segments = candidate_tuple[0].split(" ")
    # lemma_segments = lemma.split(" ")
    # j = -1
    # lcs_length = 0
    # for j in range(-1, -min(len(candidate_segments), len(lemma_segments))-1, -1):
    #     lcs_length += len(os.path.commonprefix([candidate_segments[j][::-1], lemma_segments[j][::-1]]))
    lcs_length = len(os.path.commonprefix([candidate_tuple[0][::-1], lemma[::-1]]))
    return -lcs_length

def cosine_distance(candidate_tuple, lemma):
    return 1-WV.similarity(candidate_tuple[0], "".join(lemma))

def epitran_distance(candidate_tuple, lemma):
    lemma_ipa = EPI.transliterate("".join(lemma))
    candidate_ipa = EPI.transliterate(candidate_tuple[0])
    return editdistance.eval(candidate_ipa, lemma_ipa)

def rerank_and_choose_best(lemma, candidates):
    neighbors = find_nearest_neighbors(lemma, candidates, n=10)
    if len(neighbors) == 1:
        return neighbors[0]
    best_neighbor = neighbors[0]
    best_distance =  cosine_distance(neighbors[0], lemma)
    for neighbor in neighbors[1:]:
        neighbor_distance = cosine_distance(neighbor, lemma)
        if neighbor_distance < best_distance:
            best_distance = neighbor_distance
            best_neighbor = neighbor
    return best_neighbor

def find_nearest_neighbors(lemma, candidates, distance_function=levenshtein, n=1):
    if candidates:
        sorted_candidates = sorted(candidates.items(), key=lambda x: distance_function(x, lemma))
        if sorted_candidates[0][0] == lemma and len(sorted_candidates) > 1:
            return sorted(candidates.items(), key=lambda x: distance_function(x, lemma))[1:n+1]
        else:
            return sorted(candidates.items(), key=lambda x: distance_function(x, lemma))[0:n]
    else:
        # if this combination of tags has never appeared, COPY is our best guess
        return [("".join(lemma), "".join(lemma))]

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
                    # if the hallucination is the same as the input, keep trying
                    while "".join(new_i) == "".join(i):
                        for j in range(s, e):
                            if random() > 0.75:  # arbitrary value
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
            else:
                # if unable to hallucinate, augment with nearest neighbor
                new_i, new_o = find_nearest_neighbors("".join(inputs[k]), CELL_DICT[';'.join(tags[k])])[0]
                new_i1 = list(new_i)
                new_o1 = list(new_o)

            new_inputs.append(inputs[k] + ['&'] + new_i1 + ['#'] + new_o1)
            new_outputs.append(outputs[k])
            new_tags.append(tags[k])

    elif mode == "baseline1":
        # augment with nearest neighbor found in CELL_DICT for the right tag combo
        for i, o, t in zip(inputs, outputs, tags):
            new_i, new_o = find_nearest_neighbors("".join(i), CELL_DICT[';'.join(t)])[0]
            new_inputs.append(i + ['&'] + list(new_i) + ['#'] + list(new_o))
            new_outputs.append(o)
            new_tags.append(t)

    elif mode == "oracle1.5":
        # augment with nearest neighbor found in CELL_DICT for the right tag combo
        for i, o, t in zip(inputs, outputs, tags):
            for new_i, new_o in find_nearest_neighbors("".join(i), CELL_DICT[';'.join(t)], n=10):
                new_inputs.append(i + ['&'] + list(new_i) + ['#'] + list(new_o))
                new_outputs.append(o)
                new_tags.append(t)

    elif mode == "baseline2":
        for i, o, t in zip(inputs, outputs, tags):
            new_i, new_o = find_nearest_neighbors("".join(i), CELL_DICT[';'.join(t)], 
                distance_function=common_suffix)[0]
            new_inputs.append(i + ['&'] + list(new_i) + ['#'] + list(new_o))
            new_outputs.append(o)
            new_tags.append(t)

    elif mode.startswith("baseline3"):
        for i, o, t in zip(inputs, outputs, tags):
            new_i, new_o = find_nearest_neighbors("".join(i), CELL_DICT[';'.join(t)], 
                distance_function=cosine_distance)[0]
            new_inputs.append(i + ['&'] + list(new_i) + ['#'] + list(new_o))
            new_outputs.append(o)
            new_tags.append(t)

    elif mode == "baseline4":
        for i, o, t in zip(inputs, outputs, tags):
            new_i, new_o = rerank_and_choose_best("".join(i), CELL_DICT[';'.join(t)])
            new_inputs.append(i + ['&'] + list(new_i) + ['#'] + list(new_o))
            new_outputs.append(o)
            new_tags.append(t)

    elif mode == "baseline5":
        for i, o, t in zip(inputs, outputs, tags):
            new_i, new_o = find_nearest_neighbors("".join(i), CELL_DICT[';'.join(t)], 
                distance_function=epitran_distance)[0]
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
    choices=["oracle0", "oracle1", "baseline1", "oracle1.5", "baseline2", 
             "baseline3", "baseline3.2", "baseline4", "baseline5"],
    type=str,
)
args = parser.parse_args()

DATA_PATH = args.datapath
LANG = args.language

data_files_per_lang = get_langs_and_paths(DATA_PATH, use_hall=True)
TRN_PATH, DEV_PATH, TST_PATH = data_files_per_lang[LANG]

train_ins, train_outs, train_tags = read_data(TRN_PATH)
dev_ins, dev_outs, dev_tags = read_data(DEV_PATH)
test_ins, test_outs, test_tags = read_data(TST_PATH)

# recording all lemma-form pairs per paradigm cell
CELL_DICT = defaultdict(dict)
if args.mode == "baseline3.2":
    TRN_PATH_REAL, _, _ = get_langs_and_paths(DATA_PATH, use_hall=False)[LANG]
    train_ins_real, train_outs_real, train_tags_real = read_data(TRN_PATH_REAL)
    for lemma, form, tags in zip(train_ins_real, train_outs_real, train_tags_real):
        CELL_DICT[';'.join(tags)]["".join(lemma)] = "".join(form)
else:
    for lemma, form, tags in zip(train_ins, train_outs, train_tags):
        CELL_DICT[';'.join(tags)]["".join(lemma)] = "".join(form)

train_augment_mode = "oracle1" if args.mode in ["baseline1", "baseline2", "oracle1.5", 
                        "baseline3", "baseline3.2", "baseline4", "baseline5"] else args.mode
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
        #TODO: allow adding NEW hallucinations to CELL_DICT

# augmenting the dev and test files
if test_augment_mode.startswith("baseline3") or test_augment_mode == "baseline4":
    if LANG not in fasttext_lang_codes:
        print(f"{LANG} not in FastText")
        exit()
    else:
        WV = gensim.models.fasttext. \
        load_facebook_vectors(f"/projects/tir4/users/mryskina/morphological-inflection/fasttext/cc.{fasttext_lang_codes[LANG]}.300.bin.gz")

if test_augment_mode == "baseline5":
    if LANG not in epitran_lang_codes:
        print(f"{LANG} not in Epitran")
        exit()
    else:
        EPI = epitran.Epitran(epitran_lang_codes[LANG])

ii, oo, tt = augment(dev_ins, dev_outs, dev_tags, vocab, mode=test_augment_mode)
i = [c for c in ii if c]
o = [c for c in oo if c]
t = [c for c in tt if c]

with codecs.open(f"{OUT_PATH}/{LANG}.dev.tsv", "w", "utf-8") as outp:
    for k in range(len(i)):
        outp.write(",".join(i[k] + ['$'] + t[k]) + "\t" + ",".join(o[k]) + "\n")

ii, oo, tt = augment(test_ins, test_outs, test_tags, vocab, mode=test_augment_mode)
i = [c for c in ii if c]
o = [c for c in oo if c]
t = [c for c in tt if c]

with codecs.open(f"{OUT_PATH}/{LANG}.tst.tsv", "w", "utf-8") as outp:
    for k in range(len(i)):
        outp.write(",".join(i[k] + ['$'] + t[k]) + "\t" + ",".join(o[k]) + "\n")
