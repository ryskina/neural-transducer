import sys

lang = sys.argv[1]
mode = sys.argv[2]

# for lang in 'ben', 'ceb', 'hin', 'kaz', 'kir', 'mlt', 'orm', 'sna', 'swa', 'tgk', 'tgl', 'zul':
for split in 'trn', 'dev', 'tst':
    ff = open(f"/projects/tir4/users/mryskina/morphological-inflection/task0-data/split-by-lemma/grapheme/{lang}-{mode}.{split}", "w+")
    with open(f"/projects/tir4/users/mryskina/morphological-inflection/LemmaSplitting/data_NN_TSV/{lang}.{split}.tsv") as f:
        for line in f:
            i, o = line.strip().split('\t')
            i, t = i.split('$')
            i = "".join(i.split(','))
            o = "".join(o.split(','))
            t = ";".join([x for x in t.split(',') if x])
            ff.write(f"{i}\t{o}\t{t}\n")
    ff.close()