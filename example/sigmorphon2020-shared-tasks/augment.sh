#!/bin/bash

# for lng in ben ceb hin kaz kir mlt orm sna swa tgk tgl zul; do
# for lng in elr; do
#     python example/sigmorphon2020-shared-tasks/augment.py ../LemmaSplitting/data $lng --examples 10000
# done

mode=baseline2

# for lng in ben ceb hin kaz kir mlt swa tgk tgl; do
# for lng in ben ceb hin kaz kir mlt orm sna swa tgk tgl zul; do
for lng in elr; do
    echo $lng
    python /projects/tir4/users/mryskina/morphological-inflection/neural-transducer/example/sigmorphon2020-shared-tasks/augment_nn.py ../LemmaSplitting/data $lng --mode=$mode
    python /projects/tir4/users/mryskina/morphological-inflection/neural-transducer/example/sigmorphon2020-shared-tasks/convert_tsv_to_trm.py $lng $mode
done