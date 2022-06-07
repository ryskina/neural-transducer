#!/bin/bash

# for lng in ben ceb hin kaz kir mlt orm sna swa tgk tgl zul; do
#     python example/sigmorphon2020-shared-tasks/augment.py ../LemmaSplitting/data $lng --examples 10000
# done

for lng in ben ceb hin kaz kir mlt orm sna swa tgk tgl zul; do
    echo $lng
    python example/sigmorphon2020-shared-tasks/augment_nn.py ../LemmaSplitting/data $lng --mode=oracle1
done

