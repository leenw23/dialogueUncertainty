#!/bin/bash

# margin, attention vs random, alpha0.1
python train_bert_selection_with_auxilary.py --corrupt_loss_type=margin --uw_corruption=True --uw_corrupt_ratio=0.1 --attention_for_uw_corruption=True --alpha=0.1 --margin=0.5 --batch_size=4
python train_bert_selection_with_auxilary.py --corrupt_loss_type=margin --uw_corruption=True --uw_corrupt_ratio=0.1 --attention_for_uw_corruption=False --alpha=0.1 --margin=0.5 --batch_size=4

# bce, attention vs random, alpha0.1
python train_bert_selection_with_auxilary.py --corrupt_loss_type=crossentropy --uw_corruption=True --uw_corrupt_ratio=0.1 --attention_for_uw_corruption=True --alpha=0.1 --margin=0.5 --batch_size=4
python train_bert_selection_with_auxilary.py --corrupt_loss_type=crossentropy --uw_corruption=True --uw_corrupt_ratio=0.1 --attention_for_uw_corruption=False --alpha=0.1 --margin=0.5 --batch_size=4

# margin, attention vs random, alpha0.5
python train_bert_selection_with_auxilary.py --corrupt_loss_type=margin --uw_corruption=True --uw_corrupt_ratio=0.1 --attention_for_uw_corruption=True --alpha=0.5 --margin=0.5 --batch_size=4
python train_bert_selection_with_auxilary.py --corrupt_loss_type=margin --uw_corruption=True --uw_corrupt_ratio=0.1 --attention_for_uw_corruption=False --alpha=0.5 --margin=0.5 --batch_size=4

# bce, attention vs random, alpha0.5
python train_bert_selection_with_auxilary.py --corrupt_loss_type=crossentropy --uw_corruption=True --uw_corrupt_ratio=0.1 --attention_for_uw_corruption=True --alpha=0.5 --margin=0.5 --batch_size=4
python train_bert_selection_with_auxilary.py --corrupt_loss_type=crossentropy --uw_corruption=True --uw_corrupt_ratio=0.1 --attention_for_uw_corruption=False --alpha=0.5 --margin=0.5 --batch_size=4

