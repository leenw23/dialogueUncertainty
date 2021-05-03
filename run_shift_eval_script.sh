#!/bin/sh

python eval_selection_model.py --corpus=persona --setname=dev
python eval_selection_model.py --corpus=persona --setname=test

python eval_selection_model.py --model=mcdrop --setname=dev --corpus=persona
python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=dev  --corpus=persona
python eval_selection_model.py --model=ensemble --setname=dev  --corpus=persona


python eval_selection_model.py --model=mcdrop --setname=test  --corpus=persona
python eval_selection_model.py --model=ensemble --setname=test  --corpus=persona
python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test  --corpus=persona
