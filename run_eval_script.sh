#!/bin/sh


python eval_selection_model.py --model=select --setname=test --use_annotated_testset=False --model_path=./logs/select_batch12_candi10_seed42/model --is_aux_model=False

python eval_selection_model.py --model=select --setname=test --use_annotated_testset=False --model_path=./logs/select_batch4_candi10-uw-attntop0.1-bce-alpha0.1/model  --is_aux_model=True

python eval_selection_model.py --model=select --setname=test --use_annotated_testset=False --model_path=./logs/select_batch4_candi10-uw-attntop0.1-margin0.5-alpha0.1/model --is_aux_model=True

python eval_selection_model.py  --model=select --setname=test --use_annotated_testset=False --model_path=./logs/select_batch4_candi10-uw-rand0.1-bce-alpha0.1/model --is_aux_model=True 

python eval_selection_model.py  --model=select --setname=test --use_annotated_testset=False --model_path=./logs/select_batch4_candi10-uw-rand0.1-margin0.5-alpha0.1/model --is_aux_model=True

#python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./annotation/nw_ic_attnratio0.7_contextturn4.txt --annotated_testset_attribute=ic_attnratio0.7_contextturn4 --replace_annotated_testset_into_original=False --is_ic=True
#python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./annotation/nw_ic_attnratio0.7_contextturn4.txt --annotated_testset_attribute=ic_attnratio0.7_contextturn4 --replace_annotated_testset_into_original=False --is_ic=True
#python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./annotation/nw_ic_attnratio0.7_contextturn4.txt --annotated_testset_attribute=ic_attnratio0.7_contextturn4 --replace_annotated_testset_into_original=False --is_ic=True
#python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./annotation/nw_ic_attnratio0.7_contextturn4.txt --annotated_testset_attribute=ic_attnratio0.7_contextturn4 --replace_annotated_testset_into_original=False --is_ic=True

#python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./annotation/nw_ic_attnratio0.7_contextturn4.txt --annotated_testset_attribute=ic_attnratio0.7_contextturn4 --replace_annotated_testset_into_original=True --is_ic=True
#python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./annotation/nw_ic_attnratio0.7_contextturn4.txt --annotated_testset_attribute=ic_attnratio0.7_contextturn4 --replace_annotated_testset_into_original=True --is_ic=True
#python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./annotation/nw_ic_attnratio0.7_contextturn4.txt --annotated_testset_attribute=ic_attnratio0.7_contextturn4 --replace_annotated_testset_into_original=True --is_ic=True
#python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./annotation/nw_ic_attnratio0.7_contextturn4.txt --annotated_testset_attribute=ic_attnratio0.7_contextturn4 --replace_annotated_testset_into_original=True --is_ic=True
