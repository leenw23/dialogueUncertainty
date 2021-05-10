#!/bin/sh



#5
# IC baseline origin
python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=True --is_ic=True
python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=True --is_ic=True
python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=True --is_ic=True
#python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=True --is_ic=True

#6
# IC baseline changed
python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=False --is_ic=True
python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=False --is_ic=True
python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=False --is_ic=True
#python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=False --is_ic=True

#7
# IC proposed origin
python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=True --is_ic=True
python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=True --is_ic=True
python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=True --is_ic=True
#python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=True --is_ic=True

#8
# IC proposed changed
python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=False --is_ic=True
python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=False --is_ic=True
python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=False --is_ic=True
#python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=False --is_ic=True

#5
#IC filtered origin
python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/IC_auto_test_filtered.txt --annotated_testset_attribute=IC_auto_filtered --replace_annotated_testset_into_original=True --is_ic=True
python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/IC_auto_test_filtered.txt --annotated_testset_attribute=IC_auto_filtered --replace_annotated_testset_into_original=True --is_ic=True
python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/IC_auto_test_filtered.txt --annotated_testset_attribute=IC_auto_filtered --replace_annotated_testset_into_original=True --is_ic=True
#python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/IC_auto_test_filtered.txt --annotated_testset_attribute=IC_auto_filtered --replace_annotated_testset_into_original=True --is_ic=True

#6
# IC filtered changed
python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/IC_auto_test_filtered.txt --annotated_testset_attribute=IC_auto_filtered --replace_annotated_testset_into_original=False --is_ic=True
python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/IC_auto_test_filtered.txt --annotated_testset_attribute=IC_auto_filtered --replace_annotated_testset_into_original=False --is_ic=True
python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/IC_auto_test_filtered.txt --annotated_testset_attribute=IC_auto_filtered --replace_annotated_testset_into_original=False --is_ic=True
#python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/IC_auto_test_filtered.txt --annotated_testset_attribute=IC_auto_filtered --replace_annotated_testset_into_original=False --is_ic=True


# #1
# # UW baseline origin
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_rand.txt --annotated_testset_attribute=UK_auto_test_rand --replace_annotated_testset_into_original=True --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_rand.txt --annotated_testset_attribute=UK_auto_test_rand --replace_annotated_testset_into_original=True --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_rand.txt --annotated_testset_attribute=UK_auto_test_rand --replace_annotated_testset_into_original=True --is_ic=False
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_rand.txt --annotated_testset_attribute=UK_auto_test_rand --replace_annotated_testset_into_original=True --is_ic=False

# #2
# # UW baseline changed
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_rand.txt --annotated_testset_attribute=UK_auto_test_rand --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_rand.txt --annotated_testset_attribute=UK_auto_test_rand --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_rand.txt --annotated_testset_attribute=UK_auto_test_rand --replace_annotated_testset_into_original=False --is_ic=False
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_rand.txt --annotated_testset_attribute=UK_auto_test_rand --replace_annotated_testset_into_original=False --is_ic=False

# #3
# # UW proposed origin
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_method.txt --annotated_testset_attribute=UK_auto_test_method --replace_annotated_testset_into_original=True --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_method.txt --annotated_testset_attribute=UK_auto_test_method --replace_annotated_testset_into_original=True --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_method.txt --annotated_testset_attribute=UK_auto_test_method --replace_annotated_testset_into_original=True --is_ic=False
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_method.txt --annotated_testset_attribute=UK_auto_test_method --replace_annotated_testset_into_original=True --is_ic=False

# #4
# # UW proposed changed
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_method.txt --annotated_testset_attribute=UK_auto_test_method --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_method.txt --annotated_testset_attribute=UK_auto_test_method --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_method.txt --annotated_testset_attribute=UK_auto_test_method --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/UK_auto_test_method.txt --annotated_testset_attribute=UK_auto_test_method --replace_annotated_testset_into_original=False --is_ic=False

# #5
# # UW filterd origin
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/UW_auto_test_filtered.txt --annotated_testset_attribute=UW_auto_test_filtered --replace_annotated_testset_into_original=True --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/UW_auto_test_filtered.txt --annotated_testset_attribute=UW_auto_test_filtered --replace_annotated_testset_into_original=True --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/UW_auto_test_filtered.txt --annotated_testset_attribute=UW_auto_test_filtered --replace_annotated_testset_into_original=True --is_ic=False
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/UW_auto_test_filtered.txt --annotated_testset_attribute=UW_auto_test_filtered --replace_annotated_testset_into_original=True --is_ic=False

# #6
# # UW filterd changed
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/UW_auto_test_filtered.txt --annotated_testset_attribute=UW_auto_test_filtered --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/UW_auto_test_filtered.txt --annotated_testset_attribute=UW_auto_test_filtered --replace_annotated_testset_into_original=False --is_ic=False
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/UW_auto_test_filtered.txt --annotated_testset_attribute=UW_auto_test_filtered --replace_annotated_testset_into_original=False --is_ic=False
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./0506_final_annotation/UW_auto_test_filtered.txt --annotated_testset_attribute=UW_auto_test_filtered --replace_annotated_testset_into_original=False --is_ic=False








# #5
# # IC baseline origin
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=True --is_ic=True
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=True --is_ic=True
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=True --is_ic=True
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=True --is_ic=True

# #6
# # IC baseline changed
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=False --is_ic=True
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=False --is_ic=True
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=False --is_ic=True
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.22_3.txt --annotated_testset_attribute=IC_auto_test_0.22_3 --replace_annotated_testset_into_original=False --is_ic=True

# #7
# # IC proposed origin
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=True --is_ic=True
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=True --is_ic=True
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=True --is_ic=True
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=True --is_ic=True

# #8
# # IC proposed changed
# python eval_selection_model.py --model=select --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=False --is_ic=True
# python eval_selection_model.py --model=mcdrop --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=False --is_ic=True
# python eval_selection_model.py --model=ensemble --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=False --is_ic=True
# #python eval_selection_model.py --model=nopt --model_path=./logs/select_batch12_candi10_seed{}_randinit/model --setname=test --use_annotated_testset=True --annotated_testset=./final_annotation/IC_auto_test_0.7_3.txt --annotated_testset_attribute=IC_auto_test_0.7_3 --replace_annotated_testset_into_original=False --is_ic=True
