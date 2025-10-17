#!/bin/bash
# TO RUN: bash scripts/plaid_dpo.sh | tee logs/plaid_dpo.log
export CUDA_VISIBLE_DEVICES=0

set -e  # Exit on error
set -x  # Print commands

# iteration 1 (temp 0.7)
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_temp_0.7_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_sg --output_path DPO/qwen_7b_wyckoff_sg_combined_t2 --raw --wyckoff --ratio 3 --mode basic --mode_2 sg_novel
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --dataset_name DPO/qwen_7b_wyckoff_sg_combined_t2/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7.csv --batch_size 1024 --temperature 0.7 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 0.7 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7 --num_jobs 16
done
# novelty evaluation for iteration 1
python evals/novelty.py evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7_uncon_eqv2_ehull_results.csv test.json qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it1_temp0.7 --sun_out csv_results/sun_eqv2.csv
# run_novelty aggregation for iteration 1
bash run_novelty.sh qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it1_temp0.7 0.1 eqv2

# iteration 2 (temp 0.7)
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7_uncon_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7 --output_path DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it2 --raw --wyckoff --mode basic --mode_2 sg_novel
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --dataset_name DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it2/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it2 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it2 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7.csv --batch_size 1024 --temperature 0.7 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it2 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 0.7 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7 --num_jobs 16
done
# novelty evaluation for iteration 2
python evals/novelty.py evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7_uncon_eqv2_ehull_results.csv test.json qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it2_temp0.7 --sun_out csv_results/sun_eqv2.csv
# run_novelty aggregation for iteration 2
bash run_novelty.sh qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it2_temp0.7 0.1 eqv2

# iteration 3 (temp 0.9)
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7_uncon_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7 --output_path DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it3 --raw --wyckoff --mode tieredNovel --mode_2 sg_novel
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it2 --dataset_name DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it3/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it3 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it3 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.9.csv --batch_size 1024 --temperature 0.9 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it3 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.9_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 0.9 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.9_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.9/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.9 --num_jobs 16
done
# novelty evaluation for iteration 3
python evals/novelty.py evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.9_uncon_eqv2_ehull_results.csv test.json qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it3_temp0.9 --sun_out csv_results/sun_eqv2.csv
# run_novelty aggregation for iteration 3
bash run_novelty.sh qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it3_temp0.9 0.1 eqv2

# iteration 4 (temp 0.9)
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.9_uncon_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.9 --output_path DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it4 --raw --wyckoff --mode tieredNovel --mode_2 sg_novel
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it3 --dataset_name DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it4/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it4 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it4 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it4_temp_0.9.csv --batch_size 1024 --temperature 0.9 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it4 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it4_temp_0.9_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 0.9 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it4_temp_0.9_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it4_temp_0.9/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_it4_temp_0.9 --num_jobs 16 
done
# novelty evaluation for iteration 4
python evals/novelty.py evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it4_temp_0.9_uncon_eqv2_ehull_results.csv test.json qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it4_temp0.9 --sun_out csv_results/sun_eqv2.csv
# run_novelty aggregation for iteration 4
bash run_novelty.sh qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it4_temp0.9 0.1 eqv2

# iteration 5 (temp 1.1)
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it4_temp_0.9_uncon_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it4_temp_0.9 --output_path DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it5 --raw --wyckoff --mode tieredNovel --mode_2 sg_novel
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it4 --dataset_name DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it5/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it5 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it5 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it5_temp_1.1.csv --batch_size 1024 --temperature 1.1 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it5 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it5_temp_1.1_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 1.1 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it5_temp_1.1_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it5_temp_1.1/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_it5_temp_1.1 --num_jobs 16
done
# novelty evaluation for iteration 5
python evals/novelty.py evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it5_temp_1.1_uncon_eqv2_ehull_results.csv test.json qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it5_temp1.1 --sun_out csv_results/sun_eqv2.csv
# run_novelty aggregation for iteration 5
bash run_novelty.sh qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it5_temp1.1 0.1 eqv2

# iteration 6 (temp 1.1)
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it5_temp_1.1_uncon_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it5_temp_1.1 --output_path DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it6 --raw --wyckoff --mode tieredNovel --mode_2 sg_novel
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it5 --dataset_name DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it6/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it6 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it6 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it6_temp_1.1.csv --batch_size 1024 --temperature 1.1 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it6 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it6_temp_1.1_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 1.1 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it6_temp_1.1_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it6_temp_1.1/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_it6_temp_1.1 --num_jobs 16
done
# novelty evaluation for iteration 6
python evals/novelty.py evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it6_temp_1.1_uncon_eqv2_ehull_results.csv test.json qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it6_temp1.1 --sun_out csv_results/sun_eqv2.csv
# run_novelty aggregation for iteration 6
bash run_novelty.sh qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it6_temp1.1 0.1 eqv2

# iteration 7 (temp 1.3)
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it6_temp_1.1_uncon_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it6_temp_1.1 --output_path DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it7 --raw --wyckoff --mode tieredNovel --mode_2 sg_novel
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it6 --dataset_name DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it7/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it7 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it7 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it7_temp_1.3.csv --batch_size 1024 --temperature 1.3 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it7 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it7_temp_1.3_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 1.3 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it7_temp_1.3_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it7_temp_1.3/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_it7_temp_1.3 --num_jobs 16
done
# novelty evaluation for iteration 7
python evals/novelty.py evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it7_temp_1.3_uncon_eqv2_ehull_results.csv test.json qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it7_temp1.3 --sun_out csv_results/sun_eqv2.csv
# run_novelty aggregation for iteration 7
bash run_novelty.sh qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it7_temp1.3 0.1 eqv2
