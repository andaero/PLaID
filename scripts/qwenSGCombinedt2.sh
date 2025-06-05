#!/bin/bash
# TO RUN: bash slurm/bash/qwenSGCombinedt2.sh | tee logs/qwenSGCombinedt2.log
export CUDA_VISIBLE_DEVICES=0

set -e  # Exit on error
set -x  # Print commands

# iteration 1
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_temp_0.7_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_sg --output_path DPO/qwen_7b_wyckoff_sg_combined_t2 --raw --wyckoff --ratio 3 --mode basic --mode_2 sg
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --dataset_name DPO/qwen_7b_wyckoff_sg_combined_t2/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7.csv --batch_size 1024 --temperature 0.7 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 0.7 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7 --num_jobs 16
done

# iteration 2
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7_uncon_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_temp_0.7 --output_path DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it2 --raw --wyckoff --mode basic --mode_2 sg
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2 --dataset_name DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it2/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it2 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it2 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7.csv --batch_size 1024 --temperature 0.7 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it2 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 0.7 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7_uncon.csv --num_jobs 15
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7 --num_jobs 16
done


# iteration 3
python DPO_preprocess.py --input_path evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7_uncon_eqv2_ehull_results.csv --input_path_2 evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it2_temp_0.7 --output_path DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it3 --raw --wyckoff --mode tiered2 --mode_2 sg
python DPO_train.py --model_name_or_path exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it2 --dataset_name DPO/qwen_7b_wyckoff_dpo_sg_combined_t2_it3/ --num_train_epochs 1 --logging_steps 10 --output_dir exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it3 --batch-size 8 --type 7b --report_to wandb --eval_strategy steps --eval_steps 500 --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it3 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.7.csv --batch_size 1024 --temperature 0.7 --conditions spacegroup_number --conditions_file cond_gen/input_csv/sg_in_short.csv --wyckoff --qwen --max_dir --dpo --online
python llama_sample.py --model_name 7b --model_path=exp/qwen-7b-dpo-wyckoff-sg-combined-t2-it3 --out_path=new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.7_uncon.csv --num_samples 10000 --batch_size 1024 --temperature 0.7 --wyckoff --max_dir --dpo --qwen --use_fa2 --online
for file in new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.7/*; do
    python evals/e_above_hull.py --relaxer eqv2_batched --filename $file --out_folder qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.7 --num_jobs 20
done
python evals/e_above_hull.py --relaxer eqv2_batched --filename new_llm_samples/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.7_uncon.csv --num_jobs 15
