#!/bin/bash
#SBATCH --job-name=agnews        
#SBATCH --output=run_agnewsEval.out       
#SBATCH --time=500:00:00                   
#SBATCH --ntasks=1                          
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70                  
#SBATCH --mem=200G                          
#SBATCH --partition=gpujl                   
#SBATCH --gres=gpu:4                       
                                         




eval "$(conda shell.bash hook)"


conda activate GuardSpace




python build_subspace.py \
    --model_id "./Llama-2-7B-Chat-fp16" \
    --cov_aware \
    --r 128 \
    --use_cache \
    --calib_dataset AdvBench \
    --calib_loader_size 520 \
    --null_dataset real-toxicity-prompts \
    --null_loader_size 520 \
    --mode build_adapters \
    --save_model \
    --save_path "./Decomposed_Model_llama";




PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TOKENIZERS_PARALLELISM=false \
python -u Train.py \
  --model_name_or_path "./Decomposed_Model_llama" \
  --output_dir "./finetune_llama" \
  --dataset agnews_dataset \
  --mode 1k_p_0.1 \
  --corda_mode True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 128 \
  --num_train_epochs 7 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 1 \
  --bf16 True \
  --tf32 False \
  --report_to none \
  --seed 42 \
  --data_seed 42 \
  --dataloader_num_workers 0;




conda deactivate
