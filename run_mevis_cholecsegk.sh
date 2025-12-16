#!/bin/bash
#SBATCH --partition=pri2018gpu         # Specify partition
#SBATCH --account=qoscammagpu          # Specify account
#SBATCH --time=100:00:00                # Set maximum runtime
#SBATCH --nodes=1                      # Use 1 node
#SBATCH --cpus-per-task=8              # Use 8 CPU cores per task
#SBATCH --exclude=hpc-n[600-799]       # Exclude specified nodes
#SBATCH --gres=gpu:4                   # Request 4 GPUs
#SBATCH --output=mevis_cholec8k_val.out   # Standard output file
#SBATCH --error=mevis_cholec8k_val.err    # Error output file


conda activate mevis_env3
echo "python3"


# # Compile CUDA code (if needed)
# cd .../MeViS/mask2former/modeling/pixel_decoder/ops
# if [ ! -f "ms_deform_attn.so" ]; then  # If not yet compiled
#     echo "Compiling CUDA extensions..."
#     sh make.sh
# fi

cd .../MeViS

# python train_net_SurgRef.py \
#     --config-file configs/SurgRef_SWIN_bs8.yaml \
#     --num-gpus 4 --dist-url auto \
#     MODEL.WEIGHTS .../MeViS/model_final_86143f.pkl \
#     OUTPUT_DIR outputs/train_output_cholecseg8k_divide


python train_net_SurgRef.py \
    --config-file configs/SurgRef_SWIN_bs8.yaml \
    --num-gpus 4 --dist-url auto --eval-only \
    MODEL.WEIGHTS .../MeViS/outputs/train_output_cholecseg8k_divide/model_final.pth \
    OUTPUT_DIR outputs/valu_output_cholecseg8k_divide

python tools/eval_mevis.py