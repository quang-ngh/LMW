CUDA_VISIBLE_DEVICES=0 python -m src.inference \
  --model_weights_path checkpoints_pt/solaris.pt \
  --clip_checkpoint_path checkpoints_pt/clip.pt \
  --vae_checkpoint_path checkpoints_pt/vae.pt \
  --eval_save_dir eval_out_v2 \
  --eval_data_dir datasets/eval \
  --test_dataset_name bothLookAwayEval/test \
  --dataset_name both_look_away     \
  --num_frames 257 \
  --eval_num_samples 1 \
  # --offload
