CUDA_VISIBLE_DEVICES="0,1" python -u run_decode_solver.py \
--model_dir ./diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_maxlen98_lambda0_triplet_KL_B25620240817-15:54:15 \
--seed 110 \
--bsz 2500 \
--step 10 \
--split test \
--sel_ckpt all \
--note test
## You can use steps = 10, 12, 15, 20, 25, 50, 100.
## Empirically, we find that steps in [10, 20] can generate quite good samples.
## And steps = 20 can almost converge.