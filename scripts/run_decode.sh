CUDA_VISIBLE_DEVICES=0 python -u run_decode.py \
--model_dir /home/myDiffuSeq/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_maxlen98_lambda020240528-21:10:22 \
--seed 123 \
--bsz 400 \
--sel_ckpt 4 \
--split test