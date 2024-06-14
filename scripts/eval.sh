





directory="/home/myDiffuSeq/generation_outputs/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_maxlen98_lambda0_triplet20240612-21:26:42"
echo "$directory"

for subdir in $directory/*; do
    python eval_seq2seq.py \
    --folder "$subdir" \
    # --select_step "20" \
    # --mbr
    echo -e "$subdir\n"
done
