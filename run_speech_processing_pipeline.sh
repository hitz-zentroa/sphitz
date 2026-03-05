#!/bin/bash

start_time=$(date +%s)

python speech_processing_pipeline.py \
    --run_mode="all" \
    --audio_file="./audios/common_voice_eu_18251595.wav" \
    --input_text="" \
    --out_path="./results" \
    --seg_model="./models/diar/seg_CONF.ckpt" \
    --seg_config_yml="./models/diar/seg_config.yaml" \
    --seg_option="diar" \
    --stt_model="./models/asr/stt_eu_conformer_transducer_large_v2.nemo" \
    --cp_model="./models/marianmt/eu_norm-eu" \
    --device="cuda"

echo

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

days=$((elapsed_time/86400))
hours=$(( (elapsed_time%86400)/3600 ))
minutes=$(( (elapsed_time%3600)/60 ))
seconds=$(( elapsed_time%60 ))

echo "Elapsed time: ${elapsed_time}s | ${days} days ${hours} hr ${minutes} min ${seconds} sec"
echo
