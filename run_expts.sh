#! /bin/bash

set -e

outdir="outputs"
mkdir -p ${outdir}

max_length=8192 # max model context including prompt
if [[ ! -z "${DEBUG}" ]]; then
    listen="--listen 0.0.0.0:5679 --wait-for-client"
else
    listen=""
fi

for model in "mistralai/Mistral-7B-v0.1"; do
# for model in "Qwen/Qwen3-32B"; do
# for model in "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"; do
    # for base in "folio" "proofwriter"; do
    for base in "folio"; do
        if [[ ${model} == "mistralai/Mistral-7B-v0.1" ]]; then
            batch_size=1
            precision=="fp32"
        elif [[ ${model} == "Qwen/Qwen3-32B" ]]; then
            batch_size=1
            precision="int8"
        elif [[ ${model} == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" ]]; then
            batch_size=1
            precision="int8"
        fi
        # for n in "1" "2" "4" "8"; do
        for n in "1"; do
            # for mode in "baseline" "scratchpad" "cot" "neurosymbolic" "neurosymbolicenhanced" "neurosymbolic-depthproof"; do
            for mode in "neurosymbolic"; do
                task="${base}-${mode}-${n}shot"
                run_id="${model#*/}_${task}"
                job="cd $(pwd); source activate linc; "
                job+="accelerate launch ${listen} runner.py"
                job+=" --model ${model} --precision ${precision}"
                # job+=" --use_auth_token"
                job+=" --use_auth_token --limit 1"
                job+=" --tasks ${task} --n_samples 1 --batch_size ${batch_size}"
                job+=" --max_length_generation ${max_length} --temperature 0.8"
                job+=" --allow_code_execution --trust_remote_code --output_dir ${outdir}"
                job+=" --save_generations_raw --save_generations_raw_path ${run_id}_generations_raw.json"
                job+=" --save_generations_prc --save_generations_prc_path ${run_id}_generations_prc.json"
                job+=" --save_references --save_references_path ${run_id}_references.json"
                job+=" --save_results --save_results_path ${run_id}_results.json"
                job+=" |& tee ${outdir}/${run_id}.log; exit"
                export JOB="${job}"; bash SUBMIT.sh
                echo "Submitted ${run_id}"
            done
        done
    done
done
# touch ${outdir}/run.done