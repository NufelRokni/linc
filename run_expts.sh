#! /bin/bash

set -e

outdir="outputs"
mkdir -p ${outdir}

# for model in "bigcode/starcoderplus"; do
for model in "mistralai/Mistral-7B-v0.1"; do
    # for base in "folio" "proofwriter"; do
    for base in "folio"; do
        if [[ ${model} == "bigcode/starcoderplus" ]]; then
            batch_size=1
            max_length=8192 # max model context including prompt
            precision="bf16"
        elif [[ ${model} == "mistralai/Mistral-7B-v0.1" ]]; then
            batch_size=1
            max_length=8192 # max model context including prompt
            precision="fp32"
        else
            echo "Unknown model ${model}"
            exit 1
        fi
        # for n in "1" "2" "4" "8"; do
        for n in "1"; do
            if [[ ${n} != "8" ]] && [[ ${base} != "folio" || ${model} == "gpt-4-0613" ]]; then
                continue
            fi
            # for mode in "baseline" "scratchpad" "cot" "neurosymbolic"; do
            for mode in "baseline"; do
                task="${base}-${mode}-${n}shot"
                run_id="${model#*/}_${task}"
                job="cd $(pwd); source activate linc; accelerate launch runner.py"
                job+=" --model ${model} --precision ${precision}"
                job+=" --use_auth_token --limit 5"
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
touch ${outdir}/run.done
