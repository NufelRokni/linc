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
    # for base in "folio" "proofwriter"; do
    for base in "folio"; do
        if [[ ${model} == "mistralai/Mistral-7B-v0.1" ]]; then
            batch_size=1
            precision="fp32"
        fi
        # for n in "1" "2" "4" "8"; do
        for n in "1"; do
            # for mode in "baseline" "scratchpad" "cot" "neurosymbolic"; do
                for mode in "baseline" "scratchpad" "cot" "neurosymbolic"; do
                task="${base}-${mode}-${n}shot"
                run_id="${model#*/}_${task}"
                if [[ ${model} == "mistralai/Mistral-7B-v0.1" ]]; then
                    # Single-process model-parallel: run python directly so HF device_map shards across GPUs
                    # echo "Running inside the model-parallel environment..."
                    job="cd $(pwd); source activate linc; "
                    job+="CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
                    job+="python runner.py"
                    # prefer bf16 for Mistral when sharded
                    # using device_map=auto works on this host; keep that as default
                    job+=" --model ${model} --precision ${precision} --model-parallel --device_map auto"
                else
                    # default: use accelerate launch (data-parallel)
                    # echo "Running inside the accelerate parallel environment..."
                    job="cd $(pwd); source activate linc; unset CUDA_VISIBLE_DEVICES;"
                    job+="accelerate launch ${listen} runner.py"
                    job+=" --model ${model} --precision ${precision}"
                fi
                # job+=" --use_auth_token --limit 10"
                job+=" --use_auth_token"
                job+=" --tasks ${task} --n_samples 10 --batch_size ${batch_size}"
                job+=" --max_length_generation ${max_length} --temperature 0.8"
                job+=" --allow_code_execution --trust_remote_code --output_dir ${outdir}"
                job+=" --save_generations_raw --save_generations_raw_path ${run_id}_generations_raw.json"
                job+=" --save_generations_prc --save_generations_prc_path ${run_id}_generations_prc.json"
                job+=" --save_references --save_references_path ${run_id}_references.json"
                job+=" --save_results --save_results_path ${run_id}_results.json"
                job+=" |& tee ${outdir}/${run_id}.log; exit"
                export JOB="${job}"; bash SUBMIT.sh
                echo "Submitted ${run_id}"

                # After each run, add, commit (only if there are staged changes), and push.
                # Use the repo root so this works even if the script is invoked from a subdir.
                repo_root=$(git rev-parse --show-toplevel 2>/dev/null || echo "$(pwd)")
                (
                    cd "${repo_root}"
                    git add .
                    # If there are staged changes, commit and push. Wrapped in conditional to
                    # avoid failing the whole experiment run when there is nothing to commit
                    # or when push fails.
                    if ! git diff --cached --quiet; then
                        commit_msg="auto: ${run_id} $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
                        git commit -m "${commit_msg}"
                        if git push; then
                            echo "Pushed commit for ${run_id}"
                        else
                            echo "git push failed for ${run_id} -- continuing"
                        fi
                    else
                        echo "No changes to commit for ${run_id}"
                    fi
                )
            done
        done
    done
done
# touch ${outdir}/run.done
