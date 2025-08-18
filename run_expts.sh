#! /bin/bash

set -e

outdir="outputs"
mkdir -p ${outdir}

# Auto-pick visible GPUs by free memory (exclude near-full GPUs)
# Usage: select_visible_gpus MIN_FREE_MB MAX_GPUS
select_visible_gpus() {
        local min_free_mb=${1:-8192}   # default: require at least 8 GiB free
        local max_gpus=${2:-8}
        # List GPUs with their free memory, sort by free desc, take top N, join by comma
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
            | awk -v min=${min_free_mb} 'NF>=2 { if ($2 >= min) print $1":"$2 }' \
            | sort -t: -k2,2nr \
            | cut -d: -f1 \
            | head -n ${max_gpus} \
            | paste -sd, -
}

        # Estimate free CPU cores from 1-minute load average (rough heuristic)
        # Returns at least 1. Override with FREE_CORES env to bypass.
        estimate_free_cores() {
            if [[ -n "${FREE_CORES}" ]]; then
                echo "${FREE_CORES}"
                return 0
            fi
            local total
            total=$(nproc --all 2>/dev/null || nproc)
            local load1
            load1=$(uptime | awk -F'load average: ' '{print $2}' | cut -d, -f1)
            if [[ -z "${load1}" ]]; then
                # Fallback parsing if locale/format differs
                load1=$(uptime | awk '{print $(NF-2)}' | tr -d ',')
            fi
            # ceil(load1)
            local ceil_load
            ceil_load=$(awk -v l="${load1}" 'BEGIN{printf("%d", (l==int(l)?l:int(l)+1))}')
            local free=$(( total - ceil_load ))
            if (( free < 1 )); then free=1; fi
            echo "${free}"
        }

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
        for n in "1" "2" "4" "8"; do
        # for n in "1"; do
            # for mode in "baseline" "scratchpad" "cot" "neurosymbolic"; do
            for mode in "neurosymbolic"; do
                task="${base}-${mode}-${n}shot"
                run_id="${model#*/}_${task}"
                if [[ ${model} == "mistralai/Mistral-7B-v0.1" ]]; then
                    # Single-process model-parallel: run python directly so HF device_map shards across GPUs
                    # echo "Running inside the model-parallel environment..."
                    # Dynamically exclude near-full GPUs; also cap GPUs by available CPU cores
                    FREE_CORES_EST=$(estimate_free_cores)
                    CORES_PER_GPU=${CORES_PER_GPU:-3}
                    CPU_MAX_GPUS=$(( FREE_CORES_EST / CORES_PER_GPU ))
                    if (( CPU_MAX_GPUS < 1 )); then CPU_MAX_GPUS=1; fi
                    GPU_CAP=${MAX_GPUS:-8}
                    EFFECTIVE_MAX_GPUS=$(( CPU_MAX_GPUS < GPU_CAP ? CPU_MAX_GPUS : GPU_CAP ))
                    VISIBLE_DEVICES=$(select_visible_gpus "${MIN_FREE_MB:-8192}" "${EFFECTIVE_MAX_GPUS}")
                    echo "[run_expts] free_cores=${FREE_CORES_EST} cores_per_gpu=${CORES_PER_GPU} -> max_gpus=${EFFECTIVE_MAX_GPUS}; visible=${VISIBLE_DEVICES}" >&2
                    # Fallback to all GPUs if the filter returned none (e.g., low free mem across the board)
                    if [[ -z "${VISIBLE_DEVICES}" ]]; then
                        VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
                    fi
                    job="cd $(pwd); source activate linc; "
                    job+="CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} OMP_NUM_THREADS=${CORES_PER_GPU} MKL_NUM_THREADS=${CORES_PER_GPU} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
                    job+="python runner.py"
                    # prefer bf16 for Mistral when sharded
                    # using device_map=auto works on this host; keep that as default
                    job+=" --model ${model} --precision ${precision} --model-parallel --device_map balanced_low_0"
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
                    git pull
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
