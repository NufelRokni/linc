#! /bin/bash

outdir="outputs"
mkdir -p ${outdir}

# Auto-pick visible GPUs by free memory (exclude near-full GPUs)
# Usage: select_visible_gpus MIN_FREE_MB MAX_GPUS
select_visible_gpus() {
        # Args: MIN_FREE_MB [MAX_UTIL] [MAX_GPUS]
        local min_free_mb=${1:-4096}   # default: require at least 4 GiB free
        local max_util=${2:-80}        # default: avoid GPUs with >80% utilization
        local max_gpus=${3:-8}
        # Query index, free memory, and utilization; filter by thresholds; sort by mem desc then util asc
        nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
            | awk -v min=${min_free_mb} -v maxu=${max_util} 'BEGIN{FS=", *"} { if ($2+0 >= min && $3+0 <= maxu) print $1":"$2":"$3 }' \
            | sort -t: -k2,2nr -k3,3n \
            | cut -d: -f1 \
            | head -n ${max_gpus} \
            | paste -sd, -
}

# Estimate free CPU cores from 1-minute load average (rough heuristic)
# Returns at least 1. Override with FREE_CORES env to bypass.
estimate_free_cores() {
    # If explicitly set via env var, don't calculate
    if [[ -n "${FREE_CORES}" ]]; then
        echo "${FREE_CORES}"
        return 0
    fi
    
    # Get total CPU count
    local total
    total=$(nproc --all 2>/dev/null || nproc)
    
    # Get 1-minute load average with fallback for different locale formats
    local load1
    load1=$(uptime | awk -F'load average: ' '{print $2}' | cut -d, -f1)
    if [[ -z "${load1}" ]]; then
        # Fallback parsing if locale/format differs
        load1=$(uptime | awk '{print $(NF-2)}' | tr -d ',')
    fi
    
    # Ceil the load average to get cores in use
    local ceil_load
    ceil_load=$(awk -v l="${load1}" 'BEGIN{printf("%d", (l==int(l)?l:int(l)+1))}')
    
    # Calculate free cores (minimum 1)
    local free=$(( total - ceil_load ))
    if (( free < 1 )); then free=1; fi
    
    echo "${free}"
}

# After each run, add, commit (only if there are staged changes), and push.
# Accepts one argument: run_id (used in commit message)
git_autocommit() {
    local run_id="$1"
    # Use the repo root so this works even if the script is invoked from a subdir.
    local repo_root
    repo_root=$(git rev-parse --show-toplevel 2>/dev/null || echo "$(pwd)")
    (
        cd "${repo_root}"
        git pull
        git add .
        # If there are staged changes, commit and push. Wrapped in conditional to
        # avoid failing the whole experiment run when there is nothing to commit
        # or when push fails.
        if ! git diff --cached --quiet; then
            local commit_msg
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
}

# Model configuration defaults
# Fixed max_length for generation (no adaptive override)
max_length=8192  # max model context including prompt

# Debug mode configuration
if [[ ! -z "${DEBUG}" ]]; then
    listen="--listen 0.0.0.0:5679 --wait-for-client"
else
    listen=""
fi

## (adapt_max_length removed; fixed max_length is used instead)

for model in "mistralai/Mistral-7B-v0.1"; do
    # for base in "folio" "proofwriter"; do
    for base in "folio"; do
        if [[ ${model} == "mistralai/Mistral-7B-v0.1" ]]; then
            # Always use batch_size=1 for model-parallel inference (safest default)
            batch_size=1
            # Set precision based on env var or default to fp32
            precision="${PRECISION:-fp32}"
            # Default device map - can be overridden with DEVICE_MAP_MODE env var
            device_map="${DEVICE_MAP_MODE:-balanced_low_0}"
        fi
        for n in "1" "2" "4" "8"; do
        # for n in "1"; do
            # for mode in "baseline" "scratchpad" "cot" "neurosymbolic"; do
            for mode in "baseline" "neurosymbolic"; do
                task="${base}-${mode}-${n}shot"
                run_id="${model#*/}_${task}"
                if [[ ${model} == "mistralai/Mistral-7B-v0.1" ]]; then
                    # Single-process model-parallel: run python directly so HF device_map shards across GPUs
                    # echo "Running inside the model-parallel environment..."
                    # Dynamically select GPUs that are not heavily utilized and have sufficient memory
                    FREE_CORES_EST=$(estimate_free_cores)
                    CORES_PER_GPU=${CORES_PER_GPU:-2}
                    CPU_MAX_GPUS=$(( FREE_CORES_EST / CORES_PER_GPU ))
                    if (( CPU_MAX_GPUS < 1 )); then CPU_MAX_GPUS=1; fi
                    GPU_CAP=${MAX_GPUS:-8}
                    EFFECTIVE_MAX_GPUS=$(( CPU_MAX_GPUS < GPU_CAP ? CPU_MAX_GPUS : GPU_CAP ))
                    
                    # Precision already set above (env override supported); no further override here
                    
                    # Select GPUs with >=8 GiB free and <=70% util (overridable via MIN_FREE_MB, MAX_UTIL)
                    VISIBLE_DEVICES=$(select_visible_gpus "${MIN_FREE_MB:-8192}" "${MAX_UTIL:-70}" "${EFFECTIVE_MAX_GPUS}")
                    echo "[run_expts] free_cores=${FREE_CORES_EST}, selecting ${EFFECTIVE_MAX_GPUS} GPUs with min_free=${MIN_FREE_MB:-8192}MB, max_util=${MAX_UTIL:-70}%" >&2
                    echo "[run_expts] precision=${precision}, visible_gpus=${VISIBLE_DEVICES}" >&2
                    
                    # Fallback: if no suitable GPUs found, try more permissive selection
                    if [[ -z "${VISIBLE_DEVICES}" ]]; then
                        echo "[run_expts] WARNING: No suitable GPUs found with current criteria, trying more permissive selection" >&2
                        VISIBLE_DEVICES=$(select_visible_gpus "1024" "95" "${EFFECTIVE_MAX_GPUS}")
                    fi
                    
                    # Final fallback: If still no suitable GPUs found, use all available
                    if [[ -z "${VISIBLE_DEVICES}" ]]; then
                        echo "[run_expts] WARNING: No suitable GPUs found, falling back to all GPUs (may cause OOM)" >&2
                        VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
                        # Keep precision as configured (default fp32)
                    fi
                    # Setup environment variables for better performance
                    job="cd $(pwd); source activate linc; "
                    job+="CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} "
                    job+="OMP_NUM_THREADS=${CORES_PER_GPU} "
                    job+="MKL_NUM_THREADS=${CORES_PER_GPU} "
                    job+="TOKENIZERS_PARALLELISM=false "
                    job+="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                    job+="TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
                    
                    # Launch the model with appropriate settings
                    job+="python runner.py"
                    job+=" --model ${model}" 
                    job+=" --precision ${precision}" 
                    job+=" --model-parallel" 
                    job+=" --device_map ${device_map}"
                else
                    # default: use accelerate launch (data-parallel)
                    # echo "Running inside the accelerate parallel environment..."
                    job="cd $(pwd); source activate linc; unset CUDA_VISIBLE_DEVICES;"
                    job+="accelerate launch ${listen} runner.py"
                    job+=" --model ${model} --precision ${precision}"
                fi
                # job+=" --use_auth_token --limit 10"
                job+=" --use_auth_token"
                # Always use fixed max_length; default is 8192 unless overridden via MAX_LENGTH env
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

                # After each run, auto-commit and push any changes
                git_autocommit "${run_id}"
            done
        done
    done
done
# touch ${outdir}/run.done
