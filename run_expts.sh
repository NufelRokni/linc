#! /usr/bin/env bash
set -xeuo pipefail

outdir="outputs"
mkdir -p ${outdir}

for model in "bigcode/starcoderplus"; do
  for base in "proofwriter"; do
    batch_size=5
    max_length=8192
#    for n in "1" "2" "4" "8"; do
    for n in "8"; do
#      if [[ ${n} != "8" && ${base} != "folio" ]]; then
#        continue
#      fi
#      for mode in "baseline" "scratchpad" "cot" "neurosymbolic"; do
      for mode in "baseline"; do
        task="${base}-${mode}-${n}shot"
        run_id="${model#*/}_${task}"
        job="cd $(pwd); source activate linc; accelerate launch runner.py"
        job+=" --model ${model} --precision bf16"
        job+=" --use_auth_token"
        job+=" --tasks ${task} --n_samples 10 --batch_size ${batch_size}"
	job+=" --limit 10"
        job+=" --max_length_generation ${max_length} --temperature 0.8"
        job+=" --allow_code_execution --trust_remote_code --output_dir ${outdir}"
        job+=" --save_generations_raw --save_generations_raw_path ${run_id}_generations_raw.json"
        job+=" --save_generations_prc --save_generations_prc_path ${run_id}_generations_prc.json"
        job+=" --save_references --save_references_path ${run_id}_references.json"
        job+=" --save_results --save_results_path ${run_id}_results.json"
        job+=" 2>&1 | tee ${outdir}/${run_id}.log; exit"
        export JOB=${job}; bash SUBMIT.sh
        echo "Submitted ${run_id}"
      done
    done
  done
done

touch ${outdir}/run.done
