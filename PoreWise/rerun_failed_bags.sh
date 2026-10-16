#!/usr/bin/env bash
# Retry incomplete directional runs from TIFF inputs and display live progress/ETA.
set -uo pipefail
shopt -s nullglob

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_dir="$repo_dir/porewise/data/tif"
results_dir="$repo_dir/porewise/results_directional"
logs_dir="$repo_dir/porewise/logs"

# These can be overridden, for example: STEPS=100000 ./rerun_failed_bags.sh
steps=${STEPS:-60000}
heartbeat=${HEARTBEAT:-1000}
pattern=${1:-B51_bag*.tif*}

mkdir -p "$results_dir" "$logs_dir"
files=("$data_dir"/$pattern)
retry_files=()
if (( ${#files[@]} == 0 )); then
    echo "No matching TIFF inputs found in $data_dir."
    exit 0
fi

is_complete() {
    local result_dir=$1
    [[ -s "$result_dir/fields.vti" && -s "$result_dir/permeability.json" ]] && python - "$result_dir/permeability.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    result = json.load(stream)
raise SystemExit(0 if all(f"k_{axis}_m2" in result for axis in "xyz") else 1)
PY
}

for file in "${files[@]}"; do
    name=$(basename "$file")
    name=${name%.*}
    if is_complete "$results_dir/$name"; then
        printf 'SKIP completed: %s\n' "$name"
    else
        retry_files+=("$file")
    fi
done

total=${#retry_files[@]}
if (( total == 0 )); then
    echo "Nothing to retry; all matching cases are already completed."
    exit 0
fi

printf 'Retrying %d case(s), maximum %d updates per direction.\n' "$total" "$steps"
printf 'Live batch ETA is estimated from completed case durations.\n'

batch_start=$SECONDS
completed=0
failed=0

for file in "${retry_files[@]}"; do
    name=$(basename "$file")
    name=${name%.*}
    output_dir="$results_dir/$name"
    log_file="$logs_dir/$name.log"
    case_start=$SECONDS

    printf '\n[%d/%d] Starting %s at %s\n' "$((completed+1))" "$total" "$name" "$(date '+%F %T %z')" | tee "$log_file"

    if (cd "$repo_dir" && python -m porewise "$file" \
        --backend cuda-sparse \
        --collision trt \
        --tau 0.6 \
        --solid-value 1 \
        --pore-value 0 \
        --dx 1.2e-4 \
        --steps "$steps" \
        --heartbeat "$heartbeat" \
        --verbose \
        --output "$output_dir" 2>&1 | tee -a "$log_file"); then
        status=COMPLETED
    else
        status=FAILED
        failed=$((failed+1))
    fi

    completed=$((completed+1))
    case_seconds=$((SECONDS-case_start))
    batch_seconds=$((SECONDS-batch_start))
    average_seconds=$((batch_seconds/completed))
    remaining_seconds=$((average_seconds*(total-completed)))
    printf '%s: %s; case time %02d:%02d:%02d; batch ETA %02d:%02d:%02d\n' \
        "$status" "$name" \
        "$((case_seconds/3600))" "$(((case_seconds%3600)/60))" "$((case_seconds%60))" \
        "$((remaining_seconds/3600))" "$(((remaining_seconds%3600)/60))" "$((remaining_seconds%60))" \
        | tee -a "$log_file"
done

printf '\nFinished: %d completed, %d failed.\n' "$((total-failed))" "$failed"
exit "$((failed > 0))"
