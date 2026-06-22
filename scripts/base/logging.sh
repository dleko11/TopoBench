#!/bin/bash

: "${LOGGING_ROOT_DIR:=./logs}"

run_and_log() {
    local cmd="$1"
    local log_group="$2"
    local run_name="$3"
    local root_dir="${4:-$LOGGING_ROOT_DIR}"

    local specific_log_dir="$root_dir/$log_group"
    mkdir -p "$specific_log_dir"

    local success_log="$specific_log_dir/SUCCESSFUL_RUNS.log"
    local failed_log="$specific_log_dir/FAILED_RUNS.log"

    local stdout_log="$specific_log_dir/${run_name}_stdout.log"
    local stderr_log="$specific_log_dir/${run_name}_stderr.log"

    local tmp_stdout="${stdout_log}.${BASHPID}.tmp"
    local tmp_stderr="${stderr_log}.${BASHPID}.tmp"

    local max_attempts="${MAX_ATTEMPTS:-2}"
    local retry_wait_seconds="${RETRY_WAIT_SECONDS:-15}"
    local exit_code=0

    echo "--- [START] Running: $run_name (PID: $BASHPID) ---"

    for attempt in $(seq 1 "$max_attempts"); do
        if [ "$attempt" -gt 1 ]; then
            echo "[RETRY] $run_name failed on attempt $((attempt - 1)). Waiting ${retry_wait_seconds}s before attempt $attempt..."
            sleep "$retry_wait_seconds"
        fi

        eval "$cmd" > "$tmp_stdout" 2> "$tmp_stderr"
        exit_code=$?

        if [ "$exit_code" -eq 0 ]; then
            break
        fi
    done

    if [ "$exit_code" -eq 0 ]; then
        echo "[SUCCESS] Finished: $run_name"
        (
            flock -x 200
            echo "$(date): [SUCCESS] ${run_name}" >> "$success_log"
        ) 200> "${specific_log_dir}/.success.lock"
        if [[ "${KEEP_SUCCESS_LOGS:-false}" == "true" ]]; then
            mv "$tmp_stdout" "$stdout_log"
            mv "$tmp_stderr" "$stderr_log"
        else
            rm -f "$tmp_stdout" "$tmp_stderr"
        fi
        return 0
    fi

    echo "[FAILURE] Finished: $run_name (failed after $max_attempts attempts; exit code: $exit_code)"
    mv "$tmp_stdout" "$stdout_log"
    mv "$tmp_stderr" "$stderr_log"

    (
        flock -x 200
        echo "=================================" >> "$failed_log"
        echo "FAILURE on $(date): [${run_name}]" >> "$failed_log"
        echo "Exit Code: $exit_code" >> "$failed_log"
        echo "Attempts: $max_attempts" >> "$failed_log"
        echo "Command: $cmd" >> "$failed_log"
        echo "See full logs: $stdout_log | $stderr_log" >> "$failed_log"
        echo "=================================" >> "$failed_log"
    ) 200> "${specific_log_dir}/.failed.lock"

    echo "----------------- ERROR OUTPUT ($run_name) -----------------"
    tail -n 15 "$stderr_log"
    echo "----------------------------------------------------------------"

    return 1
}
