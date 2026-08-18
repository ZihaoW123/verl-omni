#!/usr/bin/env bash
# Collect A3 diffusion-agent host-memory and Ray object-store diagnostics.
#
# Usage:
#   ./collect_a3_host_memory.sh [output.log]
#
# Continuous sampling example (10 samples, one minute apart):
#   SAMPLES=10 INTERVAL_SECONDS=60 ./collect_a3_host_memory.sh a3_hostmem.log

set -uo pipefail

OUTPUT_LOG=${1:-a3_hostmem_$(date +%Y%m%d_%H%M%S).log}
SAMPLES=${SAMPLES:-1}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-60}
TOP_MAPPINGS=${TOP_MAPPINGS:-20}

mkdir -p "$(dirname "$OUTPUT_LOG")"
touch "$OUTPUT_LOG" || {
    echo "Cannot write output log: $OUTPUT_LOG" >&2
    exit 1
}
exec > >(tee -a "$OUTPUT_LOG") 2>&1

section() {
    echo
    echo "===== $* ====="
}

run_with_timeout() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 60s "$@"
    else
        "$@"
    fi
}

collect_ray_state() {
    section "Ray status"
    if ! command -v ray >/dev/null 2>&1; then
        echo "ray CLI is not available"
        return
    fi

    run_with_timeout ray status || echo "ray status failed or timed out"

    section "Ray object-store summary"
    run_with_timeout ray memory --stats-only || echo "ray memory --stats-only failed or timed out"

    section "Ray largest object references"
    if command -v timeout >/dev/null 2>&1; then
        timeout 60s ray memory --group-by=STACK_TRACE --sort-by=OBJECT_SIZE 2>&1 |
            head -200 || true
    else
        ray memory --group-by=STACK_TRACE --sort-by=OBJECT_SIZE 2>&1 |
            head -200 || true
    fi
}

collect_worker() {
    local pid=$1
    local proc_dir=/proc/$pid

    section "DiffusionAgentLoopWorker pid=$pid"
    if [[ ! -r "$proc_dir/status" ]]; then
        echo "pid=$pid exited before it could be sampled"
        return
    fi

    echo "command: $(tr '\0' ' ' <"$proc_dir/cmdline" 2>/dev/null || true)"

    section "pid=$pid status"
    grep -E '^(Name|State|Threads|VmPeak|VmSize|VmRSS|VmData|VmLck|RssAnon|RssFile|RssShmem):' \
        "$proc_dir/status" || true

    section "pid=$pid smaps_rollup"
    if [[ -r "$proc_dir/smaps_rollup" ]]; then
        grep -E \
            '^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Anonymous|LazyFree|AnonHugePages|ShmemPmdMapped|FilePmdMapped|Locked):' \
            "$proc_dir/smaps_rollup" || true
    else
        echo "smaps_rollup is unavailable (process exited or permission denied)"
    fi

    section "pid=$pid top mappings (RSS_GB PSS_GB SHARED_DIRTY_GB PATH)"
    if [[ ! -r "$proc_dir/smaps" ]]; then
        echo "smaps is unavailable (process exited or permission denied)"
        return
    fi

    awk '
        /^[0-9a-f]+-[0-9a-f]+ / {
            path = (NF >= 6 ? $6 : "[anon]")
        }
        /^Rss:/ { rss[path] += $2 }
        /^Pss:/ { pss[path] += $2 }
        /^Shared_Dirty:/ { dirty[path] += $2 }
        END {
            for (path in rss) {
                printf "%.3f %.3f %.3f %s\n",
                    rss[path] / 1024 / 1024,
                    pss[path] / 1024 / 1024,
                    dirty[path] / 1024 / 1024,
                    path
            }
        }
    ' "$proc_dir/smaps" 2>/dev/null |
        sort -k1,1nr |
        head -n "$TOP_MAPPINGS" || true
}

collect_sample() {
    local sample=$1

    section "sample=$sample/$SAMPLES time=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "hostname: $(hostname)"
    echo "kernel: $(uname -a)"

    section "System memory"
    free -h || true
    grep -E \
        '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapCached|Active|Inactive|AnonPages|Mapped|Shmem|KReclaimable|Slab|SReclaimable|SUnreclaim|PageTables|CommitLimit|Committed_AS):' \
        /proc/meminfo || true

    section "Shared-memory filesystems"
    df -h /dev/shm /tmp 2>&1 || true

    section "Top Ray/Python processes"
    ps -eo user,pid,ppid,%mem,rss,vsz,cmd --sort=-rss |
        awk 'NR == 1 || /ray|python/ {print; count++} count >= 30 {exit}' || true

    mapfile -t worker_pids < <(
        ps -eo pid=,rss=,args= --sort=-rss |
            awk '/ray::DiffusionAgentLoopWorker/ {print $1}'
    )

    if ((${#worker_pids[@]} == 0)); then
        section "DiffusionAgentLoopWorker"
        echo "No running DiffusionAgentLoopWorker was found"
    else
        echo "Detected worker PIDs: ${worker_pids[*]}"
        for pid in "${worker_pids[@]}"; do
            collect_worker "$pid"
        done
    fi

    collect_ray_state
}

if ! [[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAMPLES must be a positive integer, got: $SAMPLES" >&2
    exit 2
fi
if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "INTERVAL_SECONDS must be a non-negative integer, got: $INTERVAL_SECONDS" >&2
    exit 2
fi

section "A3 host-memory collection started"
echo "output_log: $OUTPUT_LOG"
echo "samples: $SAMPLES"
echo "interval_seconds: $INTERVAL_SECONDS"

for ((sample = 1; sample <= SAMPLES; sample++)); do
    collect_sample "$sample"
    if ((sample < SAMPLES)); then
        sleep "$INTERVAL_SECONDS"
    fi
done

section "A3 host-memory collection finished"
echo "output_log: $OUTPUT_LOG"
