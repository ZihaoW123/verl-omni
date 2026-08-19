#!/usr/bin/env bash
# Collect A3 diffusion-agent host-memory and Ray object-store diagnostics.
#
# Usage:
#   ./collect_a3_host_memory.sh [output.log]
#
# Continuous sampling example (10 samples, one minute apart):
#   SAMPLES=10 INTERVAL_SECONDS=60 ./collect_a3_host_memory.sh a3_hostmem.log
# Compact long-running sampling (system/cgroup/all-process PSS only):
#   COMPACT=1 SAMPLES=120 INTERVAL_SECONDS=120 ./collect_a3_host_memory.sh a3_pss.log
# Use SAMPLES=0 to collect until the process is stopped.

set -uo pipefail

OUTPUT_LOG=${1:-a3_hostmem_$(date +%Y%m%d_%H%M%S).log}
SAMPLES=${SAMPLES:-1}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-60}
TOP_MAPPINGS=${TOP_MAPPINGS:-20}
TOP_PSS_PROCESSES=${TOP_PSS_PROCESSES:-30}
COMPACT=${COMPACT:-0}

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

process_class() {
    local command=$1
    case "$command" in
        *DiffusionAgentLoopWorker*) echo "DiffusionAgentLoopWorker" ;;
        *RewardLoopWorker*) echo "RewardLoopWorker" ;;
        *vLLMOmniHttpServer*) echo "vLLMOmniHttpServer" ;;
        *WorkerDict*) echo "WorkerDict" ;;
        *TaskRunner*) echo "TaskRunner" ;;
        *raylet*) echo "raylet" ;;
        *plasma_store*) echo "plasma_store" ;;
        *python* | *Python*) echo "other_python" ;;
        *) echo "other" ;;
    esac
}

collect_all_process_pss() {
    local rows
    rows=$(mktemp /tmp/a3_process_pss.XXXXXX)

    for proc_dir in /proc/[0-9]*; do
        local pid=${proc_dir##*/}
        [[ -r "$proc_dir/smaps_rollup" ]] || continue

        local command
        command=$(tr '\0' ' ' <"$proc_dir/cmdline" 2>/dev/null || true)
        [[ -n "$command" ]] || command=$(cat "$proc_dir/comm" 2>/dev/null || echo unknown)

        local class
        class=$(process_class "$command")
        local process_name
        process_name=$(tr '\t\n' '  ' <"$proc_dir/comm" 2>/dev/null || echo unknown)

        awk -v pid="$pid" -v class="$class" -v name="$process_name" '
            /^Pss:/ { pss = $2 }
            /^Pss_Anon:/ { pss_anon = $2 }
            /^Pss_File:/ { pss_file = $2 }
            /^Pss_Shmem:/ { pss_shmem = $2 }
            /^Private_Dirty:/ { private_dirty = $2 }
            /^LazyFree:/ { lazy_free = $2 }
            /^AnonHugePages:/ { anon_huge = $2 }
            END {
                printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
                    class, pid, name, pss, pss_anon, pss_file, pss_shmem,
                    private_dirty, lazy_free, anon_huge
            }
        ' "$proc_dir/smaps_rollup" >>"$rows" 2>/dev/null || true
    done

    section "All-process physical memory by class (GiB)"
    echo "CLASS PSS PSS_ANON PSS_FILE PSS_SHMEM PRIVATE_DIRTY LAZY_FREE ANON_HUGE"
    awk -F '\t' '
        {
            pss[$1] += $4; anon[$1] += $5; file[$1] += $6; shmem[$1] += $7
            dirty[$1] += $8; lazy[$1] += $9; huge[$1] += $10
        }
        END {
            for (class in pss) {
                printf "%-28s %8.2f %8.2f %8.2f %9.2f %13.2f %9.2f %9.2f\n",
                    class, pss[class] / 1048576, anon[class] / 1048576,
                    file[class] / 1048576, shmem[class] / 1048576,
                    dirty[class] / 1048576, lazy[class] / 1048576,
                    huge[class] / 1048576
            }
        }
    ' "$rows" | sort -k2,2nr

    section "Top processes by proportional set size (GiB)"
    echo "CLASS PID NAME PSS PSS_ANON PSS_FILE PSS_SHMEM PRIVATE_DIRTY LAZY_FREE ANON_HUGE"
    sort -t $'\t' -k4,4nr "$rows" |
        head -n "$TOP_PSS_PROCESSES" |
        awk -F '\t' '{
            printf "%-28s %-8s %-18s %8.2f %8.2f %8.2f %9.2f %13.2f %9.2f %9.2f\n",
                $1, $2, $3, $4 / 1048576, $5 / 1048576, $6 / 1048576,
                $7 / 1048576, $8 / 1048576, $9 / 1048576,
                $10 / 1048576
        }'

    rm -f "$rows"
}

collect_cgroup_memory() {
    local relative_path
    relative_path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2>/dev/null)
    local cgroup_dir="/sys/fs/cgroup${relative_path:-/}"

    section "Cgroup memory"
    echo "cgroup_dir: $cgroup_dir"
    for name in memory.current memory.peak memory.max memory.swap.current memory.swap.max; do
        if [[ -r "$cgroup_dir/$name" ]]; then
            echo "$name: $(cat "$cgroup_dir/$name")"
        fi
    done
    if [[ -r "$cgroup_dir/memory.stat" ]]; then
        grep -E \
            '^(anon|file|kernel|kernel_stack|pagetables|percpu|sock|shmem|file_mapped|file_dirty|inactive_anon|active_anon|inactive_file|active_file|slab_reclaimable|slab_unreclaimable|thp_fault_alloc|thp_collapse_alloc) ' \
            "$cgroup_dir/memory.stat" || true
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
    local sample_total=$SAMPLES
    if ((SAMPLES == 0)); then
        sample_total=continuous
    fi

    section "sample=$sample/$sample_total time=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "hostname: $(hostname)"
    echo "kernel: $(uname -a)"

    section "System memory"
    free -h || true
    grep -E \
        '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapCached|Active|Inactive|AnonPages|Mapped|Shmem|AnonHugePages|ShmemHugePages|ShmemPmdMapped|Unevictable|Mlocked|KReclaimable|Slab|SReclaimable|SUnreclaim|PageTables|KernelStack|CommitLimit|Committed_AS):' \
        /proc/meminfo || true

    collect_cgroup_memory

    section "Shared-memory filesystems"
    df -h /dev/shm /tmp 2>&1 || true

    section "Top Ray/Python processes"
    ps -eo user,pid,ppid,%mem,rss,vsz,cmd --sort=-rss |
        awk 'NR == 1 || /ray|python/ {print; count++} count >= 30 {exit}' || true

    collect_all_process_pss

    if [[ "$COMPACT" == "1" ]]; then
        return
    fi

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

if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "SAMPLES must be a non-negative integer (0 means continuous), got: $SAMPLES" >&2
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
echo "compact: $COMPACT"

if ((SAMPLES == 0)); then
    sample=1
    while true; do
        collect_sample "$sample"
        ((sample += 1))
        sleep "$INTERVAL_SECONDS"
    done
else
    for ((sample = 1; sample <= SAMPLES; sample++)); do
        collect_sample "$sample"
        if ((sample < SAMPLES)); then
            sleep "$INTERVAL_SECONDS"
        fi
    done
fi

section "A3 host-memory collection finished"
echo "output_log: $OUTPUT_LOG"
