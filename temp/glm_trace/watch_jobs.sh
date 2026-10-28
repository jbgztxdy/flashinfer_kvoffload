#!/usr/bin/env bash
# Quiet cluster-side watcher. Run with nohup; all output stays in a WORK log.
# Usage: bash watch_jobs.sh 1004362 1004363 1040680 1040816
set -u
root=/WORK/PUBLIC/luyouy_work/rtx
declare -A finished previous
while true; do
  if ! queue=$(squeue -h -u luyouy -o '%i %T'); then
    date -Is
    echo 'squeue failed; retrying in 600 s'
    sleep 600
    continue
  fi
  pending=0
  for id in "$@"; do
    [[ ${finished[$id]:-0} == 1 ]] && continue
    state=$(awk -v id="$id" '$1 == id {print $2}' <<< "$queue")
    if [[ -z $state ]]; then
      record=$(sacct -X -n -P -j "$id" -o State,ExitCode 2>/dev/null) || record=''
      state=${record%%|*}
      state=${state%% *}
    fi
    if [[ ${previous[$id]:-} != "$state" ]]; then
      echo "$(date -Is) job=$id state=${state:-UNKNOWN}"
      previous[$id]=$state
    fi
    case "$state" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
        finished[$id]=1
        sacct -X -j "$id" -o JobID,JobName,State,ExitCode,Elapsed
        for log in "$root"/yjw/test/glm_trace/logs/*-"$id".out \
                   "$root"/yjw/flashinfer/temp/h800/logs/*-"$id".out; do
          [[ -f $log ]] && tail -80 "$log"
        done
        ;;
      *) pending=$((pending + 1)) ;;
    esac
  done
  [[ $pending == 0 ]] && break
  sleep 600
done
