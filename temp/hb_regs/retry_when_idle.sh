#!/bin/bash
# Purpose: Retry a GPU 3 benchmark until it completes, waiting for other jobs to leave first.
# Usage: temp/hb_regs/retry_when_idle.sh <attempts> <min_free_mib> <command...>
attempts=$1; need=$2; shift 2
for i in $(seq 1 "$attempts"); do
  quiet=0
  for _ in $(seq 1 240); do
    read free util < <(nvidia-smi -i 3 --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if [ "$free" -ge "$need" ] && [ "$util" -lt 10 ]; then
      quiet=$((quiet + 1)); [ $quiet -ge 2 ] && break
    else
      quiet=0
    fi
    sleep 5
  done
  echo "attempt $i: free=${free} MiB util=${util}%"
  "$@" && exit 0
  echo "attempt $i failed; retrying"
done
exit 1
