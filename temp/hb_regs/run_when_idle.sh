#!/bin/bash
# Purpose: Wait for GPU 3 to be free of other jobs, then run a command.
# Usage: temp/hb_regs/run_when_idle.sh <min_free_mib> <max_wait_s> <command...>
need=$1; deadline=$((SECONDS + $2)); shift 2
quiet=0
while [ $SECONDS -lt $deadline ]; do
  read free util < <(nvidia-smi -i 3 --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  if [ "$free" -ge "$need" ] && [ "$util" -lt 10 ]; then
    quiet=$((quiet + 1))
    [ $quiet -ge 3 ] && break
  else
    quiet=0
  fi
  sleep 10
done
echo "starting after ${SECONDS}s wait (free=${free} MiB util=${util}%)"
"$@"
