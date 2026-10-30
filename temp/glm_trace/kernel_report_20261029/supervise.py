"""Record termination-signal provenance, honor the signal, and preserve child exit status."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

parser=argparse.ArgumentParser()
parser.add_argument("--record",type=Path,required=True)
parser.add_argument("command",nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command[1:] if args.command[:1]==["--"] else args.command
if not command:parser.error("a child command is required")
args.record.parent.mkdir(parents=True,exist_ok=True)
lock=threading.Lock()
state=dict(supervisor_pid=os.getpid(),command=command,started=time.strftime("%FT%T%z"),signals=[])


def save():
    temp=args.record.with_suffix(args.record.suffix+".new")
    temp.write_text(json.dumps(state,indent=2)+"\n");temp.replace(args.record)


watched={signal.SIGTERM,signal.SIGINT,signal.SIGHUP}
previous=signal.pthread_sigmask(signal.SIG_BLOCK,watched)
# Unblock in the child before exec so it retains normal termination behavior.
child=subprocess.Popen(command,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous))
state["child_pid"]=child.pid
save()


def receive():
    while True:
        info=signal.sigwaitinfo(watched)
        with lock:
            state["signals"].append(dict(signal=info.si_signo,sender_pid=info.si_pid,
                sender_uid=info.si_uid,code=info.si_code,timestamp=time.strftime("%FT%T%z")))
            save()
        try:child.send_signal(info.si_signo)
        except ProcessLookupError:pass


threading.Thread(target=receive,daemon=True).start()
code=child.wait()
with lock:
    state.update(child_returncode=code,ended=time.strftime("%FT%T%z"))
    save()
raise SystemExit(128-code if code<0 else code)
