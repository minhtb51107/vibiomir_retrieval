"""Export and disable ONLY the verified legacy ViBioMIR scheduled launcher.

Does not stop a running task/process. Exports remain private for rollback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.runtime.state_file import write_json

TASK='ViBioMIR_phase12_full_pubmed'
OUT=ROOT/'artifacts/final_freeze_20261027/scheduler'


def powershell(code):
    prefix='[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);$ErrorActionPreference="Stop";'
    return subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',prefix+code],encoding='utf-8-sig',stderr=subprocess.PIPE)


def inspect():
    code=f"$t=Get-ScheduledTask -TaskName '{TASK}';[pscustomobject]@{{name=$t.TaskName;path=$t.TaskPath;state=$t.State.ToString();enabled=$t.Settings.Enabled;actions=@($t.Actions|ForEach-Object{{[pscustomobject]@{{execute=$_.Execute;arguments=$_.Arguments;working_directory=$_.WorkingDirectory}}}})}}|ConvertTo-Json -Depth 5 -Compress"
    return json.loads(powershell(code))


def confirmed(data):
    actions=data.get('actions',[])
    return data.get('name')==TASK and bool(actions) and all(
        Path(a['working_directory']).resolve()==ROOT and Path(a['execute']).resolve().is_relative_to(ROOT)
        and 'competition_supervisor.py' in a['arguments'] for a in actions)


def run(apply=False):
    data=inspect()
    if not confirmed(data):raise RuntimeError('scheduled action not confirmed wholly within this project')
    if not apply:return dict(task=TASK,state=data['state'],project_owned=True,changed=False)
    if data['state']=='Running':raise RuntimeError('task is running; require graceful worker checkpoint, not Stop-ScheduledTask')
    OUT.mkdir(parents=True,exist_ok=True);xml_path=OUT/'task_before.xml'
    if xml_path.exists():raise RuntimeError('original scheduled-task export already exists; preserve it, do not overwrite')
    xml=powershell(f"Export-ScheduledTask -TaskName '{TASK}'")
    xml_path.write_bytes(xml.encode('utf-16'))
    write_json(OUT/'task_before_private.json',data)
    if inspect()!=data:raise RuntimeError('scheduled task changed during precheck')
    powershell(f"Disable-ScheduledTask -TaskName '{TASK}' | Out-Null")
    after=inspect()
    if after['enabled'] or after['state']!='Disabled':raise RuntimeError('scheduled task was not disabled')
    report=dict(task=TASK,project_owned=True,enabled_before=data['enabled'],state_before=data['state'],
        enabled_after=after['enabled'],state_after=after['state'],running_process_terminated=False,
        export_sha256=hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        rollback='Enable only this task after explicit project reopening approval; private XML export preserves original definition.')
    write_json(OUT/'result.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    try:print(json.dumps(run(args.apply)))
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write_json(OUT/'blocker.json',dict(type=type(exc).__name__,message=str(exc),task=TASK))
        print(type(exc).__name__+': scheduled-task freeze failed; see private blocker.json',file=sys.stderr)
        raise SystemExit(2)
