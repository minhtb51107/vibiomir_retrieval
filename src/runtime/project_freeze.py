"""Fail-closed guard for future supervised computation in an archived project."""
import json
from pathlib import Path


def assert_project_active(root):
    path=Path(root)/'PROJECT_FREEZE.json'
    if not path.exists():return
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
    except (ValueError,OSError) as exc:
        raise RuntimeError('PROJECT_FREEZE.json unreadable; no supervised worker may start') from exc
    if not isinstance(value,dict) or type(value.get('frozen')) is not bool:
        raise RuntimeError('Invalid project freeze schema; no supervised worker may start')
    if value['frozen']:
        raise RuntimeError('PROJECT FROZEN: no new ViBioMIR worker. Restore/backup maintenance only; reopening requires explicit user approval.')
