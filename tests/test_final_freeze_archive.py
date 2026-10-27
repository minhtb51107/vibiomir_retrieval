import json
from pathlib import Path
import tempfile
import sqlite3
import unittest

from src.runtime.project_freeze import assert_project_active
from tools.final_freeze_archive import classify, cleanup_proposal, safe_relative, sha, verify_manifest
from tools.freeze_project_schedule import confirmed, ROOT, TASK


class FreezeArchiveTests(unittest.TestCase):
    def test_cleanup_protects_unknown_descendant_and_link(self):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE files(path TEXT,bytes INTEGER,class TEXT)')
        db.executemany('INSERT INTO files VALUES(?,?,?)',[
            ('test_tmp/output',86,'C'),('test_tmp/current',0,'D'),
            ('.venv/module',20,'C'),('data/checkpoint',100,'B')])
        proposal=cleanup_proposal(db,{'test_tmp':[86,2],'.venv':[20,1],'data':[100,1]},Path('/archive'))
        self.assertEqual({p['relative_path'] for p in proposal['targets']},{'.venv','test_tmp/output'})
        self.assertEqual(proposal['logical_recoverable_bytes'],106)
        self.assertFalse(proposal['approval']);db.close()

    def test_scheduler_rejects_unrelated_actions(self):
        action=dict(execute=str(ROOT/'.venv/Scripts/python.exe'),working_directory=str(ROOT),arguments='tools/competition_supervisor.py --run-id old')
        self.assertTrue(confirmed(dict(name=TASK,actions=[action])))
        self.assertFalse(confirmed(dict(name='unrelated',actions=[action])))
        self.assertFalse(confirmed(dict(name=TASK,actions=[dict(action,working_directory=str(ROOT.parent))])))

    def test_essential_data_not_mistaken_for_disposable_cache(self):
        for p in ['data/crawl/crawl.sqlite','artifacts/run/vectors.f32','data/scores/cache.sqlite','models/weights.gguf']:
            self.assertEqual(classify(p)[0],'B')
        for p in ['src/retrieval/owner_alignment.py','configs/model.yaml','docs/TIMELINE.md']:
            self.assertEqual(classify(p)[0],'A')
        for p in ['.venv/Lib/site.py','src/__pycache__/module.pyc','test_tmp/results.json']:
            self.assertEqual(classify(p)[0],'C')
        for p in ['data/checkpoint.sqlite-wal','data/vectors.tmp','.env','unidentified.bin']:
            self.assertEqual(classify(p)[0],'D')

    def test_unsafe_archive_paths_rejected(self):
        for p in ['../outside','/absolute','C:/outside','a/../../b','\\\\server\\share']:
            with self.assertRaises(ValueError):safe_relative(p)
        self.assertEqual(str(safe_relative('data/chunks.parquet')),'data/chunks.parquet')

    def test_checksum_verifier_detects_corruption_missing_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mirror=root/'mirror';mirror.mkdir();(mirror/'a').write_bytes(b'original')
            manifest=root/'manifest.jsonl';entry=dict(path='a',bytes=8,sha256=sha(mirror/'a'))
            manifest.write_text(json.dumps(entry)+'\n',encoding='utf-8')
            result=verify_manifest(manifest,mirror)
            self.assertEqual(result['files_verified'],1);self.assertFalse(result['source_destination_deletion_authorized'])
            (mirror/'a').write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError,'checksum'):verify_manifest(manifest,mirror)
            (mirror/'a').unlink()
            with self.assertRaisesRegex(ValueError,'missing'):verify_manifest(manifest,mirror)
            (mirror/'a').write_bytes(b'original');manifest.write_text((json.dumps(entry)+'\n')*2,encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'duplicate'):verify_manifest(manifest,mirror)

    def test_empty_manifest_not_verification(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'empty';p.write_text('')
            with self.assertRaisesRegex(ValueError,'empty'):verify_manifest(p,Path(d))

    def test_freeze_blocks_and_malformed_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);assert_project_active(root)
            path=root/'PROJECT_FREEZE.json';path.write_text('{"frozen":true}')
            with self.assertRaisesRegex(RuntimeError,'FROZEN'):assert_project_active(root)
            for value in ['null','{','{"frozen":"yes"}']:
                path.write_text(value)
                with self.assertRaises(RuntimeError):assert_project_active(root)
            path.write_text('{"frozen":false}');assert_project_active(root)


if __name__=='__main__':unittest.main()
