import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.retrieval.owner_alignment import independent, records, select_documents, verify_sources
from tools.phase27_owner_alignment_ablation import checkpoint, cached_competitor


def words(prefix):
    return ' '.join(prefix+str(i) for i in range(60))


def fixture():
    chunks=[dict(doc_id=i,chunk_text=words(str(i))) for i in range(1,21)]
    chunks += [dict(doc_id=21,chunk_text=words('a')),dict(doc_id=21,chunk_text=words('b'))]
    chunks += [dict(doc_id=1,chunk_text=words('extra'+str(i))) for i in range(18)]
    return dict(id=1,relevant_docs=list(range(1,21)),relevant_chunks=chunks)


class OwnerAlignmentTests(unittest.TestCase):
    def test_historical_pure_selector_equivalence_when_available(self):
        import random
        try:
            from tools.cached_doc_evidence_experiment import select_documents as legacy
        except ImportError:
            self.skipTest('historical private experiment code not installed')
        rng=random.Random(20261024)
        for case in range(80):
            r=fixture()
            r['relevant_chunks']=[dict(doc_id=rng.randint(1,25),chunk_text=words(str(rng.randrange(12)))) for _ in range(40)]
            self.assertEqual(select_documents(r),legacy(r))

    def test_head_protected_chunks_not_mutated_and_determinism(self):
        r=fixture();before=copy.deepcopy(r);docs,changes=select_documents(r)
        self.assertEqual(docs[:19],list(range(1,20)));self.assertEqual(docs[-1],21)
        self.assertEqual(changes[0]['removed'],20);self.assertEqual(before,r)
        self.assertEqual((docs,changes),select_documents(r))

    def test_overlap_not_two_independent_windows(self):
        a=words('same');self.assertFalse(independent([(1,a),(2,a+' tail')]))
        r=fixture();r['relevant_chunks'][21]['chunk_text']=r['relevant_chunks'][20]['chunk_text']
        self.assertEqual(select_documents(r)[1],[])

    def test_multiwindow_tail_protected(self):
        r=fixture()
        for i,d in enumerate(range(16,21)):r['relevant_chunks'][22+i]['doc_id']=d
        self.assertEqual(select_documents(r)[1],[])

    def test_source_owner_and_span_fail_closed(self):
        r=fixture();c=sqlite3.connect(':memory:');c.execute('CREATE TABLE documents(doc_id INTEGER,normalized_text TEXT)')
        for d in {x['doc_id'] for x in r['relevant_chunks']}:
            c.execute('INSERT INTO documents VALUES(?,?)',(d,'\n'.join(x['chunk_text'] for x in r['relevant_chunks'] if x['doc_id']==d)))
        self.assertIn(21,verify_sources(r,c));r['relevant_chunks'][0]['chunk_text']='fabricated'
        with self.assertRaisesRegex(ValueError,'nonverbatim'):verify_sources(r,c)
        c.close()

    def test_streaming_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.json';p.write_text(json.dumps([fixture()]),encoding='utf-8')
            self.assertEqual(list(records(p)),[fixture()]);p.write_text('[{"id":1}',encoding='utf-8')
            with self.assertRaises(ValueError):list(records(p))

    def test_exact_cached_query_and_owner_not_fabricated(self):
        scores=sqlite3.connect(':memory:');scores.execute('CREATE TABLE pairs(query_id,chunk_id,query_text,chunk_text,rerank_score)')
        source=sqlite3.connect(':memory:');source.execute('CREATE TABLE documents(doc_id,normalized_text)')
        source.execute("INSERT INTO documents VALUES(21,'original text')")
        self.assertIsNone(cached_competitor(scores,source,1,21,'query')['score'])
        scores.execute("INSERT INTO pairs VALUES(1,'21-abc','query','original text',2.5)")
        self.assertEqual(cached_competitor(scores,source,1,21,'query')['score'],2.5)
        with self.assertRaisesRegex(ValueError,'query identity'):cached_competitor(scores,source,1,21,'wrong')
        scores.close();source.close()

    def test_checkpoint_signature_and_resume(self):
        import tools.phase27_owner_alignment_ablation as runner
        old=runner.ART
        try:
            with tempfile.TemporaryDirectory() as d:
                runner.ART=Path(d);c=checkpoint('signature');c.execute("INSERT INTO selections VALUES(1,'[]','[]','frozen')");c.commit();c.close()
                c=checkpoint('signature');self.assertEqual(c.execute('SELECT count(*) FROM selections').fetchone()[0],1);c.close()
                with self.assertRaisesRegex(ValueError,'signature mismatch'):checkpoint('different')
        finally:runner.ART=old

    def test_local_full_1200_query_checkpoint_replay_when_available(self):
        import tools.phase27_owner_alignment_ablation as runner
        if not (runner.ART/'ab_report.json').exists() or not (runner.ART/'selection.sqlite').exists():
            self.skipTest('private completed ablation not installed')
        report=json.loads((runner.ART/'ab_report.json').read_text(encoding='utf-8'))
        if report.get('status')!='COMPLETE':self.skipTest('ablation blocked or unfinished')
        runner.guard(start=True)
        db=runner.readonly(runner.ART/'selection.sqlite')
        count=changed=slots=0
        try:
            for row in records(runner.BEST.with_suffix('.json')):
                if count%25==0:runner.guard()
                docs,proposals=select_documents(row)
                saved=db.execute('SELECT docs,changes,chunks_sha FROM selections WHERE query_id=?',(row['id'],)).fetchone()
                self.assertEqual(json.loads(saved[0]),docs)
                self.assertEqual(json.loads(saved[1]),proposals)
                self.assertEqual(saved[2],runner.digest(row['relevant_chunks']))
                self.assertEqual(docs[:15],row['relevant_docs'][:15])
                count+=1;changed+=bool(proposals);slots+=len(proposals)
            self.assertEqual((count,changed,slots),(1200,38,40))
        finally:db.close()

    def test_local_frozen_review_queue_exact_provenance_when_available(self):
        import tools.phase27_owner_alignment_ablation as runner
        path=runner.ART/'source_review_queue.json'
        if not path.exists():self.skipTest('private frozen review queue not installed')
        queue=json.loads(path.read_text(encoding='utf-8'))['cases']
        mapping=json.loads((runner.ART/'review_mapping.json').read_text(encoding='utf-8'))
        report=json.loads((runner.ART/'ab_report.json').read_text(encoding='utf-8'))
        self.assertEqual(runner.sha(path),report['source_review_queue_sha256'])
        self.assertEqual(len(queue),16);self.assertEqual(len({r['case_id'] for r in queue}),16)
        self.assertEqual(len({r['query_id'] for r in mapping.values()}),16)
        db=runner.readonly(runner.ART/'selection.sqlite');source=runner.readonly(runner.P27/'cap16/round1/canonical_text.sqlite')
        try:
            for row in queue:
                info=mapping[row['case_id']]
                data=json.loads(db.execute('SELECT payload FROM proposal_evidence WHERE query_id=? AND inserted=? AND removed=?',
                    (info['query_id'],info['inserted'],info['removed'])).fetchone()[0])
                self.assertNotEqual(data['split'],'held_out')
                self.assertEqual(row['judgment'],'UNJUDGED')
                for side in ['LEFT','RIGHT']:
                    candidate=info['candidate_side']==side;owner=info['inserted'] if candidate else info['removed']
                    expected=[w['text'] for w in data['inserted_windows' if candidate else 'removed_selected_windows']]
                    if not expected and not candidate and data['competitor']['evidence']:expected=[data['competitor']['evidence']]
                    self.assertEqual(row['evidence_'+side.lower()],expected)
                    canonical=source.execute('SELECT normalized_text FROM documents WHERE doc_id=?',(owner,)).fetchone()[0]
                    self.assertTrue(all(text in canonical for text in expected))
        finally:db.close();source.close()


if __name__=='__main__':unittest.main()
