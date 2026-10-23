import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.multilingual_retrieval.m42_execution import measured_limits, trial


class SemanticExecutionTests(unittest.TestCase):
    def test_cold_gate_includes_runtime_import_cost(self):
        profile={'measurements':[
            {'phase':'process_start','rss_mib':40,'private_mib':100},
            {'phase':'before_model_load','rss_mib':500,'private_mib':1000,'gpu_allocated_mib':0},
            {'phase':'first_batch_written','rss_mib':800,'private_mib':4000,'gpu_allocated_mib':1100}]}
        limits=measured_limits(profile)
        self.assertEqual(limits['required_cold_ram_mib'],950)
        self.assertEqual(limits['required_warm_ram_mib'],375)

    def test_blocked_trial_never_imports_model_runtime(self):
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'artifacts/multilingual_retrieval_m42').mkdir(parents=True)
            gate={'status':'BLOCKED_RESOURCES','model_loaded':False}
            before='sentence_transformers' in sys.modules
            with patch('src.multilingual_retrieval.m42_execution.validate_manifest',return_value={}), \
                 patch('src.multilingual_retrieval.m42_execution.preflight',return_value=gate):
                self.assertEqual(trial(root),gate)
            self.assertEqual('sentence_transformers' in sys.modules,before)
            data=json.loads((root/'artifacts/multilingual_retrieval_m42/execution_resource_preflight.json').read_text())
            self.assertFalse(data['model_loaded'])
