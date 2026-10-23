import io
import unittest

from tools.phase27_final_relevance_check import records, redundancy


class BoundedAuditTests(unittest.TestCase):
    def test_array_stream(self):
        self.assertEqual(list(records(io.StringIO(' [ {"id":1}, {"id":2} ]'))),
                         [{"id": 1}, {"id": 2}])

    def test_empty_fail_closed(self):
        with self.assertRaises(ValueError):
            list(records(io.StringIO('')))

    def test_truncated_fail_closed(self):
        with self.assertRaises(ValueError):
            list(records(io.StringIO('[{"id":')))

    def test_redundancy_is_overlap_not_relevance(self):
        rows = [{"chunk_text": "one two three four five six seven"},
                {"chunk_text": "ONE two three four five six seven"},
                {"chunk_text": "different unrelated short passage"}]
        self.assertEqual(redundancy(rows), 1)


if __name__ == '__main__':
    unittest.main()
