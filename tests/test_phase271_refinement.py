import unittest
from tools.phase271_refinement import refine, overlap, promotional


def chunk(doc, text):
    return {'doc_id':doc, 'chunk_text':text}


class RefinementTests(unittest.TestCase):
    def test_same_owner_only(self):
        a=chunk(1,'one two three four five six')
        self.assertTrue(overlap(a,a))
        self.assertFalse(overlap(a,chunk(2,a['chunk_text'])))

    def test_clinical_article_footer_not_whole_article_rejection(self):
        self.assertFalse(promotional(chunk(1,('clinical explanation '*200)+'\nHotline 123. Địa chỉ: here.')))
        self.assertTrue(promotional(chunk(1,'Hotline 123. Địa chỉ: street. Đặt lịch ngay.')))

    def test_stable_selection_with_original_membership(self):
        originals=[chunk(i,f'evidence unique paragraph for document {i}') for i in range(40)]
        self.assertEqual(refine(originals, originals, 'redundancy')[0], originals)

    def test_overlap_backfill_and_repeatable(self):
        originals=[chunk(1,'one two three four five six')]*2+[chunk(i,f'other text {i}') for i in range(2,40)]
        fallback=[chunk(100,'fallback exact source')]
        first=refine(originals,fallback,'redundancy')
        self.assertEqual(first,refine(originals,fallback,'redundancy'))
        self.assertEqual(len(first[0]),40)
        self.assertIn(fallback[0],first[0])
