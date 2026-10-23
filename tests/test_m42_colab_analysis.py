import hashlib
import sqlite3
import zipfile

import numpy as np
import pytest

from tools.analyze_m42_colab import verify_archive, verify_windows, reconstruct, normalized


def test_archive_extracted_match_and_tamper(tmp_path):
    archive=tmp_path/'original.zip'; extracted=tmp_path/'extracted'; extracted.mkdir()
    with zipfile.ZipFile(archive,'w') as f: f.writestr('result.json','{}')
    (extracted/'result.json').write_text('{}')
    assert verify_archive(archive,extracted)['members_matched']==1
    (extracted/'result.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='differs'): verify_archive(archive,extracted)


def test_checkpoint_counts_and_corruption(tmp_path):
    db=tmp_path/'checkpoint.sqlite'; values=np.ones((4,1024),dtype=np.float32)
    con=sqlite3.connect(db)
    con.execute('CREATE TABLE windows(start INTEGER,count INTEGER,sha256 TEXT)')
    con.execute('CREATE TABLE metadata(signature TEXT)'); con.execute("INSERT INTO metadata VALUES('frozen')")
    con.execute('INSERT INTO windows VALUES(?,?,?)',(0,4,hashlib.sha256(values.tobytes()).hexdigest()))
    con.commit(); con.close()
    assert verify_windows(db,values)['rows_verified']==4
    values[0,0]=2
    with pytest.raises(ValueError,match='checksum'): verify_windows(db,values)


def test_cross_chunk_reconstruction_is_not_answer_label():
    rows=[{'start_offset':0,'end_offset':4,'evidence_text':'abcd'},
          {'start_offset':2,'end_offset':6,'evidence_text':'cdef'}]
    assert reconstruct(rows)=='abcdef'
    assert normalized('甲\n 乙')=='甲 乙'
    rows[1]['evidence_text']='XYef'
    with pytest.raises(ValueError,match='overlap'): reconstruct(rows)
