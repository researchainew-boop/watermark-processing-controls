"""Deployment feature and prospective-lock checks for the authorized study."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import phase6_features as f


def test_views_use_only_received_pixels():
    pixels=np.random.default_rng(3).integers(0,256,(256,256,3),dtype=np.uint8)
    original=pixels.copy()
    views=f.received_views(pixels)
    assert len(views)==2
    assert np.array_equal(views[0],original)
    assert np.array_equal(pixels,original)
    ok,encoded=cv2.imencode('.jpg',original,[cv2.IMWRITE_JPEG_QUALITY,90])
    assert ok
    assert np.array_equal(views[1],cv2.imdecode(encoded,cv2.IMREAD_COLOR))


def test_final_inference_requires_protocol(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'OUT',tmp_path)
    with pytest.raises(RuntimeError,match='Freeze method'):
        f.check_final_lock('calibration')
    assert f.check_final_lock('development') is None


def test_test_requires_calibrated_thresholds(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'OUT',tmp_path)
    f.write(tmp_path/'frozen_protocol.json',{'input_sha256':{}})
    with pytest.raises(RuntimeError,match='thresholds'):
        f.check_final_lock('test')


def test_changed_lock_input_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'OUT',tmp_path)
    monkeypatch.setattr(f,'ROOT',tmp_path)
    data=tmp_path/'method.json'
    data.write_text('frozen')
    f.write(tmp_path/'frozen_protocol.json',{'input_sha256':{'method.json':f.digest(data)}})
    data.write_text('changed')
    with pytest.raises(RuntimeError,match='Frozen input changed'):
        f.check_final_lock('calibration')


def test_registered_payloads_keep_leading_zero():
    assert len(f.PAYLOADS)==3
    assert any(p.startswith('0') for p in f.PAYLOADS)
    assert all(len(p)==16 for p in f.PAYLOADS)
