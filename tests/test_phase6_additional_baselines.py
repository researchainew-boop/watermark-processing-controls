"""Arithmetic checks for simple secondary rules, without reading test images."""
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import phase6_additional_baselines as b
import phase6_fusion as f

def data():
    return {"z":np.array([-100.,3.]),"features":np.zeros((2,3,len(f.FEATURES)))}

def test_complete_message_union_uses_exact_native_flag():
    x=data();x["features"][0,2,f.FEATURES.index("full_match")]=1
    result=b.score(x,{"kind":"full_message_union"})
    assert result[0]==2 and result[1]==expit(3)

def test_union_without_full_recovery_preserves_probe_order():
    x=data();assert np.array_equal(b.score(x,{"kind":"full_message_union"}),expit(x["z"]))

def test_tuned_confidence_applies_the_declared_bound_and_query_max():
    x=data();x["features"][:,:,f.FEATURES.index("stability")]=.75
    x["features"][:,:,f.FEATURES.index("agreement")]=.5
    x["features"][0,1,f.FEATURES.index("agreement")]=.875
    result=b.score(x,{"kind":"tuned_confidence","bound":4})
    assert np.array_equal(result,x["z"]+np.array([2.25,0.]))

def test_zero_fallback_is_the_probe():
    x=data();assert np.array_equal(b.score(x,{"kind":"probe"}),x["z"])
