"""Restore only the two released study-owned linear heads, or check their bytes."""
import argparse,base64,hashlib,io,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def validate_record(record):
    data=base64.b64decode(record['original_npz_base64'],validate=True)
    if hashlib.sha256(data).hexdigest()!=record['original_npz_sha256']:
        raise ValueError('Released coefficient checksum differs')
    if len(data)>32768:raise ValueError('Unexpected coefficient asset size')
    with np.load(io.BytesIO(data),allow_pickle=False) as values:
        if set(values.files)!={'coefficients','intercept','classes'}:raise ValueError('Unexpected head fields')
        if values['coefficients'].shape!=(1,512) or values['intercept'].shape!=(1,):raise ValueError('Unexpected linear head shape')
        if values['coefficients'].dtype!=np.float32 or values['intercept'].dtype!=np.float32:raise ValueError('Unexpected head dtype')
        if not np.array_equal(values['classes'],[0,1]):raise ValueError('Unexpected origin classes')
        if not np.isfinite(values['coefficients']).all() or not np.isfinite(values['intercept']).all():raise ValueError('Nonfinite head')
    return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--parameters',type=Path,default=ROOT/'decision_parameters/passive_heads.json')
    args=parser.parse_args()
    record=json.loads(args.parameters.read_text(encoding='utf-8'))
    if record.get('first_party_linear_decision_parameters_only') is not True or set(record['heads'])!={'resnet','clip'}:
        raise ValueError('Unexpected release parameter schema')
    for name,head in record['heads'].items():
        data=validate_record(head)
        if not args.check_only:
            destination=(ROOT/'outputs/conference_clean'/f'{name}_full.npz').resolve()
            destination.relative_to(ROOT.resolve())
            if destination.exists() and destination.read_bytes()!=data:raise FileExistsError(destination)
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(data)
        print('Verified study-owned linear coefficients:',name)


if __name__=='__main__':main()
