'''SYNTHETIC data generator (reference implementation of a PROPOSAL).

Writes clearly labeled simulated population datasets to data/synthetic/.
No real recordings, no network access, no benchmark numbers.
'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generators.dataset import SimParams, generate, save_synthetic  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / 'synthetic'
RHOS = (0.0, 0.5, 1.0)


def generate_all(out_dir=None, rhos=RHOS, seed=0):
    '''Write one SYNTHETIC dataset per sequence index rho; return the paths.'''
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, rho in enumerate(rhos):
        ds = generate(SimParams(rho=rho), seed=seed + i)
        path = out_dir / f'SYNTHETIC_rho{rho:.2f}.npz'
        save_synthetic(ds, path)
        paths.append(path)
    return paths


if __name__ == '__main__':
    for p in generate_all():
        print('wrote SYNTHETIC dataset:', p)
