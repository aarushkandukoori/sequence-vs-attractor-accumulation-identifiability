'''End-to-end demo on SYNTHETIC data (reference implementation of a PROPOSAL).

Generates (if missing) and loads SYNTHETIC datasets, computes the proposed
statistic G and two baselines, and prints whatever is computed. No results
are claimed; this only shows the pipeline runs.

Run from the repo root: python -m src.run_demo
'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.baselines import localization_baseline, participation_ratio_baseline  # noqa: E402
from data.generate_synthetic import OUT_DIR, generate_all  # noqa: E402
from src.analysis.decoders import cross_temporal_r2  # noqa: E402
from src.analysis.metrics import generalization_index  # noqa: E402
from src.generators.dataset import load_synthetic  # noqa: E402


def main():
    print('ALL DATA BELOW IS SYNTHETIC (simulated). Proposal reference code; no claims.')
    paths = sorted(OUT_DIR.glob('SYNTHETIC_rho*.npz')) or generate_all()
    for path in paths:
        ds = load_synthetic(path)
        print(f'--- {path.name} (SYNTHETIC, rho={ds.params.rho})')
        for kind in ('rates', 'fluorescence'):
            A = cross_temporal_r2(ds.observed(kind), ds.x)
            print(f'  [{kind}] G = {generalization_index(A):.3f}')
        print(f'  localization index = {localization_baseline(ds):.3f}')
        print(f'  participation ratio = {participation_ratio_baseline(ds):.3f}')


if __name__ == '__main__':
    main()
