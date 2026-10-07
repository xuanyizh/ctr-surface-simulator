"""Create the documented forward-model preview; optional dependency: matplotlib.

Run from repository root: python compare_surfaces.py
This is a deterministic scientific plot, using the same solver as the app.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from physics import preset, rod_intersections
from surface_model import SurfaceParameters, simulate_detector


def main():
    g = preset('Oblique steps')
    p = SurfaceParameters()
    result = simulate_detector(g, p)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.1), layout='constrained')
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#dce1e8')
    for column, (name, case) in enumerate(result['cases'].items()):
        s = case['surface']
        use_x = np.abs(s.x / 10) <= 700
        use_y = np.abs(s.y / 10) <= 500
        top = axes[0, column]
        height_map = top.pcolormesh(s.x[use_x] / 10, s.y[use_y] / 10,
                                   s.height_uc[np.ix_(use_y, use_x)],
                                   cmap='viridis', vmin=-4, vmax=4, shading='auto', rasterized=True)
        top.set_title(name.replace('Selected combination', 'All three together'), fontsize=11)
        top.set_xlabel('x across steps (nm)')
        if column == 0:
            top.set_ylabel('y along steps (nm)')
        bottom = axes[1, column]
        image = bottom.pcolormesh(result['u_mm'], result['v_mm'],
                                  np.maximum(case['intensity'], 1e-10), cmap=cmap,
                                  norm=LogNorm(1e-6, 1), shading='auto', rasterized=True)
        peaks = [r for r in rod_intersections(g) if r['status'] == 'Captured']
        bottom.scatter([r['u'] for r in peaks], [r['v'] for r in peaks], s=22,
                       edgecolor='#49deff', facecolor='none', linewidth=.7)
        bottom.set_xlim(-1.4, 1.4)
        bottom.set_ylim(-8.5, 8.5)
        bottom.set_xlabel('Detector u (mm)')
        if column == 0:
            bottom.set_ylabel('Detector v (mm)')
    fig.colorbar(height_map, ax=axes[0, :], shrink=.8, label='Height (unit cells)')
    fig.colorbar(image, ax=axes[1, :], shrink=.8, label='Morphology intensity M (shared scale)')
    fig.suptitle('Nonideal miscut surfaces → coherent detector scattering', fontsize=17)
    fig.supxlabel('μ = 0.10° · mean width = 224 nm · centre l = 0.5 · one coherent patch · seed 2026\n'
                  'Width CV 0.15 · edge RMS 20 nm, correlation 50 nm · island radius 60 nm, expected coverage 0.15\n'
                  'Top: central surface crop. Bottom: exact detector q, cropped view; axes have unequal scales. Cyan: ideal rod centres.\n'
                  'Same-material normalized morphology model; no material CTR factor or pixel-area integration.', fontsize=9)
    output = Path(__file__).resolve().parent / 'surface_forward_preview.png'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == '__main__':
    main()
