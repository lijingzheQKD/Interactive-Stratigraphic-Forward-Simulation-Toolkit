"""
example_run.py  –  Demo test script for sedapp_py.py
=====================================================
Replicates the original Sedapp V1.12 Example.r parameter set and
plotting behaviour **exactly**:

  * One PNG figure per time step  (200 figures total)
  * Each figure shows all time horizons up to that step as
    coloured scatter points (water-depth / sand-fraction / porosity)
  * Every 5th horizon is overdrawn with a dotted black line
  * A blue horizontal sea-level line is drawn
  * The initial topography (current subsided h0) and the template basin outline are shown
  * Y-axis range: [min(topo) − 2000, max(topo) + 5]  (same as R)
  * Colour palette: rev(topo.colors(101))  (exact R palette)

Key fixes vs original:
  1. DS = subsidence_samples * 0.68 * 4/3 / t1  (matches R formula)
  2. Scatter y-positions use h0_series[:, i] (current subsided h0 at step i)
  3. Exact R rev(topo.colors(101)) palette

Usage
-----
    python3 example_run.py

Outputs
-------
    example_output/
        plots/
            plot_t0002_0001.png  …  plot_t0400_0200.png   (200 figures)
        slope_series.csv
        z_final.csv
        z_matrix_sample.csv
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend – works everywhere
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# ── Make sure sedapp_py is importable ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sedapp_py import (run_sedapp, chazhi_function,
                       best_color_function, heng_sum,
                       guiyihua)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Parameters  (exact mirror of Example.r)
# ═════════════════════════════════════════════════════════════════════════════
L   = 15000.0    # domain length (m)
dt  = 2.0        # time step (Ma)
t1  = 400.0      # total geologic time (Ma)
nx  = 300        # number of cells

# Initial topography control points  [x (m), z (m)]
topo_samples = np.array([
    [0,    9.0],
    [1890, 0.0],
    [3900, -86.0],
    [L,    -6.8],
])

# Subsidence rate samples  [x (m), total subsidence (m) over t1]
# NOTE: These values are in METERS of total subsidence over t1=400 Ma.
# The run_sedapp function applies the R formula: DS = values * 0.68 * 4/3 / t1
# to convert to m/Ma subsidence rate.
subsidence_samples = np.array([
    [0,        0.0],
    [1890.03,  0.0],
    [2044.67,  28.0],
    [2216.49,  441.7],
    [2405.5,   722.0],
    [2731.96,  1107.0],
    [3250,     1409.0],
    [3402.06,  1530.0],
    [3436.43,  1537.0],
    [4673.5,   1530.0],
    [5910.65,  1373.0],
    [7285.22,  1239.0],
    [8556.7,   1000.0],
    [11271.5,  530.0],
    [13505.2,  77.0],
    [14003.4,  0.0],
    [15000,    0.0],
])

# Template basin (final target outline) – same as subsidence_samples x-coords
template_basin_samples = np.array([
    [0,        0.0],
    [1890.03,  0.0],
    [2044.67,  28.0],
    [2216.49,  441.7],
    [2405.5,   722.0],
    [2731.96,  1107.0],
    [3250,     1409.0],
    [3402.06,  1479.0],
    [3436.43,  1486.0],
    [4673.5,   1479.0],
    [5910.65,  1373.0],
    [7285.22,  1239.0],
    [8556.7,   1135.0],
    [11271.5,  568.0],
    [13505.2,  77.0],
    [14003.4,  0.0],
    [15000,    0.0],
])

k = int(np.ceil(t1 / dt))      # = 200 steps
sl_series = np.zeros(k)        # constant sea level = 0 m

params = {
    'nx':  nx,
    'L':   L,
    't':   0.0,
    't1':  t1,
    'dt':  dt,
    'topo_samples':    topo_samples,
    'subsidence_samples': subsidence_samples,
    'sl_series':       sl_series,
    'df_dco':          168e6,
    'df_EA':           968e2,
    'df_zhishu':       2.0,
    'env_energy':      100.0,
    'df_wdscale_sand': 0.1,
    'df_wdscale_mud':  0.075,          # 0.75 × 0.1
    'expected_sedimentation_sand': 0.135,
    'expected_sedimentation_mud':  0.135 * 5,
    'dep_ero_ratio':   1e6,
    'overlying_layer_thickness': 0.0,
    'color_mode': 1,                   # 1=water-depth, 2=sand fraction, 3=porosity
}

# ═════════════════════════════════════════════════════════════════════════════
# 2.  Colour setup  (EXACT R palette: rev(topo.colors(101)))
# ═════════════════════════════════════════════════════════════════════════════
# These are the exact hex values from R's rev(topo.colors(101))
_R_TOPO_HEX = [
    '#FFE0B3','#FFDFAD','#FFDEA7','#FFDDA2','#FFDC9C','#FFDC97','#FFDB91',
    '#FFDB8B','#FFDB86','#FFDB80','#FFDB7B','#FFDB75','#FFDB70','#FFDC6A',
    '#FFDC64','#FFDD5F','#FFDE59','#FFDF54','#FFE04E','#FFE149','#FFE343',
    '#FFE43D','#FFE638','#FFE832','#FFEA2D','#FFEC27','#FFEE21','#FFF11C',
    '#FFF316','#FFF611','#FFF90B','#FFFC06','#FFFF00','#E6FF00','#DCFF00',
    '#D2FF00','#C9FF00','#BFFF00','#B6FF00','#ACFF00','#A3FF00','#99FF00',
    '#8FFF00','#86FF00','#7CFF00','#73FF00','#69FF00','#60FF00','#56FF00',
    '#4DFF00','#43FF00','#39FF00','#30FF00','#26FF00','#1DFF00','#13FF00',
    '#0AFF00','#00FF00','#00FF0A','#00FF13','#00FF1D','#00FF26','#00FF30',
    '#00FF39','#00FF43','#00FF4D','#00E5FF','#00DCFF','#00D3FF','#00CAFF',
    '#00C1FF','#00B8FF','#00AFFF','#00A6FF','#009EFF','#0095FF','#008BFF',
    '#0083FF','#007AFF','#0071FF','#0068FF','#005EFF','#0055FF','#004CFF',
    '#0043FF','#003BFF','#0032FF','#0028FF','#001FFF','#0016FF','#000DFF',
    '#0004FF','#0400FF','#0D00FF','#1600FF','#1F00FF','#2800FF','#3100FF',
    '#3B00FF','#4300FF','#4C00FF',
]

def _hex_to_rgba(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    return (r, g, b, 1.0)

TOPO_COLORS = np.array([_hex_to_rgba(h) for h in _R_TOPO_HEX])  # (101, 4)
TOPO_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'r_topo_rev',
    [(i / 100, TOPO_COLORS[i]) for i in range(101)])


def best_color_topo(values, upper_limit=109.0, lower_limit=0.0):
    """
    Map *values* to RGBA colours using exact R rev(topo.colors(101)).
    Mirrors R's BestColorFunctionV2.1 with color.use = rev(topo.colors(101)).
    """
    v = np.asarray(values, dtype=float)
    v_clipped = np.clip(v, lower_limit, upper_limit)
    if upper_limit == lower_limit:
        normed = np.zeros_like(v_clipped)
    else:
        normed = (v_clipped - lower_limit) / (upper_limit - lower_limit)
    # Apply uniformization (guiyihua) – same as R's guiyihua()
    normed = guiyihua(normed)
    idx = np.clip(np.round(normed * 100).astype(int), 0, 100)
    return TOPO_COLORS[idx]   # (n, 4) RGBA


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Pre-compute display geometry
# ═════════════════════════════════════════════════════════════════════════════
x_value = np.linspace(L / (2 * nx), L - L / (2 * nx), nx)

# Y-axis limits: R uses huatuylim = range(min(topo)-2000, max(topo)+5)
# This is computed ONCE from the INITIAL topo samples (not tracking subsidence)
y_min_display = topo_samples[:, 1].min() - 2000.0   # = -86 - 2000 = -2086
y_max_display = topo_samples[:, 1].max() + 5.0       # = 9 + 5 = 14

# Template basin outline (negative subsidence = depth below datum)
template_x = template_basin_samples[:, 0]
template_z = -template_basin_samples[:, 1]   # R: -estimatedd.subsidence.samples[,2]
# Interpolate template to x_value grid
template_z_grid = np.interp(x_value, template_x, template_z)

# ═════════════════════════════════════════════════════════════════════════════
# 4.  Run the model
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Sedapp V1.12 – Python Port  |  Example run")
print("=" * 60)
results = run_sedapp(params)

# Unpack key results
z_mat      = results['z_matrix']              # (nx, k)
lt_mat     = results['layer_thickness_matrix']  # (nx, k)
sf_mat     = results['sand_fraction_matrix']  # (nx, k)
por_mat    = results['porosity_matrix']       # (nx, k)
wd_mat     = results['wd_matrix']             # (nx, k)
h0         = results['h0']                    # (nx,)  — final h0 (after all steps)
h0_series  = results['h0_series']             # (nx, k) — h0 at end of each step
sl_vec     = sl_series                        # (k,)
time_vec   = results['time_series']           # (k,)

# Choose colour source (color_mode=1 → water depth)
color_mode = params['color_mode']
if color_mode == 2:
    color_matrix = sf_mat
    cbar_label   = 'Sand fraction'
    upper_limit  = 1.0
elif color_mode == 3:
    color_matrix = por_mat
    cbar_label   = 'Porosity'
    upper_limit  = 1.0
else:
    color_matrix = wd_mat
    cbar_label   = 'Water depth (m)'
    upper_limit  = 109.0

# ═════════════════════════════════════════════════════════════════════════════
# 5.  Generate one figure per time step  (mirrors R plotting loop)
# ═════════════════════════════════════════════════════════════════════════════
out_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_output')
plots_dir = os.path.join(out_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

print(f"\nGenerating {k} per-step figures → {plots_dir}")
print("(This may take 2–3 minutes for 200 figures)")

for i in range(k):
    t_now = time_vec[i]
    sl_now = sl_vec[i]

    # ── h0 at this step (after i+1 subsidence applications) ──────────────────
    # Mirrors R's live h0 variable at step i: h0 = h0_initial - (i+1)*DS*dt
    h0_now = h0_series[:, i]

    fig, ax = plt.subplots(figsize=(10, 5))

    # ── Plot initial topography (grey line, same as R background) ────────────
    # R: plot(x.value, h0, ylim=huatuylim, ...) — plots current h0 as background
    ax.plot(x_value, h0_now, color='grey', linewidth=0.8, zorder=2,
            label='h0 (subsided)')

    # ── Plot each time horizon 0..i as coloured scatter + optional line ──────
    # R: BestColorFunctionV2.1 is called with the ENTIRE color.use.matrix[:,1:i]
    # so guiyihua() normalises across ALL values in ALL visible horizons at once.
    # We must replicate this: collect all color values first, then normalise.
    if i >= 0:
        # Collect all color values for horizons 0..i  shape (nx, i+1)
        color_block = color_matrix[:, :i + 1]  # (nx, i+1)
        # Apply guiyihua across all values (flatten, normalise, reshape)
        flat = color_block.flatten()
        flat_clipped = np.clip(flat, 0.0, upper_limit)
        normed = (flat_clipped - 0.0) / (upper_limit - 0.0)
        normed_gui = guiyihua(normed)  # re-normalise to stretch full range
        normed_gui = np.clip(normed_gui, 0.0, 1.0)
        idx_all = np.clip(np.round(normed_gui * 100).astype(int), 0, 100)
        rgba_all = TOPO_COLORS[idx_all].reshape(nx, i + 1, 4)  # (nx, i+1, 4)

    for ii in range(i + 1):
        # Cumulative elevation at horizon ii, using current h0 as baseline
        # This equals z.matrix[,ii] as computed at step i in R
        z_layer = h0_now + heng_sum(lt_mat[:, :ii + 1])

        # Colours already computed above with global normalisation
        c_vals = rgba_all[:, ii, :]  # (nx, 4)
        ax.scatter(x_value, z_layer,
                   c=c_vals, s=2.5, linewidths=0, zorder=3)

        # Every 5th horizon: draw a dotted black line (R: ii%%5==0)
        if (ii + 1) % 5 == 0:
            ax.plot(x_value, z_layer,
                    color='black', linewidth=0.5,
                    linestyle=':', zorder=4)

    # ── Sea level (blue horizontal line) ────────────────────────────────────
    ax.axhline(sl_now, color='blue', linewidth=1.0, zorder=5,
               label=f'Sea level = {sl_now:.0f} m')

    # ── Template basin outline (black line) ─────────────────────────────────
    ax.plot(x_value, template_z_grid,
            color='black', linewidth=0.8, zorder=6,
            label='Template basin')

    # ── Axes and labels ──────────────────────────────────────────────────────
    ax.set_xlim(0, L)
    ax.set_ylim(y_min_display, y_max_display)
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('z (m)', fontsize=10)
    ax.set_title(f't = {t_now:.0f} Ma', fontsize=11)

    # ── Colourbar ────────────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=TOPO_CMAP,
                        norm=mcolors.Normalize(vmin=0, vmax=upper_limit))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label(cbar_label, fontsize=9)

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    fname = os.path.join(plots_dir, f'plot_t{int(t_now):04d}_{i+1:04d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if (i + 1) % 20 == 0 or i == 0:
        print(f"  Saved step {i+1:3d}/{k}  →  {os.path.basename(fname)}")

print(f"\nAll {k} figures saved to: {plots_dir}")

# ═════════════════════════════════════════════════════════════════════════════
# 6.  Save numerical outputs (same as before)
# ═════════════════════════════════════════════════════════════════════════════
# Fluvial slope series
slope_df = pd.DataFrame({
    'time_Ma':         time_vec,
    'slope_fluvial':   results['slope_fluvial'],
    'slope_angle_deg': np.degrees(np.arctan(results['slope_fluvial'])),
})
slope_df.to_csv(os.path.join(out_dir, 'slope_series.csv'), index=False)
print(f"Slope series saved ({len(slope_df)} rows)")

# Final surface elevation
z_final = z_mat[:, -1]
z_df = pd.DataFrame({'x_m': x_value, 'z_m': z_final})
z_df.to_csv(os.path.join(out_dir, 'z_final.csv'), index=False)
print(f"Final z profile saved ({len(z_df)} rows)")

# z_matrix sampled every 10 steps
step_indices = list(range(0, z_mat.shape[1], 10))
z_sample = z_mat[:, step_indices]
cols = [f"step_{i}" for i in step_indices]
z_sample_df = pd.DataFrame(z_sample, columns=cols)
z_sample_df.insert(0, 'x_m', x_value)
z_sample_df.to_csv(os.path.join(out_dir, 'z_matrix_sample.csv'), index=False)
print(f"z_matrix sample saved ({len(step_indices)} time slices)")

# ═════════════════════════════════════════════════════════════════════════════
# 7.  Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Steps completed:        {k}")
print(f"  Final time:             {time_vec[-1]:.1f} Ma")
print(f"  Final mean slope:       {results['slope_fluvial'][-1]:.5f}")
print(f"  Final slope angle:      {np.degrees(np.arctan(results['slope_fluvial'][-1])):.4f}°")
print(f"  Max z (final surface):  {z_final.max():.2f} m")
print(f"  Min z (final surface):  {z_final.min():.2f} m")
print(f"  Max shoreline x:        {results['shoreline_x'].max():.1f} m")
print(f"  Total figures saved:    {k}")
print(f"  Figures directory:      {plots_dir}")
print("=" * 60)
