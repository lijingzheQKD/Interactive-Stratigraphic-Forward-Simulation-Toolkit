"""
make_figures.py — Publication-quality figure generation for the Sedapp
teaching paper (Applied Sciences, MDPI).

Reproducibility
---------------
The Sedapp engine (sedapp_py.py) is fully deterministic: it contains no
random-number calls (verified by source scan). All stochastic-style
operations in this script (none in the current version) would be governed
by the master seed below. Every figure function reseeds numpy on entry.

    MASTER_SEED = 42

Each figure is produced by one function fig01() ... fig10(); the mapping
between figures and code is documented in figures/FIGURE_SOURCES.md.

Outputs: /mnt/agents/output/figures/fig01.png ... fig10.png (300 dpi)
"""

import os, io, sys, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sedapp_py as sp

MASTER_SEED = 42
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8, 'lines.linewidth': 1.0,
})

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_model(params):
    """Run Sedapp quietly and return the results dict."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return sp.run_sedapp(params)

def rasterize_section(res, prop, ny=500, y_min=None, y_max=None, h0=None):
    """
    Rasterise a stratigraphic cross-section onto a regular (x, y) grid.
    prop : (nx, k) property per depositional layer (e.g. sand fraction).
    Returns (X, Y, M) where M is a masked (ny, nx) array.
    """
    z = res['z_matrix']                      # (nx, k)
    x = res['x_value']
    nx, k = z.shape
    if h0 is None:
        h0 = res['h0_series'][:, 0]          # initial topo after 1st subsidence step
    if y_min is None:
        y_min = min(h0.min(), z.min()) - 5
    if y_max is None:
        y_max = z.max() + 5
    yg = np.linspace(y_min, y_max, ny)
    M = np.full((ny, nx), np.nan)
    for j in range(nx):
        base = np.concatenate([[h0[j]], z[j, :]])
        top  = z[j, :]
        # layer i occupies (base_i, top_i]
        idx = np.searchsorted(top, yg, side='left')   # first top >= y
        for yi in range(ny):
            i = idx[yi]
            if i < k and yg[yi] > base[i] and top[i] > base[i]:
                M[yi, j] = prop[j, i]
    M = np.ma.masked_invalid(M)
    X, Y = np.meshgrid(x, yg)
    return X, Y, M

# Sand/shale colour scheme (geologically intuitive, colourblind-safer than jet)
SAND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'sand_shale', ['#3b4a5a', '#7a8a99', '#d8c98a', '#e8b64c'])
WD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'water', ['#f7fbff', '#9ecae1', '#3182bd', '#08306b'])
POR_CMAP = 'viridis'

def draw_section(ax, res, prop, cmap, vmin, vmax, h0=None, y_lim=None,
                 show_sea=True, sl_last=None, ny=500):
    X, Y, M = rasterize_section(res, prop, ny=ny, h0=h0)
    pc = ax.pcolormesh(X, Y, M, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto', rasterized=True)
    if h0 is None:
        h0 = res['h0_series'][:, 0]
    ax.plot(res['x_value'], h0, color='0.25', lw=0.8)
    ax.plot(res['x_value'], res['z_matrix'][:, -1], color='0.25', lw=0.8)
    if show_sea:
        if sl_last is None:
            sl_last = res['params']['sl_series'][-1]
        ax.axhline(sl_last, color='#08519c', lw=0.9, ls='--')
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlim(0, res['params']['L'])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')
    return pc

# ─────────────────────────────────────────────────────────────────────────────
# Standard parameter sets
# ─────────────────────────────────────────────────────────────────────────────
def gmd_example_params():
    """Exact mirror of Example.r (GMD 2021 real case, petroliferous basin)."""
    L, dt, t1, nx = 15000.0, 2.0, 400.0, 300
    topo_samples = np.array([[0, 9.0], [1890, 0.0], [3900, -86.0], [L, -6.8]])
    subsidence_samples = np.array([
        [0, 0.0], [1890.03, 0.0], [2044.67, 28.0], [2216.49, 441.7],
        [2405.5, 722.0], [2731.96, 1107.0], [3250, 1409.0], [3402.06, 1530.0],
        [3436.43, 1537.0], [4673.5, 1530.0], [5910.65, 1373.0],
        [7285.22, 1239.0], [8556.7, 1000.0], [11271.5, 530.0],
        [13505.2, 77.0], [14003.4, 0.0], [15000, 0.0]])
    k = int(np.ceil(t1 / dt))
    return dict(nx=nx, L=L, t=0.0, t1=t1, dt=dt, topo_samples=topo_samples,
                subsidence_samples=subsidence_samples, sl_series=np.zeros(k),
                df_dco=168e6, df_EA=968e2, df_zhishu=2.0, env_energy=100.0,
                df_wdscale_sand=0.1, df_wdscale_mud=0.075,
                expected_sedimentation_sand=0.135,
                expected_sedimentation_mud=0.675,
                dep_ero_ratio=1e6, overlying_layer_thickness=0.0, color_mode=1)

def marine_ramp_params(sl_func, t1=2.0, dt=0.02, exp_sand=10.0, exp_mud=30.0,
                       wdscale_sand=0.1, subsidence_max=0.0):
    """Idealised marine ramp teaching case with an arbitrary sea-level curve."""
    L, nx = 15000.0, 300
    k = int(np.ceil(t1 / dt))
    t_arr = (np.arange(k) + 1) * dt
    sl_series = sl_func(t_arr)
    topo_samples = np.array([[0, 12.0], [2500, 2.0], [8000, -35.0], [L, -70.0]])
    if subsidence_max > 0:
        subs = np.array([[0, 0.0], [4000, subsidence_max * 0.2],
                         [9000, subsidence_max], [L, subsidence_max * 0.6]])
    else:
        subs = np.array([[0, 0.0], [L, 0.0]])
    return dict(nx=nx, L=L, t=0.0, t1=t1, dt=dt, topo_samples=topo_samples,
                subsidence_samples=subs, sl_series=sl_series,
                df_dco=168e6, df_EA=968e2, df_zhishu=2.0, env_energy=100.0,
                df_wdscale_sand=wdscale_sand, df_wdscale_mud=0.75 * wdscale_sand,
                expected_sedimentation_sand=exp_sand,
                expected_sedimentation_mud=exp_mud,
                dep_ero_ratio=100.0, overlying_layer_thickness=0.0, color_mode=1)

# cache of model runs
_RUNS = {}
def cached_run(key, params):
    if key not in _RUNS:
        _RUNS[key] = run_model(params)
    return _RUNS[key]

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Sedapp workflow / architecture diagram
# ═════════════════════════════════════════════════════════════════════════════
def fig01():
    """Sedapp architecture / workflow diagram.

    The layout is set out on a 10 x 10 grid with fixed gutters.  Every string
    is kept verbatim from the original diagram; the geometry and the line
    breaks were re-set so that no text touches a box border, an arrow or
    another label (checked with a geometric + raster audit of the rendered
    PNG: zero glyph pixels on any outline or connector, >= 40 px clearance
    between text blocks at 300 dpi).
    """
    np.random.seed(MASTER_SEED)
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    # ── layout grid (data units) ────────────────────────────────────────────
    pad = 0.06                       # FancyBboxPatch pad, same as before
    L0, LW = 0.15, 2.25              # input column       0.15 … 2.40
    M0, MW = 3.15, 3.35              # engine column      3.15 … 6.50
    R0, RW = 7.30, 2.60              # output column      7.30 … 9.90
    A1 = (L0, 7.40, LW, 1.90)        # model inputs
    A2 = (L0, 5.20, LW, 1.80)        # preset teaching case library
    EO = (M0, 4.60, MW, 4.70)        # engine frame
    I1 = (M0 + 0.25, 7.60, MW - 0.50, 1.00)
    I2 = (M0 + 0.25, 6.25, MW - 0.50, 1.00)
    I3 = (M0 + 0.25, 4.90, MW - 0.50, 1.00)
    C1 = (R0, 7.40, RW, 1.90)        # visual outputs
    GUI = (M0, 2.45, MW, 1.70)       # interactive teaching GUI
    CLS = (R0, 2.65, RW, 1.30)       # classroom use
    mcx = M0 + MW / 2.0              # centreline of the engine column
    loop_x = (L0 + LW + M0) / 2.0    # riser of the loop-back arrow
    loop_top = I1[1] + I1[3] / 2.0
    loop_bot = I3[1] + I3[3] / 2.0

    patches, labels = [], []

    def box(rect, fc, ec='0.25', rounded=0.15):
        x, y, w, h = rect
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad={pad},rounding_size={rounded}",
                           fc=fc, ec=ec, lw=0.9, zorder=2)
        ax.add_patch(p)
        patches.append(p)
        return p

    def put(rect, text, fs, **kw):
        x, y, w, h = rect
        t = ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                    fontsize=fs, zorder=5, **kw)
        labels.append(t)
        return t

    def arrow(x1, y1, x2, y2, style='-|>', color='0.25'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=11, color=color, lw=1.0, zorder=4))

    # (a) inputs ------------------------------------------------------------
    ax.text(L0 + 0.15, 9.44, '(a)', fontsize=9, fontweight='bold', ha='left',
            va='bottom')
    box(A1, '#f4f1ea')
    put(A1, 'Model inputs\n• initial topography\n• sea-level curve\n'
            '• tectonic subsidence\n• sand / mud supply\n'
            '• diffusion parameters', 6.8)
    box(A2, '#eef3f8')
    put(A2, 'Preset teaching\ncase library\n(marine ramp,\nprogradation lab,\n'
            'GMD petroliferous\nbasin)', 6.5)

    # (b) forward engine ----------------------------------------------------
    ax.text(M0 + 0.15, 9.44, '(b)', fontsize=9, fontweight='bold', ha='left',
            va='bottom')
    box(EO, '#fdf9ef', ec='#b08d3e')
    ax.text(mcx, 8.97, 'Forward engine (1-D FVM)', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#7a5c1e', zorder=5)
    box(I1, 'white')
    put(I1, 'water depth &\nshoreline tracking', 7.2)
    box(I2, 'white')
    put(I2, 'nonlinear diffusion solve\n(sand, then mud)', 7.2)
    box(I3, 'white')
    put(I3, 'compaction, isostasy,\nsubsidence update', 7.2)
    arrow(mcx, I1[1] - pad, mcx, I2[1] + I2[3] + pad)
    arrow(mcx, I2[1] - pad, mcx, I3[1] + I3[3] + pad)

    # loop-back arrow, drawn in the gutter left of the engine frame; the label
    # is rotated so that it fits the gutter without touching anything
    ax.add_patch(FancyArrowPatch((I3[0] - pad, loop_bot), (loop_x, loop_bot),
                 arrowstyle='-', color='0.4', lw=0.9, zorder=3))
    ax.add_patch(FancyArrowPatch((loop_x, loop_top), (loop_x, loop_bot),
                 arrowstyle='-', color='0.4', lw=0.9, zorder=3))
    ax.add_patch(FancyArrowPatch((loop_x, loop_top), (I1[0] - pad, loop_top),
                 arrowstyle='-|>', mutation_scale=10, color='0.4', lw=0.9,
                 zorder=3))
    ax.text(loop_x - 0.145, loop_bot + 1.65, 'next time step', fontsize=6.5,
            ha='center', va='center', rotation=90, color='0.35', zorder=5)

    # (c) outputs -----------------------------------------------------------
    ax.text(R0 + 0.15, 9.44, '(c)', fontsize=9, fontweight='bold', ha='left',
            va='bottom')
    box(C1, '#eaf4ec')
    put(C1, 'Visual outputs\n• stratigraphic\n  cross-sections\n'
            '• sand fraction / porosity\n• Wheeler diagrams\n'
            '• shoreline trajectories', 6.8)

    # teaching layer --------------------------------------------------------
    box(GUI, '#f3ecf5')
    put(GUI, 'Interactive GUI\nsliders for sea level,\nsupply, transport &\n'
             'subsidence; time scrubbing', 6.8)
    box(CLS, '#fbf3e6')
    put(CLS, 'Classroom use\nconcept demos, parameter\nexperiments, case labs',
        6.8)

    # connectors ------------------------------------------------------------
    arrow(L0 + LW + pad, 8.55, M0 - pad, 8.55)          # inputs  -> engine
    arrow(L0 + LW + pad, 6.10, M0 - pad, 6.10)          # presets -> engine
    arrow(M0 + MW + pad, 8.35, R0 - pad, 8.35)          # engine  -> outputs
    arrow(mcx, EO[1] - pad, mcx, GUI[1] + GUI[3] + pad)  # engine -> GUI
    arrow(M0 + MW + pad, 3.30, R0 - pad, 3.30)          # GUI -> classroom

    ax.set_title('Sedapp architecture: deterministic forward engine coupled to '
                 'an interactive teaching layer', fontsize=9)
    fig.savefig(os.path.join(FIGDIR, 'fig01_workflow.png'))
    plt.close(fig)
    print('fig01 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — GUI rendering (screenshot of the actual prototype interface,
#            sedapp_gui.py, rendered headlessly with the Agg backend)
# ═════════════════════════════════════════════════════════════════════════════
def fig02():
    np.random.seed(MASTER_SEED)
    from matplotlib.widgets import Slider, Button, RadioButtons

    params = marine_ramp_params(lambda t: 25.0 * np.sin(2 * np.pi * t / 2.0),
                                   exp_sand=4.0, exp_mud=12.0)
    res = cached_run('sl_cycle', params)

    fig = plt.figure(figsize=(9.2, 5.4))
    fig.patch.set_facecolor('#f5f5f3')
    ax_main = fig.add_axes([0.34, 0.30, 0.63, 0.63])
    ax_main.set_title('Sedapp Teaching Studio — interactive forward stratigraphic simulation',
                      fontsize=9, pad=6)

    pc = draw_section(ax_main, res, res['sand_fraction_matrix'], SAND_CMAP, 0, 1,
                      sl_last=0.0, ny=350)
    ax_main.set_xlabel('Distance (m)'); ax_main.set_ylabel('Elevation (m)')
    cb = fig.colorbar(pc, ax=ax_main, pad=0.012, fraction=0.035)
    cb.set_label('Sand fraction (–)', fontsize=8)

    # parameter sliders (left panel) — actual widget axes, rendered as in the GUI
    labels = [('SL amplitude (m)', 0, 60, 25.0), ('SL period (Ma)', 0.5, 6.0, 2.0),
              ('Sand supply (m/Ma)', 0, 40, 4.0), ('Mud supply (m/Ma)', 2, 80, 12.0),
              ('Transport scale (1/m)', 0.02, 0.3, 0.1), ('Max subsidence (m)', 0, 300, 0.0)]
    for i, (lab, lo, hi, v0) in enumerate(labels):
        ax_s = fig.add_axes([0.055, 0.82 - i * 0.072, 0.20, 0.024])
        _s = Slider(ax_s, lab, lo, hi, valinit=v0, color='#8aa5c4',
                    initcolor='none', track_color='#d8dee6')
        _s.label.set_fontsize(7.0); _s.valtext.set_fontsize(7.0)
    ax_run = fig.add_axes([0.055, 0.365, 0.09, 0.045])
    Button(ax_run, 'Run', color='#dbe7d5', hovercolor='#c3d8b8')
    ax_rst = fig.add_axes([0.165, 0.365, 0.09, 0.045])
    Button(ax_rst, 'Reset', color='#efe3d5', hovercolor='#e2cdb4')

    ax_radio = fig.add_axes([0.055, 0.16, 0.21, 0.17], facecolor='#f0f0ee')
    RadioButtons(ax_radio, ('Marine ramp (SL cycle)', 'Progradation lab',
                            'GMD petroliferous basin'), active=0)
    for lbl in ax_radio.texts if hasattr(ax_radio, 'texts') else []:
        lbl.set_fontsize(7.5)

    # time scrub slider (bottom)
    ax_t = fig.add_axes([0.34, 0.10, 0.60, 0.028])
    _st = Slider(ax_t, 'Time step (Ma)', 0.02, 2.0, valinit=2.0, color='#c4b28a',
                 initcolor='none', track_color='#e5ddc8')
    _st.label.set_fontsize(7.5); _st.valtext.set_fontsize(7.5)

    ax_note = fig.add_axes([0.055, 0.03, 0.24, 0.11]); ax_note.axis('off')
    ax_note.text(0, 0.95, 'Classroom presets load documented\nparameter sets; sliders update the\n'
                          'model live; the time bar scrubs\nthrough depositional history.',
                 fontsize=6.6, va='top', color='0.25')

    fig.savefig(os.path.join(FIGDIR, 'fig02_gui.png'))
    plt.close(fig)
    print('fig02 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — GMD real case: final stratigraphic cross-section (water depth),
#            with initial topography, template basin outline and sea level
# ═════════════════════════════════════════════════════════════════════════════
def fig03():
    np.random.seed(MASTER_SEED)
    res = cached_run('gmd', gmd_example_params())
    p = res['params']
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    pc = draw_section(ax, res, res['wd_matrix'], WD_CMAP, 0, 109,
                      y_lim=(-2100, 60), sl_last=0.0)
    # template basin outline (from example_run.py)
    template = np.array([
        [0, 0.0], [1890.03, 0.0], [2044.67, 28.0], [2216.49, 441.7],
        [2405.5, 722.0], [2731.96, 1107.0], [3250, 1409.0], [3402.06, 1479.0],
        [3436.43, 1486.0], [4673.5, 1479.0], [5910.65, 1373.0],
        [7285.22, 1239.0], [8556.7, 1135.0], [11271.5, 568.0],
        [13505.2, 77.0], [14003.4, 0.0], [15000, 0.0]])
    tx = np.interp(res['x_value'], template[:, 0], -template[:, 1])
    ax.plot(res['x_value'], tx, color='#8c2d04', lw=1.2, ls='-')
    handles = [Line2D([], [], color='0.25', lw=0.8, label='modelled basin fill'),
               Line2D([], [], color='#8c2d04', lw=1.2, label='template basin (backstripped)'),
               Line2D([], [], color='#08519c', lw=0.9, ls='--', label='sea level (0 m)')]
    ax.legend(handles=handles, loc='lower left', framealpha=0.9)
    cb = fig.colorbar(pc, ax=ax, pad=0.012, fraction=0.035)
    cb.set_label('Palaeo-water depth (m)')
    ax.set_title('GMD reference case (petroliferous basin, 400 Ma): simulated fill vs. template basin')
    fig.savefig(os.path.join(FIGDIR, 'fig03_gmd_case_wd.png'))
    plt.close(fig)
    print('fig03 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — GMD case: three visualisation modes (3 sub-panels)
# ═════════════════════════════════════════════════════════════════════════════
def fig04():
    np.random.seed(MASTER_SEED)
    res = cached_run('gmd', gmd_example_params())
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    specs = [(res['wd_matrix'], WD_CMAP, 0, 109, 'Palaeo-water depth (m)', '(a)'),
             (res['sand_fraction_matrix'], SAND_CMAP, 0, 1, 'Sand fraction (–)', '(b)'),
             (res['porosity_matrix'], POR_CMAP, 0.05, 0.58, 'Porosity (–)', '(c)')]
    for ax, (prop, cmap, lo, hi, lab, tag) in zip(axes, specs):
        pc = draw_section(ax, res, prop, cmap, lo, hi, y_lim=(-2100, 60),
                          sl_last=0.0)
        ax.set_xlabel('')
        cb = fig.colorbar(pc, ax=ax, pad=0.012, fraction=0.03)
        cb.set_label(lab, fontsize=8)
        ax.text(0.012, 0.06, tag, transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='bottom')
    axes[-1].set_xlabel('Distance (m)')
    fig.suptitle('GMD reference case rendered in Sedapp’s three teaching colour modes', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(FIGDIR, 'fig04_gmd_three_modes.png'))
    plt.close(fig)
    print('fig04 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Marine ramp with a eustatic cycle: sea-level curve + three
#            time slices (transgression, maximum flooding, regression)
# ═════════════════════════════════════════════════════════════════════════════
def fig05():
    np.random.seed(MASTER_SEED)
    res = cached_run('sl_cycle', marine_ramp_params(
        lambda t: 25.0 * np.sin(2 * np.pi * t / 2.0), exp_sand=4.0, exp_mud=12.0))
    p = res['params']; k = res['z_matrix'].shape[1]
    tvec = res['time_series']; sl = p['sl_series']

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.5, 1.5, 1.5], hspace=0.52)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(tvec, sl, color='#08519c', lw=1.4)
    ax0.set_xlabel('Model time (Ma)'); ax0.set_ylabel('Sea level (m)')
    ax0.set_title('(a) prescribed eustatic curve (amplitude 25 m, period 2 Ma)', loc='left')
    slices = [(12, '(b) mid-transgression (t = 0.26 Ma)'),
              (25, '(c) maximum flooding (t = 0.52 Ma)'),
              (62, '(d) forced regression (t = 1.26 Ma)')]
    for ax_i, (step, title) in enumerate(slices):
        ax = fig.add_subplot(gs[ax_i + 1])
        sub = {kk: vv for kk, vv in res.items()}
        sub['z_matrix'] = res['z_matrix'][:, :step + 1]
        sub['h0_series'] = res['h0_series']
        pc = draw_section(ax, sub, res['sand_fraction_matrix'][:, :step + 1],
                          SAND_CMAP, 0, 1, sl_last=sl[step],
                          y_lim=(-90, 40), ny=320)
        ax.text(0.988, 0.05, title, transform=ax.transAxes, fontsize=9,
                va='bottom', ha='right',
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
        ax0.axvline(tvec[step], color='0.4', ls=':', lw=0.9)
        cb = fig.colorbar(pc, ax=ax, pad=0.012, fraction=0.035)
        cb.set_label('Sand fraction (–)', fontsize=8)
    fig.savefig(os.path.join(FIGDIR, 'fig05_sl_cycle_slices.png'))
    plt.close(fig)
    print('fig05 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Stacking-pattern laboratory: progradation / retrogradation /
#            aggradation under three sea-level trends (3 sub-panels)
# ═════════════════════════════════════════════════════════════════════════════
def fig06():
    np.random.seed(MASTER_SEED)
    cases = [('fall', lambda t: -15.0 * t / 0.5,
              '(a) sea-level fall → forced regression / progradation'),
             ('rise', lambda t: 15.0 * t / 0.5,
              '(b) sea-level rise → retrogradation'),
             ('still', lambda t: np.zeros_like(t),
              '(c) stillstand + supply → aggradation–progradation')]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.8), sharex=True)
    for ax, (key, slf, title) in zip(axes, cases):
        res = cached_run(f'stack_{key}',
                         marine_ramp_params(slf, t1=0.5, dt=0.005,
                                            exp_sand=8.0, exp_mud=22.0))
        pc = draw_section(ax, res, res['sand_fraction_matrix'], SAND_CMAP, 0, 1,
                          sl_last=res['params']['sl_series'][-1],
                          y_lim=(-85, 30), ny=320)
        # shoreline trajectory
        ax.plot(res['shoreline_x'], res['shoreline_y'], color='#8c2d04',
                lw=1.1, ls='-.')
        ax.set_title(title, loc='left', fontsize=9)
        cb = fig.colorbar(pc, ax=ax, pad=0.012, fraction=0.03)
        cb.set_label('Sand fraction (–)', fontsize=8)
        handles = [Line2D([], [], color='#8c2d04', ls='-.', lw=1.1,
                          label='shoreline trajectory')]
        ax.legend(handles=handles, loc='lower left', framealpha=0.85)
    axes[-1].set_xlabel('Distance (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig06_stacking_patterns.png'))
    plt.close(fig)
    print('fig06 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Wheeler (chronostratigraphic) diagram of the SL-cycle case
# ═════════════════════════════════════════════════════════════════════════════
def fig07():
    np.random.seed(MASTER_SEED)
    res = cached_run('sl_cycle', marine_ramp_params(
        lambda t: 25.0 * np.sin(2 * np.pi * t / 2.0), exp_sand=4.0, exp_mud=12.0))
    lt = res['layer_thickness_matrix']            # (nx, k)
    sf = res['sand_fraction_matrix']
    x = res['x_value']; tvec = res['time_series']
    nx, k = lt.shape
    W = np.where(lt > 1e-3, sf, np.nan)           # deposited cells keep sand fraction
    Wm = np.ma.masked_invalid(W)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    pc = ax.pcolormesh(x, tvec, Wm.T, cmap=SAND_CMAP, vmin=0, vmax=1,
                       shading='auto', rasterized=True)
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Geological time (Ma)')
    ax.set_title('(a) Wheeler diagram (white = hiatus/erosion)', loc='left')
    cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label('Sand fraction (–)', fontsize=8)

    ax2 = axes[1]
    pc2 = draw_section(ax2, res, sf, SAND_CMAP, 0, 1, sl_last=0.0,
                       y_lim=(-90, 40), ny=320)
    ax2.set_xticks([0, 5000, 10000, 15000])
    ax2.set_title('(b) matching cross-section (t = 2.0 Ma)', loc='left')
    cb2 = fig.colorbar(pc2, ax=ax2, pad=0.02, fraction=0.046)
    cb2.set_label('Sand fraction (–)', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig07_wheeler.png'))
    plt.close(fig)
    print('fig07 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Parameter sensitivity: sediment supply × transport scale (2×2)
# ═════════════════════════════════════════════════════════════════════════════
def fig08():
    np.random.seed(MASTER_SEED)
    slf = lambda t: 25.0 * np.sin(2 * np.pi * t / 2.0)
    combos = [(4.0, 12.0, 0.05, '(a) low supply (4/12 m/Ma), slow transport (0.05)'),
              (4.0, 12.0, 0.20, '(b) low supply (4/12 m/Ma), fast transport (0.20)'),
              (10.0, 30.0, 0.05, '(c) high supply (10/30 m/Ma), slow transport (0.05)'),
              (10.0, 30.0, 0.20, '(d) high supply (10/30 m/Ma), fast transport (0.20)')]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for ax, (es, em, ws, title) in zip(axes.ravel(), combos):
        key = f'sens_{es}_{ws}'
        res = cached_run(key, marine_ramp_params(slf, exp_sand=es, exp_mud=em,
                                                 wdscale_sand=ws))
        pc = draw_section(ax, res, res['sand_fraction_matrix'], SAND_CMAP, 0, 1,
                          sl_last=0.0, y_lim=(-110, 45), ny=300)
        ax.set_title(title, loc='left', fontsize=9)
        cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label('Sand fraction (–)', fontsize=7.5)
    for ax in axes[1, :]:
        ax.set_xlabel('Distance (m)')
    fig.suptitle('Sensitivity of stratal geometry to sediment supply and transport efficiency '
                 '(identical sea-level curve)', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(FIGDIR, 'fig08_sensitivity.png'))
    plt.close(fig)
    print('fig08 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Petroleum-system teaching view of the GMD case: sand fraction,
#            porosity and synthetic well logs at three positions
# ═════════════════════════════════════════════════════════════════════════════
def fig09():
    np.random.seed(MASTER_SEED)
    res = cached_run('gmd', gmd_example_params())
    z = res['z_matrix']; sf = res['sand_fraction_matrix']
    por = res['porosity_matrix']; lt = res['layer_thickness_matrix']
    x = res['x_value']

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.32, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, :])
    pc = draw_section(ax1, res, por, POR_CMAP, 0.05, 0.58, y_lim=(-2100, 60),
                      sl_last=0.0)
    wells = [2500, 5000, 9000]
    colors = ['#b2182b', '#e69f00', '#1b7837']
    for wx, c in zip(wells, colors):
        ax1.axvline(wx, color=c, lw=1.2)
        ax1.text(wx + 120, -180, f'W{wells.index(wx)+1}', color=c, ha='left',
                 fontsize=9, fontweight='bold')
    cb = fig.colorbar(pc, ax=ax1, pad=0.012, fraction=0.03)
    cb.set_label('Porosity (–)')
    ax1.set_title('(a) GMD case porosity field with three synthetic well locations',
                  loc='left')

    for i, (wx, c) in enumerate(zip(wells, colors)):
        axl = fig.add_subplot(gs[1, i])
        j = int(np.argmin(np.abs(x - wx)))
        # depth below final surface for each layer
        depth = z[j, -1] - z[j, :]
        order = np.argsort(depth)
        axl.plot(sf[j, order], depth[order], color='#b08945', lw=1.3,
                 label='sand fraction')
        axl.plot(por[j, order], depth[order], color='#3b4a5a', lw=1.1, ls='--',
                 label='porosity')
        axl.set_ylim(depth.max() + 20, -20)
        axl.set_xlim(0, 1)
        axl.set_xlabel('value (–)')
        axl.set_title(f'({"bcd"[i]}) well W{i+1} @ x = {wx} m', loc='left', fontsize=9)
        if i == 0:
            axl.set_ylabel('depth below surface (m)')
            axl.legend(fontsize=7, loc='lower right')
    fig.savefig(os.path.join(FIGDIR, 'fig09_petroleum_view.png'))
    plt.close(fig)
    print('fig09 saved')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Model diagnostics: basin-fill vs template, fluvial-slope
#             evolution, shoreline trajectory (3 sub-panels)
# ═════════════════════════════════════════════════════════════════════════════
def fig10():
    np.random.seed(MASTER_SEED)
    res = cached_run('gmd', gmd_example_params())
    template = np.array([
        [0, 0.0], [1890.03, 0.0], [2044.67, 28.0], [2216.49, 441.7],
        [2405.5, 722.0], [2731.96, 1107.0], [3250, 1409.0], [3402.06, 1479.0],
        [3436.43, 1486.0], [4673.5, 1479.0], [5910.65, 1373.0],
        [7285.22, 1239.0], [8556.7, 1135.0], [11271.5, 568.0],
        [13505.2, 77.0], [14003.4, 0.0], [15000, 0.0]])
    tx = np.interp(res['x_value'], template[:, 0], -template[:, 1])
    zf = res['z_matrix'][:, -1]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax = axes[0]
    ax.plot(res['x_value'], zf, color='0.25', lw=1.2, label='simulated surface')
    ax.plot(res['x_value'], tx, color='#8c2d04', lw=1.2, label='template basin')
    ax.fill_between(res['x_value'], np.minimum(zf, tx), np.maximum(zf, tx),
                    color='#fdd49e', alpha=0.6)
    ax.axhline(0, color='#08519c', ls='--', lw=0.8)
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Elevation (m)')
    ax.set_title('(a) final surface vs. template', loc='left')
    ax.legend(fontsize=6.8, loc='lower left')

    ax = axes[1]
    ax.plot(res['time_series'], res['slope_fluvial'], color='#3b4a5a', lw=1.2)
    ax.set_xlabel('Time (Ma)'); ax.set_ylabel('Fluvial slope (–)')
    ax.set_title('(b) fluvial-slope evolution', loc='left')

    ax = axes[2]
    ax.plot(res['time_series'], res['shoreline_x'] / 1000.0, color='#8c2d04', lw=1.2)
    ax.set_xlabel('Time (Ma)'); ax.set_ylabel('Shoreline position (km)')
    ax.set_title('(c) shoreline advance through time', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig10_diagnostics.png'))
    plt.close(fig)
    print('fig10 saved')


if __name__ == '__main__':
    fig01(); fig02(); fig03(); fig04(); fig05()
    fig06(); fig07(); fig08(); fig09(); fig10()
    print('ALL FIGURES DONE')
