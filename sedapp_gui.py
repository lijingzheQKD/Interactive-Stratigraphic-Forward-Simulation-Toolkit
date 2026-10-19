"""
sedapp_gui.py — Sedapp Teaching Studio: interactive GUI for the Sedapp
forward stratigraphic model (prototype accompanying the Applied Sciences
manuscript).

Run interactively (requires a display backend, e.g. TkAgg/Qt5Agg):

    python3 sedapp_gui.py

Headless rendering of the initial interface state (Agg backend) is used to
produce Figure 2 of the manuscript; see make_figures.py::fig02().

The GUI offers:
  * sliders for sea-level amplitude/period, sand & mud supply, transport
    scale (wdscale) and maximum differential subsidence;
  * a preset case library (radio buttons): marine ramp with a eustatic
    cycle, progradation laboratory, GMD petroliferous-basin example;
  * Run / Reset buttons;
  * a time slider to scrub through the depositional history of the
    computed run.

Reproducibility: the underlying engine (sedapp_py.run_sedapp) is fully
deterministic; no random numbers are used anywhere in this GUI.
"""

import os, sys, io, contextlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sedapp_py as sp
from make_figures import (marine_ramp_params, gmd_example_params,
                          draw_section, SAND_CMAP, MASTER_SEED)


class SedappTeachingStudio:
    """Interactive teaching interface around sedapp_py.run_sedapp."""

    def __init__(self):
        np.random.seed(MASTER_SEED)
        self.fig = plt.figure(figsize=(11.5, 6.6))
        self.fig.patch.set_facecolor('#f5f5f3')
        self.ax_main = self.fig.add_axes([0.34, 0.28, 0.60, 0.64])
        self.ax_main.set_title(
            'Sedapp Teaching Studio — interactive forward stratigraphic simulation')

        # ── parameter sliders ────────────────────────────────────────────
        specs = [('SL amplitude (m)', 0, 60, 25.0),
                 ('SL period (Ma)', 0.5, 6.0, 2.0),
                 ('Sand supply (m/Ma)', 0, 40, 4.0),
                 ('Mud supply (m/Ma)', 2, 80, 12.0),
                 ('Transport scale (1/m)', 0.02, 0.3, 0.10),
                 ('Max subsidence (m)', 0, 300, 0.0)]
        self.sliders = []
        for i, (lab, lo, hi, v0) in enumerate(specs):
            ax_s = self.fig.add_axes([0.055, 0.84 - i * 0.075, 0.20, 0.025])
            self.sliders.append(Slider(ax_s, lab, lo, hi, valinit=v0))

        # ── preset radio ─────────────────────────────────────────────────
        ax_radio = self.fig.add_axes([0.055, 0.20, 0.21, 0.18],
                                     facecolor='#f0f0ee')
        self.radio = RadioButtons(ax_radio, ('Marine ramp (SL cycle)',
                                             'Progradation lab',
                                             'GMD petroliferous basin'),
                                  active=0)

        # ── buttons ──────────────────────────────────────────────────────
        ax_run = self.fig.add_axes([0.055, 0.42, 0.09, 0.05])
        ax_rst = self.fig.add_axes([0.165, 0.42, 0.09, 0.05])
        self.btn_run = Button(ax_run, 'Run')
        self.btn_rst = Button(ax_rst, 'Reset')
        self.btn_run.on_clicked(self.on_run)
        self.btn_rst.on_clicked(self.on_reset)

        # ── time scrub slider ────────────────────────────────────────────
        ax_t = self.fig.add_axes([0.34, 0.10, 0.58, 0.03])
        self.time_slider = Slider(ax_t, 'Time step (Ma)', 0.0, 2.0, valinit=2.0)
        self.time_slider.on_changed(self.on_time)

        self.res = None
        self.params = None
        self.on_run(None)

    # ── callbacks ─────────────────────────────────────────────────────────
    def build_params(self):
        amp, per = self.sliders[0].val, self.sliders[1].val
        es, em = self.sliders[2].val, self.sliders[3].val
        ws, sub = self.sliders[4].val, self.sliders[5].val
        case = self.radio.value_selected
        if case == 'GMD petroliferous basin':
            return gmd_example_params()
        if case == 'Progradation lab':
            return marine_ramp_params(lambda t: np.zeros_like(t), t1=0.5,
                                      dt=0.005, exp_sand=es, exp_mud=em,
                                      wdscale_sand=ws, subsidence_max=sub)
        return marine_ramp_params(lambda t: amp * np.sin(2 * np.pi * t / per),
                                  t1=per, dt=per / 100.0, exp_sand=es,
                                  exp_mud=em, wdscale_sand=ws,
                                  subsidence_max=sub)

    def on_run(self, _event):
        self.params = self.build_params()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.res = sp.run_sedapp(self.params)
        k = self.res['z_matrix'].shape[1]
        self.time_slider.valmin = self.res['time_series'][0]
        self.time_slider.valmax = self.res['time_series'][-1]
        self.time_slider.ax.set_xlim(self.time_slider.valmin, self.time_slider.valmax)
        self.time_slider.set_val(self.res['time_series'][-1])
        self.redraw(k)

    def on_reset(self, _event):
        for s in self.sliders:
            s.reset()
        self.on_run(None)

    def on_time(self, val):
        step = int(np.argmin(np.abs(self.res['time_series'] - val))) + 1
        self.redraw(step)

    def redraw(self, step):
        step = max(1, min(step, self.res['z_matrix'].shape[1]))
        self.ax_main.clear()
        sub = dict(self.res)
        sub['z_matrix'] = self.res['z_matrix'][:, :step]
        pc = draw_section(self.ax_main, sub,
                          self.res['sand_fraction_matrix'][:, :step],
                          SAND_CMAP, 0, 1,
                          sl_last=self.params['sl_series'][step - 1], ny=300)
        self.ax_main.set_title(
            f't = {self.res["time_series"][step - 1]:.2f} Ma '
            f'(step {step}/{self.res["z_matrix"].shape[1]})', fontsize=9)
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    app = SedappTeachingStudio()
    plt.show()
