#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Specify encoding so strings can have special characters.
#
# This script, using the RunData & Mya packages, makes plots for the Run Group C experiment with CLAS12 in Hall-B.
# This script get the accumulated charge on the target for a specific period, in order to compute the target illumination.
#
# Author: Maurik Holtrop - UNH - July 2022.
#
from __future__ import print_function
import sys
import os
from datetime import datetime, timedelta
import numpy as np
import scipy
import scipy.integrate as scii

from RunData.RunData import RunData
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import chart_studio.plotly as charts

    pio.renderers.default = "browser"
    pio.templates.default = "plotly_white"

except ImportError:
    print("Sorry, but to make the nice plots, you really need a computer with 'plotly' installed.")
    sys.exit(1)


def setup_time_periods():
    """Set the time periods over which to get the data for plotting."""
    time_periods = [
        # target, start_time,   end_time,  raster diameter x, raster diameter y, color
        #(0, datetime(2022, 7, 8, 2, 0), datetime(2022, 7, 8, 7, 0), 12.5, 12.5),
        #(0, datetime(2022, 7, 9, 10, 0), datetime(2022, 7, 9, 13, 49), 12.5, 12.5)
        ('ND3', datetime(2022, 7, 8, 2, 0), datetime(2022, 7, 11, 14, 0), 12.5, 12.5, 'rgba(255, 60, 80, 0.6)'),
        ('ND3', datetime(2022, 7, 19, 10, 0), datetime.now(), 12.5, 12.5, 'rgba(255, 60, 80, 0.6)')
    ]
    return time_periods


def values_times_steps(values, times):
    """For square integrations, compute the rectangular areas that make the rectangle v[i] from t[i] to t[i+1]
       So v[i]*(t[i+1]-t[i])"""
    if type(times) is pd.core.series.Series:
        times = times.array
    if type(values) is pd.core.series.Series:
        values = values.array
    times_diff = np.diff(times, append=times[-1])
    values_times_diff = values*times_diff
    return values_times_diff


def square_integrate_total(values, times):
    """Instead of trapezoidal integration, do a square integration.
       This method 'stretches' the value v[i] at time t[i] to time t[i+1] and uses v[i]*(t[i+1]-t[i]) as the area."""
    values_times_diff = values_times_steps(values, times)
    return np.sum(values_times_diff)

def square_integrate(values, times):
    """Instead of trapezoidal integration, do a square integration.
       This method 'stretches' the value v[i] at time t[i] to time t[i+1] and uses v[i]*(t[i+1]-t[i]) as the area."""
    values_times_diff = values_times_steps(values, times)
    return np.cumsum(values_times_diff)


def main():
    """Main body of the code."""
    hostname = os.uname()[1]
    if hostname.find('clon') >= 0 or hostname.find('ifarm') >= 0 or hostname.find('jlab.org') >= 0:
        #
        # For JLAB setup the place we can find the RCDB
        #
        at_jlab = True
    else:
        at_jlab = False

    data = RunData(cache_file="", sqlcache=False, i_am_at_jlab=at_jlab)

    time_periods = setup_time_periods()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for time_p in time_periods:
        print(f"Getting data for {time_p[0]} from {time_p[1]} to {time_p[2]}")
        current = data.Mya.get('IPM2C21A', time_p[1], time_p[2])
        print(f"Got {len(current)} points.")
        current_sum_trap = scii.cumtrapz(current.value, current.ms) * 1e-9
        current_sum_trap_tot = np.trapz(current.value, current.ms) * 1e-9   # In mC
        current_sum = square_integrate(current.value, current.ms) * 1e-9
        current_sum_tot = square_integrate_total(current.value, current.ms) * 1e-9
        current_time = pd.to_datetime(current.ms, unit='ms', utc=False)

        print(f"{time_p[0]}:  {time_p[1]} - {time_p[2]}  => Total charge = {current_sum_tot}")
        print(f"{time_p[0]}:  {time_p[1]} - {time_p[2]}  => Total charge = {current_sum_trap_tot}"
              " (trapezoidal integration)")

        fig.add_trace(
            go.Scatter(x=current_time,
                       y=current.value,
                       mode="lines",
                       line=dict(color='rgba(90, 180, 88, 0.6)', width=2),
                       line_shape="hv",
                       name="Current"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=current_time,
                       y=current_sum,
                       mode="lines",
                       line=dict(color=time_p[5], width=2),
                       name=f"Current Sum {time_p[0]}"),
            secondary_y=True)

        fig.add_annotation(
            x=current_time.iloc[-1],
            y=current_sum[-1]*1.03,
            xref="x",
            yref="y2",
            text=f"<b>Total: {current_sum_tot:5.3f} mC</b>",
            showarrow=False,
            font=dict(
                family="Arial, sans-serif",
                color=time_p[5],
                size=16),
            bgcolor="#FFFFFF"
        )

    fig.update_layout(
        title=go.layout.Title(
            text="RGC 2022: charge on ND3 target",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.40),
        titlefont=dict(size=24),
        legend=dict(
            x=0.02,
            y=1.15,
            bgcolor="rgba(250,250,250,0.80)",
            font=dict(
                size=14
            ),
            orientation='h'
        )
    )

    fig.add_annotation(
        x=0.85,
        xanchor="left",
        xref="paper",
        y=-0.09,
        yanchor="bottom",
        yref="paper",
        text="Graph:<i>Maurik Holtrop, UNH</i>",
        showarrow=False,
        font=dict(
            family="Arial",
            color="rgba(170,150,200,0.3)",
            size=10)
    )

    fig.update_yaxes(
        title_text="<b>Current [nA]</b>",
        titlefont=dict(size=22),
        secondary_y=False,
        tickfont=dict(size=18),
    )
    fig.update_yaxes(
        title_text="<b>Charge [mC]</b>",
        titlefont=dict(size=22),
        secondary_y=True,
        tickfont=dict(size=18),
    )

    fig.update_xaxes(
        title_text="Date",
        titlefont=dict(size=22),
        tickfont=dict(size=18),
    )

    print("Show plots.")
    fig.write_image("RGC2022_target_illum.pdf", width=2048, height=900)
    fig.write_image("RGC2022_target_illum.png", width=2048, height=900)
    fig.write_html("RGC2022_target_illum.html")
    fig.show(width=2048, height=900)  # width=1024,height=768


if __name__ == "__main__":
    sys.exit(main())
else:
    hostname = os.uname()[1]
    if hostname.find('clon') >= 0 or hostname.find('ifarm') >= 0 or hostname.find('jlab.org') >= 0:
        #
        # For JLAB setup the place we can find the RCDB
        #
        at_jlab = True
    else:
        at_jlab = False

    data = RunData(cache_file="", sqlcache=False, i_am_at_jlab=at_jlab)
    time_periods = setup_time_periods()
    time_p = time_periods[0]
    print(f"Getting data for {time_p[0]} from {time_p[1]} to {time_p[2]}")
    current = data.Mya.get('IPM2C21A', time_p[1], time_p[2])
    print(f"Got {len(current)} points.")

