#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Specify encoding so strings can have special characters.
#
#
# From Sho's code:
# privat static double motorToAngleTop(double motor) {
# return (17.821 - motor) / 832.714;
# }
#
# private static double motorToAngleBottom(double motor) {
# return (17.397 - motor) / 832.714;
# }
#
#
#
# From Matt Solt, email 2/8/2020
#Top:
# 18.017 = -2.2178 * 0.5 + y; y = 19.126
# y(top-si) = -0.4509 × y(stage) + 8.6242
# y(top-wire) = -0.4816 × y(stage) + 1.0516
# top-angle = (18.017 - y(stage))/833.247
#
# Bottom:
# 18.250 = -2.3657 * 0.5 + y; y = 19.433
# y(bot-si) = +0.4227 × y(stage) – 8.215
# y(bot-wire) = +0.4648 × y(stage) - 1.073
# bot-angle = (18.250 - y(stage))/833.247


from __future__ import print_function
import sys
import os
from datetime import datetime, timedelta

from RunData.RunData import RunData

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import chart_studio.plotly as charts

    pio.renderers.default = "browser"
except:
    print("Sorry, but to make the nice plots, you really need a computer with 'plotly' installed.")
    sys.exit(1)

data         = 'N/A'
mya_channels = []
mya_channels_bias = []
mya_data     = 'N/A'
mya_channels_per_run = 'N/A'


# Deduced from MYA data:
#
# TOP: Motor to position:  array([ 8.62419992, -0.45089999])  (intercept, slope)
# Bottom:               :  array([-8.21500005,  0.42270001])
# SVT TOP position = 8.6242 - 0.4509* hps:svt_top:motor.RBV
# SVT BOT position = -8.215 + 0.4227* hps:svt_bos:motor.RBV
#

def SVT_pos_top(motor_top):
    return(8.6242 - 0.4509*motor_top)

def SVT_pos_bot(motor_bot):
    return(-8.215 + 0.4227*motor_bot)


def SVT_motor_to_angle_top(motor_top):
# Sho's 2015:     return( (17.821 - motor_top) / 832.714 )
# Matt Solt's 2020: top-angle = (18.017 - y(stage))/833.247
    return((18.017 - motor_top)/833.247)

def SVT_motor_to_angle_bot(motor_bot):
# Sho's 2015:    return(  (17.397 - motor_bot) / 832.714)
# Matt Solt's 2019: bot-angle = (18.250 - y(stage))/833.247
    return( (18.250 - motor_bot )/833.247 )


def plot_channel(channel,transform=None,runs=None,name=None, stride = 1):
    """Add a single channel stored in mya_channels_per_run to the figure's store of lines.
    If you specify 'name' than that will be used for the label instead of the channel name.
    If you specify 'stride' then the data will be sparsified by taking only every stride value
    in a single run.
    The routine returns a go.Scatter object that can then be updated if needed and added to a go.Figure"""

    # These are not really needed be because none of them are modified. Here to indicate we need them.
    global data
    global mya_channels_per_run

    if transform==None:
        transform = lambda x: x    # Identity transform.

    if name==None:
        name=channel

    if runs==None:
        runs = data.All_Runs.index

    xl = []
    yl = []
    ht = [] # Hover text to add to line. Nice for adding run numbers to points.
    for r in data.All_Runs.index:

        xl += [data.All_Runs.loc[r].start_time]  # Append the start time of run and channel value at start.
        yl += [transform(mya_channels_per_run[r][channel].iloc[0].value)]
        ht += ["run:{:5d} start".format(r)]

        if len(mya_channels_per_run[r][channel]) > 1:  # Check if the value changes during this run.
            # Add the data changes during the run period to the plot.
            xl += [np.datetime64(x) for x in mya_channels_per_run[r][channel].iloc[1::stride].time]
            yl += list(transform(mya_channels_per_run[r][channel].iloc[1::stride].value))
            ht += [ "run:{:5d}".format(r) for x in mya_channels_per_run[r][channel].iloc[1::stride].time]

        xl += [data.All_Runs.loc[r].end_time]  # Append the end time of run and SVT channel value at end.
        yl += [transform(mya_channels_per_run[r][channel].iloc[-1:].value.iloc[0])]  # Last value in the data.
        ht += ["run:{:5d} end".format(r)]
        xl += [data.All_Runs.loc[r].end_time + np.timedelta64(1, 's')]    # Add one more point, +1 s after run.
        yl += [None]                                                      # This is a 'None' point, forcing a line segment.
        ht += ["None"]

    xl = [q.astype('M8[ms]').astype('O') for q in xl]   # Change the np.datetime64 to a datetime.datetime.
    name_short = name.replace("SVT:bias:",'').replace("v_sens",'')
    line = go.Scatter(x=xl, y=yl, hovertext=ht, name=name_short, line=dict(shape="hv"))        # Construct a line and return.
    return(line)

def plot_bias(fig=None):
    global data
    global mya_channels
    global mya_data
    global mya_channels_per_run

    if fig == None:
        fig = go.Figure()

    for channel in mya_channels_bias:
        line = plot_channel(channel)
        fig.add_trace(line)

    fig.update_layout(
        title=go.layout.Title(
            text="HPS 2019 RUN, SVT Bias Voltage",
            yanchor="top", y=0.95,
            xanchor="center",
            x=0.5))
    fig.update_xaxes(title_text="<b>Date/time<b>")
    fig.update_yaxes(title_text="<b>V_sens [V]<b>")

    return(fig)

def plot_svt_angles(fig=None):
    global data
    global mya_channels_per_run

    top = plot_channel('hps:svt_top:motor.RBV',transform=SVT_motor_to_angle_top,name="SVT angle top")
    bot = plot_channel('hps:svt_bot:motor.RBV',transform=SVT_motor_to_angle_bot,name="SVT angle bot")

    top.update(line=dict(color="blue", width=2))
    bot.update(line=dict(color="green", width=2))

    if fig==None:
        fig = go.Figure()

    fig.add_trace(top)
    fig.add_trace(bot)

    fig.update_layout(
        title=go.layout.Title(
            text="HPS 2019 RUN, SVT angles from nominal",
            yanchor="top", y=0.95,
            xanchor="center",
            x=0.5))
    fig.update_xaxes(title_text="<b>Date/time<b>")
    fig.update_yaxes(title_text="<b>Delta Angle [rad]<b>", range=[-1e-3, 1e-3])

    return(fig)

    # charts.plot(fig, filename = 'Run2019_svt_angle', auto_open=True)

def run(username=None,password=None):

    global data
    global mya_channels
    global mya_data
    global mya_channels_per_run
    global mya_channels_bias

    hostname = os.uname()[1]
    if hostname.find('clon') >= 0 or hostname.find('ifarm') >= 0 or hostname.find('jlab.org') >= 0:
        #
        # For JLAB setup the place we can find the RCDB
        #
        at_jlab = True

    data = RunData(cache_file="HPS_run_cache.sqlite3",username=username,password=password)
    # data._cache_engine=None   # Turn OFF cache?
    data.debug = 4

    # data.Good_triggers=['hps_v7.cnf','hps_v8.cnf','hps_v9.cnf','hps_v9_1.cnf',
    #                     'hps_v9_2.cnf','hps_v10.cnf',
    #                     'hps_v11_1.cnf','hps_v11_2.cnf','hps_v11_3.cnf','hps_v11_4.cnf',
    #                     'hps_v11_5.cnf','hps_v11_6.cnf','hps_v12_1.cnf']
    data.Good_triggers = 'hps_v..?_?.?\.cnf'
    data.Production_run_type = ["PROD66", "PROD67"]

    min_event_count = 10000000  # Runs with at least 10M events.
    #    start_time = datatime(2019, 7, 17, 0, 0)  # Very start of run
    start_time = datetime(2019, 7, 25, 0, 0)  # SVT back in correct position
    end_time = datetime(2019, 9, 10, 0, 0)
    end_time = end_time + timedelta(0, 0, -end_time.microsecond)  # Round down on end_time to a second

    print("Fetching the run data from {} to {}".format(start_time, end_time))
    data.get_runs(start_time, end_time, min_event_count)
    data.select_good_runs()

# Get the MYA data for the entire dataset for each channel separately.

    mya_channels_pos=["hps:svt_top:motor.RBV", "hps:svt_bot:motor.RBV"]
    mya_channels_bias=[]
    for i in range(0,18):
        mya_channels_bias.append("SVT:bias:top:{:d}:v_sens".format(i))

    for i in range(20,38):
        mya_channels_bias.append("SVT:bias:bot:{:d}:v_sens".format(i))

    mya_channels = mya_channels_pos + mya_channels_bias

    mya_data = {}
    for channel in mya_channels:
        print("Fetching MYA channel: {}".format(channel))
        mdat = data.Mya.get(channel,start_time,end_time)
        mya_data[channel] = mdat

#
# For each run, we now check the values for each channel during the run period.
# Since the MYA data is made sparse, we want to get the *last* value *before* the start of the run
# for the first value. If there is no data during the run period, then the value was constant.
# If there are entries during the run, then these will need to be put with timestamp into the Conditions DB
# for that channel. We first store these in set of lists, one for each run, all appended to a dictionary with
# on entry per run.
#
    mya_channels_per_run = {}          # Data store for each run.

    print("Itteratively going through all the MYA data for each time period of a run. ")
    for run in data.All_Runs.index:   # For each selected run number.
        print("Processing for run {}".format(run))
        mya_channel_data = {}         # Data store for the channels.
        for channel in mya_channels:  # For each channel.
            run_start_time = data.All_Runs.loc[run].start_time
            run_end_time   = data.All_Runs.loc[run].end_time
            #
            # Get the first data point, from *before* the run_start_time
            #
            mya_data_row = mya_data[channel].loc[mya_data[channel].time <= run_start_time].tail(1)
            #
            # Get the data points *during* the run, if any. This could be empty if all was stable during the run.
            #
            mya_data_row2 = mya_data[channel].loc[ (mya_data[channel].time>run_start_time) & (mya_data[channel].time<run_end_time)]
            row = mya_data_row.append(mya_data_row2)
            #
            # Add the entries to the data store for this channel.
            #
            mya_channel_data[channel]=row
        #
        #  All the channels are stored in mya_channel_data.
        #  Store them for this run in the mya_channels_per run
        mya_channels_per_run[run]=mya_channel_data

    #
    # A big store of all the data is now available!!!
    #


if __name__ == "__main__":
    run()