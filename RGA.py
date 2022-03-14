#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Specify encoding so strings can have special characters.
#

from __future__ import print_function
import sys
import os
from datetime import datetime, timedelta
import numpy as np

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


def rga_2022_target_properties():
    """ Returns the dictionary of dictionaries for target properties. """
    target_props = {
        'names': {     # Translation table for long name to short name.
            'EMPTY': 'empty',
            'Empty cell': 'empty',
            'Empty': 'empty',
            'empty': 'empty',
            'Emptu cell': 'empty',
            'empty target': 'empty',
            'None': 'empty',
            'LH2': 'LH2',
            'Lh2': 'LH2',
            'H': 'LH2',
            'H2': 'LH2',
            '': 'LH2',       # Assume that if left blank, it was the LH2 target.
            ' ': 'LH2',
            'Liquid Hydrogen Target': 'LH2',
            'LD2': 'LD2',
            'lD2': 'LD2',
            'D2': 'LD2'
        },
        'density': {     # Units: g/cm^2
            # 'norm': 0.335,
            'LH2': 0.335,
            'empty': 0
        },
        'current': {  # Nominal current in nA.  If 0, no expected charge line will be drawn.
            ## list of currents for each beam energy period.
            'scale': [1, 1, 1],     # Special entry. Multiply sum charge by this factor,
            'LH2': [50., 50., 50.],   # for plotting with multiple beam energies, where charge rates vary a lot.
            'empty': [0., 0., 0.]
        },
        'attenuation': {     # Units: number
            'LH2':  1,
            'LD2': 1,
            'empty': 1,
        },
        'color': {  # Plot color: r,g,b,a
            'LH2':  'rgba(0, 120, 150, 0.8)',
            'empty': 'rgba(220, 220, 220, 0.8)'
        },
        'sums_color': {  # Plot color: r,g,b,a
            # 'empty': 'rgba(255, 200, 200, 0.8)',
            'LH2': 'rgba(255, 120, 150, 0.8)',
        },

    }

    return target_props


def compute_plot_runs(targets, run_config, date_min=None, date_max=None, data_loc=None):
    """This function selects the runs from data according to the target, run_configuration and date"""
    # print("Compute data for plots.")

    runs = data_loc.All_Runs.loc[data_loc.list_selected_runs(targets=targets, run_config=run_config,
                                                             date_min=date_min, date_max=date_max)]

    starts = runs["start_time"]
    ends = runs["end_time"]
    runs["center"] = starts + (ends - starts) / 2
    runs["dt"] = [(run["end_time"] - run["start_time"]).total_seconds() * 999 for num, run, in runs.iterrows()]
    runs["event_rate"] = [runs.loc[r, 'event_count'] / runs.loc[r, 'dt'] for r in runs.index]
    runs["hover"] = [f"Run: {r}<br />"
                     f"Trigger:{runs.loc[r, 'run_config']}<br />"
                     f"Start: {runs.loc[r, 'start_time']}<br />"
                     f"End: {runs.loc[r, 'end_time']}<br />"
                     f"DT:   {runs.loc[r, 'dt'] / 1000.:5.1f} s<br />"
                     f"NEvt: {runs.loc[r, 'event_count']:10,d}<br />"
                     f"Charge: {runs.loc[r, 'charge']:6.2f} mC <br />"
                     f"Lumi: {runs.loc[r, 'luminosity']:6.2f} 1/fb<br />"
                     f"<Rate>:{runs.loc[r, 'event_rate']:6.2f}kHz<br />"
                     for r in runs.index]

    return runs


def used_triggers():
    """Setup the triggers used."""
    good_triggers = '.*'
    calibration_triggers = ['None', 'RNDM']

    return good_triggers, calibration_triggers


def setup_rundata_structures(data_loc, dates):
    """Setup the data structures for parsing the databases."""

    start_time, end_time = dates
    data_loc.Good_triggers, data_loc.Calibration_triggers = used_triggers()

    data_loc.Production_run_type = "PROD.*"  # ["PROD66", "PROD66_PIN", "PROD66_noVTPread", "PROD67_noVTPread"]
    data_loc.target_properties = rga_2022_target_properties()
    data_loc.target_dens = data_loc.target_properties['density']
    data_loc.atten_dict = None
    data_loc.Current_Channel = "IPM2C21A"  # "scaler_calc1b"
    data_loc.LiveTime_Channel = "B_DAQ:livetime_pulser"
    data_loc.Useful_conditions.append('beam_energy')  # This run will have multiple beam energies.

    min_event_count = 500000  # Runs with at least 200k events.
    end_time = end_time + timedelta(0, 0, -end_time.microsecond)  # Round down on end_time to a second
    print("Fetching the data from {} to {}".format(start_time, end_time))
    data_loc.get_runs(start_time, end_time, min_event_count)
    data_loc.select_good_runs()

data = None
beam_stop_atten_time = None  # Persist, so we don't look it up all the time.
fcup_offset_time = None

def initialize_fcup_param(start_time, end_time, override=False):
    """Initialize the beam_stop_atten_time and fcup_offset_time global parameters."""

    global beam_stop_atten_time
    global fcup_offset_time

    if beam_stop_atten_time is None or override:  # Need to fill the beam_stop_atten_time dataframe:
        print("Getting beam_stop_atten.")
        beam_stop_atten = data.Mya.get(channel="beam_stop_atten",
                                       start=start_time,
                                       end=end_time,
                                       run_number=1)
        beam_stop_atten_time = beam_stop_atten.set_index(['time'])

    if fcup_offset_time is None or override:
        print("Getting fcup_offset.")
        fcup_offset = data.Mya.get(channel="fcup_offset",
                                   start=start_time,
                                   end=end_time,
                                   run_number=1)
        fcup_offset_time = fcup_offset.set_index(['time'])

    return beam_stop_atten_time, fcup_offset_time

def compute_fcup_current(rnum, data=data, override=False,
                            current_channel="scalerS2b", livetime_channel=None):
    """Compute the FCup charge for run rnum, from the FCup scaler channel and livetime_channel"""
    if current_channel is None:
        current_channel = "scalerS2b"
    if livetime_channel is None:
        livetime_channel = data.LiveTime_Channel

    if not override and \
            ( "FCup_cor" in data.All_Runs.keys()) and \
            not np.isnan(data.All_Runs.loc[rnum, current_channel]):
        return

    if data.debug > 4:
        print("compute_fcup_data, run= {:5d}".format(runnumber))

    start_time = data.All_Runs.loc[rnum, "start_time"]
    end_time   = data.All_Runs.loc[rnum, "end_time"]
    scaler = data.Mya.get(current_channel, start_time, end_time, run_number=rnum)

    # Get the "forward fill" value for start_time ==> i.e. the *value before* start_time
    bsat = beam_stop_atten_time.index.get_indexer([start_time], method='ffill')
    if bsat<0: # We asked for a time before the first beam_stop_atten_time, so instead take the [0] one
        bsat = np.array([0]) # Keep the same type.
    beam_stop_attenuation = float(beam_stop_atten_time.iloc[bsat].value)
    fcup_offset = fcup_offset_time.loc[start_time:end_time]
    # Get one more before the start_time
    fcup_prepend = fcup_offset_time.iloc[fcup_offset_time.index.get_indexer([start_time], method='ffill')]
    fcup_prepend.index=[scaler.iloc[0].time]             # Reset the index of last fcup value to start_time
    fcup_offset = pd.concat([fcup_prepend,fcup_offset])     # Add the one value to the list.

    fcup_offset_interpolate = np.interp(scaler.ms, fcup_offset.ms, fcup_offset.value)
    current_values = beam_stop_attenuation * (scaler.value - fcup_offset_interpolate) / 906.2
    scaler.value = current_values   # Override the values with the computed current.
    return scaler


def add_computed_fcup_data_to_runs(data=data, dates=None, targets=None, run_config=None, override=False,
                                   current_channel="scalerS2b", livetime_channel=None):
    """Get the mya data for beam current from the FCup using the formula:
    beam_stop_atten*(scalerS2b - fcup_offset)/906.2
    See email from Rafo: 2/8/22 10pm"""

    global beam_stop_atten_time
    global fcup_offset_time

    if dates is None:
        start_time = datetime(2018, 1, 1, 0, 0)
        end_time = datetime(2019, 12, 31, 0, 0)
    else:
        start_time = dates[0]
        end_time = dates[1]

    initialize_fcup_param(start_time, end_time, override)

    # Code modeled after RunData.add_current_data_to_runs.
    good_runs = self.list_selected_runs(targets, run_config)
    if len(good_runs) > 0:
        for rnum in self.list_selected_runs(targets, run_config):
            current = compute_fcup_current(rnum, data=data, override=override,
                                           current_channel=current_channel, livetime_channel=livetime_channel)
    else:
        # Even if there are no good runs, make sure that the "charge" column is in the table!
        # This ensure that when you write to DB the charge column exists.
        self.All_Runs.loc[:, "charge"] = np.NaN


def main(argv=None):
    import argparse
    global data
    if argv is None:
        argv = sys.argv
    else:
        argv = argv.split()
        argv.insert(0, sys.argv[0])  # add the program name.

    parser = argparse.ArgumentParser(
        description="""Make a plot, an excel spreadsheet and/or an sqlite3 database for the current run using
        conditions from the RCDB and MYA.""",
        epilog="""
        For more info, read the script ^_^, or email maurik@physics.unh.edu.""")

    parser.add_argument('-c', '--charge', action="store_true", help="Make a plot of charge not luminosity.")
    parser.add_argument('-C', '--chart', action="store_true", help="Put plot on plotly charts website.")
    parser.add_argument('-d', '--debug', action="count", help="Be more verbose if possible. ", default=0)
    parser.add_argument('-e', '--excel', action="store_true", help="Create the Excel table of the data")
    parser.add_argument('-f', '--date_from', type=str, help="Plot from date, eg '2021,11,09' ", default=None)
    parser.add_argument('-l', '--live', action="store_true", help="Show the live plotly plot.")
    parser.add_argument('-N', '--nocache', action="store_true", help="Do not use a sqlite3 cache")
    parser.add_argument('-p', '--plot', action="store_true", help="Create the plotly plots.")
    parser.add_argument('-r', '--run_period', type=int, help="Run period selector, 0=all (default)", default=0)
    parser.add_argument('-t', '--date_to', type=str, help="Plot to date, eg '2022,01,22' ", default=None)

    args = parser.parse_args(argv[1:])

    hostname = os.uname()[1]
    if hostname.find('clon') >= 0 or hostname.find('ifarm') >= 0 or hostname.find('jlab.org') >= 0:
        #
        # For JLAB setup the place we can find the RCDB
        #
        at_jlab = True
    else:
        at_jlab = False

    if not args.nocache:
        data = RunData(cache_file="RGA.sqlite3", i_am_at_jlab=at_jlab)
    else:
        data = RunData(cache_file="", sqlcache=False, i_am_at_jlab=at_jlab)
    # data._cache_engine=None   # Turn OFF cache?
    data.debug = args.debug

    run_sub_periods_available = [
            (datetime(2018, 2,  5, 20, 0), datetime(2018,  2,  8, 6, 0)),
            (datetime(2018, 9, 27,  1, 0), datetime(2018, 11, 26, 7, 0)),
            (datetime(2019, 3, 25, 18, 0), datetime(2019,  4, 15, 6, 0))
            ]

    if args.run_period == 0:
        run_sub_periods = run_sub_periods_available
    else:
        run_sub_periods = [run_sub_periods_available[args.run_period-1]]

    run_sub_energy = [2.07, 4.03, 5.99]
    run_sub_y_placement = [0.79, 0.99, 0.99]

    if args.plot:
        max_y_value_sums = 0.
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        max_expected_charge = []

    legends_data = []  # To keep track of which targets already have a legend shown.
    legends_shown = []  # To keep track of which target has the charge sum legend shown.
    #   Loop over the different run sub-periods.

    if args.excel:
        excel_output = pd.DataFrame()

    for sub_i in range(len(run_sub_periods)):
        setup_rundata_structures(data, run_sub_periods[sub_i])

        data.All_Runs['luminosity'] *= 1E-3   # Rescale luminosity from 1/pb to 1/fb

        #    data.add_current_data_to_runs()
        targets = '.*'

        # Select runs into the different categories.
        plot_runs = compute_plot_runs(targets=targets, run_config=data.Good_triggers, data_loc=data)
        calib_run_numbers = data.list_selected_runs(targets='.*', run_config=data.Calibration_triggers)
        calib_runs = plot_runs.loc[calib_run_numbers]

        plot_runs = plot_runs.loc[~plot_runs.index.isin(calib_run_numbers)]  # Take the calibration runs out.
        starts = plot_runs["start_time"]
        ends = plot_runs["end_time"]

        print("Compute cumulative charge.")
        data.compute_cumulative_charge(targets, runs=plot_runs)

        print("Adding additional current channels.")
        data.add_current_data_to_runs(current_channel="IPM2C24A", livetime_channel="B_DAQ:livetime")
        data.add_current_data_to_runs(current_channel="scaler_calc1b", livetime_channel="B_DAQ:livetime")

        print("Computing beam_stop_atten*(scalerS2b - fcup_offset)/906.2")
        add_computed_fcup_data_to_runs(data)

        if args.excel:
            excel_output = pd.concat([excel_output, plot_runs, calib_runs], sort=True)

        if args.plot:

            print(f"Build Plots for period {sub_i}")

            last_targ = None
            for targ in data.target_properties['color']:

                if args.debug:
                    print(f"Processing plot for target {targ}")
                runs = plot_runs.target.str.fullmatch(targ.replace('(', r"\(").replace(')', r'\)'))

                if np.count_nonzero(runs) > 1 and sub_i == len(run_sub_periods) - 1 and \
                        targ in data.target_properties['sums_color']:
                    last_targ = targ    # Store the last target name with data for later use.

                if targ in legends_data or np.count_nonzero(runs) <= 1:  # Do we show this legend?
                    show_data_legend = False                             # This avoids the same legend shown twice.
                else:
                    show_data_legend = True
                    legends_data.append(targ)

                fig.add_trace(
                    go.Bar(x=plot_runs.loc[runs, 'center'],
                           y=plot_runs.loc[runs, 'event_rate'],
                           width=plot_runs.loc[runs, 'dt'],
                           hovertext=plot_runs.loc[runs, 'hover'],
                           name="run with " + targ,
                           marker=dict(color=data.target_properties['color'][targ]),
                           legendgroup="group1",
                           showlegend=show_data_legend
                           ),
                    secondary_y=False, )

            fig.add_trace(
                 go.Bar(x=calib_runs['center'],
                        y=calib_runs['event_rate'],
                        width=calib_runs['dt'],
                        hovertext=calib_runs['hover'],
                        name="Calibration runs",
                        marker=dict(color='rgba(150,150,150,0.5)'),
                        legendgroup="group1",
                        ),
                 secondary_y=False, )

            if args.charge:
                current_plotting_scale = data.target_properties['current']['scale'][sub_i]
                sumcharge = plot_runs.loc[:, "sum_charge"] * current_plotting_scale
                max_y_value_sums = plot_runs.sum_charge_targ.max() * current_plotting_scale

                plot_sumcharge_t = [starts.iloc[0], ends.iloc[0]]
                plot_sumcharge_v = [0, sumcharge.iloc[0]]

                for i in range(1, len(sumcharge)):
                    plot_sumcharge_t.append(starts.iloc[i])
                    plot_sumcharge_t.append(ends.iloc[i])
                    plot_sumcharge_v.append(sumcharge.iloc[i - 1])
                    plot_sumcharge_v.append(sumcharge.iloc[i])

                for targ in data.target_properties['sums_color']:
                    sumch = plot_runs.loc[plot_runs["target"] == targ, "sum_charge_targ"]*current_plotting_scale
                    st = plot_runs.loc[plot_runs["target"] == targ, "start_time"]
                    en = plot_runs.loc[plot_runs["target"] == targ, "end_time"]

                    if len(sumch) > 1:
                        # Complication: When a target was taken out and then later put back in there is an interruption
                        # that should not count for the expected charge.

                        # Setup this initial entries for the plot lines.
                        plot_sumcharge_target_t = [st.iloc[0], en.iloc[0]]
                        plot_sumcharge_target_v = [0, sumch.iloc[0]]
                        if data.target_properties['current'][targ][sub_i] > 0.:
                            plot_expected_charge_t = [st.iloc[0]]
                            plot_expected_charge_v = [0]

                            current_expected_sum_charge = 0

                        for i in range(1, len(sumch)):
                            # Step through all the runs for this target that have "sum_charge_targ".
                            #
                            # We also need to detect if there is a run number gap.
                            # Check if all the intermediate runs have the same target. If so, these were calibration
                            # or junk runs, and we continue normally. If there were other targets, we make a break in the
                            # line.
                            if sumch.keys()[i] - sumch.keys()[i - 1] > 1 and \
                                    not np.all(data.All_Runs.loc[sumch.keys()[i-1]:sumch.keys()[i]].target == targ):
                                # There is a break in the run numbers (.keys()) and a target change occurred.
                                plot_sumcharge_target_t.append(st.iloc[i])  # Add the last time stamp again.
                                plot_sumcharge_target_v.append(None)        # None causes the solid line to break.

                                fig.add_trace(                 # Add a horizontal dotted line for the target sum_charge.
                                    go.Scatter(x=[en.iloc[i-1], st.iloc[i]],
                                               y=[plot_sumcharge_target_v[-2], plot_sumcharge_target_v[-2]],
                                               mode="lines",
                                               line=dict(color=data.target_properties['sums_color'][targ], width=1,
                                                         dash="dot"),
                                               name=f"Continuation line {targ}",
                                               showlegend=False),
                                    secondary_y=True)

                                if data.target_properties['current'][targ][sub_i] > 0.:  # Are we plotting expected charge?
                                    plot_expected_charge_t.append(en.iloc[i-1])  # Add the time stamp
                                    current_expected_sum_charge += (en.iloc[i-1] - plot_expected_charge_t[-2]).\
                                        total_seconds() * data.target_properties['current'][targ][sub_i] * 1e-6 * 0.5
                                    plot_expected_charge_v.append(current_expected_sum_charge)
                                    # Current is in nA, Charge is in mC, at 50% efficiency.
                                    plot_expected_charge_t.append(en.iloc[i-1])
                                    plot_expected_charge_v.append(None)

                                    if i+1 < len(sumch):  # Add the start of the next line segment
                                        plot_expected_charge_t.append(st.iloc[i])
                                        plot_expected_charge_v.append(current_expected_sum_charge)
                                        fig.add_trace(
                                            go.Scatter(x=plot_expected_charge_t[-2:],
                                                       y=[current_expected_sum_charge, current_expected_sum_charge],
                                                       mode='lines',
                                                       line=dict(color='rgba(90, 180, 88, 0.6)', width=2, dash="dot"),
                                                       name=f"Continuation line",
                                                       showlegend=False
                                                       # Only one legend at the end.
                                                       ),
                                            secondary_y=True
                                        )


                            plot_sumcharge_target_t.append(st.iloc[i])
                            plot_sumcharge_target_t.append(en.iloc[i])
                            plot_sumcharge_target_v.append(sumch.iloc[i - 1])
                            plot_sumcharge_target_v.append(sumch.iloc[i])

                        if data.target_properties['current'][targ][sub_i] > 0.:
                            plot_expected_charge_t.append(plot_sumcharge_target_t[-1])
                            i = len(plot_expected_charge_v)-1
                            while plot_expected_charge_v[i] is None and i > 0:
                                i -= 1
                            current_expected_sum_charge = plot_expected_charge_v[i]
                            current_expected_sum_charge += \
                                (plot_sumcharge_target_t[-1] - plot_expected_charge_t[-2]).total_seconds() * \
                                data.target_properties['current'][targ][sub_i] * 1e-6 * 0.5
                            plot_expected_charge_v.append(current_expected_sum_charge*current_plotting_scale)

                        if targ in legends_shown:
                            show_legend_ok = False
                        else:
                            show_legend_ok = True
                            legends_shown.append(targ)


                        fig.add_trace(
                            go.Scatter(x=plot_sumcharge_target_t,
                                       y=plot_sumcharge_target_v,
                                       mode="lines",
                                       line=dict(color=data.target_properties['sums_color'][targ], width=3),
                                       name=f"Total Charge on {targ}",
                                       legendgroup="group2",
                                       showlegend=show_legend_ok
                                       ),
                            secondary_y=True)

                        # Decorative: add a dot at the end of the curve.
                        fig.add_trace(
                            go.Scatter(x=[plot_sumcharge_target_t[-1]],
                                       y=[plot_sumcharge_target_v[-1]],
                                       marker=dict(color=data.target_properties['sums_color'][targ],
                                                   size=6),
                                       showlegend=False),
                            secondary_y=True)

                        # Decorative: add a box with an annotation of the total charge on this target.
                        actual_charge = plot_sumcharge_target_v[-1]/current_plotting_scale
                        fig.add_annotation(
                            x=plot_sumcharge_target_t[-1],
                            y=plot_sumcharge_target_v[-1] + max_y_value_sums*0.015,
                            xref="x",
                            yref="y2",
                            text=f"<b>total: {actual_charge:3.2f} mC</b>",
                            showarrow=False,
                            font=dict(
                                family="Arial, sans-serif",
                                color=data.target_properties['sums_color'][targ],
                                size=16),
                            bgcolor="#FFFFFF"
                        )

                        # # Annotate - add a curve for the expected charge at 50% efficiency.
                        showlegend = True if targ == last_targ and sub_i == 2 else False
                        if args.debug:
                            print(f"last_targ = {last_targ}  targ: {targ}, sub_i = {sub_i}, showlegend = {showlegend}")
                        if data.target_properties['current'][targ][sub_i] > 0.:
                            fig.add_trace(
                                go.Scatter(x=plot_expected_charge_t,
                                           y=plot_expected_charge_v,
                                           mode='lines',
                                           line=dict(color='rgba(90, 180, 88, 0.6)', width=4),
                                           name=f"Expected charge at 50% up",
                                           showlegend=True if targ == last_targ and sub_i == 2 else False,
                                           # Only one legend at the end.
                                           legendgroup="group2",
                                           ),
                                secondary_y=True
                            )
                            max_expected_charge.append(plot_expected_charge_v[-1])
                            # print(f"max_expected_charge = {max_expected_charge}")


    #################################################################################################################
    #                     Luminosity
    #################################################################################################################
            else:
                # sumlumi = plot_runs.loc[:, "sum_lumi"]
                # plot_sumlumi_t = [starts.iloc[0], ends.iloc[0]]
                # plot_sumlumi = [0, sumlumi.iloc[0]]
                #
                # for i in range(1, len(sumlumi)):
                #     plot_sumlumi_t.append(starts.iloc[i])
                #     plot_sumlumi_t.append(ends.iloc[i])
                #
                #     plot_sumlumi.append(sumlumi.iloc[i - 1])
                #     plot_sumlumi.append(sumlumi.iloc[i])
                #
                # fig.add_trace(
                #     go.Scatter(x=plot_sumlumi_t,
                #                y=plot_sumlumi,
                #                line=dict(color='#FF3030', width=3),
                #                name="Luminosity Live"),
                #     secondary_y=True)

                # We get the sums per target in two steps. Clumsy, but only way to get the maximum available in second loop
                plot_sumlumi_target = {}                       # Store sum results.
                for targ in data.target_properties['sums_color']:
                    selected = plot_runs.target == targ
                    if len(plot_runs.loc[selected, "luminosity"]) > 0:
                        plot_sumlumi_target[targ] = np.cumsum(plot_runs.loc[selected, "luminosity"])
                        # Store overall max.
                        max_y_value_sums = max(float(plot_sumlumi_target[targ].max()), max_y_value_sums)

                for targ in plot_sumlumi_target.keys():  # data.target_properties['sums_color']:
                    plot_sumlumi_starts = plot_runs.loc[plot_runs["target"] == targ, "start_time"]
                    plot_sumlumi_ends = plot_runs.loc[plot_runs["target"] == targ, "end_time"]
                    if len(plot_sumlumi_target[targ]) > 1:
                        plot_sumlumi_target_t = [plot_sumlumi_starts.iloc[0], plot_sumlumi_ends.iloc[0]]
                        plot_sumlumi_target_v = [0, plot_sumlumi_target[targ].iloc[0]]
                        for i in range(1, len(plot_sumlumi_target[targ])):
                            if plot_sumlumi_target[targ].keys()[i] - plot_sumlumi_target[targ].keys()[i - 1] > 1 and \
                                    not np.all(data.All_Runs.loc[plot_sumlumi_target[targ].keys()[i-1]:
                                               plot_sumlumi_target[targ].keys()[i]].target == targ):
                                plot_sumlumi_target_t.append(plot_sumlumi_starts.iloc[i])
                                plot_sumlumi_target_v.append(None)

                                fig.add_trace(
                                    go.Scatter(x=[plot_sumlumi_starts.iloc[i-1], plot_sumlumi_starts.iloc[i]],
                                               y=[plot_sumlumi_target_v[-2], plot_sumlumi_target_v[-2]],
                                               mode="lines",
                                               line=dict(color=data.target_properties['sums_color'][targ], width=1,
                                                         dash="dot"),
                                               name=f"Continuation line {targ}",
                                               legendgroup="group2",
                                               ),
                                    secondary_y=True)

                            plot_sumlumi_target_t.append(plot_sumlumi_starts.iloc[i])
                            plot_sumlumi_target_t.append(plot_sumlumi_ends.iloc[i])
                            plot_sumlumi_target_v.append(plot_sumlumi_target[targ].iloc[i - 1])
                            plot_sumlumi_target_v.append(plot_sumlumi_target[targ].iloc[i])

                        fig.add_trace(
                            go.Scatter(x=plot_sumlumi_target_t,
                                       y=plot_sumlumi_target_v,
                                       mode="lines",
                                       line=dict(color=data.target_properties['sums_color'][targ], width=3),
                                       name=f"Sum luminosity on {targ}",
                                       legendgroup="group2",
                                       ),
                            secondary_y=True)

                        fig.add_trace(
                            go.Scatter(x=[plot_sumlumi_target_t[-1]],
                                       y=[plot_sumlumi_target_v[-1]],
                                       marker=dict(
                                           color=data.target_properties['sums_color'][targ],
                                           size=6
                                           ),
                                       showlegend=False
                                       ),
                            secondary_y=True
                        )

                        fig.add_annotation(
                            x=plot_sumlumi_target_t[-1],
                            y=plot_sumlumi_target_v[-1] + max_y_value_sums*0.015,
                            xref="x",
                            yref="y2",
                            text=f"<b>total: {plot_sumlumi_target_v[-1]:3.2f} 1/fb</b>",
                            showarrow=False,
                            font=dict(
                                family="Arial, sans-serif",
                                color=data.target_properties['sums_color'][targ],
                                size=16),
                            bgcolor="#FFFFFF"
                        )

    if args.excel:
        print("Write new Excel table.")
        excel_output.to_excel("RGA_progress.xlsx",
                              #columns=['start_time', 'end_time', 'target', 'beam_energy', 'run_config', 'selected',
                              #         'event_count', 'sum_event_count', 'charge', 'sum_charge', 'luminosity',
                              #         'sum_lumi', 'evio_files_count', 'megabyte_count', 'operators', 'user_comment']
                              )

        # End sub run period loop.
    if args.plot:
        fig.add_annotation(
            x=0.85,
            xanchor="left",
            xref="paper",
            y=-0.1,
            yanchor="bottom",
            yref="paper",
            text="Graph:<i>Maurik Holtrop, UNH</i>",
            showarrow=False,
            font=dict(
                family="Arial",
                color="rgba(170,150,200,0.4)",
                size=12)
        )
        # Set x-axis title
        fig.update_layout(
            title=go.layout.Title(
                text="RGA Runs",
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

        # Set y-axes titles
        fig.update_yaxes(
            title_text="<b>Event rate kHz</b>",
            titlefont=dict(size=22),
            secondary_y=False,
            tickfont=dict(size=18),
            range=[0, 1.05*max(25., plot_runs.loc[runs, 'event_rate'].max())]
        )

        if args.charge:
            max_expected_charge.append(max_y_value_sums)
            if args.debug:
              print(f"max_expected_charge: {max_expected_charge}")
            max_y_2nd_scale = 1.1*np.max(max_expected_charge)
            # max_y_2nd_scale = 7.
            if args.debug:
                print(f"max_y_2nd_scale = {max_y_2nd_scale}")
            fig.update_yaxes(title_text="<b>Accumulated Charge (mC)</b>",
                             titlefont=dict(size=22),
                             range=[0, max_y_2nd_scale],
                             secondary_y=True,
                             tickfont=dict(size=18)
                             )
        else:
            fig.update_yaxes(title_text="<b>Integrated Luminosity (1/fb)</b>",
                             titlefont=dict(size=22),
                             range=[0, 1.05*max_y_value_sums],
                             secondary_y=True,
                             tickfont=dict(size=18)
                             )

        fig.update_xaxes(
            title_text="Date",
            titlefont=dict(size=22),
            tickfont=dict(size=18),
        )

        if (args.date_from is not None) or (args.date_to is not None):
            if args.date_from is not None:
                date_from = datetime.strptime(args.date_from, '%Y,%m,%d')
            else:
                date_from = starts.iloc[0]

            if args.date_to is not None:
                date_to = datetime.strptime(args.date_to, '%Y,%m,%d')
            else:
                date_to = ends.iloc[-1]

            fig.update_xaxes(
                range=[date_from, date_to]
            )

        print("Show plots.")
        fig.write_image("RGA_progress.pdf", width=2048, height=900)
        fig.write_image("RGA_progress.png", width=2048, height=900)
        fig.write_html("RGA_progress.html")
        if args.chart:
            charts.plot(fig, filename='RGA_edit', width=2048, height=900, auto_open=True)
        if args.live:
            fig.show(width=2048, height=900)  # width=1024,height=768

if __name__ == "__main__":
    sys.exit(main())
else:
    print("Imported the RGA info. Setting up data.")
    data = RunData(cache_file="RGA.sqlite3", sqlcache=True, i_am_at_jlab=False)
    data.debug = 10
    print("setup_rundata_structures(data,(datetime(), datetime()))")
