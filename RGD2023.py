#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Specify encoding so strings can have special characters.
#
# This script, using the RunData pacakge, makes plots for the Run Group D experiment with CLAS12 in Hall-B.
#
# Author: Maurik Holtrop - UNH - June 2022.
#
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


def target_properties():
    """ Returns the dictionary of dictionaries for target properties. """
    target_props = {
        'names': {     # Translation table for long name to short name.
            'Empty cell': 'empty',
            'Empty': 'empty',
            'empty': 'empty',
            'None': 'empty',
            'LH2': 'LH2',
            'H': 'LH2',
            'Liquid Hydrogen Target': 'LH2',
            'LD2': 'LD2',
            'D2': 'LD2',
            'Liquid Deuterium Target': 'LD2',
            'C': 'C',
            'C12': 'C',
            '12C': 'C',
            'Carbon target 2 mm': 'C',
        },
        'density': {     # Units: g/cm^2
            # 'norm': 0.335,
            'C': 0.440,
            'empty': 0,
        },
        'current': {  # Nominal current in nA.  If 0, no expected charge line will be drawn.
            # list of currents for each beam energy period.
            'scale': [10., 1, 1, 1],     # Special entry. Multiply sum charge by this factor,
            'empty': [10., 20., 40., 20.],
            'C': [2.5, 5., 10., 5.],
        },
        'attenuation': {     # Units: number
            'empty': 1,
            'C': 1,
        },
        'color': {  # Plot color: r,g,b,a
            'empty': 'rgba(160, 110, 110, 0.7)',
            'C': 'rgba(120, 120, 200, 0.7)',
            'CH2': 'rgba(100, 255, 255, 0.7)',
            'calibration': 'rgba(220,220,220,0.5)',
        },
        'sums_color': {  # Plot color: r,g,b,a
            'empty': 'rgba(150, 90, 90, 0.8)',
            'C': 'rgba(100, 100, 180, 0.8)',
            'CH2': 'rgba(80, 200, 200, 0.8)',
            'expected': 'rgba(0, 0, 0, 0.7)',
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
    runs.loc[(runs.target_polarization == "None"), 'target_polarization'] = 0.   # Fix RCDB unpleasantries.

    runs["hover"] = [f"Run: {r}<br />"
                     f"Trigger:{runs.loc[r, 'run_config']}<br />"
                     f"Start: {runs.loc[r, 'start_time']}<br />"
                     f"End: {runs.loc[r, 'end_time']}<br />"
                     f"DT:   {runs.loc[r, 'dt'] / 1000.:5.1f} s<br />"
                     f"NEvt: {runs.loc[r, 'event_count']:10,d}<br />"
                     f"Target Pol: {runs.loc[r, 'target_polarization']}<br />"
                     f"Half W Plate: {runs.loc[r, 'half_wave_plate']} <br />"
                     f"Charge: {runs.loc[r, 'charge']:6.2f} mC <br />"
                     f"Lumi: {runs.loc[r, 'luminosity']:6.2f} 1/fb<br />"
                     f"<Rate>:{runs.loc[r, 'event_rate']:6.2f}kHz<br />"
                     for r in runs.index]

    return runs


def used_triggers():
    """Setup the triggers used."""
    good_triggers = '.*'
    calibration_triggers = ['rgc_300MeV_v1.2_zero.cnf', 'rgc_300MeV_v1.3_zero.cnf', 'rgc_300MeV_v1.4_zero.cnf',
                            'rgc_300MeV_v1.5_zero.cnf', 'rgc_300MeV_v1.6_zero.cnf']

    return good_triggers, calibration_triggers


def setup_rundata_structures(data_loc, dates):
    """Setup the data structures for parsing the databases."""

    start_time, end_time = dates
    data_loc.Good_triggers, data_loc.Calibration_triggers = used_triggers()

    data_loc.Production_run_type = "PROD.*"
    data_loc.target_properties = target_properties()
    data_loc.target_dens = data_loc.target_properties['density']
    data_loc.atten_dict = None
    data_loc.Current_Channel =  "scaler_calc1b" # "IPM2C21A"
    data_loc.LiveTime_Channel = "B_DAQ:livetime"
    data_loc.Useful_conditions.append('beam_energy')

    min_event_count = 500000  # Runs with at least 500k events.

    end_time = end_time + timedelta(0, 0, -end_time.microsecond)  # Round down on end_time to a second
    print("Fetching the data from {} to {}".format(start_time, end_time))
    data_loc.get_runs(start_time, end_time, min_event_count)
    data_loc.select_good_runs()


dat = None


def main(argv=None):
    import argparse
    global dat
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

    parser.add_argument('-d', '--debug', action="count", help="Be more verbose if possible. ", default=0)
    parser.add_argument('-N', '--nocache', action="store_true", help="Do not use a sqlite3 cache")
    parser.add_argument('-p', '--plot', action="store_true", help="Create the plotly plots.")
    parser.add_argument('-l', '--live', action="store_true", help="Show the live plotly plot.")
    parser.add_argument('-e', '--excel', action="store_true", help="Create the Excel table of the data")
    parser.add_argument('-c', '--charge', action="store_true", help="Make a plot of charge not luminosity.")
    parser.add_argument('-C', '--chart', action="store_true", help="Put plot on plotly charts website.")
    parser.add_argument('-f', '--date_from', type=str, help="Plot from date, eg '2021,11,09' ", default=None)
    parser.add_argument('-t', '--date_to', type=str, help="Plot to date, eg '2022,01,22' ", default=None)
    parser.add_argument('-m', '--max_rate', type=float, help="Maximum for date rate axis ", default=None)
    parser.add_argument('-M', '--max_charge', type=float, help="Maximum for charge axis ", default=None)
    parser.add_argument('-r', '--runperiod', type=int,
                        help="Run sub-period, 0=all, 1, 2, 3, 4, or 12 for 1+2", default=4),
    parser.add_argument('--return_data', action="store_true", help="Internal use: return the data at end.", default=None)

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
        data = RunData(cache_file="RGC_2022.sqlite3", i_am_at_jlab=at_jlab)
    else:
        data = RunData(cache_file="", sqlcache=False, i_am_at_jlab=at_jlab)
    # data._cache_engine=None   # Turn OFF cache?
    data.debug = args.debug

    run_sub_periods = [(datetime(2023, 9, 18,  0, 0),  datetime.now())
                       ]

    run_sub_energy = [99.]
    run_sub_y_placement = [0.90]

    run_period_name = ""
    run_period_sub_num = range(len(run_sub_periods))
    if args.runperiod == 0:
        run_period_name = "_all"
    elif args.runperiod <= len(run_sub_periods):
        run_period_sub_num = [args.runperiod-1]
        if args.runperiod == len(run_sub_periods):  # Last period == current period, no postfix.
            run_period_name = ""
        else:
            run_period_name = "_r"+str(args.runperiod)
    else:
        print("Incorrect choice for runperiod argument.")
        return

    if args.plot:
        max_y_value_sums = 0.
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        max_expected_charge = []

    if args.excel:
        excel_output = pd.DataFrame()

    legends_data = []  # To keep track of which targets already have a legend shown.
    legends_shown = []  # To keep track of which target has the charge sum legend shown.
    #   Loop over the different run sub-periods.

    for sub_i in run_period_sub_num:

        data.clear()
        setup_rundata_structures(data, run_sub_periods[sub_i])
        if data.All_Runs is None:
            return

        if "luminosity" in data.All_Runs.keys():
            data.All_Runs['luminosity'] *= 1E-3   # Rescale luminosity from 1/pb to 1/fb

        targets = '.*'

        # Select runs into the different categories.
        if data.All_Runs is not None:
            plot_runs = compute_plot_runs(targets=targets, run_config=data.Good_triggers, data_loc=data)
        else:
            plot_runs = pd.DataFrame()

        calib_run_numbers = data.list_selected_runs(targets='.*', run_config=data.Calibration_triggers)

        calib_runs = plot_runs.loc[calib_run_numbers]
    #    calib_starts = calib_runs["start_time"]
    #    calib_ends = calib_runs["end_time"]

        plot_runs = plot_runs.loc[~plot_runs.index.isin(calib_run_numbers)]  # Take the calibration runs out.

        starts = plot_runs["start_time"]
        ends = plot_runs["end_time"]

        # if args.debug:
        #     print("Calibration runs: ", calib_runs)

        print(f"Compute cumulative charge for period {sub_i+1}.")

        if "charge" not in plot_runs.keys():
            print(f"No charge data available for run period {sub_i+1}, skipping period.")
            continue
        data.compute_cumulative_charge(targets, runs=plot_runs)

        if args.excel:
            excel_output = pd.concat([excel_output, plot_runs, calib_runs]).sort_index()

        if args.plot:
            print(f"Build Plots for period {sub_i+1}")

            last_targ = None
            for targ in data.target_properties['attenuation']:

                if args.debug:
                    print(f"Processing plot for target {targ}")
                runs = plot_runs.target.str.fullmatch(targ.replace('(', r"\(").replace(')', r'\)'))


                if np.count_nonzero(runs) > 1 and sub_i == len(run_period_sub_num) - 1 and \
                        targ in data.target_properties['sums_color']:
                    last_targ = targ  # Store the last target name with data for later use.

                if targ in legends_data or np.count_nonzero(runs) < 1:  # Do we show this legend?
                    show_data_legend = False  # This avoids the same legend shown twice.
                else:
                    show_data_legend = True
                    legends_data.append(targ)

                data_x = plot_runs.loc[runs, 'center']
                data_y = plot_runs.loc[runs, 'event_rate']
                data_width = plot_runs.loc[runs, 'dt']
                data_hover = plot_runs.loc[runs, 'hover']
                data_color = data.target_properties['color'][targ]

                fig.add_trace(
                    go.Bar(x=data_x,
                           y=data_y,
                           width=data_width,
                           hovertext=data_hover,
                           name="run with " + targ,
                           marker=dict(color=data_color),
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
                        marker=dict(color=data.target_properties['color']['calibration']),
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

                        plot_sumcharge_target_t = [st.iloc[0], en.iloc[0]]
                        plot_sumcharge_target_v = [0, sumch.iloc[0]]
                        if data.target_properties['current'][targ][sub_i] > 0.:
                            plot_expected_charge_t = [st.iloc[0]]
                            plot_expected_charge_v = [0]

                        for i in range(1, len(sumch)):
                            # Detect if there is a run number gap and also
                            # check if all the intermediate runs have the same target. If so, these were calibration
                            # or junk runs and we continue normally. If there were other targets, we make a break in the
                            # line.
                            if sumch.keys()[i] - sumch.keys()[i - 1] > 1 and \
                                    not np.all(data.All_Runs.loc[sumch.keys()[i-1]:sumch.keys()[i]].target == targ):
                                plot_sumcharge_target_t.append(st.iloc[i])
                                plot_sumcharge_target_v.append(None)        # None causes line break.

                                fig.add_trace(
                                    go.Scatter(x=[en.iloc[i-1], st.iloc[i]],
                                               y=[plot_sumcharge_target_v[-2], plot_sumcharge_target_v[-2]],
                                               mode="lines",
                                               line=dict(color=data.target_properties['sums_color'][targ], width=1,
                                                         dash="dot"),
                                               name=f"Continuation line {targ}",
                                               showlegend=False),
                                    secondary_y=True)

                                if data.target_properties['current'][targ][sub_i] > 0.:
                                    plot_expected_charge_t.append(en.iloc[i-1])
                                    current_expected_sum_charge = plot_expected_charge_v[-1]
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
                                                       line=dict(color=data.target_properties['sums_color']['expected'],
                                                                 width=1, dash="dot"),
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

                # The lines below would draw a horizontal line to the end of the graph for each target.
                #                plot_sumcharge_target_t[t].append(ends.iloc[-1])
                #                plot_sumcharge_target_v[t].append(sumch.iloc[-1])

                # The lines below would add a curve for the sum total charge, all targets added together.
                # fig.add_trace(
                #     go.Scatter(x=plot_sumcharge_t,
                #                y=plot_sumcharge_v,
                #                line=dict(color='#F05000', width=3),
                #                name="Total Charge Live"),
                #     secondary_y=True)

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
                        # Decorative: add a box with an annotation of the total charge on this target.
                        actual_charge = plot_sumcharge_target_v[-1]/current_plotting_scale

                        fig.add_trace(
                            go.Scatter(x=[plot_sumcharge_target_t[-1]],
                                       y=[plot_sumcharge_target_v[-1]],
                                       mode="markers+text",
                                       marker=dict(color=data.target_properties['sums_color'][targ],
                                                   size=6),
                                       # text=f"<b>total: {actual_charge:4.2f} mC</b>",
                                       # textfont=dict(
                                       #      family="Arial, sans-serif",
                                       #      color=data.target_properties['sums_color'][targ],
                                       #      size=16),
                                       # textposition="top center",
                                       showlegend=False),
                            secondary_y=True)

                        annotation_text = f"<b>total: {actual_charge:4.2f} mC</b>"
                        text_extra_y_shift = 0
                        if targ in ["NH3", "ND3"]:
                            annotation_y_offset = max_y_value_sums*0.03
                            text_extra_y_shift = 12
                            charge_up = plot_runs.loc[
                                (plot_runs.target == targ) & (plot_runs.target_polarization >= 0.)].charge.sum()
                            charge_down = plot_runs.loc[
                                (plot_runs.target == targ) & (plot_runs.target_polarization < 0.)].charge.sum()
#                            annotation_text += f"<br />⬆︎ {charge_up:5.3f} mC ⬇︎ {charge_down:5.3f} mC"
                            fig.add_annotation(
                                x=plot_sumcharge_target_t[-1],
                                y=plot_sumcharge_target_v[-1],
                                xref="x",
                                yref="y2",
                                yanchor="bottom",
                                yshift=6,
                                text=f"⬆︎ {charge_up:5.3f} mC ⬇︎ {charge_down:5.3f} mC",
                                showarrow=False,
                                font=dict(
                                    family="Arial, sans-serif",
                                    color=data.target_properties['sums_color'][targ],
                                    size=10),
                                bgcolor="rgba(255,255,255,0.7)"
                            )

                        fig.add_annotation(
                            x=plot_sumcharge_target_t[-1],
                            y=plot_sumcharge_target_v[-1],
                            xref="x",
                            yref="y2",
                            yanchor="bottom",
                            yshift=6 + text_extra_y_shift,
                            text=annotation_text,
                            showarrow=False,
                            font=dict(
                                family="Arial, sans-serif",
                                color=data.target_properties['sums_color'][targ],
                                size=16),
                            bgcolor="rgba(255,255,255,0.6)"
                        )

                        # # Annotate - add a curve for the expected charge at 50% efficiency.
                        showlegend = True if targ == last_targ and sub_i == len(run_period_sub_num)-1 else False
                        if args.debug:
                            print(f"last_targ = {last_targ}  targ: {targ}, sub_i = {sub_i}, showlegend = {showlegend}")
                        if data.target_properties['current'][targ][sub_i] > 0.:
                            fig.add_trace(
                                go.Scatter(x=plot_expected_charge_t,
                                           y=plot_expected_charge_v,
                                           mode='lines',
                                           line=dict(color=data.target_properties['sums_color']['expected'], width=2),
                                           name=f"Expected charge at 50% up",
                                           showlegend=showlegend,
                                           # Only one legend at the end.
                                           legendgroup="group2",
                                           ),
                                secondary_y=True
                            )
                            max_expected_charge.append(plot_expected_charge_v[-1])
                            # print(f"max_expected_charge = {max_expected_charge}")

            run_sub_annotation = f"<b>E<sub>b</sub> = {run_sub_energy[sub_i]} GeV</b><br>"
            if data.target_properties['current']['scale'][sub_i] != 1.:
                run_sub_annotation += f"Current scaled {data.target_properties['current']['scale'][sub_i]:3.0f}x"

            mid_time = run_sub_periods[sub_i][0] + (run_sub_periods[sub_i][1] - run_sub_periods[sub_i][0])/2
            fig.add_annotation(
                x=mid_time,
                xanchor="center",
                xref="x",
                y=run_sub_y_placement[sub_i],
                yanchor="middle",
                yref="paper",
                text=run_sub_annotation,
                showarrow=False,
                font=dict(
                    family="Times",
                    color="#FF0000",
                    size=20),
                bgcolor="#FFEEEE"
            )

    # End sub run period loop.
    if args.plot:
        # Set x-axis title
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

        if args.runperiod == 4:
            title = "<b>RGC 23 FTon Progress</b>"
        elif args.runperiod == 3:
            title = "<b>RGC 22 FToff Progress</b>"
        elif args.runperiod == 0:
            title = "<b>RGC 22 FTon Progress</b>"
        else:
            title = "<b>RGC 22 FTon Progress</b>"
        fig.update_layout(
            title=go.layout.Title(
                text=title,
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=0.40),
            titlefont=dict(size=24, color='rgba(0,0,100,1.)'),
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

        if args.max_rate is not None and args.max_rate > 0:
            max_rate = args.max_rate
        else:
            all_runs = (plot_runs.selected == True)
            max_rate = 1.05*plot_runs.loc[all_runs, 'event_rate'].max()

        # Set y-axes titles
        fig.update_yaxes(
            title_text="<b>Event rate kHz</b>",
            titlefont=dict(size=22),
            secondary_y=False,
            tickfont=dict(size=18),
            # range=[0, 1.05*max(25., plot_runs.loc[runs, 'event_rate'].max())]
            range=[0, max_rate]
        )

        if args.charge:
            max_expected_charge.append(max_y_value_sums)
            if args.debug:
              print(f"max_expected_charge: {max_expected_charge}")
            max_y_2nd_scale = 1.1*np.max(max_expected_charge)
            # max_y_2nd_scale = 7.
            if args.debug:
                print(f"max_y_2nd_scale = {max_y_2nd_scale}")

            if args.max_charge is not None and args.max_charge > 0.:
                max_charge = args.max_charge
            else:
                max_charge = max_y_2nd_scale
            fig.update_yaxes(title_text="<b>Accumulated Charge (mC)</b>",
                             titlefont=dict(size=22),
                             range=[0, max_charge],
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
        fig.write_image("RGC2022_progress"+run_period_name+".pdf", width=2048, height=900)
        fig.write_image("RGC2022_progress"+run_period_name+".png", width=2048, height=900)
        fig.write_html("RGC2022_progress"+run_period_name+".html")
        if args.chart:
            charts.plot(fig, filename='RGC_edit', width=2048, height=900, auto_open=True)
        if args.live:
            fig.show(width=2048, height=900)  # width=1024,height=768

    if args.excel:
        print("Write new Excel table.")
        excel_output.to_excel("RGC2022_progress"+run_period_name+".xlsx",
                              columns=['start_time', 'end_time', 'target', 'target_polarization', 'beam_energy',
                                       'half_wave_plate', 'run_config', 'selected',
                                       'event_count', 'sum_event_count', 'charge', 'sum_charge', 'sum_charge_targ',
                                       'evio_files_count', 'megabyte_count',
                                       'operators', 'user_comment'])

    if args.return_data:
        return data, plot_runs


if __name__ == "__main__":
    sys.exit(main())
else:
    print("RGC setup.")
    # arguments = "--return_data -d -d -d "  # No plot or excel.
    # print("data, plot_runs, wave_plate, polarization = main(arguments)")
