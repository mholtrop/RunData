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


def rgm_2021_target_properties():
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
            'He': 'L4He',
            'L4He': 'L4He',
            'Liquid 4He target': 'L4He',
            '40Ca': '40Ca',
            '48Ca': '48Ca',
            'C': 'C',
            'Carbon target 2 mm': 'C',
            'C (x4)': 'C (x4)',
            'Carbon target 2 mm (4x)': 'C (4x)',
            'Sn (x4)': 'Sn (x4)',
            '300 um Sn': 'Sn (x4)',
            'LAr': 'LAr',
            'Ar': 'LAr',
            'Liquid Argon': 'LAr'
        },
        'density': {     # Units: g/cm^2
            'empty': 0,
            # 'norm': 0.335,
            'LH2': 0.335,
            'LD2': 0.820,
            'L4He': 0.625,
            '40Ca': 0.310,
            '48Ca': 0.310,
            'C': 0.440,
            'C (x4)': 0.440,
            'Sn (x4)': 0.205,
            'LAr': 0.698
        },
        'attenuation': {     # Units: number
            'empty': 1,
            'LH2':  1,
            'LD2': 1,
            'L4He': 1,
            '40Ca': 1,
            '48Ca': 1,
            'C': 1,
            'C (x4)': 1,
            'Sn (x4)': 1,
            'LAr': 1
        },
        'color': {  # Plot color: r,g,b,a
            'empty': 'rgba(220, 220, 220, 0.8)',
            'LH2':  'rgba(0, 120, 150, 0.8)',
            'LD2': 'rgba(20, 80, 255, 0.8)',
            'L4He': 'rgba(120, 120, 80, 0.8)',
            '40Ca': 'rgba(0, 80, 0, 0.8)',
            '48Ca': 'rgba(0, 150, 0, 0.8)',
            'C': 'rgba(120, 120, 200, 0.8)',
            'C(x4)': 'rgba(120, 120, 120, 0.8)',
            'Sn (x4)': 'rgba(120, 0, 200, 0.8)',
            'LAr': 'rgba(120, 120, 0, 0.8)'
        },
        'color_charge': {  # Plot color: r,g,b,a
            'empty': 'rgba(255, 200, 200, 0.8)',
            'LH2': 'rgba(255, 120, 150, 0.8)',
            'LD2': 'rgba(255, 80, 255, 0.8)',
            'L4He': 'rgba(255, 120, 80, 0.8)',
            '40Ca': 'rgba(255, 80, 0, 0.8)',
            '48Ca': 'rgba(255, 150, 0, 0.8)',
            'C': 'rgba(255, 120, 200, 0.8)',
            'C(x4)': 'rgba(255, 120, 120, 0.8)',
            'Sn (x4)': 'rgba(255, 0, 200, 0.8)',
            'LAr': 'rgba(255, 120, 0, 0.8)'
        },

    }

    return target_props


def compute_plot_runs(targets, run_config, date_min=None, date_max=None, data=None):
    """This function selects the runs from data according to the target, run_configuration and date"""
    # print("Compute data for plots.")

    runs = data.All_Runs.loc[data.list_selected_runs(targets=targets, run_config=run_config,
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

    Good_triggers = '.*'
    Calibration_triggers = ['None', 'RNDM']

    return Good_triggers, Calibration_triggers

def setup_rundata_structures(data):
    """Setup the data structures for parsing the databases."""
    data.Good_triggers, data.Calibration_triggers = used_triggers()

    data.Production_run_type = ["PROD66", "PROD66_PIN"]
    data.target_properties = rgm_2021_target_properties()
    data.target_dens = data.target_properties['density']
    data.atten_dict = None
    data.Current_Channel = "IPM2C21A"  # "scaler_calc1b"
    data.LiveTime_Channel = "B_DAQ:livetime"

    min_event_count = 500000  # Runs with at least 200k events.
    start_time = datetime(2021, 11, 10, 8, 0)  # Start of run.
#    end_time = datetime(2022, 01, 31, 8, 11)
    end_time = datetime.now()
    end_time = end_time + timedelta(0, 0, -end_time.microsecond)  # Round down on end_time to a second
    print("Fetching the data from {} to {}".format(start_time, end_time))
    data.get_runs(start_time, end_time, min_event_count)
    data.select_good_runs()


def main(argv=None):
    import argparse

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

    args = parser.parse_args(argv[1:])

    hostname = os.uname()[1]
    if hostname.find('clon') >= 0 or hostname.find('ifarm') >= 0 or hostname.find('jlab.org') >= 0:
        #
        # For JLAB setup the place we can find the RCDB
        #
        at_jlab = True
    else:
        at_jlab = False

    data = None
    if not args.nocache:
        data = RunData(cache_file="RGM_2021.sqlite3", i_am_at_jlab=at_jlab)
    else:
        data = RunData(cache_file="", sqlcache=False, i_am_at_jlab=at_jlab)
    # data._cache_engine=None   # Turn OFF cache?
    data.debug = args.debug
    setup_rundata_structures(data)
    data.All_Runs['luminosity'] *= 1E-3   # Rescale luminosity from 1/pb to 1/fb

    #    data.add_current_data_to_runs()
    targets = '.*'

    # Select runs into the different categories.
    plot_runs = compute_plot_runs(targets=targets, run_config=data.Good_triggers, data=data)

    calib_run_numbers = data.list_selected_runs(targets='.*', run_config=data.Calibration_triggers)

    calib_runs = plot_runs.loc[calib_run_numbers]
    calib_starts = calib_runs["start_time"]
    calib_ends = calib_runs["end_time"]

    plot_runs = plot_runs.loc[~plot_runs.index.isin(calib_run_numbers)]  # Take the calibration runs out.
    plot_runs = plot_runs.loc[~plot_runs.index.isin([1506, 1507, 1508, 1509])]  # Take error numbers out.
    starts = plot_runs["start_time"]
    ends = plot_runs["end_time"]

    # if args.debug:
    #     print("Calibration runs: ", calib_runs)

    print("Compute cumulative charge.")
    data.compute_cumulative_charge(targets, runs=plot_runs)

    if args.excel:
        print("Write new Excel table.")
        data.All_Runs.to_excel("HPSRun2021_progress.xlsx",
                               columns=['start_time', 'end_time', 'target', 'run_config', 'selected', 'event_count',
                                        'sum_event_count', 'charge', 'sum_charge', 'luminosity', 'sum_lumi',
                                        'operators', 'user_comment'])

    #    print(data.All_Runs.to_string(columns=['start_time','end_time','target','run_config','selected','event_count','charge','user_comment']))
    #    data.All_Runs.to_latex("hps_run_table.latex",columns=['start_time','end_time','target','run_config','selected','event_count','charge','operators','user_comment'])

    if args.plot:

        print("Build Plots.")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for targ in data.target_properties['color']:
            if args.debug:
                print(f"Processing plot for target {targ}")
            runs = plot_runs.target.str.contains(targ)
            fig.add_trace(
                go.Bar(x=plot_runs.loc[runs, 'center'],
                       y=plot_runs.loc[runs, 'event_rate'],
                       width=plot_runs.loc[runs, 'dt'],
                       hovertext=plot_runs.loc[runs, 'hover'],
                       name="run with " + targ,
                       marker=dict(color=data.target_properties['color'][targ])
                       ),
                secondary_y=False, )

        fig.add_trace(
             go.Bar(x=calib_runs['center'],
                    y=calib_runs['event_rate'],
                    width=calib_runs['dt'],
                    hovertext=calib_runs['hover'],
                    name="Calibration runs",
                    marker=dict(color='rgba(150,150,150,0.5)')
                    ),
             secondary_y=False, )

        if args.charge:
            sumcharge = plot_runs.loc[:, "sum_charge"]

            plot_sumcharge_t = [starts.iloc[0], ends.iloc[0]]
            plot_sumcharge_v = [0, sumcharge.iloc[0]]


            for i in range(1, len(sumcharge)):
                plot_sumcharge_t.append(starts.iloc[i])
                plot_sumcharge_t.append(ends.iloc[i])
                plot_sumcharge_v.append(sumcharge.iloc[i - 1])
                plot_sumcharge_v.append(sumcharge.iloc[i])

            # sumcharge_norm = plot_runs.loc[:, "sum_charge_norm"]
            # plot_sumcharge_norm_t = [starts.iloc[0], ends.iloc[0]]
            # plot_sumcharge_norm_v = [0, sumcharge_norm.iloc[0]]
            #
            # for i in range(1, len(sumcharge_norm)):
            #     plot_sumcharge_norm_t.append(starts.iloc[i])
            #     plot_sumcharge_norm_t.append(ends.iloc[i])
            #     plot_sumcharge_norm_v.append(sumcharge_norm.iloc[i - 1])
            #     plot_sumcharge_norm_v.append(sumcharge_norm.iloc[i])

            plot_sumcharge_target_t = {}
            plot_sumcharge_target_v = {}

            for t in data.target_dens:
                sumch = plot_runs.loc[plot_runs["target"] == t, "sum_charge_targ"]
                st = plot_runs.loc[plot_runs["target"] == t, "start_time"]
                en = plot_runs.loc[plot_runs["target"] == t, "end_time"]

                if len(sumch > 3):
                    plot_sumcharge_target_t[t] = [st.iloc[0], en.iloc[0]]
                    plot_sumcharge_target_v[t] = [0, sumch.iloc[0]]
                    for i in range(1, len(sumch)):
                        plot_sumcharge_target_t[t].append(st.iloc[i])
                        plot_sumcharge_target_t[t].append(en.iloc[i])
                        plot_sumcharge_target_v[t].append(sumch.iloc[i - 1])
                        plot_sumcharge_target_v[t].append(sumch.iloc[i])
            #                plot_sumcharge_target_t[t].append(ends.iloc[-1])
            #                plot_sumcharge_target_v[t].append(sumch.iloc[-1])

            # fig.add_trace(
            #     go.Scatter(x=plot_sumcharge_t,
            #                y=plot_sumcharge_v,
            #                line=dict(color='#F05000', width=3),
            #                name="Total Charge Live"),
            #     secondary_y=True)

            for targ in data.target_dens:
                if targ in plot_sumcharge_target_v:
                    fig.add_trace(
                        go.Scatter(x=plot_sumcharge_target_t[targ],
                                   y=plot_sumcharge_target_v[targ],
                                   line=dict(color=data.target_properties['color_charge'][targ], width=3),
                                   name=f"Total Charge on {targ}"),
                        secondary_y=True)



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

            for targ in data.target_dens:
                selected = plot_runs.target == targ
                plot_sumlumi_target = np.cumsum(plot_runs.loc[selected, "luminosity"])
                plot_sumlumi_starts = plot_runs.loc[plot_runs["target"] == targ, "start_time"]
                plot_sumlumi_ends = plot_runs.loc[plot_runs["target"] == targ, "end_time"]
                if len(plot_sumlumi_target) > 1:
                    plot_sumlumi_target_t = [plot_sumlumi_starts.iloc[0], plot_sumlumi_ends.iloc[0]]
                    plot_sumlumi_target_v = [0, plot_sumlumi_target.iloc[0]]
                    for i in range(1, len(plot_sumlumi_target)):
                        plot_sumlumi_target_t.append(plot_sumlumi_starts.iloc[i])
                        plot_sumlumi_target_t.append(plot_sumlumi_ends.iloc[i])
                        plot_sumlumi_target_v.append(plot_sumlumi_target.iloc[i - 1])
                        plot_sumlumi_target_v.append(plot_sumlumi_target.iloc[i])

                    fig.add_trace(
                        go.Scatter(x=plot_sumlumi_target_t,
                                   y=plot_sumlumi_target_v,
                                   line=dict(color=data.target_properties['color_charge'][targ], width=3),
                                   name=f"Sum luminosity on {targ}"),
                        secondary_y=True)

                    fig.add_trace(
                        go.Scatter(x=[plot_sumlumi_target_t[-1]],
                                   y=[plot_sumlumi_target_v[-1]],
                                   marker=dict(
                                       color=data.target_properties['color_charge'][targ],
                                       size=6
                                       ),
                                   showlegend=False
                                   ),
                        secondary_y=True
                    )


                    fig.add_annotation(
                        x=plot_sumlumi_target_t[-1],
                        y=plot_sumlumi_target_v[-1]+0.25,
                        xref="x",
                        yref="y2",
                        text=f"<b>total: {plot_sumlumi_target_v[-1]:3.2f} 1/fb</b>",
                        showarrow=False,
                        font=dict(
                            family="Arial, sans-serif",
                            color=data.target_properties['color_charge'][targ],
                            size=16),
                        bgcolor="#FFFFFF"
                    )


        # Set x-axis title
        fig.update_layout(
            title=go.layout.Title(
                text="RGM 2021 Progress",
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
            fig.update_yaxes(title_text="<b>Accumulated Charge (mC)</b>",
                             titlefont=dict(size=22),
                             range=[0, 5],  # proposed_charge
                             secondary_y=True,
                             tickfont=dict(size=18)
                             )
        else:
            fig.update_yaxes(title_text="<b>Integrated Luminosity (1/fb)</b>",
                             titlefont=dict(size=22),
                             range=[0, 1.05*np.max(plot_sumlumi_target_v)],  # proposed_lumi[-1]
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
        fig.write_image("RGM2021_progress.pdf", width=2048, height=900)
        fig.write_image("RGM2021_progress.png", width=2048, height=900)
        fig.write_html("RGM2021_progress.html")
        if args.chart:
            charts.plot(fig, filename='RGM2021_edit', width=2048, height=900, auto_open=True)
        if args.live:
            fig.show(width=2048, height=900)  # width=1024,height=768


if __name__ == "__main__":
    sys.exit(main())
else:
    print("Imported the RGM2021 info. Setting up data.")
    data = RunData(cache_file="RGM_2021.sqlite3", sqlcache=True, i_am_at_jlab=False) #
    setup_rundata_structures(data)
