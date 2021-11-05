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


def rgm_2021_targets():
    """ Returns the dictionary of target name to target thickness.
        One extra entry, named 'norm' is used for normalization of the charge curve.
        Target thickness is in units of cm."""
    targets = {
        'norm': (20.e-4, 'rgba(255,100,255,0.8)'),      # Value to normalize to.
        '8 um W ': (8.e-4, 'rgba(20,80,255,0.8)'),
        '20 um W ': (20.e-4, 'rgba(0,120,150,0.8)')
    }

    return targets


def attennuations_with_targ_thickness():
    """ During the run we have observed that the beam attenuation depends on the target thickness too.
    So this dictionary provides target<->attenuation dictionary """

    # From logbook: https://logbooks.jlab.org/entry/3900778
    # 0 um 36.556800
    # 8 um 32.860550
    # 20 um 27.330850
    #
    attenuations = {

        'Empty': 36.556800,
        'empty': 36.556800,
        'Unknown': 36.556800,
        '8 um W':  32.860550,
        '20 um W': 27.330850
    }

    return attenuations


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
                     f"Lumi: {runs.loc[r, 'luminosity']:6.2f} 1/pb<br />"
                     f"<Rate>:{runs.loc[r, 'event_rate']:6.2f}kHz<br />"
                     for r in runs.index]

    return runs, starts, ends


def main(argv=None):
    import argparse

    if argv is None:
        argv = sys.argv
    else:
        argv = argv.split()
        argv.insert(0, sys.argv[0])  # add the program name.

    # total_days_in_proposed_run - The calandar days (NOT PAC DAYS) this run was scheduled for.
    proposed_lumi_rate = 9.470415818323063e-05  # 1(pb s) = 0.09470415818323064 * 1/(nb s)
    total_days_in_proposed_run = 7*7
    total_proposed_luminosity = proposed_lumi_rate * total_days_in_proposed_run * 24 * 3600 / 2
    print(f"Total days expected in run= {total_days_in_proposed_run} = {total_days_in_proposed_run*24*3600} s")
    print(f"Total proposed luminosity = {total_proposed_luminosity}")

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
    parser.add_argument('-f', '--date_from', type=str, help="Plot from date, eg '2021,07,03' ", default=None)
    parser.add_argument('-t', '--date_to', type=str, help="Plot to date, eg '2022,11,22' ", default=None)

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

    # data.Good_triggers=['hps_v7.cnf','hps_v8.cnf','hps_v9.cnf','hps_v9_1.cnf',
    #                     'hps_v9_2.cnf','hps_v10.cnf',
    #                     'hps_v11_1.cnf','hps_v11_2.cnf','hps_v11_3.cnf','hps_v11_4.cnf',
    #                     'hps_v11_5.cnf','hps_v11_6.cnf','hps_v12_1.cnf']
    data.Good_triggers = '.*'
    data.Calibration_triggers = ' '

    data.Production_run_type = ["PROD77", "PROD77_PIN"]
    data.target_dict = rgm_2021_target_thickness()
    data.atten_dict = attennuations_with_targ_thickness()
    data.Current_Channel = "scaler_calc1b"

    min_event_count = 10000000  # Runs with at least 10M events.
    #    start_time = datatime(2019, 7, 17, 0, 0)  # Very start of run
    start_time = datetime(2021, 11, 9, 0, 0)  # DAQ Issues resolved.
    # end_time = datetime(2021, 9, 11, 0, 0)
    end_time = datetime.now()
    end_time = end_time + timedelta(0, 0, -end_time.microsecond)  # Round down on end_time to a second

    print("Fetching the data from {} to {}".format(start_time, end_time))
    data.get_runs(start_time, end_time, min_event_count)
    data.select_good_runs()
    #    data.add_current_data_to_runs()
    targets = '.*um W *'

    # Select runs into the different catagories.
    plot_runs, starts, ends = compute_plot_runs(targets=targets, run_config=data.Good_triggers,
                                                data=data)

    calib_runs, c_starts, c_ends = compute_plot_runs(targets=targets, run_config=data.Calibration_triggers, data=data)
    if args.debug:
        print("Calibration runs: ", calib_runs)

    print("Compute cumulative charge.")
    data.compute_cumulative_charge(targets, runs=plot_runs)  # Only the tungsten targets count.

    if args.excel:
        print("Write new Excel table.")
        data.All_Runs.to_excel("HPSRun2021_progress.xlsx",
                               columns=['start_time', 'end_time', 'target', 'run_config', 'selected', 'event_count',
                                        'sum_event_count', 'charge', 'sum_charge', 'luminosity', 'sum_lumi',
                                        'operators', 'user_comment'])

    #    print(data.All_Runs.to_string(columns=['start_time','end_time','target','run_config','selected','event_count','charge','user_comment']))
    #    data.All_Runs.to_latex("hps_run_table.latex",columns=['start_time','end_time','target','run_config','selected','event_count','charge','operators','user_comment'])

    if args.plot:
        sumcharge = plot_runs.loc[:, "sum_charge"]
        sumlumi = plot_runs.loc[:, "sum_lumi"]
        plot_sumcharge_t = [starts.iloc[0], ends.iloc[0]]
        plot_sumcharge_v = [0, sumcharge.iloc[0]]
        plot_sumlumi = [0, sumlumi.iloc[0]]

        for i in range(1, len(sumcharge)):
            plot_sumcharge_t.append(starts.iloc[i])
            plot_sumcharge_t.append(ends.iloc[i])
            plot_sumcharge_v.append(sumcharge.iloc[i - 1])
            plot_sumcharge_v.append(sumcharge.iloc[i])
            plot_sumlumi.append(sumlumi.iloc[i-1])
            plot_sumlumi.append(sumlumi.iloc[i])

        sumcharge_norm = plot_runs.loc[:, "sum_charge_norm"]
        plot_sumcharge_norm_t = [starts.iloc[0], ends.iloc[0]]
        plot_sumcharge_norm_v = [0, sumcharge_norm.iloc[0]]

        for i in range(1, len(sumcharge_norm)):
            plot_sumcharge_norm_t.append(starts.iloc[i])
            plot_sumcharge_norm_t.append(ends.iloc[i])
            plot_sumcharge_norm_v.append(sumcharge_norm.iloc[i - 1])
            plot_sumcharge_norm_v.append(sumcharge_norm.iloc[i])

        plot_sumcharge_target_t = {}
        plot_sumcharge_target_v = {}

        for t in data.target_dict:
            sumch = plot_runs.loc[plot_runs["target"] == t, "sum_charge_targ"]
            sumlum = plot_runs.loc[plot_runs["target"] == t, "sum_charge_targ"]
            st = plot_runs.loc[plot_runs["target"] == t, "start_time"]
            en = plot_runs.loc[plot_runs["target"] == t, "end_time"]

            if len(sumch > 3):
                plot_sumcharge_target_t[t] = [starts.iloc[0], st.iloc[0], en.iloc[0]]
                plot_sumcharge_target_v[t] = [0, 0, sumch.iloc[0]]
                for i in range(1, len(sumch)):
                    plot_sumcharge_target_t[t].append(st.iloc[i])
                    plot_sumcharge_target_t[t].append(en.iloc[i])
                    plot_sumcharge_target_v[t].append(sumch.iloc[i - 1])
                    plot_sumcharge_target_v[t].append(sumch.iloc[i])
                plot_sumcharge_target_t[t].append(ends.iloc[-1])
                plot_sumcharge_target_v[t].append(sumch.iloc[-1])

        print("Build Plots.")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for targ in targ_cols:
            runs = plot_runs.target.str.contains(targ)
            fig.add_trace(
                go.Bar(x=plot_runs.loc[runs, 'center'],
                       y=plot_runs.loc[runs, 'event_rate'],
                       width=plot_runs.loc[runs, 'dt'],
                       hovertext=plot_runs.loc[runs, 'hover'],
                       name="run with " + targ,
                       marker=dict(color=target_colors[targ])
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
            fig.add_trace(
                go.Scatter(x=plot_sumcharge_t,
                           y=plot_sumcharge_v,
                           line=dict(color='#B09090', width=3),
                           name="Total Charge Live"),
                secondary_y=True)

            fig.add_trace(
                go.Scatter(x=plot_sumcharge_norm_t,
                           y=plot_sumcharge_norm_v,
                           line=dict(color='red', width=3),
                           name="Tot Charge * targ thick/20 µm"),
                secondary_y=True)

            t = '8 um W '
            fig.add_trace(
                go.Scatter(x=plot_sumcharge_target_t[t],
                           y=plot_sumcharge_target_v[t],
                           line=dict(color='#990000', width=2),
                           name="Charge on 8 µm W"),
                secondary_y=True)

            t = '20 um W '
            fig.add_trace(
                go.Scatter(x=plot_sumcharge_target_t[t],
                           y=plot_sumcharge_target_v[t],
                           line=dict(color='#009940', width=3),
                           name="Charge on 20 µm W"),
                secondary_y=True)

            proposed_charge = (ends.iloc[-1] - starts.iloc[0]).total_seconds() * 120.e-6 * 0.5
            fig.add_trace(
                go.Scatter(x=[starts.iloc[0], ends.iloc[-1]],
                           y=[0, proposed_charge],
                           line=dict(color='#88FF99', width=2),
                           name="120nA on 20µm W 50% up"),
                secondary_y=True
            )

#################################################################################################################
#                     Luminosity
#################################################################################################################
        else:
            fig.add_trace(
                go.Scatter(x=plot_sumcharge_t,
                           y=plot_sumlumi,
                           line=dict(color='#FF3030', width=3),
                           name="Luminosity Live"),
                secondary_y=True)

            # starts_lumi = starts.copy()
            ends_lumi = ends.copy()
            end_time_proposed_run = starts.iloc[0] + timedelta(days=total_days_in_proposed_run)
            num_runs_before_eight_week_end = np.count_nonzero(ends_lumi < end_time_proposed_run)  # Drop the last run
            # print(f"Run end: {end_time_proposed_run} has {num_runs_before_eight_week_end} runs.")
            proposed_lumi = [0] + [(ends_lumi.iloc[i] - starts.iloc[0]).total_seconds() * proposed_lumi_rate * 0.5
                                   for i in range(num_runs_before_eight_week_end)]  # len(ends)

            # The last run completed proposed run time but kept going.
            if ends_lumi.iloc[num_runs_before_eight_week_end] > end_time_proposed_run:
                print(f"Fixing at index {num_runs_before_eight_week_end} of {len(ends_lumi)} ")
                # proposed_lumi[num_runs_before_eight_week_end] = total_proposed_luminosity
                ends_lumi = ends_lumi.append(ends_lumi.iloc[-1:])   # Duplicate last value
                ends_lumi.iloc[num_runs_before_eight_week_end] = end_time_proposed_run
                proposed_lumi += [total_proposed_luminosity]        # Add another value at the end.
                print(ends_lumi.iloc[-5:])

            if len(ends) > num_runs_before_eight_week_end:
                # Extend the curve for runs past the proposed end of run, i.e for the extension time.
                proposed_lumi += [total_proposed_luminosity for i in range(num_runs_before_eight_week_end, len(ends))]

            fig.add_trace(
                go.Scatter(x=[starts.iloc[0]] + [ends_lumi.iloc[i] for i in range(len(ends_lumi))],
                           y=proposed_lumi,
                           line=dict(color='#FFC030', width=3),
                           name="120nA on 20µm W 50% up"),
                secondary_y=True)

            fig.add_trace(
                go.Scatter(x=[ends.iloc[-1],ends.iloc[-1]],
                           y=[plot_sumlumi[-1],plot_sumlumi[-1]],
                           line=dict(color='#FF0000', width=1),
                           name=f"Int. Lumi. = {plot_sumlumi[-1]:4.1f} /pb = "
                                f"{100*plot_sumlumi[-1]/200.:3.1f}% of 200 1/pb."),
                secondary_y=True)

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
            range=[0, 35.]
        )

        if args.charge:
            fig.update_yaxes(title_text="<b>Accumulated Charge (mC)</b>",
                             titlefont=dict(size=22),
                             range=[0, max(proposed_charge, plot_sumcharge_v[-1])],
                             secondary_y=True,
                             tickfont=dict(size=18)
                             )
        else:
            fig.update_yaxes(title_text="<b>Integrated Luminosity (1/pb)</b>",
                             titlefont=dict(size=22),
                             range=[0, 1.05*max(proposed_lumi[-1], plot_sumlumi[-1])],
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
        fig.write_image("HPSRun2021_progress.pdf", width=2048, height=900)
        fig.write_image("HPSRun2021_progress.png", width=2048, height=900)
        fig.write_html("HPSRun2021_progress.html")
        if args.chart:
            charts.plot(fig, filename='Run2021_edit', width=2048, height=900, auto_open=True)
        if args.live:
            fig.show(width=2048, height=900)  # width=1024,height=768


if __name__ == "__main__":
    sys.exit(main())