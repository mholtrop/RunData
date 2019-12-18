#!/usr/bin/env python
from __future__ import print_function
import sys
import os
from datetime import datetime, timedelta

from RunData.RunData import RunData

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import chart_studio.plotly as charts

    pio.renderers.default = "browser"
except:
    print("Sorry, but to make the nice plots, you really need a computer with 'plotly' installed.")
    sys.exit(1)


def HPS_2019_Run_Target_Thickness():
    ''' Returns the dictionary of target name to target thickness.
        One extra entry, named 'norm' is used for filling the
        Target thickness is in units of cm.'''
    targets = {
        'norm': 8.e-4,
        '4 um W ': 4.e-4,
        '8 um W ': 8.e-4,
        '15 um W ': 15.e-4,
        '20 um W ': 20.e-4
    }
    return targets


def AttennuationsWithTargThickness():
    ''' During the run we have observed that the beam attenuation depends on the target thickness too.
    So this dictionary provides target<->attenuation dictionary '''

    Attenuations = {
        'Empty': 29.24555,
        'empty': 29.24555,
        'Unknown': 29.24555,
        '4 um W': 28.40418,
        '8 um W': 27.56255,
        '15 um W': 26.16205,
        '20 um W': 25.38535
    }

    return Attenuations


if __name__ == "__main__":

    hostname = os.uname()[1]
    if hostname.find('clon') >= 0 or hostname.find('ifarm') >= 0 or hostname.find('jlab.org') >= 0:
        #
        # For JLAB setup the place we can find the RCDB
        #
        at_jlab = True

    data = RunData()
    # data._cache_engine=None   # Turn OFF cache?
    data.debug = 4

    # data.Good_triggers=['hps_v7.cnf','hps_v8.cnf','hps_v9.cnf','hps_v9_1.cnf',
    #                     'hps_v9_2.cnf','hps_v10.cnf',
    #                     'hps_v11_1.cnf','hps_v11_2.cnf','hps_v11_3.cnf','hps_v11_4.cnf',
    #                     'hps_v11_5.cnf','hps_v11_6.cnf','hps_v12_1.cnf']
    data.Good_triggers = 'hps_v..?_?.?\.cnf'
    data.Production_run_type = ["PROD66", "PROD67"]
    data.target_dict = HPS_2019_Run_Target_Thickness()
    data.atten_dict  = AttennuationsWithTargThickness()

    min_event_count = 1000000  # Runs with at least 1M events.
    start_time = datetime(2019, 7, 25, 0, 0)  # SVT back in correct position
    end_time =   datetime(2019, 9, 10, 0, 0)
    end_time = end_time + timedelta(0, 0, -end_time.microsecond)  # Round down on end_time to a second

    print("Fetching the data from {} to {}".format(start_time, end_time))
    data.get_runs(start_time, end_time, min_event_count)
    data.select_good_runs()
    data.add_current_data_to_runs()
    targets = '.*um W *'
    print("Compute cumulative charge.")
    data.compute_cumulative_charge(targets)  # Only the tungsten targets count.
    print("Write new Excel table.")
    data.All_Runs.to_excel("hps_run_table.xlsx",
                           columns=['start_time', 'end_time', 'target', 'run_config', 'selected', 'event_count',
                                    'sum_event_count', 'charge', 'sum_charge', 'operators', 'user_comment'])

    #    print(data.All_Runs.to_string(columns=['start_time','end_time','target','run_config','selected','event_count','charge','user_comment']))
    #    data.All_Runs.to_latex("hps_run_table.latex",columns=['start_time','end_time','target','run_config','selected','event_count','charge','operators','user_comment'])

    print("Compute data for plots.")
    Plot_Runs = data.All_Runs.loc[data.list_selected_runs(targets=targets)]
    starts = Plot_Runs["start_time"]
    ends = Plot_Runs["end_time"]
    Plot_Runs["center"] = starts + (ends - starts) / 2
    Plot_Runs["dt"] = [(run["end_time"] - run["start_time"]).total_seconds() * 999 for num, run, in
                       Plot_Runs.iterrows()]
    Plot_Runs["hover"] = ["Run: {}<br />Start: {}<br />End: {}<br /><Rate>:{:6.2f}kHz<br />Trigger:{}".format(
        r, Plot_Runs.loc[r, "start_time"], Plot_Runs.loc[r, "end_time"],
        999 * Plot_Runs.loc[r, "event_count"] / 1e3 / Plot_Runs.loc[r, "dt"], Plot_Runs.loc[r, "run_config"]) for r in
        Plot_Runs.index]

    sumcharge = Plot_Runs.loc[:, "sum_charge"]
    plot_sumcharge_t = [starts.iloc[0], ends.iloc[0]]
    plot_sumcharge_v = [0, sumcharge.iloc[0]]

    for i in range(1, len(sumcharge)):
        plot_sumcharge_t.append(starts.iloc[i])
        plot_sumcharge_t.append(ends.iloc[i])
        plot_sumcharge_v.append(sumcharge.iloc[i - 1])
        plot_sumcharge_v.append(sumcharge.iloc[i])

    sumcharge_norm = Plot_Runs.loc[:, "sum_charge_norm"]
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
        sumch = Plot_Runs.loc[Plot_Runs["target"] == t, "sum_charge_targ"]
        st = Plot_Runs.loc[Plot_Runs["target"] == t, "start_time"]
        en = Plot_Runs.loc[Plot_Runs["target"] == t, "end_time"]

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

    targ_cols = {
        '4 um W ': 'rgba(255,100,255,0.8)',
        '8 um W ': 'rgba(20,80,255,0.8)',
        '15 um W ': 'rgba(0,255,255,0.8)',
        '20 um W ': 'rgba(0,120,150,0.8)'
    }

    for targ in targ_cols:
        runs = Plot_Runs.target.str.contains(targ)
        fig.add_trace(
            go.Bar(x=Plot_Runs.loc[runs, 'center'],
                   y=Plot_Runs.loc[runs, 'event_count'],
                   width=Plot_Runs.loc[runs, 'dt'],
                   hovertext=Plot_Runs.loc[runs, 'hover'],
                   name="#evt for " + targ,
                   marker=dict(color=targ_cols[targ])
                   ),
            secondary_y=False, )

    fig.add_trace(
        go.Scatter(x=plot_sumcharge_t, y=plot_sumcharge_v, line=dict(color='#B09090', width=3),
                   name="Total Charge Live"),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=plot_sumcharge_norm_t, y=plot_sumcharge_norm_v, line=dict(color='red', width=3),
                   name="Tot Charge * targ thick/8 µm"),
        secondary_y=True,
    )

    t = '8 um W '
    fig.add_trace(
        go.Scatter(x=plot_sumcharge_target_t[t], y=plot_sumcharge_target_v[t], line=dict(color='#990000', width=2),
                   name="Charge on 8 µm W"),
        secondary_y=True,
    )

    t = '20 um W '
    fig.add_trace(
        go.Scatter(x=plot_sumcharge_target_t[t], y=plot_sumcharge_target_v[t], line=dict(color='#009940', width=3),
                   name="Charge on 20 µm W"),
        secondary_y=True,
    )

    proposed_charge = (ends.iloc[-1] - starts.iloc[0]).total_seconds() * 150.e-6
    # fig.add_trace(
    #     go.Scatter(x=[starts.iloc[0],ends.iloc[-1]], y=[0,proposed_charge],line=dict(color='yellow', width=2),name="300nA in 8µm W 50% up"),
    #     secondary_y=True,
    # )
    fig.add_trace(
        go.Scatter(x=[starts.iloc[0], ends.iloc[-1]], y=[0, proposed_charge / 2], line=dict(color='#88FF99', width=2),
                   name="150nA on 8µm W 50% up"),
        secondary_y=True,
    )

    fig.update_layout(
        title=go.layout.Title(
            text="HPS Run 2019 Progress",
            yanchor="top",
            y=0.95,
            xanchor="center",
            x=0.5),
    )

    a_index = []
    a_x = []
    a_y = []
    a_text = []
    a_ax = []
    a_ay = []

    index = Plot_Runs.index[Plot_Runs.loc[:, "end_time"] > datetime(2019, 8, 2, 8, 25)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index, "end_time"])
    a_y.append(sumcharge.loc[index])
    a_text.append("Hall-A Wien Flip,<br />difficulty restoring beam.")
    a_ax.append(10)
    a_ay.append(-540)

    index = Plot_Runs.index[Plot_Runs.loc[:, "end_time"] > datetime(2019, 8, 4, 12, 11)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index, "end_time"])
    a_y.append(sumcharge.loc[index])
    a_text.append("DAQ problem,<br />followed by<br />beam restore issues.")
    a_ax.append(10)
    a_ay.append(-480)

    index = Plot_Runs.index[Plot_Runs.loc[:, "end_time"] > datetime(2019, 8, 6, 14, 52)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index, "end_time"])
    a_y.append(sumcharge.loc[index])
    a_text.append("Beam Halo,<br />retuning beam.")
    a_ax.append(10)
    a_ay.append(-590)

    index = Plot_Runs.index[Plot_Runs.loc[:, "end_time"] > datetime(2019, 8, 7, 14, 22)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index, "end_time"])
    a_y.append(sumcharge.loc[index])
    a_text.append("Thunder storm,<br />followed by retune<br />followed by DAQ issues.")
    a_ax.append(-10)
    a_ay.append(-630)

    index = Plot_Runs.index[Plot_Runs.loc[:, "end_time"] > datetime(2019, 8, 13, 4, 7)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index, "end_time"])
    a_y.append(sumcharge.loc[index])
    a_text.append(
        "Calibration run <br /> Target replacement during beam studies<br />SVT motor issues.<br />CHL Event and Beam Tuning")
    a_ax.append(40)
    a_ay.append(-440)

    index = Plot_Runs.index[Plot_Runs.loc[:, "end_time"] > datetime(2019, 8, 14, 1, 35)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index, "end_time"])
    a_y.append(sumcharge.loc[index])
    a_text.append("SVT motor issues")
    a_ax.append(0)
    a_ay.append(-390)

    index = Plot_Runs.index[Plot_Runs.loc[:, "end_time"] > datetime(2019, 8, 22, 4, 53)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index, "end_time"])
    a_y.append(sumcharge_norm.loc[index])
    a_text.append("Scheduled down<br />then beam tuning")
    a_ax.append(30)
    a_ay.append(-260)

    a_annot = []
    for i in range(len(a_x)):
        a_annot.append(
            go.layout.Annotation(
                x=a_x[i],
                y=a_y[i],
                xref="x",
                yref="y2",
                text=a_text[i],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#505050",
                ax=a_ax[i],
                ay=a_ay[i],
                font={
                    "family": "Times",
                    "size": 10,
                    "color": "#0040C0"
                }
            )
        )

    fig.update_layout(
        annotations=a_annot + []
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Time")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Number of events</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Accumulated Charge (mC)</b>", range=[0, proposed_charge], secondary_y=True)

    print("Show plots.")
    fig.write_image("HPSRun2019_progress.pdf", width=2000, height=1200)
    fig.write_image("HPSRun2019_progress.png")
    #    charts.plot(fig, filename = 'Run2019_edit', auto_open=True)
    fig.show(width=2048, height=900)  # width=1024,height=768
