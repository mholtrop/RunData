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

# Globals are actually a pain in the ass.
# data = None
# data.beam_stop_atten_time = None  # Persist in data, so we don't look it up all the time.
# data.fcup_offset_time = None


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
            # list of currents for each beam energy period.
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


def compute_plot_runs(targets, run_config, date_min=None, date_max=None, data=None):
    """This function selects the runs from data according to the target, run_configuration and date"""
    # print("Compute data for plots.")

    runs = data.All_Runs.loc[data.list_selected_runs(targets=targets, run_config=run_config,
                                                     date_min=date_min, date_max=date_max)]

    starts = runs["start_time"]
    ends = runs["end_time"]
    runs["center"] = starts + (ends - starts) / 2
    runs["dt"] = [(run["end_time"] - run["start_time"]).total_seconds() * 1000 for num, run, in runs.iterrows()]
    runs["event_rate"] = [runs.loc[r, 'event_count'] / runs.loc[r, 'dt'] if runs.loc[r, 'dt'] > 0 else 0
                          for r in runs.index]
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


def read_rafo_start_end_times():
    """Read the run start and end times that Rafo extracted from the data and return in a DataFrame."""
    rafo_times1 = pd.read_csv("RunSummaryinfo_F18_Inb_Earlyruns.dat",
                              names=["run_number", "charge", "t_start", "t_end"], index_col=0)
    rafo_times2 = pd.read_csv("RunSummaryinfo_F18inbend.dat", names=["run_number", "charge", "t_start", "t_end"],
                              index_col=0)
    rafo_times1["data_start"] = rafo_times1.t_start.map(datetime.fromtimestamp)
    rafo_times1["data_end"] = rafo_times1.t_end.map(datetime.fromtimestamp)
    rafo_times2["data_start"] = rafo_times2.t_start.map(datetime.fromtimestamp)
    rafo_times2["data_end"] = rafo_times2.t_end.map(datetime.fromtimestamp)
    rafo_times = pd.concat([rafo_times1, rafo_times2])
    return rafo_times


def setup_rundata_structures(data, runs=None):
    """Setup the data structures for parsing the databases."""

    data.Good_triggers, data.Calibration_triggers = used_triggers()
    data.Production_run_type = "PROD.*"  # ["PROD66", "PROD66_PIN", "PROD66_noVTPread", "PROD67_noVTPread"]
    data.target_properties = rga_2022_target_properties()
    data.target_dens = data.target_properties['density']
    data.atten_dict = None
    data.Current_Channel = "IPM2C21A"  # "scaler_calc1b"
    data.LiveTime_Channel = "B_DAQ:livetime_pulser"
    data.Useful_conditions.append('beam_energy')  # This run will have multiple beam energies.

    min_event_count = 0  # Get all the runs even with zero counts, because RCDB has bad entries.

    print(f"Getting runs: {runs[0]} to {runs[1]}")
    data.fix_bad_rcdb_start_times = True
    data.get_runs_from_rcdb_by_run_number(run_min=runs[0], run_max=runs[1], min_event_count=min_event_count)

    # We need to check the runs that have missing start_time or end_time.
    # If start or end time is missing, use the data tables from Rafo.

    print("Get the data based start times.")
    if not hasattr(data, "rafo_times"):
        data.rafo_times = read_rafo_start_end_times()

    # Add the rafo_times into the All_Runs dataframe
    data.All_Runs["data_start"] = data.rafo_times["data_start"]
    data.All_Runs["data_end"] = data.rafo_times["data_end"]
    data.All_Runs["data_charge"] = data.rafo_times["charge"]

    # if 5381 in data.All_Runs.index and 5382 in data.All_Runs.index:
    #     # This run seems to have a bad start time from the RCDB database. If so, we fix it here.
    #     if data.All_Runs.loc[5382].start_time < data.All_Runs.loc[5381].end_time:
    #         if data.All_Runs.loc[5382].data_start is not None:
    #             data.All_Runs.loc[5382, "start_time"] = data.All_Runs.loc[5381, "data_start"]
    #         else:
    #             data.All_Runs.loc[5382, "start_time"] = data.All_Runs.loc[5381, "end_time"] + timedelta(seconds=1)

    # Bad time entries from the RCDB
    for rnum in [4985, 4990, 4991, 4998, 5000, 5001, 5381, 5382]:
        if rnum in data.All_Runs.index:
            data.All_Runs.loc[rnum, "start_time"] = data.All_Runs.loc[rnum, "data_start"]
            data.All_Runs.loc[rnum, "end_time"] = data.All_Runs.loc[rnum, "data_end"]

    data.All_Runs.loc[data.All_Runs.start_time.isnull(), "start_time"] = data.All_Runs.loc[
        data.All_Runs.start_time.isnull(), "data_start"]
    data.All_Runs.loc[data.All_Runs.end_time.isnull(), "end_time"] = data.All_Runs.loc[
        data.All_Runs.end_time.isnull(), "data_end"]


def initialize_fcup_param(periods, data, no_cache=False, override=False, debug=0):
    """Initialize the beam_stop_atten_time and fcup_offset_time parameters.
    If beam_stop_atten is already set in data.beam_stop_atten, then keep that unless override is True.
    If fcup_offset_time is already in data.fcup_offset_time, then keep that unless override is True.
    If none of that is the case, then get the data using Mya.get(...), passing no_cache parameter.
    So, override=True and no_cache=True means the data is freshly fetched from epicsweb.
    The obtained values for beam_atten_time and fcup_offset_time are put in data and returned.
    """

    if periods is None:
        periods = [
                (datetime(2018, 2,  5, 20, 0), datetime(2018,  2,  8, 6, 0)),
                (datetime(2018, 9, 27,  1, 0), datetime(2018, 11, 26, 7, 0)),
                (datetime(2019, 3, 25, 18, 0), datetime(2019,  4, 15, 6, 0))
                ]

    if not hasattr(data, "beam_stop_atten_time"):
        data.beam_stop_atten_time = None

    if not hasattr(data, "fcup_offset_time"):
        data.fcup_offset_time = None

    if type(periods[0]) is not list and type(periods[0]) is not tuple:
        periods = [periods]

    if data.beam_stop_atten_time is None or override or no_cache:  # Need to fill the beam_stop_atten_time dataframe:
        if debug > 1:
            print("Getting beam_stop_atten from Mya.get() for extended time range.")

        data.beam_stop_atten_time = data.Mya.get(channel="beam_stop_atten",
                                                 start=datetime(2018, 1, 1, 0, 0),
                                                 end=datetime(2019, 12, 31, 0, 0),
                                                 run_number=1, no_cache=no_cache)
        data.beam_stop_atten_time.set_index(['time'], inplace=True)

    if data.fcup_offset_time is None or override or no_cache:

        if debug > 1:
            print("Getting fcup_offset with Mya.get().")

        fcup_offset_period = []
        for i in range(len(periods)):
            fcup_offset = data.Mya.get(channel="fcup_offset",
                                       start=periods[i][0],
                                       end=periods[i][1],
                                       run_number=i+1, no_cache=no_cache)

            if fcup_offset.iloc[0].value is not None:
                fcup_offset_period.append(fcup_offset.copy())

        if len(fcup_offset_period) > 0:
            data.fcup_offset_time = pd.concat(fcup_offset_period, ignore_index=True)
            data.fcup_offset_time.set_index(['time'], inplace=True)
        else:
            data.fcup_offset_time = None

    return data.beam_stop_atten_time, data.fcup_offset_time


def compute_fcup_current(rnum, data, override=False, current_channel="scalerS2b"):
    """Compute the FCup charge for run rnum, from the FCup scaler channel and livetime_channel"""

    if not hasattr(data, "beam_stop_atten_time") or data.beam_stop_atten_time is None \
            or not hasattr(data, "fcup_offset_time") or data.fcup_offset_time is None:
        print("The data variables data.beam_stop_atten_time and data.fcup_offset_time must be initialized. Abort.")
        return None

    if current_channel is None:
        current_channel = "scalerS2b"

    if not override and \
            ("FCup_cor" in data.All_Runs.keys()) and \
            not np.isnan(data.All_Runs.loc[rnum, current_channel]):
        return

    if data.debug > 4:
        print("compute_fcup_data, run= {:5d}".format(rnum))

    start_time = data.All_Runs.loc[rnum, "start_time"]
    end_time = data.All_Runs.loc[rnum, "end_time"]
    if pd.isnull(start_time) or pd.isnull(end_time):
        return(None)
    else:
        scaler = data.Mya.get(current_channel, start_time, end_time, run_number=rnum)

    if len(scaler) <= 2:
        return None

    # Get the "forward fill" value for start_time ==> i.e. the *value before* start_time
    bsat = data.beam_stop_atten_time.index.get_indexer([start_time], method='ffill')
    if bsat < 0:  # We asked for a time before the first beam_stop_atten_time, so instead take the [0] one
        bsat = np.array([0])  # Keep the same type.
    beam_stop_attenuation = float(data.beam_stop_atten_time.iloc[bsat].value)

    # fcup_offset = fcup_offset_time.loc[start_time:end_time]
    # Get one more before the start_time
    # fcup_prepend = fcup_offset_time.iloc[fcup_offset_time.index.get_indexer([start_time], method='ffill')]
    # fcup_prepend.index = [scaler.iloc[0].time]             # Reset the index of last fcup value to start_time
    # fcup_offset = pd.concat([fcup_prepend, fcup_offset])     # Add the one value to the list.
    # fcup_offset_interpolate = np.interp(scaler.ms, fcup_offset.ms, fcup_offset.value)

    # We don't want interpolated, we want the previous valid value.
    times = scaler.time.to_list()
    if times[0] < data.fcup_offset_time.iloc[0].name:  # There is no fcup_offset for this time span.
        fcup_offset_tmp = scaler[(scaler.value < 600)].value.mean()  # Sort of calculate it.
        current_values = beam_stop_attenuation * (scaler.value - fcup_offset_tmp) / 906.2
    else:
        if not data.fcup_offset_time.index.is_monotonic:
            print(f"{rnum} -- Check fcup_time: is not monotonic! len = {len(data.fcup_offset_time)}")
            return
        if np.any(np.roll(data.fcup_offset_time.index.values, 1) == data.fcup_offset_time.index.values):
            print("fcup_offset_time still has repeat indexes.")
            return
        fcup_offset_indexes = data.fcup_offset_time.index.get_indexer(times, method="ffill")
        fcup_offset_interpolate = data.fcup_offset_time.iloc[fcup_offset_indexes].value.to_list()
        try:
            current_values = beam_stop_attenuation * (scaler.value - fcup_offset_interpolate) / 906.2
            scaler.value = current_values
        except Exception as e:
            print(e)
            print(f"rnum = {rnum}  len(scaler) = {len(scaler)}")
            print(type(scaler.value), ":", scaler.value)
            print("Courageously continuing....")
            return None
    scaler.value = current_values   # Override the values with the computed current.
    return scaler


def compute_fcup_current_livetime_correction(runnumber, current, data, livetime_channel=None):
    """Take the current array (computed with compute_fcup_current) and do the livetime correction.
       The resulting corrected current is returned.
       Also, the current is added into the data.All_Runs for rnum."""

    # Code templated on RunData.add_current_cor()

    if livetime_channel is None:
        livetime_channel = data.LiveTime_Channel

    start = data.All_Runs.loc[runnumber, "start_time"]
    end = data.All_Runs.loc[runnumber, "end_time"]
    live_time = data.Mya.get(livetime_channel, start, end, run_number=runnumber)
    if len(live_time) < 2:
        if data.debug > 1:
            print(f"compute_fcup_current_livetime_correction: Issue with live_time for run {runnumber} - len<2 ")
        live_time = pd.DataFrame({'ms': [start.timestamp() * 1000, end.timestamp() * 1000],
                      'value': [100., 100.],
                      'time': [start, end]})
    elif len(live_time) < 3:
        live_time.fillna(100., inplace=True)  # Replace Nan or None with 1 - no data returned.
    else:
        live_time.fillna(0, inplace=True)  # Replace Nan or None with 0
        live_time.loc[live_time.value.isna(), 'value'] = 0

    #
    # The sampling of the current and live_time are NOT guaranteed to be the same.
    # We interpolate the live_time at the current time stamps to compensate.
    #
    try:
        live_time_corr = np.interp(current.ms, live_time.ms, live_time.value) / 100.  # convert to fraction from %
    except Exception as e:
        print("live_time_corr: There is a problem with the data for run {}".format(runnumber))
        print(e)
        return None

    #
    # Now we can just multiply the live_time_corr with the current.
    #
    try:
        current_corr = current.value * live_time_corr
    except Exception as e:
        print("current_corr: There is a problem with the data for run {}".format(runnumber))
        print(e)
        return None
    #
    # We need to do a proper trapezoidal integration over the current data points.
    # Store the result in the data frame.
    #
    # Scale conversion:  I is in nA, dt is in ms, so I*dt is in nA*ms = 1e-9 A*1e-3 s = 1e-12 C
    # If we want mC instead of Coulombs, the factor is 1e-12*1e3 = 1e-9
    #

    data.All_Runs.loc[runnumber, "Fcup_charge"] = np.trapz(current.value, current.ms) * 1e-9  # mC
    data.All_Runs.loc[runnumber, "Fcup_charge_corr"] = np.trapz(current_corr, current.ms) * 1e-9  # mC

    return current_corr


def add_computed_fcup_data_to_runs(data, dates=None, targets=None, run_config=None,
                                   current_channel="scalerS2b", livetime_channel=None, override=False):
    """Get the mya data for beam current from the FCup using the formula:
    beam_stop_atten*(scalerS2b - fcup_offset)/906.2
    See email from Rafo: 2/8/22 10pm"""

    if dates is None:
        start_time = datetime(2018, 1, 1, 0, 0)
        end_time = datetime(2019, 12, 31, 0, 0)
    else:
        start_time = dates[0]
        end_time = dates[1]

    # Make sure the beam_stop_atten and fcup_offset data are available.
    # This call will not do anything if they are, unless override = True
    data.beam_stop_atten_time, data.fcup_offset_time = \
        initialize_fcup_param([start_time, end_time],
                              data=data,
                              override=override
                              )

    # Code modeled after RunData.add_current_data_to_runs.
    good_runs = data.list_selected_runs(targets, run_config)
    if len(good_runs) > 0:
        for rnum in data.list_selected_runs(targets, run_config):
            current = compute_fcup_current(rnum, data=data, override=override,
                                           current_channel=current_channel)
            if current is not None:
                compute_fcup_current_livetime_correction(rnum, current, data=data,
                                                         livetime_channel=livetime_channel)

    else:
        # Even if there are no good runs, make sure that the "charge" column is in the table!
        # This ensure that when you write to DB the charge column exists.
        data.All_Runs.loc[:, "Fcup_charge"] = np.NaN
        data.All_Runs.loc[:, "Fcup_charge_corr"] = np.NaN


def main(argv=None):
    import argparse
    import os.path as p

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
    parser.add_argument('-5', '--hdf5', action="store_true", help="Store all data in an hdf5 file.")
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

    # run_sub_periods_available = [
    #         (datetime(2018, 2,  5, 20, 0), datetime(2018,  2,  8, 6, 0)),
    #         (datetime(2018, 9, 27,  1, 0), datetime(2018, 11, 26, 7, 0)),
    #         (datetime(2019, 3, 25, 18, 0), datetime(2019,  4, 15, 6, 0))
    #         ]

    run_sub_periods_run_numbers = [
        (3018, 3086),
        (4764, 5666),
        (6608, 6783)
        ]

    run_sub_periods_times = [
        (datetime(2018, 2, 5, 20, 0), datetime(2018, 2, 8, 6, 0)),
        (datetime(2018, 9, 27, 1, 0), datetime(2018, 11, 26, 7, 0)),
        (datetime(2019, 3, 25, 18, 0), datetime(2019, 4, 15, 6, 0))
    ]

    if args.run_period == 0:
        run_sub_periods = run_sub_periods_run_numbers
    else:
        run_sub_periods = [run_sub_periods_run_numbers[args.run_period-1]]
        run_sub_periods_times = [run_sub_periods_times[args.run_period-1]]

    if not args.nocache:
        dat = RunData(cache_file="RGA.sqlite3", i_am_at_jlab=at_jlab)
    else:
        dat = RunData(cache_file="", sqlcache=False, i_am_at_jlab=at_jlab)

    dat.debug = args.debug

    beam_stop_atten_time, fcup_offset_time = initialize_fcup_param(run_sub_periods_times, dat)

#    run_sub_energy = [2.07, 4.03, 5.99]
#    run_sub_y_placement = [0.79, 0.99, 0.99]

    if args.excel:
        excel_output = pd.DataFrame()

    if args.hdf5:
        from pandas import HDFStore
        run_data_hdf = HDFStore('RGA_RunData.h5')

    for sub_i in range(len(run_sub_periods)):

        dat.All_Runs = None   # Delete the content of the previous period, if any.
        setup_rundata_structures(dat, runs=run_sub_periods[sub_i])

        targets = '.*'

        print("Adding current channels.")
        dat.add_current_data_to_runs(current_channel="IPM2C21A")
        dat.add_current_data_to_runs(current_channel="IPM2C24A")
        dat.add_current_data_to_runs(current_channel="scaler_calc1b")

        calib_run_numbers = dat.list_selected_runs(targets='.*', run_config=dat.Calibration_triggers)
        calib_runs = dat.All_Runs.loc[calib_run_numbers]

        starts = dat.All_Runs["start_time"]
        ends = dat.All_Runs["end_time"]

#        print("Compute cumulative charge.")
#        dat.compute_cumulative_charge(targets, runs=plot_runs)

        print("Computing beam_stop_atten*(scalerS2b - fcup_offset)/906.2")
        add_computed_fcup_data_to_runs(data=dat)
        # Copy the computed columns over to plot_runs for plotting later.

        if args.excel:
            excel_output = pd.concat([excel_output, dat.All_Runs], sort=True)

        if args.hdf5:
            sub_dir = f"period_{sub_i}/"
            run_data_hdf.put(sub_dir + "All_Runs", dat.All_Runs)
            run_data_hdf.put(sub_dir + "fcup_offset_time", fcup_offset_time)
            run_data_hdf.put(sub_dir + "beam_stop_atten_time", beam_stop_atten_time)

    if args.excel:
        print("Write new Excel table.")
        ordered_columns = ['start_time', 'end_time', 'data_start', 'data_end', 'target', 'beam_energy',
                           'beam_current_request', 'run_config',
                           'selected', 'event_count', 'sum_event_count', 'event_rate', 'evio_files_count',
                           'megabyte_count', 'is_valid_run_end',
                           'B_DAQ:livetime_pulser', 'data_charge', 'Fcup_charge', 'Fcup_charge_corr',
                           'IPM2C21A', 'IPM2C21A_corr',
                           'IPM2C24A', 'IPM2C24A_corr', 'scaler_calc1b', 'scaler_calc1b_corr',
                           'sum_charge', 'sum_charge_targ',
                           'operators', 'user_comment'
                           ]
        ordered_columns_copy = ordered_columns.copy()
        for c in ordered_columns_copy:
            if c not in excel_output.columns:
                print(f"Excel output, column {c} removed.")
                ordered_columns.remove(c)
        # print(f"ordered_columns = {ordered_columns}")
        # print(f"excel_output.columns = {excel_output.columns}")

        excel_output.to_excel("RGA_all_runs.xlsx", columns=ordered_columns)
        selected_output = excel_output.loc[~excel_output.data_start.isnull()]
        selected_output.to_excel("RGA_selected_runs.xlsx", columns=ordered_columns)
        # End sub run period loop.


if __name__ == "__main__":
    sys.exit(main())
else:
    print("Imported the RGA info.")
    print("Setup 'data' with: data = RunData(cache_file='RGA.sqlite3', sqlcache=True)")
#    data = RunData(cache_file="RGA.sqlite3", sqlcache=True)
#    data.debug = 10

