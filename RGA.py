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


def setup_rundata_structures(data, dates):
    """Setup the data structures for parsing the databases."""

    start_time, end_time = dates
    data.Good_triggers, data.Calibration_triggers = used_triggers()

    data.Production_run_type = "PROD.*"  # ["PROD66", "PROD66_PIN", "PROD66_noVTPread", "PROD67_noVTPread"]
    data.target_properties = rga_2022_target_properties()
    data.target_dens = data.target_properties['density']
    data.atten_dict = None
    data.Current_Channel = "IPM2C21A"  # "scaler_calc1b"
    data.LiveTime_Channel = "B_DAQ:livetime_pulser"
    data.Useful_conditions.append('beam_energy')  # This run will have multiple beam energies.

    min_event_count = 500000  # Runs with at least 200k events.
    end_time = end_time + timedelta(0, 0, -end_time.microsecond)  # Round down on end_time to a second
    print("Fetching the data from {} to {}".format(start_time, end_time))
    data.get_runs(start_time, end_time, min_event_count)
    if 5381 in data.All_Runs.index and 5382 in data.All_Runs.index:
        # This run seems to have a bad start time from the RCDB database. If so, we fix it here.
        if data.All_Runs.loc[5382].start_time < data.All_Runs.loc[5381].end_time:
            data.All_Runs.loc[5382, "start_time"] = data.All_Runs.loc[5381, "end_time"] + timedelta(seconds=1)

    data.select_good_runs()


def use_once_sparsified_fcup_offset(periods, data):
    """Use this once to get a sparsified set of values for the fcup_offset parameter, and also beam_stop_atten.
    The output is placed in data.beam_stop_atten_time and data.fcup_offset_time    """

    print("Getting the fcup_offset from Mya. This may take some time!")

    # We first get the beam_stop_atten and fcup_offset data, if not already available (probably not!)
    data.beam_stop_atten_time, data.fcup_offset_time = initialize_fcup_param(periods, data=data)
    run_data = []
    #
    # Fill the All_Runs run table for all the run periods.
    # This will also fill the sqlite3 cache with these runs, and get the data_loc.Current_Channel and
    # data_loc.LiveTime_Channel
    # ("IPM2C21A" and "B_DAQ:livetime_pulser").
    # We request the "IPM2C24A", "scalerS2b" as well for each run below.
    #
    for i in range(len(periods)):
        setup_rundata_structures(data, periods[i])
        run_data.append(data.All_Runs.copy())
        data.All_Runs = None

    data.All_Runs = pd.concat(run_data)

    mya_cache = data.Mya.cache_engine
    fcup_offset_sparse_time = []
    for run_num, row in data.All_Runs.iterrows():
        # Here we compute the sparse version of the fcup_offset data.

        # Check if this data is already in the cache. If it is, do not write it again.
        if mya_cache is not None and \
                not data.Mya.check_if_data_is_in_cache("fcup_offset_sparse",
                                                       start=row.start_time,
                                                       end=row.end_time,
                                                       run_number=run_num):

            if data.debug > 4:
                print(f"Getting the Mya channels for run {run_num}")

            # First get the scalerS2b (and also IPM2C24A
            scalers2b = data.Mya.get(channel="scalerS2b", start=row.start_time, end=row.end_time, run_number=run_num)
            ipm2c24a = data.Mya.get(channel="IPM2C24A", start=row.start_time, end=row.end_time, run_number=run_num)

            if scalers2b.iloc[0].time < row.start_time:
                # curious error where we get too much data at the start.
                if data.debug > 0:
                    print(f"ScalerS2b starts too early for run {run_num}: "
                          f"start {scalers2b.iloc[0].time} < {row.start_time}")
                scalers2b.drop(
                    scalers2b[(scalers2b.time < row.start_time)].index, inplace=True)
            # Next: Get an index for the fcup_offset at each time for the scalerS2b
            # where there is beam: value>1000, finding the previous fcup_offset value.
            if scalers2b.iloc[0].time < data.fcup_offset_time.iloc[0].name or \
                    scalers2b.value.max() < 1000 or (len(scalers2b) < 3 and np.any(np.isnan(scalers2b.value))):
                # fcup_offset_time doesn't have a value or there never was any beam for the run!
                scaler_local_mean = scalers2b[(scalers2b.value < 1000)].value.mean()
                fcup_offset_sparse_run = pd.DataFrame(data={'ms': [scalers2b.iloc[0].ms],
                                                            'index': [0],
                                                            'value': [scaler_local_mean],
                                                            'time': [scalers2b.iloc[0].time]
                                                            }
                                                      )
            else:
                if data.debug > 4:
                    print(f"Sparsifying for run = {run_num}")
                fcup_offset_indexes = data.fcup_offset_time.index.\
                    get_indexer(scalers2b[scalers2b.value > 1000].time.to_list(), method="ffill")
                # Make a new dataframe of all these fcup_offset, dropping ones with duplicate values.
                if len(fcup_offset_indexes) > 0:
                    fcup_offset_sparse_tmp = data.fcup_offset_time.iloc[fcup_offset_indexes]
                    fcup_offset_sparse_first_entry = fcup_offset_sparse_tmp.iloc[0:1]
                    # Set the time of this entry to the start of the run. This avoids having multiple entries with the
                    # exact same time.

                    fcup_offset_sparse_first_entry.reset_index(inplace=True)
                    fcup_offset_sparse_first_entry.iloc[0, fcup_offset_sparse_first_entry.columns.get_loc('time')] = row.start_time
                    fcup_offset_sparse_first_entry.set_index('time', inplace=True)

                    fcup_offset_sparse_run = pd.concat(
                        [
                            fcup_offset_sparse_first_entry,
                            fcup_offset_sparse_tmp[
                             ((fcup_offset_sparse_tmp.shift().value - fcup_offset_sparse_tmp.value).abs() > 0.01)
                            ]
                         ])
                    fcup_offset_sparse_run.reset_index(inplace=True)   # Make 'time' a column again.
                else:
                    print(f"No fcup_offset for run {run_num}. Probably no beam? "
                          f"scaler2b.value.max={scalers2b.value.max()} -- We should never get here!! ERROR.")
                    fcup_offset_sparse_run = pd.DataFrame(data={'ms': [scalers2b.iloc[0].ms],
                                                                'index': [0],
                                                                'value': [0],
                                                                'time': [scalers2b.iloc[0].time]
                                                                }
                                                          )

            fcup_offset_sparse_time.append(fcup_offset_sparse_run)

            if len(fcup_offset_sparse_run.columns) > 4:
                print(f"FCup_offset: {run_num} has too many columns: ", fcup_offset_sparse_run.columns)
                for c in fcup_offset_sparse_run.columns:
                    if c not in ['time', 'ms', 'value']:
                        fcup_offset_sparse_run.drop(labels=[c], axis=1)

            fcup_offset_sparse_run["run_num"] = run_num
            data.Mya.add_to_mya_data_range("fcup_offset_sparse", start=row.start_time, end=row.end_time,
                                           run_number=run_num, data_length=len(fcup_offset_sparse_run))

            if "index" in fcup_offset_sparse_run.columns:
                fcup_offset_sparse_run.drop(columns=['index'], inplace=True)
                # We dropped the not useful "index" column to the to_sql can write the index as "index".

            if data.debug > 6:
                print(f"fcup_offset_sparse_run: columns: {fcup_offset_sparse_run.columns}")

            fcup_offset_sparse_run.to_sql("fcup_offset_sparse", mya_cache, if_exists="append")

        else:  # The data is in the cache.
            fcup_offset_sparse_time.append(
                data.Mya.get(channel="fcup_offset_sparse",
                             start=row.start_time,
                             end=row.end_time,
                             run_number=run_num)
            )

    for t in fcup_offset_sparse_time:
        t["orig_index"] = t.index      # To save the original index when concat these dataframes.

    fcup_offset_sparse = pd.concat(fcup_offset_sparse_time, ignore_index=True)
    if "index" in fcup_offset_sparse.columns:
        fcup_offset_sparse.drop(columns=['index'], inplace=True)   # Drop the not useful "index" column

    fcup_offset_sparse.set_index(['time'], inplace=True)

    data.fcup_offset_time = fcup_offset_sparse

    return data.beam_stop_atten_time, fcup_offset_sparse


def initialize_fcup_param(periods, data,
                          no_cache=False, override=False, debug=0):
    """Initialize the beam_stop_atten_time and fcup_offset_time parameters.
    If beam_stop_atten is already set in data.beam_stop_atten, then keep that unless override is True.
    If fcup_offset_time is already in data.fcup_offset_time, then keep that unless override is True.
    If fcup_offset_time is not in data, but fcup_offset_sparse is in the cache, then get the sparse version.
    If it is not in the cache, but fcup_offset is in the cache, then get that version.
    If none of that is the case, then get the data using Mya.get(...), passing no_cache parameter.
    So, override=True and no_cache=True means the data is freshly fetched from epicsweb.
    The obtained values for beam_atten_time and fcup_offset_time are put in data and returned.
    """

    if not hasattr(data, "beam_stop_atten_time"):
        data.beam_stop_atten_time = None

    if not hasattr(data, "fcup_offset_time"):
        data.fcup_offset_time = None

    if type(periods[0]) is not list and type(periods[0]) is not tuple:
        periods = [periods]

    if data.beam_stop_atten_time is None or override or no_cache:  # Need to fill the beam_stop_atten_time dataframe:
        if debug > 1:
            print("Getting beam_stop_atten.")

        data.beam_stop_atten_time = data.Mya.get(channel="beam_stop_atten",
                                                 start=datetime(2018, 1, 1, 0, 0),
                                                 end=datetime(2019, 12, 31, 0, 0),
                                                 run_number=1, no_cache=no_cache)
        data.beam_stop_atten_time.set_index(['time'], inplace=True)

    if data.fcup_offset_time is None or override or no_cache:
        if data.Mya.check_if_table_is_in_cache("fcup_offset_sparse") and not no_cache:
            # We have the sparse data in cache... assume it is complete.
            if debug > 1:
                print("Getting fcup_offset from fcup_offset_sparse.")
            data.fcup_offset_time = data.Mya.get_channel_from_cache("fcup_offset_sparse")
            data.fcup_offset_time.set_index(['time'], inplace=True)

        else:
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
            print("Check fcup_time: is not monotonic!")
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

    run_sub_periods_available = [
            (datetime(2018, 2,  5, 20, 0), datetime(2018,  2,  8, 6, 0)),
            (datetime(2018, 9, 27,  1, 0), datetime(2018, 11, 26, 7, 0)),
            (datetime(2019, 3, 25, 18, 0), datetime(2019,  4, 15, 6, 0))
            ]

    if args.run_period == 0:
        run_sub_periods = run_sub_periods_available
    else:
        run_sub_periods = [run_sub_periods_available[args.run_period-1]]

    if not args.nocache:
        if not p.exists("RGA.sqlite3"):
            # If the cache file does not exist, we call use_once_sparsified_fcup_offset AFTER initalizeing data.
            init_data_once = True
        else:
            init_data_once = False
        dat = RunData(cache_file="RGA.sqlite3", i_am_at_jlab=at_jlab)
        dat.debug = args.debug
        # Assume that if the cache file exists, the cache is already build.
        if init_data_once:
            beam_stop_atten_time, fcup_offset_time = use_once_sparsified_fcup_offset(run_sub_periods_available, dat)

    else:
        dat = RunData(cache_file="", sqlcache=False, i_am_at_jlab=at_jlab)
        dat.debug = args.debug
    # data._cache_engine=None   # Turn OFF cache?

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

    if args.hdf5:
        from pandas import HDFStore
        run_data_hdf = HDFStore('RGA_RunData.h5')

    for sub_i in range(len(run_sub_periods)):
        dat.All_Runs = None   # Delete the content of the previous period, if any.
        setup_rundata_structures(dat, run_sub_periods[sub_i])

        dat.All_Runs['luminosity'] *= 1E-3   # Rescale luminosity from 1/pb to 1/fb

        dat.add_current_data_to_runs()
        targets = '.*'

        print("Adding additional current channels.")
        dat.add_current_data_to_runs(current_channel="IPM2C24A")
        dat.add_current_data_to_runs(current_channel="scaler_calc1b")

        # Select runs into the different categories.
        plot_runs = compute_plot_runs(targets=targets, run_config=dat.Good_triggers, data=dat)
        calib_run_numbers = dat.list_selected_runs(targets='.*', run_config=dat.Calibration_triggers)
        calib_runs = plot_runs.loc[calib_run_numbers]

        plot_runs = plot_runs.loc[~plot_runs.index.isin(calib_run_numbers)]  # Take the calibration runs out.
        starts = plot_runs["start_time"]
        ends = plot_runs["end_time"]

        print("Compute cumulative charge.")
        dat.compute_cumulative_charge(targets, runs=plot_runs)

        # Copy the computed columns over to All_Runs so they show up in the spread sheet.
        dat.All_Runs.loc[plot_runs.index, "sum_charge"] = plot_runs["sum_charge"]
        dat.All_Runs.loc[plot_runs.index, "sum_charge_targ"] = plot_runs["sum_charge_targ"]

        print("Computing beam_stop_atten*(scalerS2b - fcup_offset)/906.2")
        add_computed_fcup_data_to_runs(data=dat)
        # Copy the computed columns over to All_Runs so they show up in the spread sheet.
        plot_runs["Fcup_charge"] = dat.All_Runs.loc[plot_runs.index, "Fcup_charge"]
        plot_runs["Fcup_charge_corr"] = dat.All_Runs.loc[plot_runs.index, "Fcup_charge_corr"]



        if args.excel:
            excel_output = pd.concat([excel_output, plot_runs, calib_runs], sort=True)

        if args.plot:

            print(f"Build Plots for period {sub_i}")

            last_targ = None
            for targ in dat.target_properties['color']:

                if args.debug:
                    print(f"Processing plot for target {targ}")
                runs = plot_runs.target.str.fullmatch(targ.replace('(', r"\(").replace(')', r'\)'))

                if np.count_nonzero(runs) > 1 and sub_i == len(run_sub_periods) - 1 and \
                        targ in dat.target_properties['sums_color']:
                    last_targ = targ    # Store the last target name with data for later use.

                if targ in legends_data or np.count_nonzero(runs) <= 1:  # Do we show this legend?
                    show_data_legend = False                             # This avoids the same legend shown twice.
                else:
                    show_data_legend = True
                    legends_data.append(targ)

                fig.add_trace(
                    go.Bar(x=plot_runs.loc[runs, 'center'],
                           y=plot_runs.loc[runs, 'event_rate'],
                           width=plot_runs.loc[runs, 'dt']*999/1000,
                           hovertext=plot_runs.loc[runs, 'hover'],
                           name="run with " + targ,
                           marker=dict(color=dat.target_properties['color'][targ]),
                           legendgroup="group1",
                           showlegend=show_data_legend
                           ),
                    secondary_y=False, )

            fig.add_trace(
                 go.Bar(x=calib_runs['center'],
                        y=calib_runs['event_rate'],
                        width=calib_runs['dt']*999/1000,
                        hovertext=calib_runs['hover'],
                        name="Calibration runs",
                        marker=dict(color='rgba(150,150,150,0.5)'),
                        legendgroup="group1",
                        ),
                 secondary_y=False, )

            if args.charge:
                current_plotting_scale = dat.target_properties['current']['scale'][sub_i]
                sumcharge = plot_runs.loc[:, "sum_charge"] * current_plotting_scale
                max_y_value_sums = plot_runs.sum_charge_targ.max() * current_plotting_scale

                plot_sumcharge_t = [starts.iloc[0], ends.iloc[0]]
                plot_sumcharge_v = [0, sumcharge.iloc[0]]

                for i in range(1, len(sumcharge)):
                    plot_sumcharge_t.append(starts.iloc[i])
                    plot_sumcharge_t.append(ends.iloc[i])
                    plot_sumcharge_v.append(sumcharge.iloc[i - 1])
                    plot_sumcharge_v.append(sumcharge.iloc[i])

                for targ in dat.target_properties['sums_color']:
                    sumch = plot_runs.loc[plot_runs["target"] == targ, "sum_charge_targ"]*current_plotting_scale
                    st = plot_runs.loc[plot_runs["target"] == targ, "start_time"]
                    en = plot_runs.loc[plot_runs["target"] == targ, "end_time"]

                    if len(sumch) > 1:
                        # Complication: When a target was taken out and then later put back in there is an interruption
                        # that should not count for the expected charge.

                        # Setup this initial entries for the plot lines.
                        plot_sumcharge_target_t = [st.iloc[0], en.iloc[0]]
                        plot_sumcharge_target_v = [0, sumch.iloc[0]]
                        if dat.target_properties['current'][targ][sub_i] > 0.:
                            plot_expected_charge_t = [st.iloc[0]]
                            plot_expected_charge_v = [0]

                            current_expected_sum_charge = 0

                        for i in range(1, len(sumch)):
                            # Step through all the runs for this target that have "sum_charge_targ".
                            #
                            # We also need to detect if there is a run number gap.
                            # Check if all the intermediate runs have the same target. If so, these were calibration
                            # or junk runs, and we continue normally. If there were other targets,
                            # we make a break in the line.
                            if sumch.keys()[i] - sumch.keys()[i - 1] > 1 and \
                                    not np.all(dat.All_Runs.loc[sumch.keys()[i - 1]:sumch.keys()[i]].target == targ):
                                # There is a break in the run numbers (.keys()) and a target change occurred.
                                plot_sumcharge_target_t.append(st.iloc[i])  # Add the last time stamp again.
                                plot_sumcharge_target_v.append(None)        # None causes the solid line to break.

                                fig.add_trace(                 # Add a horizontal dotted line for the target sum_charge.
                                    go.Scatter(x=[en.iloc[i-1], st.iloc[i]],
                                               y=[plot_sumcharge_target_v[-2], plot_sumcharge_target_v[-2]],
                                               mode="lines",
                                               line=dict(color=dat.target_properties['sums_color'][targ], width=1,
                                                         dash="dot"),
                                               name=f"Continuation line {targ}",
                                               showlegend=False),
                                    secondary_y=True)

                                if dat.target_properties['current'][targ][sub_i] > 0.:  # Are we plotting expected charge?
                                    plot_expected_charge_t.append(en.iloc[i-1])  # Add the time stamp
                                    current_expected_sum_charge += (en.iloc[i-1] - plot_expected_charge_t[-2]).\
                                        total_seconds() * dat.target_properties['current'][targ][sub_i] * 1e-6 * 0.5
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

                        if dat.target_properties['current'][targ][sub_i] > 0.:
                            plot_expected_charge_t.append(plot_sumcharge_target_t[-1])
                            i = len(plot_expected_charge_v)-1
                            while plot_expected_charge_v[i] is None and i > 0:
                                i -= 1
                            current_expected_sum_charge = plot_expected_charge_v[i]
                            current_expected_sum_charge += \
                                (plot_sumcharge_target_t[-1] - plot_expected_charge_t[-2]).total_seconds() * \
                                dat.target_properties['current'][targ][sub_i] * 1e-6 * 0.5
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
                                       line=dict(color=dat.target_properties['sums_color'][targ], width=3),
                                       name=f"Total Charge on {targ}",
                                       legendgroup="group2",
                                       showlegend=show_legend_ok
                                       ),
                            secondary_y=True)

                        # Decorative: add a dot at the end of the curve.
                        fig.add_trace(
                            go.Scatter(x=[plot_sumcharge_target_t[-1]],
                                       y=[plot_sumcharge_target_v[-1]],
                                       marker=dict(color=dat.target_properties['sums_color'][targ],
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
                                color=dat.target_properties['sums_color'][targ],
                                size=16),
                            bgcolor="#FFFFFF"
                        )

                        # # Annotate - add a curve for the expected charge at 50% efficiency.
                        showlegend = True if targ == last_targ and sub_i == 2 else False
                        if args.debug:
                            print(f"last_targ = {last_targ}  targ: {targ}, sub_i = {sub_i}, showlegend = {showlegend}")
                        if dat.target_properties['current'][targ][sub_i] > 0.:
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

                # We get the sums per target in two steps.
                # Clumsy, but only way to get the maximum available in second loop
                plot_sumlumi_target = {}                       # Store sum results.
                for targ in dat.target_properties['sums_color']:
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
                                    not np.all(dat.All_Runs.loc[plot_sumlumi_target[targ].keys()[i - 1]:
                                               plot_sumlumi_target[targ].keys()[i]].target == targ):
                                plot_sumlumi_target_t.append(plot_sumlumi_starts.iloc[i])
                                plot_sumlumi_target_v.append(None)

                                fig.add_trace(
                                    go.Scatter(x=[plot_sumlumi_starts.iloc[i-1], plot_sumlumi_starts.iloc[i]],
                                               y=[plot_sumlumi_target_v[-2], plot_sumlumi_target_v[-2]],
                                               mode="lines",
                                               line=dict(color=dat.target_properties['sums_color'][targ], width=1,
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
                                       line=dict(color=dat.target_properties['sums_color'][targ], width=3),
                                       name=f"Sum luminosity on {targ}",
                                       legendgroup="group2",
                                       ),
                            secondary_y=True)

                        fig.add_trace(
                            go.Scatter(x=[plot_sumlumi_target_t[-1]],
                                       y=[plot_sumlumi_target_v[-1]],
                                       marker=dict(
                                           color=dat.target_properties['sums_color'][targ],
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
                                color=dat.target_properties['sums_color'][targ],
                                size=16),
                            bgcolor="#FFFFFF"
                        )

        if args.hdf5:
            sub_dir = f"period_x{sub_i}/"
            run_data_hdf.put(sub_dir + "All_Runs", dat.All_Runs)


    if args.excel:
        print("Write new Excel table.")
        all_columns = excel_output.columns.drop(['hover', 'center', 'run_start_time', 'run_end_time'])
        ordered_columns = ['start_time', 'end_time', 'target', 'beam_energy', 'beam_current_request', 'run_config',
                           'selected', 'event_count', 'sum_event_count', 'event_rate', 'evio_files_count',
                           'megabyte_count', 'is_valid_run_end',
                           'B_DAQ:livetime_pulser',	'Fcup_charge', 'Fcup_charge_corr', 'IPM2C21A', 'IPM2C21A_corr',
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

        excel_output.to_excel("RGA_progress.xlsx", columns=ordered_columns)

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
    print("Imported the RGA info.")
    print("Setup 'data' with: data = RunData(cache_file='RGA.sqlite3', sqlcache=True)")
#    data = RunData(cache_file="RGA.sqlite3", sqlcache=True)
#    data.debug = 10

