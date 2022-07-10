#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Class RunData
# author: Maurik Holtrop, UNH, 2019 - 2021
#
# This python class will read the RCDB and the Mya database to construct a Pandas DataFrame that is useful
# for computing information about the run.
#
# Required externally supplied information:
#
# self.target_dens  = {}
#          Dictionary of target areal density in gram/cm^2  OR
#          a list of target thickness in cm, target density in gram/cm^3, molecular mass in gram/mole
# self.atten_dict        = {}  # Dictionary of attenuation values for each target.
#
# RunData.All_Runs  - Pandas DataFrame with run number of index.
#
# Example Contents of DataFrame:
#
# start_time            datetime                      2019-09-08 04:48:02
# end_time              datetime                      2019-09-08 06:06:25
# is_valid_run_end      bool                                         True
# user_comment          str
# run_type              str                                        PROD66
# target                str                                       20 um W
# beam_current_request  str                                           120
# operators             str   expert: Mathew Graham, worker: Hakop Voskanyan
# event_count           int                                      96056017
# events_rate           float                                     169.888     =
# run_config            str                                 hps_v12_1.cnf
# status                int                                             0
# evio_files_count      int                                           558
# megabyte_count        int                                       1140687
# run_start_time        datetime                      2019-09-08 04:48:02
# run_end_time          datetime v                    2019-09-08 06:06:25
# selected              bool                                         True   Is this run selected by select_good_runs()?
# scaler_calc1b                                          470013449.718153   scaler_calc1b from Mya
# live_time                                              405775888.513366   Trapezoidal integrated live_time * dt (ms)
# charge                                                         0.443503   Total charge on target in mC
#
# Notes:
# live_time is a percentage * time in ms. So to get the runs average live time in percent you need for run 10722:
# data.All_Runs.loc[10722,'live_time']/
#   (data.All_Runs.loc[10722,'run_end_time']  - data.All_Runs.loc[10722,'run_start_time']).total_seconds()/1000.
#
import sys
import os
import re

try:
    import sqlalchemy
except ImportError:
    print("We need the sqlalchemy installed for the database, but I could not find it in:")
    print("sys.path: ", sys.path)
    sys.exit(1)

try:
    from rcdb.model import Run, ConditionType, Condition, all_value_types
    from rcdb.provider import RCDBProvider
except ImportError:
    print("Please set your PYTHONPATH to a copy of the rcdb Python libraries.\n")
    print("sys.path: ", sys.path)
    sys.exit(1)

# import requests
# from requests.packages.urllib3.exceptions import InsecureRequestWarning
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
#
from datetime import datetime, timedelta

try:
    import pandas as pd
except ImportError:
    print("Sorry, but you really need a computer with 'pandas' installed.")
    print("Try 'anaconda' python, which should have both.")
    sys.exit(1)

import numpy as np
from .MyaData import MyaData

import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')   # Turn off the warning for regex with () in it.

#
# Some global configuration settings.
# These will need to go to input values on a web form.
#


class RunData:

    def __init__(self, i_am_at_jlab=False, cache_file=None, sqlcache=True,
                 username=None, password=None):
        """ Set things up. If not at JLab you will be asked for CUE username and password.
        sqlcache=False will prevent caching querries in a local sqlite3 database.
        sqlcache='mysql://user.pwd@host/db' will use that DB as the cache. String is a sqlalchemy style DB string.
        sqlcache=True  will use a local sqlite3 file for caching.
        """
        self.Production_run_type = "PROD.*"  # Type of production runs to consider.
        # List of conditions to put in tables.
        self.Useful_conditions = ['is_valid_run_end', 'user_comment', 'run_type',
                                  'target', 'beam_current_request', 'operators', 'event_count',
                                  'events_rate', 'run_config', 'status',
                                  'evio_files_count', 'megabyte_count', 'run_start_time', 'run_end_time']

        # Regex string or list of trigger conditions to use for run selection.
        self.Good_triggers = r'hps.*\.cnf'

        self.not_good_triggers = []

        # This runs to be excluded for other reasons. (I.e. marked in online spreadsheet.
        self.ExcludeRuns = []

        self.min_event_count = 1000000
        self.target_properties = {}
        self.target_dens = {}
        # This is a dictionary of target dependent correction factors for correcting the FCup current.
        self.atten_dict = {}
        self.at_jlab = i_am_at_jlab
        self.All_Runs = None
        self._debug = 0

        self.Current_Channel = "scaler_calc1b"  # Mya Channel for the current from FCUP.
        self.LiveTime_Channel = "B_DAQ_HPS:TS:livetime"  # Mya Channel for the livetime.
        self._db = None
        self._session = None

        self._cache_engine = None
        self._cache_known_data = None
        self._cache_file_name = cache_file

        self.fix_bad_rcdb_start_times = False  # Set to true to query RCDB twice, once for time, once for run numbers.

        self.start_rcdb()
        self.start_cache(sqlcache)
        self.Mya = MyaData(i_am_at_jlab, username=username, password=password, cache=self._cache_engine)

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, debug_level):
        self._debug = debug_level
        self.Mya.debug = debug_level

    def __str__(self):
        """Return a table with some of the information, to see what is in the All_Runs conveniently. """
        out = str(self.All_Runs.loc[:, ["start_time", "end_time", "target", "run_config", "event_count"]])
        return out

    def clear(self):
        """Reset the run tables only. Done by setting self.All_Runs = None"""
        self.All_Runs = None

    def start_rcdb(self):
        """Setup a connection to the RCDB
        return: an RCDB handle"""
        if "RCDB_CONNECTION" in os.environ:
            connection_string = os.environ["RCDB_CONNECTION"]
        else:
            connection_string = "mysql://rcdb@clasdb.jlab.org/rcdb"
            # print("Using standard connection string from HallB")

        try:
            self._db = RCDBProvider(connection_string)
        except:
            print("WARNING: Cannot connect to the RCDB. Will try with data from Cache only.")
            self._db = None

    def start_cache(self, connector_string=True):
        """Start up the cache backend according to connector_string.
        If connector_string is True, use a local sqlite3 file."""
        #
        # This is NOT some super smart caching setup. I just hope it is better than nothing.
        #
        if connector_string is False:
            self._cache_engine = None
            return
        elif connector_string is True:
            connector_string = "sqlite:///" + self._cache_file_name
        elif type(connector_string) is str and "///" not in connector_string:
            connector_string = "sqlite:///" + connector_string

        self._cache_engine = sqlalchemy.create_engine(connector_string)
        #
        # We typically query a time range and an event number cut.
        # This complicates caching, because the next request may be incorporated
        # in a timerange, but have a different (lower) event cut.
        # The Known_Data_Range table tries to help with this.
        if not sqlalchemy.inspect(self._cache_engine).has_table("Known_Data_Ranges"):
            print("Creating the run data cache.")
            # sql='''CREATE TABLE IF NOT EXISTS "Runs_Table" (
            #   "index"  INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            #   "start_time" TIMESTAMP,
            #   "end_time" TIMESTAMP,
            #   "min_event_count" INTEGER,
            #   "min_run_number"  INTEGER,
            #   "max_run_number"  INTEGER
            # );
            # '''
            meta = sqlalchemy.MetaData()
            sqlalchemy.Table('Known_Data_Ranges', meta,
                             sqlalchemy.Column('index', sqlalchemy.Integer, primary_key=True),
                             sqlalchemy.Column('start_time', sqlalchemy.DateTime),
                             sqlalchemy.Column('end_time', sqlalchemy.DateTime),
                             sqlalchemy.Column('min_event_count', sqlalchemy.Integer))
            meta.create_all(self._cache_engine)

        self._cache_known_data = pd.read_sql("Known_Data_Ranges", self._cache_engine, index_col="index")

    def _check_for_cache_hits(self, start, end):
        # Cache Hit logic: We need to determine if we have a cache hit.
        # There are multiple cases:
        #     1 - No Hit
        #     2 - Everything in cache.
        #     3 - One chunck in cache - extend cache before and/or after
        #     4 - Multiple chuncks in cache - Punt this! Just overwrite all but one
        #
        start = start + timedelta(0, 0, -start.microsecond)  # Round down on start
        if end.microsecond != 0:
            end = end + timedelta(0, 0, 1000000 - end.microsecond)  # Round up on end.
        cache_overlaps = []
        cache_extend_before = []
        cache_extend_after = []
        for index in range(len(self._cache_known_data)):  # Loop by iloc so index order is early to late
            # Check if "start_time" < [start,end] < "end_time"
            cache_data = self._cache_known_data.iloc[index]
            if (cache_data["start_time"] <= np.datetime64(start) < cache_data["end_time"]) and \
                    (cache_data["start_time"] < np.datetime64(end) <= cache_data["end_time"]):
                cache_overlaps.append(index)
                continue  # No need to check extending, it's not needed.

            # Check if start is before "end_time", and end is after "end_time", we can extend after.
            # I.e.  end_time is in [start,end]
            if np.datetime64(start) <= cache_data["end_time"] < np.datetime64(end):
                cache_extend_after.append(index)
            # Check if "start_time" is inside [start,end], if so we can extend before.
            if np.datetime64(start) < cache_data["start_time"] <= np.datetime64(end):
                cache_extend_before.append(index)

        return cache_overlaps, cache_extend_before, cache_extend_after

    def _cache_fill_runs(self, start, end, min_event):
        """Fill the cache with runs from start to end and put the result in All_Runs."""

        start = start + timedelta(0, 0, -start.microsecond)  # Round down on start
        if end.microsecond != 0:
            end = end + timedelta(0, 0, 1000000 - end.microsecond)  # Round up on end.

        if self.debug > 0:
            print("cache_fill_runs: {} - {} minevt: {}".format(start, end, min_event))

        num_runs = self.get_runs_from_rcdb(start, end, min_event)  # Get the new data from the RCDB.

        if num_runs == 0 and self._db is not None:
            # We still need to adjust the "end" time, so we don't end up in a endless loop.
            # Now data.All_Runs is likely empty, so use RCDB to get end of last run.
            rcdb_runs = self._db.session.query(Run).order_by(Run.start_time.desc()).limit(20)  # Set a limit to speedup
            use_index = 0

            while not rcdb_runs[use_index].get_condition_value("is_valid_run_end"):
                use_index += 1
                if self.debug > 4:
                    print("num_runs == 0: end adjust found: ", use_index, " = ",
                          rcdb_runs[use_index].get_condition_value("is_valid_run_end"))

            end = rcdb_runs[use_index].end_time
            if self.debug > 2:
                print("Adjusted end for empty rcdb fetch to: ", end)
            return num_runs, end

        # Two scenarios now:
        # 1: The run period we just added is somewhere in the middle of all possible runs. It is thus extremely unlikely
        #    that a new run is added in between the 'end' and the last run we just fetched. It is thus OK to set the end
        #    of this cache period to the end requested. We want to do this, because it improves the chances we get
        #    overlaps of the run sections.
        #
        # 2: The run period we just added stretches beyond the last possible run (e.g. now() or later).
        #    It is thus likely that runs are added later, which may be added before the 'end' time selected.
        #    To avoid missing such runs, we need to set the 'end' if the cache period to the end of the last run.
        #    This is further complicated by the possibility that the last run is still ongoing, and thus has an
        #    incorrect number of events, files etc.
        #
        # To detect between 1 and 2, get the very last run from the database.

        last = self._db.session.query(Run).order_by(Run.start_time.desc()).limit(1).first()

        if end > last.start_time:  # This is scenario 2, adjust the end time
            # The 'end' for the cache period must be the 'end' of the last run,
            # or if that is not valid (run end not recorded)
            # then 'end' should be the end of the run one before.
            end_of_last_run_valid = self.All_Runs[(self.All_Runs["start_time"] > start) &
                                                  (self.All_Runs["end_time"] < end)].iloc[-1].is_valid_run_end

            if end_of_last_run_valid is None or not end_of_last_run_valid:
                if self.debug:
                    print("WARNING: Last run does not have is_valid_run_end set.")
                # If the last run does not have a valid end, could it be that it is still ongoing?
                # Expect a run to not be more than 8 hours.
                if self.All_Runs[(self.All_Runs["start_time"] > start) &
                                 (self.All_Runs["end_time"] < end)].iloc[-1].end_time > (datetime.now() -
                                                                                         timedelta(hours=8)):
                    if self.debug:
                        print("WARNING: Run with is_valid_run_end not set is withing last 8 hours,"
                              " not adding to cache.")
                    # We drop the run from the table, so it will not go into the cache.
                    self.All_Runs.drop(self.All_Runs.index[-1], inplace=True)
                    num_runs -= 1
                    if len(self.All_Runs[(self.All_Runs["start_time"] > start) &
                                         (self.All_Runs["end_time"] < end) &
                                         (self.All_Runs["is_valid_run_end"])]
                           ) <= 1:  # there is no run left.
                        if self.debug:
                            print("No runs left to add.")
                        last_2_runs = self._db.session.query(Run).order_by(Run.start_time.desc()).limit(2)
                        if last_2_runs.count() != 2:
                            print("Problem I do not understand. The DB did not return 2 runs.")
                            sys.exit(1)
                        end = last_2_runs[1].end_time
                        if self.debug:
                            print("Set the end to:", end)
                        return num_runs, end

            # Get the end of the last run:
            end_of_last_run = self.All_Runs[(self.All_Runs["start_time"] > start) &
                                            (self.All_Runs["end_time"] < end) &
                                            (self.All_Runs["is_valid_run_end"])].iloc[-1].end_time

            if end_of_last_run is None or type(end_of_last_run) is not pd.Timestamp:
                # This seems to be really, really rare.
                print("WARNING: Last run added to cache does not have a proper end time!!!")
                # end_of_last_run = self.All_Runs[(self.All_Runs["start_time"] > start) &
                #                                 (self.All_Runs["end_time"] < extend_to)].iloc[-1].start_time
                # end_of_last_run -= timedelta(0,1)
                # One second before last start, so the next time this run will get updated.

            if self.debug > 2:
                print("Change the end time from {} to {}".format(end, end_of_last_run))
            end = end_of_last_run

        # Now update the chache times table with the new entry from "start" to "end"
        if len(self._cache_known_data) == 0 or "index" not in self._cache_known_data:
            self._cache_known_data = pd.DataFrame({"index": [0], "start_time": [start],
                                                  "end_time": [end], "min_event_count": [1]})
        else:
            cache_known_data_add = pd.DataFrame({"index": [self._cache_known_data["index"].max()+1],
                                                "start_time": [start], "end_time": [end], "min_event_count": [1]})
            self._cache_known_data = pd.concat([self._cache_known_data, cache_known_data_add],
                                               ignore_index=True, sort=False)
        self._cache_known_data = self._cache_known_data.sort_values("start_time")

        if self.debug > 2:
            print("cache_known_data:")
            print(self._cache_known_data)

        # Add the new runs to the run cache table in SQL:
        self.All_Runs.to_sql("Runs_Table", self._cache_engine, if_exists="append")
        # And the new chuck to the data ranges table:
        self._cache_known_data.to_sql("Known_Data_Ranges", self._cache_engine, if_exists='replace')

        # We only want to keep those runs with the min_event criteria in the All_Runs table
        test_num_too_small = self.All_Runs["event_count"] < min_event
        # Drop the ones with too few counts:
        self.All_Runs = self.All_Runs.drop(self.All_Runs[test_num_too_small].index)

        return num_runs, end  # Return number of runs and possibly modified end time.

    def _cache_get_runs(self, start, end, min_event):
        """Get runs directly from the cache and only from the cache.
        Result is stored in All_Runs if it was empty, or appended to All_Runs if not already there, and then sorted.
        return the number of runs added."""
        # All the data is available in the cache, so just get it.

        if not sqlalchemy.inspect(self._cache_engine).has_table("Runs_Table"):
            #
            # Table is not there, so cache was not initialized, just return so data will be fetched
            # from RCDB.
            #
            return 0

        start = start + timedelta(0, 0, -start.microsecond)  # Round down on start
        if end.microsecond != 0:
            end = end + timedelta(0, 0, 1000000 - end.microsecond)  # Round up on end.

        if self.debug > 0:
            print("Getting data from cache for {} - {} > {} evt. ".format(start, end, min_event))

        # Get the runs ordered already.
        sql = f"select * from Runs_Table where start_time >= '{start}' and " \
              f"end_time <= '{end}' and event_count>= '{min_event}' order by number"
        new_runs = pd.read_sql(sql, self._cache_engine, index_col="number", parse_dates=["start_time", "end_time"])

        if self.All_Runs is None or len(self.All_Runs) == 0:
            self.All_Runs = new_runs
        else:
            #
            # It is possible some (or all) of the new runs had already been fetched earlier.
            # We do not want overlaps, so we need to sort this out.
            if self.debug > 1:
                print("Checking overlap for runs {} to All_Runs.".format(len(new_runs)))
            new_runs.drop(new_runs[new_runs.index.isin(self.All_Runs.index)].index, inplace=True)
            # Drop the overlapping runs.
            if self.debug > 1:
                print("Number of runs that will be appended: {}".format(len(new_runs)))
            if len(new_runs) > 0:
                self.All_Runs = pd.concat([self.All_Runs, new_runs])
                self.All_Runs.sort_index(inplace=True)
                if np.any(self.All_Runs.duplicated()):
                    print("ARGGGGG: we have duplicates in All_Runs")
                    self.All_Runs.drop_duplicates(inplace=True)

        if self.debug > 1:
            print("Got {} runs from cache.".format(len(new_runs)))

        return len(new_runs)

    def _cache_consolidate(self):
        """Goes through cached regions and checks if any have 'grown' to touch or overlap.
        If so, combine those regions."""
        # We go from early to late, but make sure there is a +1
        if self.debug > 1:
            print("Cache consolidate: ")
            print(self._cache_known_data)

        for index in range(len(self._cache_known_data)):
            if index + 1 < len(self._cache_known_data):
                if (self._cache_known_data.loc[self._cache_known_data.index[index], "end_time"] >=
                        self._cache_known_data.loc[self._cache_known_data.index[index + 1], "start_time"]):
                    # Eliminate the second in favor of the first.
                    self._cache_known_data.loc[self._cache_known_data.index[index], "end_time"] = \
                        self._cache_known_data.loc[self._cache_known_data.index[index + 1], "end_time"]
                    self._cache_known_data = self._cache_known_data.drop(self._cache_known_data.index[index + 1])
                    if self.debug > 1:
                        print("------ Eliminated {} in favor of {}".format(index + 1, index))
                        print(self._cache_known_data)
                        print()
        # Make sure the cache is still sorted (it should be.)
        self._cache_known_data = self._cache_known_data.sort_values("start_time")
        # Write out the new table to SQL.

        # TODO: update sql instead
        self._cache_known_data.to_sql("Known_Data_Ranges", self._cache_engine, if_exists='replace')

    def get_runs(self, start, end, min_event):
        """Get runs with get_runs_only, and then run the select_good_runs and add_current_data_to_runs."""
        self.get_runs_only(start, end, min_event)
        good_runs = self.select_good_runs()
        self.add_current_data_to_runs()
        return len(self.All_Runs)

    def get_runs_only(self, start, end, min_event):
        """Fetch the runs from start time to end time with at least min_event events.
        Checking local cache if runs have already been fetched (if self._cache_engine is not None)
        If not get them from the rcdb and update the cache (if not None).
        Times are rounded down to the second for start and up to the second for end."""

        # TODO: This whole algorithm has gotten away from itself. Too complicated! It should be replaced.
        # Currently this is too difficult to trace, verify, debug and fix.

        self.min_event_count = min_event

        start = start + timedelta(0, 0, -start.microsecond)  # Round down on start
        if end.microsecond != 0:
            end = end + timedelta(0, 0, 1000000 - end.microsecond)  # Round up on end.

        if self.debug:
            print("get_runs from {} - {} ".format(start, end))

        if self._cache_engine is None or self._cache_engine is False:  # No cache, so just get the runs.
            if self.debug > 2:
                print("Getting runs bypassing cache, for start={}, end={}, min_event={}".format(start, end, min_event))
            num_runs = self.get_runs_from_rcdb(start, end, min_event)
            return num_runs

        num_runs_cache = self._cache_get_runs(start, end, min_event)  # Get whatever we have in the cache already.

        if self._db is None:  # No DB so we are done
            if self.All_Runs:
                return len(self.All_Runs)
            else:
                return 0

        # Check for overlaps of request with cache.
        cache_overlaps, cache_extend_before, cache_extend_after = self._check_for_cache_hits(start, end)

        if len(cache_overlaps) + len(cache_extend_before) + len(cache_extend_after) == 0:  # No overlaps at all.
            num_runs, tmp = self._cache_fill_runs(start, end, min_event)  # so get the date for the entire stretch.
            return num_runs + num_runs_cache

        if len(cache_overlaps) > 1:
            print("Cache is dirty: multiple full overlaps.")

        while len(cache_overlaps) == 0:
            if self.debug > 2:
                print("check_cache_hits: ", cache_overlaps, cache_extend_before, cache_extend_after, start, end)
            #
            # Iteratively extend the cache with time periods, starting from below.
            #
            if len(cache_extend_before) > 0:
                min_before = np.min(cache_extend_before)
            else:
                min_before = None

            if len(cache_extend_after) > 0:
                min_after = np.min(cache_extend_after)
            else:
                min_after = None

            if (min_after is None) or ((min_before is not None) and (min_before <= min_after)):
                # Start extending the before of the earliest overlap period to "start"

                save_all_runs = None
                save_all_runs, self.All_Runs = self.All_Runs, save_all_runs  # Save what is in All_Runs.
                min_before_start = self._cache_known_data.loc[self._cache_known_data.index[min_before], "start_time"]
                if self.debug > 2:
                    print("Extending {} before from {}  to {}".format(min_before, min_before_start, start))
                # Add the new data to cache and All_Runs up to min_before_start
                num_runs, new_end = self._cache_fill_runs(start, min_before_start, min_event)
                if num_runs == 0:
                    self.All_Runs = save_all_runs
                else:
                    #  self.All_Runs = self.All_Runs.append(save_all_runs, sort=True)  # Append the saved runs.
                    self.All_Runs = pd.concat([self.All_Runs, save_all_runs], sort=True)
            else:
                # The earliest overlap is in extend_after, so "start" is inside this overlap period.
                # We need to extend out to the earliest of "end" or the "start" of the next period.
                min_after_end = self._cache_known_data.loc[self._cache_known_data.index[min_after], "end_time"]
                if (min_before is not None) and (self._cache_known_data.loc[self._cache_known_data.index[min_before],
                                                                            "start_time"] < end):
                    # extend to min_before_start
                    extend_to = self._cache_known_data.loc[self._cache_known_data.index[min_before], "start_time"]
                else:
                    # Try to extend to the requested end time.
                    extend_to = end

                if self.debug > 2:
                    print("Extending {} after from {}  to {}".format(min_after, min_after_end, extend_to))
                save_all_runs = None
                save_all_runs, self.All_Runs = self.All_Runs, save_all_runs  # Save what is in All_Runs.
                num_runs, end = self._cache_fill_runs(min_after_end, extend_to, min_event)
                # Add the new data to cache and All_Runs.
                # Note we re-set the end to the possibly corrected end.
                # This is needed because otherwise we keep checking!
                if num_runs != 0:
                    if self.debug > 2:
                        print("Appending runs. ")
                    self.All_Runs = pd.concat([save_all_runs, self.All_Runs], sort=True)
                else:
                    # No new runs, so restore what we had before.
                    # TODO: Fix this. When no new runs, because no new data, this can cause an infinite loop.
                    # Do we set the loop to break out?
                    self.All_Runs = save_all_runs
                    end = min_after_end

                if self.All_Runs is None or len(self.All_Runs) == 0:  # There are no runs at all. Just quit.
                    return 0

            if self.debug > 2:
                print("New cache config:")
                print(self._cache_known_data)

            #
            #  Now go through until there is full overlap.
            #
            self._cache_consolidate()  # If regions are now bordering, combine them.
            cache_overlaps, cache_extend_before, cache_extend_after = self._check_for_cache_hits(start, end)

        # We are done
        return len(self.All_Runs)

    def get_excluded_runs(self, filename):

        if os.path.exists(filename):
            with open(filename) as ff:
                for line in ff:
                    line = line.replace('\n', '')

                    if line not in self.ExcludeRuns:
                        self.ExcludeRuns.append(line)

    def beam_aten_corr(self, run):
        """This method computes a correction to the Faraday Cup current from sclare_calc1b Epics channel from Mya
        This correction is sometimes needed, such as in HPS runs with a beam blocker in front of the FCup.
        If self.atten_dict is None, this correction is ignored.
        if self.atten_dict is a dictionary, and the current target is in the dictionary then
              if atten_dict is a list or tuple, the run number is compared to [0] < run < [1] if true use correction.
              else use correction
        The correction is equal to atten_dict(target_name)/atten_dict('Empty')
        """
        corr = 1.
        # print( self.All_Runs.loc[run, 'target'] )
        targ_name_no_spaces = (self.All_Runs.loc[run, 'target']).rstrip()
        if self.atten_dict and targ_name_no_spaces in self.atten_dict:
            if (type(self.atten_dict[targ_name_no_spaces]) is list) or \
                    (type(self.atten_dict[targ_name_no_spaces]) is tuple):
                if self.atten_dict[targ_name_no_spaces][0] < run < self.atten_dict[targ_name_no_spaces][1]:
                    corr = self.atten_dict[targ_name_no_spaces][2] / self.atten_dict['Empty'][2]
                    if self.debug > 3:
                        print(f"Using a beam attenuation correction of {corr} for run {run}")
            else:
                corr = self.atten_dict[targ_name_no_spaces] / self.atten_dict['Empty']
                if self.debug > 3:
                    print(f"Using a beam attenuation correction of {corr}")
        # print ('Corr is ' + str(corr))

        return corr

    def get_runs_from_rcdb_by_run_number(self, run_min, run_max, min_event_count):
        """Return a dictionary with a list of runs with run numbers from run_min to run_max.
         This will get the list directly from the rcdb database, not looking at the local cache.
         The output is then parsed through process_runs_from_rcdb."""

        runs = self._db.get_runs(run_min, run_max)
        self.process_runs_from_rcdb(runs)
        # The RCDB is by and large a piece of ..., er, very unreliable. Translate "None" to 0.
        self.All_Runs.loc[(self.All_Runs.event_count == "None"), "event_count"] = 0
        self.All_Runs = self.All_Runs.loc[self.All_Runs.event_count >= min_event_count]  # Filter the low count runs.

        return len(self.All_Runs)

    def get_runs_from_rcdb(self, start_time, end_time, min_event_count):
        """Return a dictionary with a list of runs for each target in the run period from start_time to end_time.
        This will get the list directly from the rcdb database, not looking at the local cache.
        The output is then parsed through process_runs_from_rcdb."""
        # A fundamental issue with how the tables are setup is that you cannot construct
        # a simple query filtering on two different conditions. You would need to
        # construct a rather complicated query with a sub query, at which point you may be
        # better off just using MySQL directly. Such queries get really complicated, and this
        # isn't really needed here. Better to do one query and then filter using Python.

        if self.debug:
            print("Getting runs from RCDB: {} - {} minevt: {}".format(start_time, end_time, min_event_count))

        if self._db is not None:
            q = self._db.session.query(Run).join(Run.conditions).join(Condition.type) \
                .filter(Run.start_time >= start_time).filter(Run.start_time <= end_time) \
                .filter((ConditionType.name == "event_count") & (Condition.int_value >= min_event_count))
        else:
            return 0

        if self.debug:
            print("Found {} runs.\n".format(q.count()))
        num_runs = q.count()
        if num_runs == 0:
            return 0
        all_runs = q.all()

        if self.fix_bad_rcdb_start_times:
            first_run_number = all_runs[0].number
            last_run_number = all_runs[-1].number
            if self.debug > 0:
                print(f"Getting runs from RCDB from {first_run_number} to {last_run_number}")
            num_runs = self.get_runs_from_rcdb_by_run_number(first_run_number, last_run_number, min_event_count)
        else:
            num_runs = self.process_runs_from_rcdb(all_runs)
            # The RCDB is by and large a piece of ..., er, very unreliable. Translate "None" to 0.
            self.All_Runs.loc[(self.All_Runs.event_count == "None"), "event_count"] = 0
            self.All_Runs = self.All_Runs.loc[self.All_Runs.event_count >= min_event_count]  # Filter the low count runs.

        return num_runs

    def process_runs_from_rcdb(self, db_runs):
        """Process the rund from the RCDB and enter them into the Pandas DataFrame. """

        num_runs = len(db_runs)
        runs = []

        for R in db_runs:
            run_dict = {"number": R.number, "start_time": R.start_time, "end_time": R.end_time}

            if str(R.number) in self.ExcludeRuns:
                # print ("Excluding" + str(R.number) + "Since it it in Cameron's list")
                continue

            for c in self.Useful_conditions:
                value = R.get_condition_value(c)
                if value is not None:          # Try to scrub for junk input. This may need expanding depending on junk.
                    run_dict[c] = value
                else:
                    run_dict[c] = "None"

                # Do Target name translation if the target_properties are set and contain an entry, "names"
                if (c == "target") and (self.target_properties is not None) and ("names" in self.target_properties):
                    if run_dict[c].strip() in self.target_properties["names"]:
                        run_dict[c] = self.target_properties["names"][run_dict[c].strip()]
                    else:
                        print(f"Warning. The target name {run_dict[c]} does not appear in the translation table.")

            if run_dict["run_start_time"] is not None and run_dict["run_start_time"] != "None":
                run_dict["start_time"] = run_dict["run_start_time"]  # Use the run_start_time and run_end_time
            elif run_dict["start_time"] is None or run_dict["start_time"] == "None":
                if self.debug > 0:
                    print(f"Run {R.number} has no proper start time!!!")
                if not self.fix_bad_rcdb_start_times:
                    continue
            if run_dict["run_end_time"] is not None and run_dict["run_end_time"] != "None":
                run_dict["end_time"] = run_dict["run_end_time"]      # from the RCDB records.
            elif run_dict["end_time"] is None or run_dict["end_time"] == "None":
                if self.debug > 0:
                    print(f"Run {R.number} has no proper end time!!!")
                if not self.fix_bad_rcdb_start_times:
                    continue

            # This allows for start/end corrections.
            runs.append(run_dict)

        self.All_Runs = pd.DataFrame(runs)
        self.All_Runs["target"] = [x.strip() for x in self.All_Runs["target"]]   # Squeeze spaces.

        self.All_Runs.loc[:, "selected"] = True  # Default to selected
        # Rewrite the run_config to eliminate the long directory name, which is not useful.
        self.All_Runs.loc[:, "run_config"] = [self.All_Runs.loc[r, "run_config"].split('/')[-1]
                                              for r in self.All_Runs.index]
        self.All_Runs.set_index('number', inplace=True)
        return num_runs

    def add_current_cor(self, runnumber, override=False, current_channel=None, livetime_channel=None):
        """Add the livetime corrected charge and luminosity for a run to the pandas DataFrame.
        for that run.
        arguments:
             runnumber  - The run number for which you want to fetch the data.
             override   - Set to true if data should be fetched even if it seems it was already.
             current_channel  - Override the RunData.Current_Channel with this channel.
             livetime_channel - Override the RunData.LiveTime_Channel with this channel.
        """

        if current_channel is None:
            current_channel = self.Current_Channel
        if livetime_channel is None:
            livetime_channel = self.LiveTime_Channel

        if not override and \
                (current_channel in self.All_Runs.keys()) and \
                not np.isnan(self.All_Runs.loc[runnumber, current_channel]):
            return

        if self.debug > 4:
            print(f"add_current_cor, run= {runnumber:5d}  start={self.All_Runs.loc[runnumber,'start_time']} "
                  f"end={self.All_Runs.loc[runnumber, 'end_time']}")

        if pd.isnull(self.All_Runs.loc[runnumber, "start_time"]) or \
           pd.isnull(self.All_Runs.loc[runnumber, "end_time"]):
            # We cannot compute the current if the times are not known.
            return

        current = self.Mya.get(current_channel,
                               self.All_Runs.loc[runnumber, "start_time"],
                               self.All_Runs.loc[runnumber, "end_time"],
                               run_number=runnumber)

        current.fillna(0, inplace=True)     # Replace Nan or None with 0
        live_time = self.Mya.get(livetime_channel,
                                 self.All_Runs.loc[runnumber, "start_time"],
                                 self.All_Runs.loc[runnumber, "end_time"],
                                 run_number=runnumber)
        # If there is bad data in the live_time, None or nan, then set those to zero.
        if len(live_time) < 2:
            start = self.All_Runs.loc[runnumber, 'start_time']
            end = self.All_Runs.loc[runnumber, 'end_time']
            print(f"RunData: live_time < 2. This should not be possible -- Cache Corrupted?"
                  f" run number:{runnumber}  start:{start} end:{end} ")
            live_time = pd.DataFrame({'ms': [start.timestamp() * 1000, end.timestamp() * 1000],
                                      'value': [100., 100.],
                                      'time': [start, end]})

        if len(live_time) < 3:
            live_time.fillna(1., inplace=True)  # Replace Nan or None with 1 - no data returned.
        else:
            live_time.fillna(0, inplace=True)  # Replace Nan or None with 0
            live_time.loc[live_time.value.isna(), 'value'] = 0

        if current_channel == "scaler_calc1b":
            # Getting the target thickness dependend FCup charge correction
            curr_correction = self.beam_aten_corr(runnumber)
            # Applying the correction
            current['value'] *= curr_correction

        #
        # The sampling of the current and live_time are NOT guaranteed to be the same.
        # We interpolate the live_time at the current time stamps to compensate.
        #
        try:
            live_time_corr = np.interp(current.ms, live_time.ms, live_time.value) / 100.  # convert to fraction from %
        except Exception as e:
            print(f"RunData(live_time_corr) There is a problem with the data for run {runnumber}")
            print(e)
            return

        #
        # Now we can just multiply the live_time_corr with the current.
        #
        try:
            current_corr = current.value * live_time_corr
        except Exception as e:
            print(f"RunData(current_corr) There is a problem with the data for run {runnumber}")
            print(e)
            return

        #
        # We need to do a proper trapezoidal integration over the current data points.
        # Store the result in the data frame.
        #
        # Scale conversion:  I is in nA, dt is in ms, so I*dt is in nA*ms = 1e-9 A*1e-3 s = 1e-12 C
        # If we want mC instead of Coulombs, the factor is 1e-12*1e3 = 1e-9
        #

        self.All_Runs.loc[runnumber, current_channel] = np.trapz(current.value, current.ms) * 1e-9  # mC
        self.All_Runs.loc[runnumber, livetime_channel] = np.trapz(live_time.value, live_time.ms)
        self.All_Runs.loc[runnumber, current_channel+"_corr"] = np.trapz(current_corr, current.ms) * 1e-9  # mC
        self.All_Runs.loc[runnumber, "charge"] = self.All_Runs.loc[runnumber, current_channel+"_corr"]
        # self._Mya_cache[runnumber] = pd.DataFrame({"time":current.time,"current":current.value,
        #          "live_time":live_time_corr,"current_cor":current_corr})
        #
        target = self.All_Runs.loc[runnumber, "target"]

        if target in self.target_dens:
            if (type(self.target_dens[target]) is float) or (type(self.target_dens[target]) is int):
                target_dens = self.target_dens[target]
            else:
                target_thick = self.target_dens[target][0]
                target_rho = self.target_dens[target][1]
                target_mmass = self.target_dens[target][2]
                target_dens = target_rho * target_thick / target_mmass
        else:
            target_dens = 1.

        # 1 C/e = 6.241509074460763e+18  so 1 mC/e = 6.241509074460763e+15
        charge = self.All_Runs.loc[runnumber, "charge"]  # Integrated number of e- in beam
        # Avogadro = 6.02214076e+23 / mole
        # (1.*u.avogadro_constant*1.*u.cm * 1.*u.g/(1*u.Unit('cm^3'))/(1.*u.Unit('g/mol'))).to('1/fb') =
        #  6.02214076e-16 <Unit('1 / femtobarn')>
        # 1 C/e = 6.241509074460763e+18  so 1 mC/e = 6.241509074460763e+15
        # ((1.*u.avogadro_constant*1.*u.cm * 1.*u.g/(1*u.Unit('cm^3'))/(1.*u.Unit('g/mol')))*
        #    (1*u.mC/(1*u.elementary_charge))).to('1/fb') =
        # 6.241509074460763e+15 * 6.02214076e-16 = 3.758724620122004
        lumi = 3758.724620122003 * charge * target_dens   # In 1/pb
        self.All_Runs.loc[runnumber, current_channel+"_lumi"] = lumi
        self.All_Runs.loc[runnumber, "luminosity"] = lumi
        return

    def add_current_data_to_runs(self, targets=None, run_config=None,
                                 override=False, current_channel=None, livetime_channel=None):
        """Add the mya data for beam current to all the runs with selected flag set.
        You can select other runs by specifying 'targets' and 'run_config' similarly to list_selected_runs()"""
        # We want to sort the data by target type.
        # We also want to veto runs that do not have the correct configuration,
        # such as pulser runs, FEE runs, etc.
        #
        good_runs = self.list_selected_runs(targets, run_config)
        if self.debug > 3:
            print(good_runs)
        if len(good_runs) > 0:
            for rnum in self.list_selected_runs(targets, run_config):
                self.add_current_cor(rnum, override=override,
                                     current_channel=current_channel, livetime_channel=livetime_channel)
        else:
            # Even if there are no good runs, make sure that the "charge" column is in the table!
            # This ensure that when you write to DB the charge column exsists.
            self.All_Runs.loc[:, "charge"] = np.NaN

    def select_good_runs(self):
        """Select the runs that have the selection criteria of Production_run_type
        and good_triggers.
        self.Production_run_type and self.Good_triggers can be either a list of strings to match
        or a regular expression to match. Use '.*' to match everything. """

        if self.debug > 5:
            print("select_good_runs: Production_run_type = ", self.Production_run_type)
        good_runs = []
        for rnum in self.All_Runs.index:
            test1 = False
            if self.Production_run_type is None:
                test1 = True
            elif type(self.Production_run_type) is list:
                test1 = self.All_Runs.loc[rnum, "run_type"] in self.Production_run_type
            elif type(self.Production_run_type) is str:
                if self.debug > 5:
                    print(f"select_good_runs: rnum={rnum:6d} run_type: '{self.All_Runs.loc[rnum, 'run_type']}'")
                if self.All_Runs.loc[rnum, "run_type"] is not None and \
                        re.match(self.Production_run_type, self.All_Runs.loc[rnum, "run_type"]):
                    test1 = True
            else:
                print("Incorrect type for self.Production_run_type:", type(self.Production_run_type))
                sys.exit(1)

            test2 = False
            trigger = self.All_Runs.loc[rnum, "run_config"]  # .split('/')[-1] ## The split is already done.
            if type(self.Good_triggers) is list:
                test2 = trigger in self.Good_triggers
            elif type(self.Good_triggers) is str:
                if re.match(self.Good_triggers, trigger):
                    test2 = True

            if not test2:
                if trigger not in self.not_good_triggers:
                    self.not_good_triggers.append(trigger)
            self.All_Runs.loc[rnum, "selected"] = test1 & test2
            if test1 & test2:
                good_runs.append(rnum)

        return good_runs

    def list_selected_runs(self, targets=None, run_config=None, date_min=None, date_max=None, runs=None):
        """return the list of run numbers with the selected flag set. If 'targets' is specified, list only the runs with
        those targets.
        Arguments:
        targets       - None, False, string or list
        run_config    - None, False, string or list.
        date_min      - Earliest date to consider, must be datetime object or None
        date_max      - Latest date to consider, must be datetime object or None
        runs          - Runs DateFrame to operate on, or self.All_Runs if None

        If targets is None or False, all targets will be listed.
        If targets is a string, it will be matched case insensitive regex style.
        I.e. "um W" will give all thickness Tungsten.
        If targets is a list, it will be matched only for the exact strings in the list.
        So "4 um W" will be empty, because the space after the W is missing.

        If run_config is None, then only the pre-selected runs (set with All_Runs.selected True or 1) are used.
        If run_config is False, then all run_configs (triggers) are used.
        If run_config is a string, is will be matched case insensitive regex style.
        If run_config is a list, then that list will be used instead of the data.Good_triggers list.
        """

        if runs is None:
            runs = self.All_Runs

        test_run_config = None
        if run_config is None:
            test_run_config = runs.selected
        elif run_config is False or len(run_config) == 0:
            test_run_config = np.array([True] * len(runs))  # I.e. select all. Yes, funny logic.
        elif type(run_config) is str:
            test_run_config = runs["run_config"].str.contains(run_config, case=False)
        elif type(run_config) is list:
            # test_run_config = runs["run_config"].isin(run_config)  #-- This works, but requires exact match
            test_run_config = np.array([False] * len(runs))  # Set All to false
            for t_str in run_config:
                test_run_config = test_run_config | runs["run_config"].str.contains(t_str, case=False)
        else:
            print("I do not know what to do with run_config = ", type(run_config))

        test_target = None
        if targets is None or targets is False or len(targets) == 0:
            test_target = np.array([True] * len(runs))
        elif type(targets) is str:
            test_target = runs["target"].str.contains(targets, case=False)
        elif type(targets) is list:
            test_target = runs.target.isin(targets)
        elif type(targets) is dict:
            test_target = runs.target.isin(targets.keys())
        else:
            print("I do not know what to do with target = ", type(targets))

        test_date_min = None
        if date_min is None:
            test_date_min = np.array([True] * len(runs))
        elif type(date_min) is datetime:
            test_date_min = (date_min < runs.start_time)

        test_date_max = None
        if date_max is None:
            test_date_max = np.array([True] * len(runs))
        elif type(date_max) is datetime:
            test_date_max = (runs.end_time < date_max)

        return runs.index[test_run_config & test_target & test_date_min & test_date_max]

    def compute_cumulative_charge(self, targets=None, run_config=None, date_min=None, date_max=None, runs=None):
        """Compute the cumulative charge, luminosity, and event count for the runs in the current All_Runs table.
           The increasing numbers are put in the table in 'sum_charge', 'sum_charge_norm' and 'sum_event_count',
           and 'sum_lumi'
           The runs are selected with list_selected_runs with the same 'targets' and 'run_config' and date arguments.
           If runs is set to a Pandas data frame of runs, compute the sums only for those runs.
           Returns the end value of: (sum_charge,sum_charge_norm,sum_event_count)"""

        # filter the runs on target, run_config, date_min, date_max.
        selected = self.list_selected_runs(targets=targets, run_config=run_config,
                                           date_min=date_min, date_max=date_max, runs=runs)

        if runs is None:
            runs = self.All_Runs

        if self.debug > 1:
            print("Computing cumulative charge, lumi, and event count for runs:", list(selected))

        runs.loc[selected, "sum_event_count"] = np.cumsum(runs.loc[selected, "event_count"])
        runs.loc[selected, "sum_charge"] = np.cumsum(runs.loc[selected, "charge"])
        runs.loc[selected, "sum_lumi"] = np.cumsum(runs.loc[selected, "luminosity"])

        sum_charge_per_target = self.target_dens.copy()
        for k in sum_charge_per_target:
            sum_charge_per_target[k] = 0.

        cumsum_charge_norm = 0.
        for run in selected:
            target = runs.loc[run, "target"]
            if target in self.target_dens:
                target_norm = 1.
                if 'norm' in self.target_dens:
                    if (type(self.target_dens[target]) is float) or (type(self.target_dens[target]) is int):
                        target_norm = self.target_dens[target]/self.target_dens['norm']
                    else:
                        target_norm = self.target_dens[target][0]/self.target_dens['norm'][0]

                if not np.isnan(runs.loc[run, "charge"]):
                    sum_charge_per_target[runs.loc[run, "target"]] += runs.loc[run, "charge"]
                    runs.loc[run, "sum_charge_targ"] = sum_charge_per_target[
                        runs.loc[run, "target"]]
                    cumsum_charge_norm += runs.loc[run, "charge"] * target_norm
                    runs.loc[run, "sum_charge_norm"] = cumsum_charge_norm

        if len(runs.loc[selected]):
            if "sum_charge_norm" in runs.keys():
                return (runs.loc[selected, "sum_charge"].iloc[-1],
                        runs.loc[selected, "sum_charge_norm"].iloc[-1],
                        runs.loc[selected, "sum_event_count"].iloc[-1])
            else:
                return (runs.loc[selected, "sum_charge"].iloc[-1],
                        None,
                        runs.loc[selected, "sum_event_count"].iloc[-1])
        else:
            return 0, 0, 0


if __name__ == "__main__":
    import argparse

    print("Do something!")
