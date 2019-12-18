#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import re

try:
    from rcdb.model import Run, ConditionType, Condition, all_value_types
    from rcdb.provider import RCDBProvider
except:
    print("Please set your PYTHONPATH to a copy of the rcd Python libraries.\n")
    print("sys.path: ",sys.path)
    sys.exit(1)

try:
    import sqlalchemy
except:
    print("We need the sqlalchemy installed for the database, but I could not find it in:")
    print("sys.path: ",sys.path)
    sys.exit(1)


# import requests
# from requests.packages.urllib3.exceptions import InsecureRequestWarning
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
#
from datetime import datetime,timedelta

try:
    import pandas as pd
except:
    print("Sorry, but you really need a computer with 'pandas' installed.")
    print("Try 'anaconda' python, which should have both.")
    sys.exit(1)

import json
import numpy as np

from RunData.MyaData import MyaData
#
# Some global configuration settings.
# These will need to go to input values on a web form.
#
class RunData:

    def __init__(self,I_am_at_jlab=False,sqlcache=True):
        ''' Set things up. If not at JLab you will be asked for CUE username and password.
        sqlcache=False will prevent caching querries in a local sqlite3 database.
        sqlcache='mysql://user.pwd@host/db' will use that DB as the cache. String is a sqlalchemy style DB string.
        sqlcache=True  will use a local sqlite3 file for caching.
        '''
        self.Production_run_type = "PROD.*"     # Type of production runs to consider.
        self.Useful_conditions=['is_valid_run_end', 'user_comment', 'run_type',
        'target', 'beam_current_request', 'operators','event_count',
        'events_rate','run_config', 'status',
         'evio_files_count', 'megabyte_count', 'run_start_time', 'run_end_time']   # List of conditions to put in tables.
        self.Good_triggers='hps_v..?_?.?\.cnf'   # Regex string or list of trigger conditions to use for run selection.
        self.not_good_triggers=[]
        self.ExcludeRuns = []  # This runs are not present in Cameron's list which he obtained parsing the google spreadsheet, and maybe other sources too
        self.min_event_count = 1000000
        self.target_dict={}
        self.atten_dict={}     # This is a dictionary of target dependent correction factors for correcting the FCup current.
        self.at_jlab=I_am_at_jlab
        self.All_Runs=None
        self.debug=0

        self.Current_Channel="scaler_calc1b"    # Mya Channel for the current from FCUP.
        self.LiveTime_Channel="B_DAQ_HPS:TS:livetime"   # Mya Channel for the livetime.
        self._db=None
        self._session=None

        self._cache_engine=None
        self._cache_known_data=None
        self._cache_file_name="run_data_cache.sqlite3"


        self.start_rcdb()
        self.Mya = MyaData(I_am_at_jlab)
        self.start_cache(sqlcache)

    def __str__(self):
        '''Return a table with some of the information, to see what is in the All_Runs conveniently. '''
        out=str(self.All_Runs.loc[:,["start_time","end_time","target","run_config","event_count"]])
        return(out)

    def start_rcdb(self):
        '''Setup a connection to the RCDB
        return: an RCDB handle'''
        try:
            connection_string = os.environ["RCDB_CONNECTION"]
        except:
            connection_string = "mysql://rcdb@clasdb.jlab.org/rcdb"
            # print("Using standard connection string from HallB")

        self._db = RCDBProvider(connection_string)

    def start_cache(self,connector_string=True):
        '''Start up the cache backend according to connector_string.
        If connector_string is True, use a local sqlite3 file.'''
        #
        # This is NOT some super smart caching setup. I just hope it is better than nothing.
        #
        if connector_string is False:
            self._cache_engine=None
            return

        if connector_string is True:
            connector_string = "sqlite:///"+self._cache_file_name

        self._cache_engine = sqlalchemy.create_engine(connector_string)
        #
        # We typically query a time range and an event number cut.
        # This complicates caching, because the next request may be incorporated
        # in a timerange, but have a different (lower) event cut.
        # The Known_Data_Range table tries to help with this.
        if not self._cache_engine.dialect.has_table(self._cache_engine, "Known_Data_Ranges"):
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
            kdr = sqlalchemy.Table('Known_Data_Ranges',meta,
                sqlalchemy.Column('index', sqlalchemy.Integer, primary_key=True),
                sqlalchemy.Column('start_time',sqlalchemy.DateTime),
                sqlalchemy.Column('end_time',sqlalchemy.DateTime),
                sqlalchemy.Column('min_event_count',sqlalchemy.Integer))
            meta.create_all(self._cache_engine)

        self._cache_known_data = pd.read_sql("Known_Data_Ranges",self._cache_engine,index_col="index")

    def _check_for_cache_hits(self,start,end):
        # Cache Hit logic: We need to determine if we have a cache hit.
        # There are multiple cases:
        #     1 - No Hit
        #     2 - Everything in cache.
        #     3 - One chunck in cache - extend cache before and/or after
        #     4 - Multiple chuncks in cache - Punt this! Just overwrite all but one
        #
        start = start+timedelta(0,0,-start.microsecond)      # Round down on start
        if end.microsecond != 0: end   = end  +timedelta(0,0,1000000-end.microsecond) # Round up on end.
        cache_overlaps=[]
        cache_extend_before=[]
        cache_extend_after=[]
        for index in range(len(self._cache_known_data)):   # Loop by iloc so index order is early to late
            # Check if "start_time" < [start,end] < "end_time"
            cache_data = self._cache_known_data.iloc[index]
            if (cache_data["start_time"] <= start < cache_data["end_time"]) and (cache_data["start_time"] < end <= cache_data["end_time"]):
                cache_overlaps.append(index)
                continue                       # No need to check extending, it's not needed.

            # Check if start is before "end_time", and end is after "end_time", we can extend after.
            # I.e.  end_time is in [start,end]
            if ( start <= cache_data["end_time"] < end):
                cache_extend_after.append(index)
            # Check if "start_time" is inside [start,end], if so we can extend before.
            if  start  < cache_data["start_time"] <= end:
                cache_extend_before.append(index)

        return(cache_overlaps,cache_extend_before,cache_extend_after)

    def _cache_fill_runs(self,start,end,min_event):
        '''Fill the cache with runs from start to end.
        Filter on min_event and leave the rest in All_Runs.'''

        start = start+timedelta(0,0,-start.microsecond)      # Round down on start
        if end.microsecond != 0: end   = end  +timedelta(0,0,1000000-end.microsecond) # Round up on end.

        if self.debug>0:
            print("cache_fill_runs: {} - {}".format(start,end))
        num_runs = self.get_runs_from_rcdb(start,end,1)           # Get the new data from the RCDB.
        if num_runs == 0:
            # We still need to adjust the "end" time, so we don't end up in a endless loop.
            # Now data.All_Runs is likely empty, so use RCDB to get end of last run.
            rcdb_runs=self._db.session.query(Run).order_by(Run.start_time.desc()).limit(2)
            use_index=0
            if not rcdb_runs[0].get_condition_value("is_valid_run_end"): use_index=1
            end = rcdb_runs[use_index].end_time
            if self.debug>2: print("Adjusted end for empty rcdb fetch to: ",end)
            return num_runs,end

        good_runs=self.select_good_runs()
        self.add_current_data_to_runs()                           # Fill in the missing current info

        # Two scenarios now:
        # 1: The run period we just added is somewhere in the middle of all possible runs. It is thus extremely unlikely
        #    that a new run is added in between the 'end' and the last run we just fetched. It is thus OK to set the end
        #    of this cache period to the end requested. We want to do this, because it improves the chances we get overlaps
        #    of the run sections.
        #
        # 2: The run period we just added stretches beyond the last possible run (e.g. now() or later). It is thus likely
        #    that runs are added later, which may be added before the 'end' time selected. To avoid missing such runs, we
        #    need to set the 'end' if the cache period to the end of the last run.
        #    This is further complicated by the possibility that the last run is still ongoing, and thus has an incorrect
        #    number of events, files etc.
        #
        # To detect between 1 and 2, get the very last run from the database.
        last=self._db.session.query(Run).order_by(Run.start_time.desc()).limit(1).first()
        if end > last.start_time : # This is scenario 2, adjust the end time
            # The 'end' for the cache period must be the 'end' of the last run, or if that is not valid (run end not recorded)
            # then 'end' should be the end of the run one before.
            end_of_last_run_valid = self.All_Runs[(self.All_Runs["start_time"]>start) & ( self.All_Runs["end_time"]< end )].iloc[-1].is_valid_run_end

            if end_of_last_run_valid is None or not end_of_last_run_valid:
                if self.debug: print("WARNING: Last run does not have is_valid_run_end set.")
                # If the last run does not have a valid end, could it be that it is still ongoing? Expect a run to not be more than 8 hours.
                if self.All_Runs[(self.All_Runs["start_time"]>start) & ( self.All_Runs["end_time"]< end )].iloc[-1].end_time > (datetime.now() - timedelta(hours=8)):
                    if self.debug: print("WARNING: Run with is_valid_run_end not set is withing last 8 hours, not adding to cache.")
                    # We drop the run from the table, so it will not go into the cache.
                    self.All_Runs.drop(self.All_Runs.index[-1],inplace=True)
                    num_runs -= 1
                    if len(self.All_Runs[(self.All_Runs["start_time"]>start) & ( self.All_Runs["end_time"]< end )])<=1:  # there is no run left.
                        if self.debug: print("No runs left to add.")
                        last_2_runs=self._db.session.query(Run).order_by(Run.start_time.desc()).limit(2)
                        if last_2_runs.count()!=2:
                            print("Problem I do not understand. The DB did not return 2 runs.")
                            sys.exit(1)
                        end=last_2_runs[1].end_time
                        if self.debug: print("Set the end to:",end)
                        return 0,end

            end_of_last_run = self.All_Runs[(self.All_Runs["start_time"]>start) & ( self.All_Runs["end_time"]< end )].iloc[-1].end_time  # Get the end of the last run.

            if end_of_last_run is None or type(end_of_last_run) is not pd.Timestamp:   # This seems to be really, really rare.
                print("WARNING: Last run added to cache does not have a proper end time!!!")
                end_of_last_run = self.All_Runs[(self.All_Runs["start_time"]>start) & ( self.All_Runs["end_time"]< extend_to )].iloc[-1].start_time
                end_of_last_run -= timedelta(0,1) # One second before last start, so the next time this run will get updated.

            if self.debug>2: print("Change the end time from {} to {}".format(end,end_of_last_run))
            end = end_of_last_run

        # Now update the chache times table with the new entry from "start" to "end"
        self._cache_known_data=self._cache_known_data.append({"start_time":start,
                        "end_time":end,"min_event_count":1},ignore_index=True,sort=False)
        self._cache_known_data=self._cache_known_data.sort_values("start_time")

        if self.debug>2:
            print("cache_known_data:")
            print(self._cache_known_data)


        self.All_Runs.to_sql("Runs_Table",self._cache_engine,if_exists="append") # Add the new runs to the run cache table in SQL
        self._cache_known_data.to_sql("Known_Data_Ranges",self._cache_engine,if_exists='replace') # And the new chuck to the data ranges table.

        # We only want to keep those runs with the min_event criteria in the All_Runs table
        test_num_too_small  = self.All_Runs["event_count"]< min_event
        self.All_Runs= self.All_Runs.drop(self.All_Runs[test_num_too_small].index)   # Drop the ones with too few counts.

        return num_runs,end     # Return number of runs and possibly modified end time.

    def _cache_get_runs(self,start,end,min_event):
        '''Get runs directly from the cache and only from the cache.
        Result is stored in All_Runs if it was empty, or appended to All_Runs if not already there, and then sorted.
        return the number of runs added.'''
        # All the data is available in the cache, so just get it.

        if not self._cache_engine.dialect.has_table(self._cache_engine, "Runs_Table"):
            #
            # Table is not there, so cache was not initialized, just return so data will be fetched
            # from RCDB.
            #
            return 0


        start = start+timedelta(0,0,-start.microsecond)      # Round down on start
        if end.microsecond != 0: end   = end  +timedelta(0,0,1000000-end.microsecond) # Round up on end.

        if self.debug>0:
            print("Getting data from cache for {} - {} > {} evt. ".format(start,end,min_event))

        # Get the runs ordered already.
        sql = "select * from Runs_Table where start_time >= '{}' and end_time <= '{}' and event_count>= '{}' order by number".format(start,end,min_event)
        New_Runs = pd.read_sql(sql,self._cache_engine,index_col="number",parse_dates=["start_time","end_time"])

        if self.All_Runs is None or len(self.All_Runs) == 0:
            self.All_Runs = New_Runs
        else:
            #
            # It is possible some (or all) of the new runs had already been fetched earlier.
            # We do not want overlaps, so we need to sort this out.
            if self.debug>1:
                print("Checking overlap for runs {} to All_Runs.".format(len(New_Runs)))
            New_Runs.drop(New_Runs[New_Runs.index.isin(self.All_Runs.index)].index,inplace=True) # Drop the overlapping runs.
            if self.debug>1:
                print("Number of runs that will be appended: {}".format(len(New_Runs)))
            if len(New_Runs)>0:
                self.All_Runs = self.All_Runs.append(New_Runs)
                self.All_Runs.sort_index(inplace=True)
                if np.any(self.All_Runs.duplicated()):
                    print("ARGGGGG: we have duplicates in All_Runs")
                    self.All_Runs.drop_duplicates(inplace=True)

        if self.debug>1:
            print("Got {} runs from cache.".format(len(New_Runs)))

        return(len(New_Runs))

    def _cache_consolidate(self):
        '''Goes through cached regions and checks if any have 'grown' to touch or overlap.
        If so, combine those regions.'''
        # We go from early to late, but make sure there is a +1
        if self.debug>1:
            print("Cache consolidate: ")
            print(self._cache_known_data)

        for index in range(len(self._cache_known_data)):
           if index+1 < len(self._cache_known_data):
               if (self._cache_known_data.loc[self._cache_known_data.index[index  ],"end_time"  ] >= \
                   self._cache_known_data.loc[self._cache_known_data.index[index+1],"start_time"]):
                   # Eliminate the second in favor of the first.
                   self._cache_known_data.loc[self._cache_known_data.index[index],"end_time"] = \
                   self._cache_known_data.loc[self._cache_known_data.index[index+1],"end_time"]
                   self._cache_known_data=self._cache_known_data.drop(self._cache_known_data.index[index+1])
                   if self.debug > 1:
                       print("------ Eliminated {} in favor of {}".format(index+1,index))
                       print(self._cache_known_data)
                       print()
        # Make sure the cache is still sorted (it should be.)
        self._cache_known_data=self._cache_known_data.sort_values("start_time")
        # Write out the new table to SQL.
        self._cache_known_data.to_sql("Known_Data_Ranges",self._cache_engine,if_exists='replace') # TODO: update sql instead



    def get_runs(self,start,end,min_event):
        '''Fetch the runs from start time to end time with at least min_event events.
        Checking local cache if runs have already been fetched (if self._cache_engine is not None)
        If not get them from the rcdb and update the cache (if not None).
        Times are rounded down to the second for start and up to the second for end.'''

        self.min_event_count = min_event

        start = start+timedelta(0,0,-start.microsecond)      # Round down on start
        if end.microsecond != 0: end   = end  +timedelta(0,0,1000000-end.microsecond) # Round up on end.
        if self.debug: print("get_runs from {} - {} ".format(start,end))

        if self._cache_engine is None or self._cache_engine is False:  # No cache, so just get the runs.
            if self.debug>2: print("Getting runs bypassing cache, for start={}, end={}, min_event={}".format(start,end,min_event))
            num_runs = self.get_runs_from_rcdb(start,end,min_event)
            self.select_good_runs()
            self.add_current_data_to_runs()                                 # Fill in the missing current info
            return num_runs

        num_runs_cache = self._cache_get_runs(start,end,min_event)  # Get whatever we have in the cache already.

        cache_overlaps,cache_extend_before,cache_extend_after = self._check_for_cache_hits(start,end) # Check for overlaps of request with cache.

        if(len(cache_overlaps)+len(cache_extend_before)+len(cache_extend_after) == 0): # No overlaps at all.
            num_runs,tmp = self._cache_fill_runs(start,end,min_event)                      # so get the date for the entire stretch.
            return num_runs + num_runs_cache

        if(len(cache_overlaps)>1):
            print("Cache is dirty: multiple full overlaps.")

        while(len(cache_overlaps)==0):
            if self.debug > 2: print("check_cache_hits: ",cache_overlaps,cache_extend_before,cache_extend_after,start,end )
            #
            # Iteratively extend the cache with time periods, starting from below.
            #
            if len(cache_extend_before)>0:
                min_before = np.min(cache_extend_before)
            else:
                min_before = None

            if len(cache_extend_after)>0:
                min_after  = np.min(cache_extend_after)
            else:
                min_after = None

            if (min_after is None) or ( (min_before is not None) and (min_before<=min_after) ):
                # Start extending the before of the earliest overlap period to "start"

                Save_Runs=None
                Save_Runs, self.All_Runs = self.All_Runs, Save_Runs     # Save what is in All_Runs.
                min_before_start = self._cache_known_data.loc[self._cache_known_data.index[min_before],"start_time"]
                if self.debug>2: print("Extending {} before from {}  to {}".format(min_before,min_before_start,start))
                num_runs,new_end = self._cache_fill_runs(start,min_before_start,min_event) # Add the new data to cache and All_Runs up to min_before_start
                self.All_Runs = self.All_Runs.append(Save_Runs,sort=True)         # Append the saved runs.
            else:
                # The earliest overlap is in extend_after, so "start" is inside this overlap period.
                # We need to extend out to the earliest of "end" or the "start" of the next period.
                min_after_end = self._cache_known_data.loc[self._cache_known_data.index[min_after],"end_time"]
                if(min_before is not None) and ( self._cache_known_data.loc[self._cache_known_data.index[min_before],"start_time"]< end):
                    # extend to min_before_start
                    extend_to = self._cache_known_data.loc[self._cache_known_data.index[min_before],"start_time"]
                else:
                    # Try to extend to the requested end time.
                    extend_to = end

                if self.debug>2: print("Extending {} after from {}  to {}".format(min_after,min_after_end,extend_to))
                Save_Runs=None
                Save_Runs, self.All_Runs = self.All_Runs, Save_Runs     # Save what is in All_Runs.
                num_runs,end = self._cache_fill_runs(min_after_end,extend_to,min_event) # Add the new data to cache and All_Runs.
                # Note we re-set the end to the possibly corrected end. This is needed because otherwise we keep checking!
                if num_runs != 0:
                    if self.debug>2: print("Appending runs. ")
                    self.All_Runs = Save_Runs.append(self.All_Runs,sort=True)
                else:
                    # No new runs, so restore what we had before.
                    self.All_Runs = Save_Runs

                if self.All_Runs is None or len(self.All_Runs) == 0:  # There are no runs at all. Just quit.
                    return 0


            if self.debug>2:
                print("New cache config:")
                print(self._cache_known_data)

            #
            #  Now go through until there is full overlap.
            #
            self._cache_consolidate()  # If regions are now bordering, combine them.
            cache_overlaps,cache_extend_before,cache_extend_after = self._check_for_cache_hits(start,end)

        # We are done
        return len(self.All_Runs)

    def get_ExcludedRuns(self, fileName):

        if (os.path.exists(fileName)):
            with open(fileName) as ff:
                for line in ff:
                    line = line.replace('\n', '')

                    if line not in self.ExcludeRuns:
                        self.ExcludeRuns.append(line)

    def BeamAtenCorr(self, run):

        corr = 1.

        if (run < 10448):
            # print( self.All_Runs.loc[run, 'target'] )
            targnameNoSpaces = (self.All_Runs.loc[run, 'target']).rstrip()
            corr = self.atten_dict[targnameNoSpaces] / self.atten_dict['Empty']
            # print ('Corr is ' + str(corr))

        return corr

    def get_runs_from_rcdb(self,start_time,end_time,min_event_count):
        '''Return a dictionary with a list of runs for each target in the run period.
        This will get the list directly from the rcdb database, not looking at the local cache.'''
        # A fundamental issue with how the tables are setup is that you cannot construct
        # a simple query filtering on two different conditions. You would need to
        # construct a rather complicated query with a sub query, at which point you may be
        # better off just using MySQL directly. Such queries get really complicated, and this
        # isn't really needed here. Better to do one query and then filter using Python.

        if self.debug: print("Getting runs from RCDB: {} - {}".format(start_time,end_time))
        q=self._db.session.query(Run).join(Run.conditions).join(Condition.type)\
        .filter(Run.start_time > start_time).filter(Run.start_time < end_time)\
        .filter( (ConditionType.name == "event_count") & (Condition.int_value > min_event_count))

        if self.debug: print("Found {} runs.\n".format(q.count()))
        num_runs = q.count()
        if num_runs == 0:
            return(0)
        all_runs = q.all()
        runs=[]
        for R in all_runs:
            run_dict = {"number":R.number,"start_time":R.start_time,"end_time":R.end_time}

            if str(R.number) in self.ExcludeRuns:
                #print ("Excluding" + str(R.number) + "Since it it in Cameron's list")
                continue

            for c in self.Useful_conditions:
                run_dict[c]=R.get_condition_value(c)

            run_dict["start_time"] = run_dict["run_start_time"]; # Use the run_start_time and run_end_time
            run_dict["end_time"] = run_dict["run_end_time"];     # from the RCDB records. This allows for start/end corrections.
            runs.append(run_dict)

        self.All_Runs = pd.DataFrame(runs)
        self.All_Runs.loc[:,"selected"]=True            # Default to selected
        # Rewrite the run_config to eliminate the long directory name, which is not useful.
        self.All_Runs.loc[:,"run_config"]=[self.All_Runs.loc[r,"run_config"].split('/')[-1]   for r in self.All_Runs.index]
        self.All_Runs.set_index('number',inplace=True)
        return(num_runs)

    def add_current_cor(self,runnumber,override=False):
        '''Add the livetime corrected charge for a run to the pandas DataFrame.
        for that run.
        arguments:
             runnumber  - The run number for which you want to fetch the data.
             override   - Set to true if data should be fetched even if it seems it was already.
        '''

        if not override and ("charge" in self.All_Runs.keys()) and not np.isnan(self.All_Runs.loc[runnumber,"charge"]):
            return

        current =  self.Mya.get(self.Current_Channel,
            self.All_Runs.loc[runnumber,"start_time"],
            self.All_Runs.loc[runnumber,"end_time"]   )
        live_time = self.Mya.get(self.LiveTime_Channel,
            self.All_Runs.loc[runnumber,"start_time"],
            self.All_Runs.loc[runnumber,"end_time"]       )


        if self.Current_Channel == "scaler_calc1b":
            # Getting the target thickness dependend FCup charge correction
            currCorrection = self.BeamAtenCorr( runnumber)
            # Applying the correction
            current['value'] *= currCorrection

        #
        # The sampling of the current and live_time are NOT guaranteed to be the same.
        # We interpolate the live_time at the current time stamps to compensate.
        #
        try:
            live_time_corr = np.interp(current.ms,live_time.ms,live_time.value)/100.  # convert to fraction from %
        except Exception as e:
            print("There is a problem with the data for run {}".format(runnumber))
            print(e)

        #
        # Now we can just multiply the live_time_corr with the current.
        #
        try:
            current_corr = current.value*live_time_corr
        except Exception as e:
            print("There is a problem with the data for run {}".format(runnumber))
            print(e)

        #
        # We need to do a proper trapezoidal integration over the current data points.
        # Store the result in the data frame.
        #
        # Scale conversion:  I is in nA, dt is in ms, so I*dt is in nA*ms = 1e-9 A*1e-3 s = 1e-12 C
        # If we want mC instead of Coulombs, the factor is 1e-12*1e3 = 1e-9
        #
        self.All_Runs.loc[runnumber,"charge"]=np.trapz(current_corr,current.ms)*1e-9 # mC
        # self._Mya_cache[runnumber] = pd.DataFrame({"time":current.time,"current":current.value,
        #          "live_time":live_time_corr,"current_cor":current_corr})
        return

    def add_current_data_to_runs(self,targets=None,run_config=None):
        '''Add the mya data for beam current to all the runs with selected flag set.
        You can select other runs by specifying 'targets' and 'run_config' similarly to list_selected_runs()'''
        # We want to sort the data by target type.
        # We also want to veto runs that do not have the correct configuration,
        # such as pulser runs, FEE runs, etc.
        #
        good_runs = self.list_selected_runs(targets,run_config)
        if len(good_runs)>0:
            for rnum in self.list_selected_runs(targets,run_config):
                self.add_current_cor(rnum)
        else:
            # Even if there are no good runs, make sure that the "charge" column is in the table!
            # This ensure that when you write to DB the charge column exsists.
            self.All_Runs.loc[:,"charge"]=np.NaN


    def select_good_runs(self):
        '''Select the runs that have the selection criteria of Production_run_type
        and good_triggers.
        self.Production_run_type and self.Good_triggers can be either a list of strings to match
        or a regular expression to match. Use '.*' to match everything. '''

        good_runs=[]
        for rnum in self.All_Runs.index:
            test1 = False
            if self.Production_run_type is None:
                test1=True
            elif type(self.Production_run_type) is list:
                test1 = self.All_Runs.loc[rnum,"run_type"] in self.Production_run_type
            elif type(self.Production_run_type) is str:
                if re.match(self.Production_run_type,self.All_Runs.loc[rnum,"run_type"]):
                    test1 = True
            else:
                print("Incorrect type for self.Production_run_type:",type(self.Production_run_type))
                sys.exit(1)

            test2 = False
            trigger = self.All_Runs.loc[rnum,"run_config"]     #  .split('/')[-1] ## The split is already done.
            if type(self.Good_triggers) is list:
                test2 =  trigger in self.Good_triggers
            elif type(self.Good_triggers) is str:
                if re.match(self.Good_triggers,trigger):
                    test2=True

            if not test2:
                if trigger not in self.not_good_triggers:
                    self.not_good_triggers.append(trigger)
            self.All_Runs.loc[rnum,"selected"] = test1 & test2
            if test1 & test2:
                good_runs.append(rnum)

        return(good_runs)

    def list_selected_runs(self,targets=None,run_config=None):
        '''return the list of run numbers with the selected flag set. If 'targets' is specified, list only the runs with
        those targets.
        Arguments:
        targets       - None, False, string or list
        run_config    - None, False, string or list.
        If targets is None or False, all targets will be listed.
        If targets is a string, it will be matched case insensitive regex style. I.e. "um W" will give all thickness Tungsten.
        If targets is a list, it will be matched only for the exact strings in the list. So "4 um W" will be empty, because
        the space after the W is missing.

        If run_config is None, then only the pre-selected runs (set with All_Runs.selected True or 1) are used.
        If run_config is False, then all run_configs (triggers) are used.
        If run_config is a string, is will be matched case insensitive regex style.
        If run_config is a list, then that list will be used instead of the data.Good_triggers list.
        '''

        if run_config is None:
            test_run_config = self.All_Runs.selected
        elif run_config is False or len(run_config)==0:
            test_run_config = np.array([True]*len(self.All_Runs))     # I.e. select all. Yes, funny logic.
        elif type(run_config) is str:
            test_run_config =  self.All_Runs["run_config"].str.contains(run_config,case=False)
        elif type(run_config) is list:
            test_run_congig = self.All_Runs["run_config"].isin(run_config)
        else:
            print("I do not know what to do with run_config = ",type(run_config))

        if targets is None or targets is False or len(targets)==0:
            test_target = np.array([True]*len(self.All_Runs))
        elif type(targets) is str:
            test_target = self.All_Runs["target"].str.contains(targets,case=False)
        elif type(targets) is list:
            test_target = self.All_Runs.target.isin(targets)
        elif type(targets) is dict:
            test_target = self.All_Runs.target.isin(targets.keys())
        else:
            print("I do not know what to do with target = ",type(targets))

        return(self.All_Runs.index[ (test_run_config) & (test_target) ])


    def compute_cumulative_charge(self,targets=None,run_config=None):
        '''Compute the cumulative charge and event count for the runs in the current All_Runs table.
           The increasing numbers are put in the table in 'sum_charge', 'sum_charge_norm' and 'sum_event_count'.
           The runs are selected with list_selected_runs with the same 'targets' and 'run_config' arguments.
           Returns the end value of: (sum_charge,sum_charge_norm,sum_event_count)'''

        # filter the runs with a target it.
        selected = self.list_selected_runs(targets,run_config)

        if self.debug>1:
            print("Computing cumulative charge and event count for runs:",list(selected))

        self.All_Runs.loc[selected,"sum_event_count"] = np.cumsum(self.All_Runs.loc[selected,"event_count"])
        self.All_Runs.loc[selected,"sum_charge"] = np.cumsum(self.All_Runs.loc[selected,"charge"])

        sum_charge_per_target=self.target_dict.copy()
        for k in sum_charge_per_target:
            sum_charge_per_target[k]=0.

        if 'norm' in self.target_dict:
            cumsum_charge_norm=0.
            for run in selected:
                if self.All_Runs.loc[run,"target"] in self.target_dict:
                    target_norm = self.target_dict[self.All_Runs.loc[run,"target"]]/self.target_dict['norm']
                    if not np.isnan(self.All_Runs.loc[run,"charge"]):
                        sum_charge_per_target[self.All_Runs.loc[run,"target"]]+= self.All_Runs.loc[run,"charge"]
                        self.All_Runs.loc[run,"sum_charge_targ"]=sum_charge_per_target[self.All_Runs.loc[run,"target"]]
                        cumsum_charge_norm += self.All_Runs.loc[run,"charge"]*target_norm
                        self.All_Runs.loc[run,"sum_charge_norm"] = cumsum_charge_norm

        if len(self.All_Runs.loc[selected]):
            return(self.All_Runs.loc[selected,"sum_charge"].iloc[-1],
            self.All_Runs.loc[selected,"sum_charge_norm"].iloc[-1],
            self.All_Runs.loc[selected,"sum_event_count"].iloc[-1])
        else:
            return(0,0,0)


if __name__ == "__main__":

    import argparse
    print("Do something!")
