#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import logging
import argparse
#import dash
#import dash_core_components as dcc
#import dash_html_components as html
#from dash.dependencies import Input, Output, State
#import plotly.graph_objs as go

#import pandas as pd
#import MySQLdb

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



import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from datetime import datetime,timedelta

try:
    import pandas as pd
    plotly_panda_ok=True
except:
    print("Sorry, but you really need a computer with 'pandas' installed.")
    print("Try 'anaconda' python, which should have both.")

import json
import numpy as np


class MyaData:
    '''A class to help retrieve and store the Mya data from myQuery requests to the JLab EPICS server.'''

    def __init__(self,I_am_at_jlab=False,user_name=None,password=None):
        '''Connect to the Mya database using MyQuery.
        At JLab, no password is needed, but offsite we need a password to connect to the server.
        If needed and not provided, ask for the CUE username and password to setup the connection.
        Sets up the session handle for the requests'''

        self.at_jlab=I_am_at_jlab
        self.debug=0
        self._session = requests.session()

    #
    # When onsite, no password is needed so don't bother the user.
    # Setup the connection to the Mya Database through the MyQuery web interface.
    #
        if self.at_jlab:
            self._url_head = "https://myaweb.acc.jlab.org/myquery/interval"
        else:
            self._url_head = "https://epicsweb.jlab.org/myquery/interval"
            import getpass
            if user_name is None:
                print("Please enter your CUE login credentials.")
                print("Username: ",file=sys.stderr,end="") # so stdout can be piped.
                if sys.version_info.major == 2:
                    user_name = raw_input("")
                else:
                    user_name = input("")
            if password is None:
                password = getpass.getpass("Password: ")

            url="https://epicsweb.jlab.org/"
            page = self._session.get(url)
            payload = {'httpd_username':user_name,'httpd_password':password,"login": "Login"}
            page= self._session.post(url,data=payload)
            # print(page.cookies.items())

    def get(self,channel,start,end):
        '''Get a series of Mya data with a myQuery call for channel, from start to end time.
        Returns two numpy arrays, one of time stamps, and one of values.
        '''
        #
        # Get the value from Mya over the run period
        #
        params={
            'c':channel,
            'b':start,
            'e':end,
            't':'event',
            'u':'on'  }

        try:
            data=self._session.get(self._url_head,verify=False,params=params)
        except ConnectionError:
            print("Could not connect to the Mya myQuery website. Was the password correctly entered? ")
            sys.exit(1)

        if data is None:
            print("Error, could not get the current data for run {}".format(run.number));
            return( pd.DataFrame({'ms':[start.timestamp()*1000,end.timestamp()*1000],'value':[0.,0.],'time':[start,end]}))

        dat_len = len(data.json()['data'])
        if dat_len == 0:                                           # EPICS sparsified the zeros.
            return( pd.DataFrame({'ms':[start.timestamp()*1000,end.timestamp()*1000],'value':[0.,0.],'time':[start,end]}))

        pd_frame = pd.DataFrame(data.json()['data'])

        if len(pd_frame.columns)>2:          # If there are issues with time stamps, 2 extra columns are added: 't' and 'x'
            if self.debug>3:
                print("There is trouble with the Mya data for channel {} in the time period {} - {}".format(channel,start,end))
                print(pd_frame.columns)
            try:
                for i in range(len(pd_frame)): # We need to clean it up
                     if type(pd_frame.loc[i,"t"]) is not float:
                         if self.debug>3: print("Clean up frame: ",pd_frame.loc[i])
                         pd_frame.drop(i,inplace=True)
                pd_frame.drop(['t','x'],inplace=True,axis=1)
            except Exception as e:
                print("Could not fix the issue.")
                print(e)
                sys.exit(1)

        pd_frame.columns = ["ms","value"]                         # Rename the columns
        #
        # Convert the ms timestamp to a datetime in the correct time zone.
        #
        pd_frame.loc[:,'time']=pd.Series([ datetime.fromtimestamp(x/1000) for x in pd_frame.ms])

        # If you want with encoded timezone, you can do:
        #pd.Series(pd.to_datetime([ datetime.fromtimestamp(x/1000) for x in pd_frame.ms]).tz_localize("US/Eastern"),dtype=object)
        # or
        #pd.Series(pd.to_datetime(pd_frame.loc[:,'ms'],unit='ms',utc=True).dt.tz_convert("US/Eastern"),dtype=object)
        #
        # But these are quite a bit slower.
        #
        return(pd_frame)

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
        self.Production_run_type = ["PROD66"]
        self.Useful_conditions=['is_valid_run_end', 'user_comment', 'run_type',
        'target', 'beam_current_request', 'operators','event_count',
        'events_rate','run_config', 'status',
         'evio_files_count', 'megabyte_count']
        self.good_triggers=[]
        self.not_good_triggers=[]
        self.min_event_count = 1000000
        self.at_jlab=I_am_at_jlab
        self.All_Runs=None
        self.debug=2

        self._db=None
        self._session=None

        self._cache_engine=None
        self._cache_known_data=None


        self.start_rcdb()
        self.Mya = MyaData(I_am_at_jlab)
        self.start_cache(sqlcache)

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
            connector_string = "sqlite:///run_data_cache.sqlite3"

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
            return
        self.select_good_runs()
        self.add_current_data_to_runs()                                 # Fill in the missing current info
        self.All_Runs.to_sql("Runs_Table",self._cache_engine,if_exists="append") # Add the new runs to the cache table
        # We only want to keep the runs with the min_event criteria
        test_num_too_small  = self.All_Runs["event_count"]< min_event
        self.All_Runs= self.All_Runs.drop(self.All_Runs[test_num_too_small].index)   # Drop the ones with too few counts.

    def _cache_get_runs(self,start,end,min_event):
        '''Get runs directly from the cache and only from the cache.
        Result is stored in All_Runs, or appended to All_Runs and then sorted.
        return the number of runs added.'''
        # All the data is available in the cache, so just get it.

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
            self.All_Runs = self.All_Runs.append(New_Runs)
            if(len(self.All_Runs)>0):
                self.All_Runs.sort_index(inplace=True)
                if np.any(self.All_Runs.duplicated()):
                    print("ARGGGGG: we have duplicates in All_Runs")
                    self.All_Runs.drop_duplicates(subset="number",inplace=True)

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


    def get_runs(self,start,end,min_event):
        '''Return a dictionary with a list of runs for each target in the run period.
        Checking local cache if runs have already been fetched (if self._cache_engine is not None)
        If not get them from the rcdb and put them in the cache.
        Times are rounded down to the second for start and up to the second for end.'''

        self.min_event_count = min_event

        start = start+timedelta(0,0,-start.microsecond)      # Round down on start
        if end.microsecond != 0: end   = end  +timedelta(0,0,1000000-end.microsecond) # Round up on end.
        if self.debug: print("get_runs from {} - {} ".format(start,end))

        if self._cache_engine is None or self._cache_engine is False:
            if self.debug>0: print("Getting runs bypassing cache, for start={}, end={}, min_event={}".format(start,end,min_event))
            self.get_runs_from_rcdb(start,end,min_event)
            self.select_good_runs()
            self.add_current_data_to_runs()                                 # Fill in the missing current info
            return

        self._cache_get_runs(start,end,min_event)  # Get whatever we have in the cache.

        cache_overlaps,cache_extend_before,cache_extend_after = self._check_for_cache_hits(start,end)

        if(len(cache_overlaps)+len(cache_extend_before)+len(cache_extend_after) == 0): # No overlaps at all.
            self._cache_fill_runs(start,end,min_event)

            # Update a new cache record with the new run period.
            self._cache_known_data=self._cache_known_data.append({"start_time":start,
                            "end_time":end,"min_event_count":1},ignore_index=True)

            self._cache_known_data=self._cache_known_data.sort_values("start_time")
            if self.debug>1:
                print("cache_known_data:")
                print(self._cache_known_data)

            self._cache_known_data.to_sql("Known_Data_Ranges",self._cache_engine,if_exists='replace') # TODO: update sql instead
            return

        if(len(cache_overlaps)>1):
            print("Cache is dirty: multiple full overlaps.")

        while(len(cache_overlaps)==0):
            if self.debug > 1: print("check_cache_hits: ",cache_overlaps,cache_extend_before,cache_extend_after )
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
                if self.debug>1: print("Extending {} before from {}  to {}".format(min_before,min_before_start,start))
                self._cache_fill_runs(start,min_before_start,min_event) # Add the new data to cache and All_Runs up to min_before_start
                self.All_Runs = self.All_Runs.append(Save_Runs)         # Append the saved runs.
                self._cache_known_data.loc[self._cache_known_data.index[min_before],"start_time"]=start # Update the cache record.
            else:
                # The earliest overlap is in extend_after, so "start" is inside this overlap period.
                # We need to extend out to the earliest of "end" or the "start" of the next period.
                min_after_end = self._cache_known_data.loc[self._cache_known_data.index[min_after],"end_time"]
                if(min_before is not None) and ( self._cache_known_data.loc[self._cache_known_data.index[min_before],"start_time"]< end):
                    # extend to min_before_start
                    extend_to = self._cache_known_data.loc[self._cache_known_data.index[min_before],"start_time"]
                else:
                    extend_to = end

                if self.debug>1: print("Extending {} after from {}  to {}".format(min_after,min_after_end,extend_to))
                Save_Runs=None
                Save_Runs, self.All_Runs = self.All_Runs, Save_Runs     # Save what is in All_Runs.
                self._cache_fill_runs(min_after_end,extend_to,min_event) # Add the new data to cache and All_Runs.
                self.All_Runs = Save_Runs.append(self.All_Runs)
                self._cache_known_data.loc[self._cache_known_data.index[min_after],"end_time"]=extend_to

            if self.debug>1:
                print("New cache config:")
                print(self._cache_known_data)

            #
            #  Now go through until there is full overlap.
            #
            self._cache_consolidate()  # If regions are now bordering, combine them.
            cache_overlaps,cache_extend_before,cache_extend_after = self._check_for_cache_hits(start,end)

        # We are done, write the new cache index to DB.
        self._cache_known_data.to_sql("Known_Data_Ranges",self._cache_engine,if_exists='replace') # TODO: update sql instead

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
            for c in self.Useful_conditions:
                run_dict[c]=R.get_condition_value(c)
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

        current =  self.Mya.get('IPM2C21A',
            self.All_Runs.loc[runnumber,"start_time"],
            self.All_Runs.loc[runnumber,"end_time"]   )
        live_time = self.Mya.get('B_DAQ_HPS:TS:livetime',
            self.All_Runs.loc[runnumber,"start_time"],
            self.All_Runs.loc[runnumber,"end_time"]       )
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
        for rnum in self.list_selected_runs(targets,run_config):
            self.add_current_cor(rnum)

    def select_good_runs(self):
        '''Select the runs that have the selection criteria of Production_run_type
        and good_triggers.'''

        good_runs=[]
        for rnum in self.All_Runs.index:
            test1 = self.All_Runs.loc[rnum,"run_type"] in self.Production_run_type
            trigger = self.All_Runs.loc[rnum,"run_config"]     #  .split('/')[-1] ## The split is already done.
            test2 =  trigger in self.good_triggers
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
        If run_config is a list, then that list will be used instead of the data.good_triggers list.
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
        else:
            print("I do not know what to do with target = ",type(targets))

        return(self.All_Runs.index[ (test_run_config) & (test_target) ])


    def compute_cumulative_charge(self,targets=None,run_config=None):
        '''Compute the cumulative charge and event count for the runs in the current All_Runs table.
           The increasing numbers are put in the table in 'sum_charge' and 'sum_event_count'.
           The runs are selected with list_selected_runs with the same 'targets' and 'run_config' arguments.
           Returns the end value of: (sum_charge,sum_event_count)'''

        # filter the runs with a target it.
        selected = self.list_selected_runs(targets,run_config)
        # Sum_Runs = data.All_Runs[data.All_Runs["target"].isin(targets) & data.All_Runs["selected"]]

        self.All_Runs.loc[selected,"sum_event_count"] = np.cumsum(self.All_Runs.loc[selected,"event_count"])
        self.All_Runs.loc[selected,"sum_charge"] = np.cumsum(self.All_Runs.loc[selected,"charge"])

        if len(self.All_Runs.loc[selected]):
            return(self.All_Runs.loc[selected,"sum_charge"].iloc[-1],self.All_Runs.loc[selected,"sum_event_count"].iloc[-1])
        else:
            return(0,0)

if __name__ == "__main__":

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
        import chart_studio.plotly as charts
        pio.renderers.default="browser"
    except:
        print("Sorry, but to make the nice plots, you really need a computer with 'plotly' installed.")
        sys.exit(1)

    hostname=os.uname()[1]
    if hostname.find('clon')>=0 or hostname.find('ifarm')>=0 or hostname.find('jlab.org')>=0:
        #
        # For JLAB setup the place we can find the RCDB
        #
        sys.path.insert(0,"/home/holtrop/rcdb/python3")
        at_jlab=True

    data = RunData()
    # data._cache_engine=None   # Turn OFF cache?

    data.good_triggers=['hps_v7.cnf','hps_v8.cnf','hps_v9.cnf','hps_v9_1.cnf','hps_v9_2.cnf','hps_v10.cnf']
    data.Production_run_type=["PROD66"]
    min_event_count = 1000000              # Runs with at least 1M events.
    start_time = datetime(2019,7,25,0,0)  # SVT back in correct position
    end_time   = datetime.now()
    end_time = end_time+timedelta(0,0,-end_time.microsecond)      # Round down on end_time to a second

    data.get_runs(start_time,end_time,min_event_count)

    targets=['8 um W ','4 um W ']
    data.compute_cumulative_charge(targets)

    data.All_Runs.to_excel("hps_run_table.xlsx",columns=['start_time','end_time','target','run_config','selected','event_count','sum_event_count','charge','sum_charge','operators','user_comment'])

#    print(data.All_Runs.to_string(columns=['start_time','end_time','target','run_config','selected','event_count','charge','user_comment']))
#    data.All_Runs.to_latex("hps_run_table.latex",columns=['start_time','end_time','target','run_config','selected','event_count','charge','operators','user_comment'])

    Plot_Runs = data.All_Runs.loc[data.list_selected_runs(targets=targets)]
    hover = ["Run: {} Start time: {}".format(r,Plot_Runs["start_time"][r]) for r in Plot_Runs.index ]
    starts = Plot_Runs["start_time"]
    ends = Plot_Runs["end_time"]
    center=starts + (ends-starts)/2
    runlen= [(run["end_time"]-run["start_time"]).total_seconds()*999 for num,run, in Plot_Runs.iterrows()]
    sumcharge = Plot_Runs.loc[:,"sum_charge"]
    plot_sumcharge_t=[starts.iloc[0],ends.iloc[0]]
    plot_sumcharge_v=[0,sumcharge.iloc[0]]

    for i in range(1,len(sumcharge)):
        plot_sumcharge_t.append(starts.iloc[i])
        plot_sumcharge_t.append(ends.iloc[i])
        plot_sumcharge_v.append(sumcharge.iloc[i-1])
        plot_sumcharge_v.append(sumcharge.iloc[i])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=center, y=Plot_Runs['event_count'],width=runlen,hovertext=hover,name="Number of events"),
                 secondary_y=False,)
    fig.add_trace(
        go.Scatter(x=plot_sumcharge_t, y=plot_sumcharge_v,line=dict(color='red', width=3),name="Total Charge Live"),
        secondary_y=True,
    )

    proposed_charge = (ends.iloc[-1]-starts.iloc[0]).total_seconds()*150.e-6
    fig.add_trace(
        go.Scatter(x=[starts.iloc[0],ends.iloc[-1]], y=[0,proposed_charge],line=dict(color='yellow', width=2),name="Proposal Charge"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=[starts.iloc[0],ends.iloc[-1]], y=[0,proposed_charge/2],line=dict(color='#88FF99', width=2),name="150nA on 8µm W 50% up"),
        secondary_y=True,
    )


    fig.update_layout(
        title=go.layout.Title(
            text="HPS Run 2019 Progress",
            yanchor="top",
            y=0.86,
            xanchor="center",
            x=0.5),
    )

    a_index=[]
    a_x=[]
    a_y=[]
    a_text=[]
    a_ax=[]
    a_ay=[]

    index=Plot_Runs.index[Plot_Runs.loc[:,"end_time"]>datetime(2019,8,2,8,25)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index,"end_time"])
    a_y.append(sumcharge.loc[index] )
    a_text.append("Hall-A Wien Flip,<br />difficulty restoring beam.")
    a_ax.append(80)
    a_ay.append(-100)

    index=Plot_Runs.index[Plot_Runs.loc[:,"end_time"]>datetime(2019,8,4,12,11)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index,"end_time"])
    a_y.append(sumcharge.loc[index] )
    a_text.append("DAQ problem,<br />followed by<br />beam restore issues.")
    a_ax.append(30)
    a_ay.append(-140)

    index=Plot_Runs.index[Plot_Runs.loc[:,"end_time"]>datetime(2019,8,6,14,52)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index,"end_time"])
    a_y.append(sumcharge.loc[index] )
    a_text.append("Beam Halo,<br />retuning beam.")
    a_ax.append(0)
    a_ay.append(-160)

    index=Plot_Runs.index[Plot_Runs.loc[:,"end_time"]>datetime(2019,8,7,14,22)][0]
    a_index.append(index)
    a_x.append(Plot_Runs.loc[index,"end_time"])
    a_y.append(sumcharge.loc[index] )
    a_text.append("Thunder storm,<br />followed by retune<br />followed by DAQ issues.")
    a_ax.append(-30)
    a_ay.append(-240)




    a_annot=[]
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
            font = {
                "family": "Times",
                "size": 8,
                "color": "#0040C0"
            }
            )
        )

    fig.update_layout(
        annotations=a_annot+[]
        )

    # Set x-axis title
    fig.update_xaxes(title_text="Time")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Number of events</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Accumulated Charge (mC)</b>", range=[0,proposed_charge],secondary_y=True)
#    fig.write_image("HPSRun2019_progress.pdf")
#    fig.write_image("HPSRun2019_progress.png")
    charts.plot(fig, filename = 'HPSRun2019', auto_open=True)
#    fig.show()
