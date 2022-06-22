#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os

try:
    import sqlalchemy
except ImportError:
    print("We need the sqlalchemy installed for the database, but I could not find it in:")
    print("sys.path: ", sys.path)
    sys.exit(1)

import requests
# from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


class MyaData:
    """A class to help retrieve and store the Mya data from myQuery requests to the JLab EPICS server.

    Typical use is to retrieve the time sequence of a single channel with .get("channel",start_time,end_time),
    which can then be used for processing or plotting.
    The function .get_multi() can be used to retrieve multiple channels, where the second (third, etc) channel
    will be interpolated to have the same time intervals as the first channel.
    """

    def __init__(self, i_am_at_jlab=False, username=None, password=None, cache=None):
        """Connect to the Mya database using MyQuery.
        At JLab, no password is needed, but offsite we need a password to connect to the epicsweb server.
        If needed and not provided, ask for the CUE username and password to setup the connection.
        The code then sets up the session handle for the requests.
        options:
              cache  - Default is None, so no cache database is utilized.
                     - You can specify an sqlalchemy engine (i.e. RunData._cache_engine) and MyaData will store
                       the retrieved data there, and look there before requesting from epicsweb. This speeds up
                       operations on already retrieved data.

        Note: The "cache" is not intended to be a local mini replica of the Mya database. Doing so becomes quite
              complicated for the logic of what has been stored already and what has not. So the cache stores
              Mya data for a particular channel for a *specific run*, not for generalized time periods. You can
              game this by creating run numbers for any arbirary run period.
        """

        self.at_jlab = i_am_at_jlab
        self._debug = 0
        self._session = requests.session()
        self._cache_engine = None
        self.start_cache_engine(cache)
        #
    # When onsite, no password is needed so don't bother the user.
    # Setup the connection to the Mya Database through the MyQuery web interface.
    #
        if self.at_jlab:
            self._url_head = "https://myaweb.acc.jlab.org/myquery/interval"
        else:
            self._url_head = "https://epicsweb.jlab.org/myquery/interval"
            if os.path.exists(os.environ['HOME']+'/.password-store/JLAB/username.gpg'):
                # We can use the 'pass' utility to get the password safely.
                if self.debug:
                    print("Using the pass utility to get the JLAB username and password.")
                import subprocess
                if username is None:
                    sub_out = subprocess.run(['pass', 'show', 'JLAB/username'], capture_output=True, text=True)
                    username = sub_out.stdout.strip()
                sub_out = subprocess.run(['pass', 'show', 'JLAB/login'], capture_output=True, text=True)
                if password is None:
                    password = sub_out.stdout.strip()
            else:
                import getpass
                if username is None:
                    print("Please enter your CUE login credentials.")
                    print("Username: ", file=sys.stderr, end="")  # so stdout can be piped.
                    username = input("")
                if password is None:
                    password = getpass.getpass("Password: ")

            url = "https://epicsweb.jlab.org/"
            try:
                # page =
                self._session.get(url)
                payload = {'httpd_username': username, 'httpd_password': password, "login": "Login"}
                # page =
                self._session.post(url, data=payload)
                # print(page.cookies.items())
            except requests.exceptions.ConnectionError as e:
                print(e)
                print("Session connecting to epicsweb.jlab.org failed. ")
                self._session = None

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, debug_level):
        self._debug = debug_level

    @property
    def cache_engine(self):
        """Return the cache engine in use.
        This allows you to do direct manipulations on the sqlalchemy object. This is read only."""
        return self._cache_engine

    def start_cache_engine(self, cache):
        """Setup and/or check the cache database."""

        if cache is None or cache is False:
            self._cache_engine = None
            return
        elif type(cache) is sqlalchemy.engine.base.Engine:
            self._cache_engine = cache
        elif type(cache) is str:
            if '///' not in cache:
                connector_string = "sqlite:///" + cache
            else:
                connector_string = cache

            self._cache_engine = sqlalchemy.create_engine(connector_string)

        if not self.check_if_table_is_in_cache("Mya_Data_Ranges"):
            meta = sqlalchemy.MetaData()
            sqlalchemy.Table('Mya_Data_Ranges', meta,
                             sqlalchemy.Column('index', sqlalchemy.Integer, primary_key=True),
                             sqlalchemy.Column('run_number', sqlalchemy.Integer),
                             sqlalchemy.Column('start_time', sqlalchemy.DateTime),
                             sqlalchemy.Column('end_time', sqlalchemy.DateTime),
                             sqlalchemy.Column('channel', sqlalchemy.String),
                             sqlalchemy.Column('data_length', sqlalchemy.Integer))
            meta.create_all(self._cache_engine)

    def check_if_table_is_in_cache(self, table_name):
        """Check if the table for channel 'table_name' is in the cache."""
        if self._cache_engine is None:
            return False

        return sqlalchemy.inspect(self._cache_engine).has_table(table_name)

    def check_if_data_is_in_cache(self, channel, start, end, run_number):
        """Check if the Mya_Data_Ranges table in the cache has an entry for channel and run_number, and
        check that start and end are within the limits for this data."""

        if self._cache_engine is None:
            return False

        sql = f'select * from Mya_Data_Ranges where run_number = {run_number} and channel = "{channel}";'
        run_data_mya = pd.read_sql(sql, self._cache_engine, index_col="run_number",
                                   parse_dates=["start_time", "end_time"])
        if len(run_data_mya) == 1:
            if self._debug > 2:
                print(f"MyaData:: Data found in cache for {run_number} {channel}")
            if start is not None and start < run_data_mya.iloc[0].start_time:
                print(f"MyaData: WARNING- asking from cache before start: {start} < {run_data_mya.iloc[0].start_time}")
            if end is not None and end > run_data_mya.iloc[0].end_time:
                print(f"MyaData: WARNING - asking from cache after end: {end} > {run_data_mya.iloc[0].end_time}")
            return True

        if len(run_data_mya) > 1:
            print(f"MyaData Cache corruption: Multiple entries found for {run_number} {channel} ")

        return False

    def get_channel_from_cache(self, channel, start=None, end=None, run_number=None,
                               data_values='"index", ms, value, time', index_col="index"):
        """Get a channel from the cache from start to end, or all of it if None. Do not check for run number.
        Options: data_values  - The values to get from the cache.
        Standard, the cache will contain \"index\", ms, value, time. We do not want the index, so
        default data_values is set to 'ms, value, time' """

        if self._cache_engine is None:
            return None

        sql = f"select {data_values} from '{channel}' "
        if (start is not None) or (end is not None):
            sql += " where "
        if start is not None:
            sql = sql + f"time >= '{start}' "
            if end is not None:
                sql = sql + "and "
        if end is not None:
            # For database reasons, add a fraction of a second so end is included.
            end = end + timedelta(seconds=0.0001)
            sql = sql + f"time <= '{end}'"
        if run_number is not None:
            if (start is not None) or (end is not None):
                sql = sql + " and "
            sql = sql + f"run_number = {run_number}"
        if self._debug > 2:
            print(f"Getting the data from cache. \nSQL={sql}")
        pd_frame = pd.read_sql(sql, self._cache_engine, parse_dates=["time"], index_col=index_col)
        return pd_frame

    def get(self, channel, start, end, do_not_clean=False, run_number=None, no_cache=False):
        """Get a series of Mya data with a myQuery call for channel, from start to end time.
        IF run_number is specified, then first look in the cache to see if the data is already there. If it is not
        then the data will be retreived from Mya and added to the cache for that run number.
        Returns a Pandas data frame with index, and keys: "ms", "value" and "time". Where "ms" is the
        MYA millisecond time stamp, "value" is the requested channel value, and "time" is a Pandas timestamp.
        Usually the data will be cleaned, unless do_not_clean is set to True. Cleaning in this case means dropping
        all entries that have not-a-number values.
        """
        #
        # Get the value from Mya over the run period
        #

        if self._session is None and self._cache_engine is None:
            return None

        if run_number is not None and not no_cache:
            if self.check_if_data_is_in_cache(channel, start, end, run_number):  # Data is available.
                return self.get_channel_from_cache(channel, start=start, end=end)

        if (start is None) or (end is None):
            print("MyaData.get():: ERROR = If data is not in cache, you *must* suply a start and end time.")
            return None

        data_age = (datetime.now() - start).days
        if data_age > 2*365:   # More than two years old, get from history deployment.
            deployment = 'history'
        else:
            deployment = 'ops'

        params = {
            'c': channel,
            'b': start,
            'e': end,
            'm': deployment,
            't': 'event',
            'u': 'on',  # u = on - Return the values in "ms" timestamp format.
            'a': 'on'}  # a = on - Adjust the ms timestamp to the server timezone.

        if self.debug > 3:
            print("Fetching channel '{}'".format(channel))

        try:
            if self.debug > 5:
                print(f"Asking for data: {self._url_head} params={params}")
            my_dat = self._session.get(self._url_head, verify=False, params=params)
        except ConnectionError:
            print("Could not connect to the Mya myQuery website. Was the password correctly entered? ")
            raise ConnectionError("Could not connect to ", self._url_head)

        if not my_dat.ok:
            print("Error, could not get the data for channel: {}".format(channel))
            print("Webserver responded with status: ", my_dat.status_code)
            print("Where your CUE login credential typed correctly? ")
            raise ConnectionError("Could not connect to ", self._url_head)

        dat_len = len(my_dat.json()['data'])

        if self.debug > 5:
            print(f"Number of data points returned = {dat_len}")

        if dat_len == 0:                                           # EPICS sparsified the data?
            if self._debug > 0:
                print(f"run {run_number} - No data received for channel: {channel} between {start} and {end}.")
            pd_frame = pd.DataFrame({'ms': [start.timestamp() * 1000, end.timestamp() * 1000],
                                     'value': [np.nan, np.nan],
                                     'time': [start, end],
                                     'run_number': np.nan})
        else:
            pd_frame = pd.DataFrame(my_dat.json()['data'])
            if len(pd_frame.columns) > 2 and not do_not_clean:
                # If there are issues with time stamps, 2 extra columns are added: 't' and 'x'
                if self.debug > 3:
                    print(f"There is trouble with the Mya data for channel {channel} in the time period {start} - {end}")
                    print(pd_frame.columns)
                try:
                    pd_frame.drop(pd_frame.loc[pd.isna(pd_frame['v'])].index, inplace=True)
                    if 'x' in pd_frame.columns:
                        pd_frame.drop(['x'], axis=1, inplace=True)
                    if 't' in pd_frame.columns:
                        pd_frame.drop(['t'], axis=1, inplace=True)
                except Exception as e:
                    print("Could not fix the issue.")
                    print(e)
                    raise

            pd_frame.rename(columns={"d": "ms", "v": "value"}, inplace=True)         # Rename the columns
            pd_frame.sort_values('ms', inplace=True)  # Rare, but sometimes Mya does not give a sorted list, so sort it.
            #
            # Convert the ms timestamp to a datetime in the correct time zone.
            # TODO: Replace with map
            #
            pd_frame.loc[:, 'time'] = [np.datetime64(x, 'ms') for x in pd_frame.ms]

            pd_frame['run_number'] = run_number
            # If you want with encoded timezone, you can do:
            # pd.Series(pd.to_datetime([ datetime.fromtimestamp(x/1000) for x in pd_frame.ms]) \
            #   .tz_localize("US/Eastern"),dtype=object)
            # or
            #pd.Series(pd.to_datetime(pd_frame.loc[:,'ms'],unit='ms',utc=True).dt.tz_convert("US/Eastern"),dtype=object)
            #
            # But these are quite a bit slower.
            #

        if self._cache_engine is not None and run_number is not None:
            # We want to now store the data to the cache.
            # We use the pd.DataFrame functionality to do so.
            self.add_to_mya_data_range(channel, start, end, run_number, dat_len)
            pd_frame.to_sql(channel, self._cache_engine, if_exists="append")

        return pd_frame

    def add_to_mya_data_range(self, channel, start, end, run_number, data_length):
        """Add an entry to the Mya_Data_Ranges table in the cache, so you can lookup
        the data by run number as well as by time slot. This is mostly an internal method."""
        if self._cache_engine is not None and run_number is not None:
            result = self._cache_engine.execute('select max("index") from Mya_Data_Ranges;')
            result_fetch = result.fetchall()
            if result_fetch[0][0] is not None:
                max_index = result_fetch[0][0] + 1
            else:
                max_index = 0
            data_range_add = pd.DataFrame({"run_number": [run_number], "start_time": [start], "end_time": [end],
                                           "channel": [channel], "data_length": [data_length]}, index=[max_index])
            data_range_add.to_sql("Mya_Data_Ranges", self._cache_engine, if_exists="append")
            return data_range_add
        else:
            return None

    def get_multi(self, channels, start, end):
        """Get multiple channels in the list 'channels' into a single dataframe and return.
        To do so, the first channel's time stamps are used as master. All the other channels are fetched,
        and their time stamps are re-aligned with the first channel timestamps by interpolation.
        arguments:
            channels  - a list of channels to fetch, or a dictionary. If dict, then translate channel names.
            start     - start time.
            end       - end time"""

        translate = False
        if type(channels) is str:
            return self.get(channels, start, end)
        if type(channels) is dict:
            channels_dict = channels
            channels = list(channels_dict.keys())
            translate = True

        pd_frame = self.get(channels[0], start, end)
        columns = list(pd_frame.columns)

        if translate:
            columns[1] = channels_dict[channels[0]]
        else:
            columns[1] = channels[0]

        pd_frame.columns = columns                     # Rename the "value" column to the channel name.

        for i in range(1, len(channels)):
            pd_tmp = self.get(channels[i], start, end)
            tmp_corr = np.interp(pd_frame.ms, pd_tmp.ms, pd_tmp.value)
            if translate:
                pd_frame[channels_dict[channels[i]]] = tmp_corr
            else:
                pd_frame[channels[i]] = tmp_corr       # Add the interpolated data into the data frame.

        return pd_frame
