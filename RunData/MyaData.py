#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# from datetime import datetime, timedelta

import sys
import os

import requests
# from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

if hasattr(__builtins__, 'raw_input'):
    input_str = raw_input
else:
    input_str = input


class MyaData:
    """A class to help retrieve and store the Mya data from myQuery requests to the JLab EPICS server.

    Typical use is to retrieve the time sequence of a single channel with .get("channel",start_time,end_time),
    which can then be used for processing or plotting.
    The function .get_multi() can be used to retrieve multiple channels, where the second (third, etc) channel
    will be interpolated to have the same time intervals as the first channel.
    """

    def __init__(self, i_am_at_jlab=False, username=None, password=None):
        """Connect to the Mya database using MyQuery.
        At JLab, no password is needed, but offsite we need a password to connect to the server.
        If needed and not provided, ask for the CUE username and password to setup the connection.
        Sets up the session handle for the requests"""

        self.at_jlab = i_am_at_jlab
        self.debug = 0
        self._session = requests.session()

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
                    username = input_str("")
                if password is None:
                    password = getpass.getpass("Password: ")

            url = "https://epicsweb.jlab.org/"
            page = self._session.get(url)
            payload = {'httpd_username': username, 'httpd_password': password, "login": "Login"}
            page = self._session.post(url, data=payload)
            # print(page.cookies.items())

    def get(self, channel, start, end, do_not_clean=False):
        """Get a series of Mya data with a myQuery call for channel, from start to end time.
        Returns a Pandas data frame with index, and keys: "ms", "value" and "time". Where "ms" is the
        MYA millisecond time stamp, "value" is the requested channel value, and "time" is a Pandas timestamp.
        """
        #
        # Get the value from Mya over the run period
        #
        params = {
            'c': channel,
            'b': start,
            'e': end,
            't': 'event',
            'u': 'on',  # u = on - Return the values in "ms" timestamp format.
            'a': 'on'}  # a = on - Adjust the ms timestamp to the server timezone.

        if self.debug:
            print("Fetching channel '{}'".format(channel))

        try:
            my_dat = self._session.get(self._url_head,verify=False,params=params)
        except ConnectionError:
            print("Could not connect to the Mya myQuery website. Was the password correctly entered? ")
            raise ConnectionError("Could not connect to ",self._url_head)

        if not my_dat.ok:
            print("Error, could not get the data for channel: {}".format(channel))
            print("Webserver responded with status: ",my_dat.status_code)
            print("Where your CUE login credential typed correctly? ")
            raise ConnectionError("Could not connect to ",self._url_head)

        dat_len = len(my_dat.json()['data'])
        if dat_len == 0:                                           # EPICS sparsified the data?
            return( pd.DataFrame({'ms':[start.timestamp()*1000,end.timestamp()*1000],'value':[None,None],'time':[start,end]}))

        pd_frame = pd.DataFrame(my_dat.json()['data'])

        if len(pd_frame.columns)>2 and not do_not_clean:          # If there are issues with time stamps, 2 extra columns are added: 't' and 'x'
            if self.debug>3:
                print("There is trouble with the Mya data for channel {} in the time period {} - {}".format(channel,start,end))
                print(pd_frame.columns)
            try:
#
# TODO: See if these two cases can be consolidated into one, which just looks for nan in 'v'
#
# Test this:
# Set to zero:
#               pd_frame.loc[ pd.isna(pd_frame['v'] ),'v'] = 0
# Drop them:
#               pd_frame.drop(pd_frame.loc[ pd_frame['x'] == True].index,inplace=True)
# or
                pd_frame.drop(pd_frame.loc[ pd.isna(pd_frame['v']) ].index,inplace=True)

                # if 't' in pd_frame.keys():
                #     pd_frame.drop(['t'], inplace=True, axis=1) # Finally, remove entire 't' column.
                #
                # if 'x' in pd_frame.keys():
                #     pd_frame.drop(['x'], inplace=True, axis=1)  # Finally, remove entire 't' column.

            # # The case with 't' and 'x' seems to occur with current data (BPM and/or FCup)
                # if 't' in pd_frame.keys() and 'x' in pd_frame.keys():
                #     for i in range(len(pd_frame)): # We need to clean it up
                #          if type(pd_frame.loc[i,"t"]) is not float:
                #              if self.debug>3: print("Clean up frame: ",pd_frame.loc[i])
                #              pd_frame.drop(i,inplace=True)
                #     pd_frame.drop(['t','x'],inplace=True,axis=1)
                #
                # # Case with a 't' column:
                # elif 't' in pd_frame.keys():
                #     for i in range(len(pd_frame)):         # We need to clean it up. Look for 'nan' values in 'v'
                #         if pd.isna(pd_frame.loc[i,'v']):   # Found one.
                #             if self.debug > 3: print("Clean up frame: ", pd_frame.loc[i])
                #             pd_frame.drop(i, inplace=True) # So drop that data.
                #     pd_frame.drop(['t'], inplace=True, axis=1) # Finally, remove entire 't' column.

            except Exception as e:
                print("Could not fix the issue.")
                print(e)
                sys.exit(1)

        pd_frame.rename(columns={"d":"ms","v":"value"},inplace=True)                         # Rename the columns
        #
        # Convert the ms timestamp to a datetime in the correct time zone.
        #
        pd_frame.loc[:,'time']= [ np.datetime64(x,'ms') for x in pd_frame.ms]

        # If you want with encoded timezone, you can do:
        #pd.Series(pd.to_datetime([ datetime.fromtimestamp(x/1000) for x in pd_frame.ms]).tz_localize("US/Eastern"),dtype=object)
        # or
        #pd.Series(pd.to_datetime(pd_frame.loc[:,'ms'],unit='ms',utc=True).dt.tz_convert("US/Eastern"),dtype=object)
        #
        # But these are quite a bit slower.
        #
        return(pd_frame)

    def get_multi(self,channels,start,end):
        '''Get multiple channels in the list 'channels' into a single dataframe and return.
        To do so, the first channel's time stamps are used as master. All the other channels are fetched,
        and their time stamps are re-aligned with the first channel timestamps by interpolation.
        arguments:
            channels  - a list of channels to fetch, or a dictionary. If dict, then translate channel names.
            start     - start time.
            end       - end time'''

        translate=False
        if type(channels) is str:
            return(self.get(channels,start,end))
        if type(channels) is dict:
            channels_dict = channels
            channels = list(channels_dict.keys())
            translate=True


        pd_frame = self.get(channels[0],start,end)
        columns = list(pd_frame.columns)

        if translate:
            columns[1]=channels_dict[channels[0]]
        else:
            columns[1]=channels[0]

        pd_frame.columns=columns                     # Rename the "value" column to the channel name.

        for i in range(1,len(channels)):
            pd_tmp = self.get(channels[i],start,end)
            tmp_corr = np.interp(pd_frame.ms,pd_tmp.ms,pd_tmp.value)
            if translate:
                pd_frame[channels_dict[channels[i]]] = tmp_corr
            else:
                pd_frame[channels[i]] = tmp_corr       # Add the interpolated data into the data frame.

        return(pd_frame)
