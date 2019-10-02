import sys
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import pandas as pd
from datetime import datetime,timedelta

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

        if self.debug: print("Fetching channel '{}'".format(channel))

        try:
            my_dat=self._session.get(self._url_head,verify=False,params=params)
        except ConnectionError:
            print("Could not connect to the Mya myQuery website. Was the password correctly entered? ")
            sys.exit(1)

        if my_dat.ok == False:
            print("Error, could not get the data for run {}".format(channel))
            print("Webserver responded with status: ",my_dat.status_code)
            return( pd.DataFrame({'ms':[start.timestamp()*1000,end.timestamp()*1000],'value':[0.,0.],'time':[start,end]}))

        dat_len = len(my_dat.json()['data'])
        if dat_len == 0:                                           # EPICS sparsified the zeros.
            return( pd.DataFrame({'ms':[start.timestamp()*1000,end.timestamp()*1000],'value':[0.,0.],'time':[start,end]}))

        pd_frame = pd.DataFrame(my_dat.json()['data'])

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
