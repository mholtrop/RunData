#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Specify encoding so strings can have special characters.
#
#
# From Sho's code:
# privat static double motorToAngleTop(double motor) {
# return (17.821 - motor) / 832.714;
# }
#
# private static double motorToAngleBottom(double motor) {
# return (17.397 - motor) / 832.714;
# }
#
#
#
# From Matt Solt, email 2/8/2020
# Top:
# 18.017 = -2.2178 * 0.5 + y; y = 19.126
# y(top-si) = -0.4509 × y(stage) + 8.6242
# y(top-wire) = -0.4816 × y(stage) + 1.0516
# top-angle = (18.017 - y(stage))/833.247
#
# Bottom:
# 18.250 = -2.3657 * 0.5 + y; y = 19.433
# y(bot-si) = +0.4227 × y(stage) – 8.215
# y(bot-wire) = +0.4648 × y(stage) - 1.073
# bot-angle = (18.250 - y(stage))/833.247


from __future__ import print_function

import sys

if hasattr(__builtins__, 'raw_input'):
    input_str = raw_input
else:
    input_str = input

import os

from datetime import datetime, timedelta

from RunData.RunData import RunData

import pandas as pd
import numpy as np
import MySQLdb
import sqlalchemy

Plotting_enabled = True

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io
    plotly.io.renderers.default = "browser"

except ImportError:
    print("Sorry, but to make the nice plots, you really need a computer with 'plotly' installed.")
    print("Plotting functionality will be disabled.")
    Plotting_enabled = False
    go = None
    plotly = None

class HPSSVTStatus:

    def __init__(self, db_host=None, db_user=None, db_passwd=None, db_name=None):

        self.data = None
        self.mya_data = None
        self.mya_channels_per_run = None

        self.mya_channels_pos = ["hps:svt_top:motor.RBV", "hps:svt_bot:motor.RBV"]
        self.SVT_motor_to_angle = [self.svt_motor_to_angle_top, self.svt_motor_to_angle_bot]

        self.mya_channels_bias = []
        for i in range(0, 18):
            self.mya_channels_bias.append("SVT:bias:top:{:d}:v_sens".format(i))

        for i in range(20, 38):
            self.mya_channels_bias.append("SVT:bias:bot:{:d}:v_sens".format(i))

        self.mya_channels = self.mya_channels_bias + self.mya_channels_pos
        self.mya_channels.sort()  # Speed things up.

        self.start_time = datetime(2019, 7, 25, 0, 0)  # SVT back in correct position
        # self.start_time = datetime(2019, 9, 1, 0, 0)  # SVT back in correct position
        self.end_time = datetime(2019, 9, 10, 0, 0)
        self.end_time = self.end_time - timedelta(0, 0, self.end_time.microsecond)  # Round down on end_time to a second

        self.debug = 0
        self.no_exec = False

        self.db_host = db_host
        self.user = db_user
        self.passwd = db_passwd
        self.db_name = db_name
        self.db = None

    # Deduced from MYA data:
    #
    # TOP: Motor to position:  array([ 8.62419992, -0.45089999])  (intercept, slope)
    # Bottom:               :  array([-8.21500005,  0.42270001])
    # SVT TOP position = 8.6242 - 0.4509* hps:svt_top:motor.RBV
    # SVT BOT position = -8.215 + 0.4227* hps:svt_bos:motor.RBV
    #

    @staticmethod
    def svt_pos_top(motor_top):
        return 8.6242 - 0.4509 * motor_top

    @staticmethod
    def svt_pos_bot(motor_bot):
        return -8.215 + 0.4227 * motor_bot

    @staticmethod
    def svt_motor_to_angle_top(motor_top):
        # Sho's 2015:     return( (17.821 - motor_top) / 832.714 )
        # Matt Solt's 2020: top-angle = (18.017 - y(stage))/833.247
        return (18.017 - motor_top) / 833.247

    @staticmethod
    def svt_motor_to_angle_bot(motor_bot):
        # Sho's 2015:    return(  (17.397 - motor_bot) / 832.714)
        # Matt Solt's 2019: bot-angle = (18.250 - y(stage))/833.247
        return (18.250 - motor_bot) / 833.247

    @staticmethod
    def identity(x):  # Identity, do nothing, function.
        return x

    def plot_channel(self, channel, transform=None, runs=None, name=None, stride=1):
        """Add a single channel stored in mya_channels_per_run to the figure's store of lines.
        If you specify 'name' than that will be used for the label instead of the channel name.
        If you specify 'stride' then the data will be sparsified by taking only every stride value
        in a single run.
        The routine returns a go.Scatter object that can then be updated if needed and added to a go.Figure"""

        # These are not really needed be because none of them are modified. Here to indicate we need them.

        if Plotting_enabled:
            if transform is None:
                transform = self.identity

            if name is None:
                name = channel

            if runs is None:
                runs = self.data.All_Runs.index

            xl = []
            yl = []
            ht = []  # Hover text to add to line. Nice for adding run numbers to points.
            for run in runs:
                run_channel = self.mya_channels_per_run.loc[(channel, run), ]  # Slice of the data for run, channel
                xl += list(run_channel.time.iloc[0::stride])  # Append the start time of run and channel value at start.
                yl += transform(list(run_channel.value.iloc[0::stride]))
                ht += ["run:{:5d} start".format(run)]*len(run_channel.iloc[0::stride])

                # .astype('M8[ms]').astype('O') changes the np.datetime64 to a datetime.datetime.
                xl += [self.data.All_Runs.loc[run].end_time.astype('M8[ms]').astype('O')]
                # Append the end time of run and SVT channel value at end.
                yl += transform(list(run_channel.iloc[-1:].value))  # Last value in the data.
                ht += ["run:{:5d} end".format(run)]
                xl += [(self.data.All_Runs.loc[run].end_time + np.timedelta64(1, 's')).astype('M8[ms]').astype('O')]
                # Add one more point, +1 s after run.
                yl += [None]  # This is a 'None' point, forcing a line segment.
                ht += ["None"]

            name_short = name.replace("SVT:bias:", '').replace("v_sens", '')
            # Construct a line and return.
            return go.Scatter(x=xl, y=yl, hovertext=ht, name=name_short, line=dict(shape="hv"))

    def plot_bias(self, fig=None):

        if Plotting_enabled:
            if fig is None:
                fig = go.Figure()

            for channel in self.mya_channels_bias:
                line = self.plot_channel(channel)
                fig.add_trace(line)

            fig.update_layout(
                title=go.layout.Title(
                    text="HPS 2019 RUN, SVT Bias Voltage",
                    yanchor="top", y=0.95,
                    xanchor="center",
                    x=0.5))
            fig.update_xaxes(title_text="<b>Date/time<b>")
            fig.update_yaxes(title_text="<b>V_sens [V]<b>")

            return fig

    def plot_svt_angles(self, fig=None):

        if Plotting_enabled:
            top = self.plot_channel('hps:svt_top:motor.RBV',
                                    transform=self.svt_motor_to_angle_top, name="SVT angle top")
            bot = self.plot_channel('hps:svt_bot:motor.RBV',
                                    transform=self.svt_motor_to_angle_bot, name="SVT angle bot")

            top.update(line=dict(color="blue", width=2))
            bot.update(line=dict(color="green", width=2))

            if fig is None:
                fig = go.Figure()

            fig.add_trace(top)
            fig.add_trace(bot)

            fig.update_layout(
                title=go.layout.Title(
                    text="HPS 2019 RUN, SVT angles from nominal",
                    yanchor="top", y=0.95,
                    xanchor="center",
                    x=0.5))
            fig.update_xaxes(title_text="<b>Date/time<b>")
            fig.update_yaxes(title_text="<b>Delta Angle [rad]<b>", range=[-1e-3, 1e-3])

            return fig

    def open_conditions_database(self, host=None, user=None, passwd=None, name=None):
        """Open the database connection for on host, for user with password to database with name"""

        if host is not None:
            self.db_host = host

        if user is not None:
            self.user = user

        if passwd is not None:
            self.passwd = passwd

        if name is not None:
            self.db_name = name

        if self.no_exec:
            print("Not opening conditions database (-x)")
            return

        if self.db_host is None:
            print("Please enter the database host and login credentials.")
            print("DB hostname: ", file=sys.stderr, end="")
            self.db_host = input_str("")

        if self.user is None:
            print("DB Username: ", file=sys.stderr, end="")  # so stdout can be piped.
            self.user = input_str("")

        if self.passwd is None:
            import getpass
            self.passwd = getpass.getpass("DB Password: ")

        if self.db_name is None:
            self.db_name = "hps_conditions"

        self.db = MySQLdb.connect(host=self.db_host, user=self.user, passwd=self.passwd, db=self.db_name)

    def write_entry_to_conditions_db(self, table_name, run_start, run_end, start_time, end_time, names, values,
                                     comment="no comment"):
        """Write a new entry into the conditions database for a condition table.
        Arguments:
        table_name -- Name of the table to write, i.e. 'svt_motor_positions'
        run_start  -- first run in the range for these values.
        run_end    -- end run in the range.
        start_time -- first time for which the values are valid.
        end_time   -- last time for which the values are valid.
        names      -- list of names of the columns, i.e. ['top','bottom']
        values     -- list of lists of the values, i.e. [ [0.001,0.0011,...], [0.0031,0.0033,...]]
        Note: We try to be smart. If names is a string, not a list, then a single value is entered into the table.
        In that case, values should be a list of numbers, or a single number. If names is a list, but values is a
        list of numbers (instead of a list of lists) then a single value for each named column is entered.
        """

        # Examples for the tables.
        # Collections table:
        # +------+---------------------+---------- ----+------------------------------+---------------------+
        # | id   | table_name          | log           | description                  | created             |
        # +------+---------------------+---------------+------------------------------+---------------------+
        # | 2343 | svt_motor_positions | created by me | run ranges for SVT positions | 2016-08-08 20:45:42 |
        # | 2344 | svt_motor_positions | created by me | run ranges for SVT positions | 2016-08-08 20:45:42 |
        #
        # Conditions table:
        #
        # +------+-----------+---------+---------------------+---------------------+------+------------+
        # --------------------+---------------------+---------------------+---------------+
        # | id   | run_start | run_end | updated             | created             | tag  | created_by |
        # notes          | name                | table_name          | collection_id |
        # +------+-----------+---------+---------------------+---------------------+------+------------+
        # --------------------+---------------------+---------------------+---------------+
        # | 1755 |      7567 |    7567 | 2016-08-09 00:45:42 | 2016-08-08 20:45:42 | NULL |      spaul |
        # constants from mya | svt_motor_positions | svt_motor_positions |          2343 |
        # | 1756 |      7579 |    7579 | 2016-08-09 00:45:42 | 2016-08-08 20:45:42 | NULL |      spaul |
        # constants from mya | svt_motor_positions | svt_motor_positions |          2344 |
        #

        # svt_motor_positions table:
        #
        # Note: the start and end are "ms" time stamps in the server time.
        # +------+---------------+----------------------+----------------------+---------------+---------------+
        # | id   | collection_id | top                  | bottom               | start         | end           |
        # +------+---------------+----------------------+----------------------+---------------+---------------+
        # | 1173 |          2343 | 0.021401105301460045 | 0.020891926880057254 | 1456136619532 | 1456564765520 |
        # | 1174 |          2344 | 0.021401105301460045 | 0.020891926880057254 | 1456136619532 | 1456564765520 |
        # +------+---------------+----------------------+----------------------+---------------+---------------+
        #
        # If the motor moved during the run, there are multiple entries with the same collection_id but
        # different time stamps.

        # Step one - Store a row in collections table:
        sql = "insert into collections (table_name,log, description, created) " \
              "values ('{}', 'Entered by {} using HPSSVTStatus.py', " \
              "'{}', curdate());".format(table_name, self.user, comment)

        cur = None
        if self.no_exec:
            print("SQL: ", sql)
        else:
            cur = self.db.cursor()
            cur.execute(sql)
            self.db.commit()

        # Now retreive the id which was set by auto increment:
        sql = "select id from collections where table_name='{}' order by id desc limit 1;".format(table_name)

        if self.no_exec:
            print("SQL: ", sql)
            collection_id = 12345
        else:
            cur.execute(sql)
            collection_id = cur.fetchone()[0]

        # Create an entry in the Conditions table.
        sql = "insert into conditions (run_start, run_end, updated, created, tag, created_by, " \
              "notes, name,table_name, collection_id) values ({}, {}, curdate(), curdate(), NULL, '{}',' {}', " \
              "'{}', 'svt_motor_positions', {})".format(run_start, run_end, self.user, comment, table_name,
                                                        collection_id)

        if self.no_exec:
            print("SQL: ", sql)
        else:
            cur.execute(sql)

        if type(names) is list:
            if (type(values) is not list) and (type(values) is not np.ndarray):  # Error condition.
                print("write_entry_to_conditions db - error - names is list, but values is not.")
                raise ValueError("values needs to be a list.")
            else:
                if (type(values[0]) is not list) and (type(values[0]) is not np.ndarray):
                    values = [[x] for x in values]  # Wrap it in a list
                    start_time = [start_time]
                    end_time = [end_time]
        else:
            names = [names]
            # Names is not a list, so we enter a single list of values, or a single value.
            if (type(values) is not list) and (type(values) is not np.ndarray):
                values = [[values]]   # From value to list of list.
                start_time = [start_time]
                end_time = [end_time]
            else:
                values = [values]     # From list to list of list

        if self.debug > 2:
            print("Write collection id = {} run_start= {}  run_end = {} N={}".
                  format(collection_id, run_start, run_end, len(values[0])))

        for i in range(len(values[0])):
            sql = "insert into {} (collection_id, start, end, ".format(table_name)
            for n in names:
                sql += "{},".format(n)
            sql = sql[0:-1]  # Erase the last ,
            sql += ") values ({},{},{},".format(collection_id, start_time[i], end_time[i])
            for vl in values:
                sql += "{},".format(vl[i])
            sql = sql[0:-1]    # Again erase last ,
            sql += ")"
            if self.no_exec:
                print("SQL: ", sql)
            else:
                cur.execute(sql)
                self.db.commit()

    def motor_positions_to_conditions_db(self):
        """Store the positions of the SVT motor in the HPS conditions database.
        Argument: db = a handle to an openened database. """

        table_name = 'svt_motor_positions'
        comment = 'Angle relative to nominal from Mya motor position.'

        svt_angle_tolerance = 1e-7

        start_irun = 0
        start_run = self.data.All_Runs.index[start_irun]
        start_values = np.array([self.SVT_motor_to_angle[i](
            self.mya_channels_per_run.loc[(self.mya_channels_pos[i], start_run, 0), 'value']) for i in (0, 1)])

        # Make sure the times are converted to np.datetime64, if not already.
        start_time = np.datetime64(self.data.All_Runs.iloc[start_irun].start_time, 'ms')

        for irun in range(len(self.data.All_Runs.index)):

            run = self.data.All_Runs.index[irun]
            lengths = [len(self.mya_channels_per_run.loc[(self.mya_channels_pos[i], run,), 'value']) for i in (0, 1)]

            if self.debug > 0:
                print("Motor positions for run {:5d} lengths:".format(run), lengths)

            if lengths[0] > 1 or lengths[1] > 1:
                # Motors moved during the run. We need a separate entry for it.
                # First write the previous run range out to the DB.
                if irun > 0:
                    last_run = self.data.All_Runs.index[irun-1]
                    last_lengths = [len(self.mya_channels_per_run.loc[(self.mya_channels_pos[i], last_run, ), 'value'])
                                    for i in (0, 1)]
                    if last_lengths[0] > 1 or last_lengths[1] > 1:
                        if self.debug > 1:
                            print("No need to write, last run written already.")
                        # Last run the motor also moved, so that run was already completely written out.
                        pass
                    else:
                        last_time = np.datetime64(self.data.All_Runs.iloc[irun-1].end_time, 'ms')
                        self.write_entry_to_conditions_db(table_name, start_run, last_run,
                                                          start_time.astype(int),
                                                          last_time.astype(int),
                                                          ['top', 'bottom'],
                                                          [start_values[0], start_values[1]],
                                                          comment)
                # Now collect a list of positions and times.

                # The table and Mya data complicate life here. One of the two or both motors may be moving
                # and the times are not synchronized, but the database table pretends that they are.
                # For merging tables see: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

                merged = pd.merge_ordered(self.mya_channels_per_run.loc[self.mya_channels_pos[0], run, ],
                                          self.mya_channels_per_run.loc[self.mya_channels_pos[1], run, ],
                                          on='ms', suffixes=("_top", "_bot"), how='outer')

                # The merge procedure will leave lots of nan values for the missing data.
                # We fill in the nan with the previous value for the top or bottom

                for i in merged.index[1:]:
                    if np.isnan(merged.loc[i, 'value_top']):
                        merged.loc[i, 'value_top'] = merged.loc[i - 1, 'value_top']
                    if np.isnan(merged.loc[i, 'value_bot']):
                        merged.loc[i, 'value_bot'] = merged.loc[i - 1, 'value_bot']

                # The table is now complete, but it is too long.
                # Reduce the table so that either delta_t > 10 sec, or the motor shaft never changes by
                # more than 0.5 between consecutive entries.

                motor_shaft_tolerance = 0.5
                delta_t_max = 10. * 10000
                keep_list = [merged.index[0]]  # Always keep the first entry, which corresponds to start of run.
                last_top = merged.iloc[0].value_top
                last_bot = merged.iloc[0].value_bot
                last_ms = merged.iloc[0].ms
                for i in merged.index[1:]:
                    if (merged.loc[i].ms - last_ms > delta_t_max) or \
                            np.abs(merged.loc[i].value_top - last_top) > motor_shaft_tolerance or \
                            np.abs(merged.loc[i].value_bot - last_bot) > motor_shaft_tolerance:
                        keep_list.append(i)
                        last_top = merged.loc[i].value_top
                        last_bot = merged.loc[i].value_bot
                        last_ms = merged.loc[i].ms

                keep_list.append(merged.index[-1])  # Always keep the last value

                top_vals = list(self.svt_motor_to_angle_top(merged.loc[keep_list, 'value_top']))  # Make the lists.
                bot_vals = list(self.svt_motor_to_angle_bot(merged.loc[keep_list, 'value_bot']))
                start_time = list(merged.loc[keep_list, 'ms'])
                # End times are offset by one, with the last value the end of the run.
                end_time = start_time[1:]
                end_time.append(np.datetime64(self.data.All_Runs.loc[run].end_time, 'ms').astype(int))

                # Write the whole list to the DB:
                self.write_entry_to_conditions_db(table_name, run, run, start_time, end_time,
                                                  ['top', 'bottom'],
                                                  [top_vals, bot_vals],
                                                  comment)

                # Now reset the triggers.
                start_values = np.array([10000, 10000])  # Really big, because we must trigger a write for the next run.
                # Next start_run will need to be the next run.
                if irun < len(self.data.All_Runs.index):  # We are not at the end yet.
                    start_irun = irun+1
                    start_run = self.data.All_Runs.index[start_irun]
                    start_time = np.datetime64(self.data.All_Runs.iloc[start_irun].start_time, 'ms')

            else:
                # This is not a run with many values, there is only one.
                # Get the motor values for the run.

                this_values = np.array(
                    [self.SVT_motor_to_angle[i](self.mya_channels_per_run.loc[
                                (self.mya_channels_pos[i], run, 0), 'value']) for i in (0, 1)])

                if self.debug > 1:
                    print("run: {:5d} values: ".format(run), this_values)

                if np.any(np.abs(this_values - start_values) > svt_angle_tolerance):
                    #
                    # Different from before, so need to add a new entry for the LAST run,
                    # unless it was done already.
                    if self.debug > 1:
                        print("Values changed for run: {:5d}".format(run), this_values, start_values)
                    last_run = self.data.All_Runs.index[irun-1]
                    last_lengths = [len(self.mya_channels_per_run.loc[(self.mya_channels_pos[i], last_run, ), 'value'])
                                    for i in (0, 1)]

                    if last_lengths[0] > 1 or last_lengths[1] > 1:
                        if self.debug > 1:
                            print("No need to write out, was written already.")
                        # Last run the motor moved, so that run was already completely written out.
                        # There is nothing to do until the motor changes again.
                        pass
                    else:
                        last_time = np.datetime64(self.data.All_Runs.iloc[irun-1].end_time, 'ms')

                        # Write a single entry to the DB for the range on runs.
                        self.write_entry_to_conditions_db(table_name, start_run, last_run,
                                                          start_time.astype(int),
                                                          last_time.astype(int),
                                                          ['top', 'bottom'],
                                                          [start_values[0], start_values[1]],
                                                          comment)

                    # Done, reset the values
                    start_irun = irun
                    start_run = self.data.All_Runs.index[start_irun]
                    start_values = this_values
                    start_time = np.datetime64(self.data.All_Runs.iloc[start_irun].start_time, 'ms')

        # The loop is done, but we still need to write the data for the last run set out.
        last_run = self.data.All_Runs.index[-1]
        last_time = np.datetime64(self.data.All_Runs.loc[last_run].end_time, 'ms')
        self.write_entry_to_conditions_db(table_name, start_run, last_run,
                                          start_time.astype(int),
                                          last_time.astype(int),
                                          ['top', 'bottom'],
                                          [start_values[0], start_values[1]],
                                          comment)

        # All done.

    def bias_conditions_to_db(self):
        """Write the SVT bias data to the conditions database."""

        # The current framework for the bias data in HPS Java is basically on or off, and represents only a single
        # channel.
        # Here, we set minimum thresholds for the bias voltages, and if any channel falls below the threshold,
        # we write an "off" (0 V) to the database, otherwise we write "on" ( 180 V).

        table_name = "svt_bias_constants"
        comment = "Combined bias for all Mya bias channels."

        # Collect the average and stdev value for each channel for N=0

        for channel in self.mya_channels_bias:
            # This adds a new column to the data with the average value of the N=0 readings.
            # Since *most of the time* N=0 has a good bias, the average is close to the correct value for the channel.
            self.mya_channels_per_run.loc[(channel, ), 'average'] = np.average(
                self.mya_channels_per_run.loc[channel, slice(None), 0].value)
            # self.mya_channels_per_run.loc[ (channel,),'stdev'] =
            #      np.std(self.mya_channels_per_run.loc[channel, slice(None), 0].value)

        self.mya_channels_per_run['okay'] = self.mya_channels_per_run.value > 0.9*self.mya_channels_per_run.average

        start_run = self.data.All_Runs.index[0]

        # Make sure the times are all of type np.datetime64.
        start_time = np.datetime64(self.data.All_Runs.loc[start_run].start_time, 'ms')

        for i_run in range(len(self.data.All_Runs.index)):
            # For each run,
            # We need to check each channel to make sure bias was > than 0.9*average.

            # Setup a select list of all data for this particular run.
            run = self.data.All_Runs.index[i_run]

            if self.debug > 0:
                print("Processing run {:5d}".format(run))

            if not np.all(self.mya_channels_per_run.loc[(self.mya_channels_bias, run), :].okay):
                # Not all bias channels are OK all run.
                # Store the last group of runs as a set of ALL ok, unless i_run == 0.
                if i_run > 0:
                    last_run = self.data.All_Runs.index[i_run - 1]
                    # check that the last run was all okay otherwise it is written out already.
                    if np.all(self.mya_channels_per_run.loc[(self.mya_channels_bias, last_run), :].okay):
                        # Write out a set of runs with bias = 180V
                        last_time = np.datetime64(self.data.All_Runs.iloc[i_run - 1].end_time, 'ms')
                        self.write_entry_to_conditions_db(table_name, start_run, last_run,
                                                          start_time.astype(int),
                                                          last_time.astype(int),
                                                          ['value'],
                                                          [180.],
                                                          comment)
                # We now need to build a list of periods during the run where all bias is good, or not good.
                # Make a view of the data we are interested in:
                dat_sel = self.mya_channels_per_run.loc[(self.mya_channels_bias, run), :]
                dat_sel = dat_sel.sort_values('ms')  # Sort the list.
                dat_sel.reset_index(inplace=True)    # Make the index a simple list from 0 to N

                start_time = [dat_sel.iloc[0].ms]       # We start with the very first value
                status = [dat_sel.iloc[0].okay]       # And we store the status at the start.
                #
                # The complication here is that when you transition from okay = True to okay = False,
                # you need to take the *first* occurrence of False (any channel). If you go from False
                # to True, you need to make sure *all channels* are okay.

                channel_not_ok = []
                for idx in range(len(dat_sel)):
                    if dat_sel.iloc[idx].okay:
                        # True, so channel is good.
                        if dat_sel.iloc[idx].channel in channel_not_ok:       # if it is in the bad channel list
                            channel_not_ok.remove(dat_sel.iloc[idx].channel)  # remove the value from the channel list
                    else:
                        # False, so channel is not good.
                        if dat_sel.iloc[idx].channel not in channel_not_ok:   # Not in the list yet
                            channel_not_ok.append(dat_sel.iloc[idx].channel)  # So add it to the list.

                    # If the channel_not_ok list is empty, then the current timestamp is all good.
                    if status[-1]:  # Last status was all OK.
                        if len(channel_not_ok) > 0:  # Now there is a channel not OK.
                            status.append(False)     # So switch to False = not all good.
                            start_time.append(dat_sel.iloc[idx].ms)  # Save timestamp.
                    else:           # Last status was not OK.
                        if len(channel_not_ok) == 0:  # Now all channels are OK
                            status.append(True)       # So switch to True = all good.
                            start_time.append(dat_sel.iloc[idx].ms)  # Save timestamp.

                end_time = start_time[1:]
                end_time.append(np.datetime64(self.data.All_Runs.loc[run].end_time, 'ms').astype(int))

                values = []
                for stat in status:  # Convert the status into values 180 or 0.
                    if stat:
                        values.append(180.)
                    else:
                        values.append(0.)
                # Write to the database for this run only.
                self.write_entry_to_conditions_db(table_name, run, run,
                                                  start_time,
                                                  end_time,
                                                  ['value'],
                                                  [values],
                                                  comment)
                # RESET if not last run
                if i_run < len(self.data.All_Runs.index)-1:
                    start_run = self.data.All_Runs.index[i_run+1]
                    start_time = np.datetime64(self.data.All_Runs.loc[start_run].start_time, 'ms')

        # End of runs loop.
        # Write the last entry.
        last_run = self.data.All_Runs.index[-1]
        # check that the last run was all okay otherwise it is written out already.
        if np.all(self.mya_channels_per_run.loc[(self.mya_channels_bias, last_run), :].okay):
            # Write out a set of runs with bias = 180V
            last_time = np.datetime64(self.data.All_Runs.iloc[- 1].end_time, 'ms')
            self.write_entry_to_conditions_db(table_name, start_run, last_run,
                                              start_time.astype(int),
                                              last_time.astype(int),
                                              ['value'],
                                              [180.],
                                              comment)
        # Should be all done now.

    def get_run_data(self, start_time, end_time, username=None, password=None):
        """Get the run data information from RunData. """

        hostname = os.uname()[1]
        if hostname.find('clon') >= 0 or hostname.find('ifarm') >= 0 or hostname.find('jlab.org') >= 0:
            at_jlab = True
        else:
            at_jlab = False

        self.data = RunData(cache_file="HPS_run_cache.sqlite3", username=username, password=password,
                            i_am_at_jlab=at_jlab)

        self.data.debug = 0

        self.data.Good_triggers = r"hps_v..?_?.?\.cnf"
        self.data.Production_run_type = ["PROD66", "PROD67"]

        min_event_count = 10000000  # Runs with at least 10M events.
        #    start_time = datatime(2019, 7, 17, 0, 0)  # Very start of run

        print("Fetching the run data from {} to {}".format(start_time, end_time))
        self.data.get_runs(start_time, end_time, min_event_count)
        self.data.select_good_runs()

    def get_channel_data(self, username=None, password=None):

        if self.data is None:
            self.get_run_data(self.start_time, self.end_time, username, password)

        # Get the MYA data for the entire dataset for each channel separately.

        start_time = self.start_time - timedelta(minutes=60)
        end_time = self.end_time

        self.mya_data = {}
        for channel in self.mya_channels:
            if self.debug > 0:
                print("Fetching MYA channel: {}".format(channel))
            mdat = self.data.Mya.get(channel, start_time, end_time)
            self.mya_data[channel] = mdat

        #
        # For each run, we now check the values for each channel during the run period.
        # Since the MYA data is made sparse, we want to get the *last* value *before* the start of the run
        # for the first value. If there is no data during the run period, then the value was constant.
        # If there are entries during the run, then these will need to be put with timestamp into the Conditions DB
        # for that channel. We first store these in set of lists, one for each run, all appended to a dictionary with
        # on entry per run.
        #
        self.mya_channels_per_run = pd.DataFrame(columns=['ms', 'value', 'time'],
                                                 index=pd.MultiIndex.from_arrays([[], [], []],
                                                                                 names=["channel", "run", "N"]))

        for channel in self.mya_channels:  # For each channel.
            if self.debug > 0:
                print("Processing channel: {}".format(channel))

            for run in self.data.All_Runs.index:  # For each selected run number.
                # print("Processing for run {}".format(run))
                run_start_time = self.data.All_Runs.loc[run].start_time
                run_end_time = self.data.All_Runs.loc[run].end_time

                #
                # Get the first data point, from *before* the run_start_time
                #
                mya_data_row = self.mya_data[channel].loc[self.mya_data[channel].time <= run_start_time].tail(1)
                if len(mya_data_row) == 0:
                    print("Warning!!! The MYA data fetched does not start early enough!")
                    continue

                mya_data_row.time = run_start_time
                mya_data_row.ms = int(np.datetime64(run_start_time).astype(np.int64) // 1000)
                self.mya_channels_per_run.loc[channel, run, 0] = mya_data_row.iloc[0]

                mya_data_row2 = self.mya_data[channel].loc[
                    (self.mya_data[channel].time > run_start_time) &
                    (self.mya_data[channel].time < run_end_time)]

                if len(mya_data_row2) > 0:
                    mya_data_row2.index = [(channel, run, i+1) for i in range(len(mya_data_row2))]
                    self.mya_channels_per_run = self.mya_channels_per_run.append(mya_data_row2[['ms', 'value', 'time']])

        self.mya_channels_per_run.sort_index(inplace=True)  # Speed up further access by sorting the index.

        #
        # A big store of all the data is now available!!!
        #

    def store_tables_in_db(self):

        connector_string = "sqlite:///hps_channel_run_db.sqlite3"
        dbs = sqlalchemy.create_engine(connector_string)
        # meta = sqlalchemy.MetaData()
        # kdr = sqlalchemy.Table('hps_channel_run', meta,
        #                        sqlalchemy.Column('index', sqlalchemy.Integer, primary_key=True),
        #                        sqlalchemy.Column('channel', sqlalchemy.Text),
        #                        sqlalchemy.Column('run', sqlalchemy.Integer),
        #                        sqlalchemy.Column('N', sqlalchemy.Integer),
        #                        sqlalchemy.Column('ms', sqlalchemy.Integer),
        #                        sqlalchemy.Column('value', sqlalchemy.Float),
        #                        sqlalchemy.Column('time', sqlalchemy.DateTime))
        # meta.create_all(dbs)

        self.mya_channels_per_run.to_sql("channels_per_run", dbs, if_exists='replace')
        self.data.All_Runs.to_sql("all_runs", dbs, if_exists='replace')

    def read_tables_from_db(self):

        connector_string = "sqlite:///hps_channel_run_db.sqlite3"
        dbs = sqlalchemy.create_engine(connector_string)

        sql = "select * from channels_per_run"
        self.mya_channels_per_run = pd.read_sql(sql, dbs, parse_dates=["time"])
        self.mya_channels_per_run.set_index(['channel', 'run', 'N'], inplace=True)

        sql = "select * from all_runs"
        if self.data is None:
            self.data = RunData(i_am_at_jlab=True)  # Do not ask for username / password

        self.data.All_Runs = pd.read_sql(sql, dbs, parse_dates=["start_time", "end_time"])
        self.data.All_Runs.set_index(['number'], inplace=True)


def main(argv=None):

    import argparse

    if argv is None:
        argv = sys.argv
    else:
        argv = argv.split()
        argv.insert(0, sys.argv[0])  # add the program name.

    parser = argparse.ArgumentParser(
        description="""Submit SVT bias and motor positions to the conditions database.""",
        epilog="""
        Hint: First time you run this, you may want to use 'HPSSVTStatus.py -x -s', which will not execute the
        sql commands, but print them to the screen instead. The -s stores the MYA and RunData tables in a local
        sqlite3 file so the next time you can run with '-l' to speed up data retrieval. 
        For more info, read the script ^_^, or email maurik@physics.unh.edu.""")

    parser.add_argument('-d', '--debug', action="count", help="Be more verbose if possible. ", default=0)
    parser.add_argument('-x', '--noexec', action="store_true", help="Do not execute sql, print instead.")

    parser.add_argument('-H', '--host', type=str, help="Host name for Conditions DB.", default=None)
    parser.add_argument('-u', '--user', type=str, help="User name for Conditions DB.", default=None)
    parser.add_argument('-p', '--passwd', type=str, help="Password for Conditions DB.", default=None)
    parser.add_argument('-n', '--dbname', type=str, help="Name of Conditions DB.", default="hps_conditions")

    parser.add_argument('-m', '--motorplot', action="store_true", help="Create plot of motor position.")
    parser.add_argument('-b', '--biasplot', action="store_true", help="Create plot of bias values.")

    parser.add_argument('-l', '--local', action="store_true", help="Use local version of Mya database.")
    parser.add_argument('-s', '--store', action="store_true", help="Store data in local database for re-user with -l")

    args = parser.parse_args(argv[1:])

    hps = HPSSVTStatus(db_host=args.host, db_user=args.user, db_passwd=args.passwd, db_name=args.dbname)
    hps.debug = args.debug
    hps.no_exec = args.noexec

    if args.local:
        print("Reading MYA tables for SVT condition from local sqlite3 file.")
        hps.read_tables_from_db()
    else:
        print("Reading MYA tables for SVT condition from web server.")
        hps.get_channel_data()
        if args.store:
            print("Storing MYA tables for SVT conditions to local sqlite3 file.")
            hps.store_tables_in_db()

    print("Open conditions database.")
    hps.open_conditions_database()
    print("Process motor positions.")
    hps.motor_positions_to_conditions_db()
    print("Now processing bias conditions.")
    hps.bias_conditions_to_db()


if __name__ == "__main__":
    sys.exit(main())
