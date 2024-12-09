#!/usr/bin/env python
# Get the current data and livetime data from MYA at JLab and save to an HDF5 file.
# Author: Maurik Holtrop (UNH)
# Date: 2024-02-14

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from pandas import HDFStore
from RunData.RunData import RunData

data = RunData(sqlcache=False, i_am_at_jlab=False)

# Commissioning period ???
#start_time = datetime(2023, 10, 4, 3, 24, 0)
#end_time = datetime(2023, 10, 6, 0, 0, 0)
#output_file_name = "RGD_raw_mya_data_commissioning.h5"

#
# Fetch the relevant MYA channels for the entrire run period.
# This querries the MYA database through the http interface, which we use through the RunData class so that the 
# login details can be handled gracefully. This is a somewhat time consuming step, depening on the network.
#
# Start time and End time of the data period to fetch.

start_time = datetime(2023, 10, 4, 9, 0, 0)
end_time = datetime(2023, 12, 15, 8, 0, 0)
output_file_name = "RGD_raw_mya_data.h5"

#
# Get the uncorrected BPM currents.
#
current_IPM2C21A = data.Mya.get("IPM2C21A", start_time, end_time,
                                do_not_clean=True, run_number=1)
current_IPM2C24A = data.Mya.get("IPM2C24A", start_time, end_time,
                                do_not_clean=True, run_number=1)
current_IPM2H01 = data.Mya.get("IPM2H01", start_time, end_time,
                               do_not_clean=True, run_number=1)

#
# Get the Faraday Cup information.
#
current_scalerS2b = data.Mya.get("scalerS2b", start_time, end_time,
                                 do_not_clean=True, run_number=1)
current_FCup = data.Mya.get("scaler_calc1b", start_time, end_time,
                            do_not_clean=True, run_number=1)
#
# These offsets should be re-computed, which we do below.
# The slope is *always* the same number: 906.2, interspersed with Nan. Get it anyway, it is only a few points.
#
FCup_offset = data.Mya.get("fcup_offset", start_time, end_time,
                           do_not_clean=True, run_number=1)
FCup_slope = data.Mya.get("fcup_slope", start_time, end_time,
                          do_not_clean=True, run_number=1)
#
# The FCup_beam_stop lets us know when the beam stop was out ~0, or in ~68.
#
FCup_beam_stop = data.Mya.get("beam_stop", start_time, end_time,
                              do_not_clean=True, run_number=1)
#
# These would be the attenuation factors used, if they were entered. This is not reliable. At all.
#
FCup_beam_atten = data.Mya.get("beam_stop_atten", start_time, end_time,
                               do_not_clean=True, run_number=1)

#
# Get the live times from different channels. Only LT_DAQ was found to be consistently usefull.
#
LT_DAQ = data.Mya.get("B_DAQ:livetime", start_time, end_time,
                      do_not_clean=True, run_number=1)
LT_Trigger = data.Mya.get("B_DET_TRIG_DISC_00:cTrg:livetime", start_time, end_time,
                          do_not_clean=True, run_number=1)
LT_clock = data.Mya.get("B_DAQ:livetime_pulser", start_time, end_time,
                        do_not_clean=True, run_number=1)

#
# Clean up the MYA data. Sometimes extra columns, 'x' an 't' are added. The seem to be caused when there was a network interruptopn.
# We need to drop them. We also need to drop entries with the same time. These are rare. They are not perfectly correlated with the
# network issues.
#
for dat in [current_IPM2C21A, current_IPM2C24A, current_IPM2H01, current_scalerS2b, current_FCup, FCup_offset, FCup_slope, FCup_beam_stop, FCup_beam_atten, LT_DAQ, LT_Trigger, LT_clock]:
    if 'x' in dat.keys():
        dat.drop(columns=['x','t'], inplace=True)
    duplicates = dat.loc[((dat.time.shift(-1)==dat.time))].index  # Selects all but the last duplicate.
    if len(duplicates):
        dat.drop(duplicates, inplace=True)  # Drop them.
#
# We use the IPM2C21A channel to detect the periods where the beam was tripped. 
# We find the trip times with the condition that the beam *current* is less than 0.01, and the measurement before was
# at least 0.2nA higher. This was found to be fairly reliable. See RGD_Currents_Compute.ipynb for the method and the associated graphs.
#
current = current_IPM2C21A
select_zero_start = (np.diff(current.value,prepend=0) < -0.2)& (current.value < .01)

# Now we step through the times of beam drop we found.
# I do not think this can be done without looping through the start points where the current drops to zero.
# For each zero_start time, we get the associated index from the current, and make sure we are not at the very end.
# The zero_end times are now identified simply by the next entry in the current channel. This works because the MYA data is
# sparsified to eliminate consecutive data points with the same value (or nearly the same value.) If this was not the case we would
# need to scan the current for the first non zero entry.
# If the start time is at current.loc[idx].ms, the stop time will be at current.loc[idx+1].ms We subtract an addional 2.5 seconds 
# to make extra sure that there is no current. This small delta-t takes care of synchronisation errors between the readout of the BPM
# and the Faraday Cup, and also the possibility that there was already *some* current before that BPM sample.
# We also use a cut that there have to be at least 30 samples in the data, i.e. 30 seconds of no beam, and the standard
# deviation of the "no current counts" if the FCup has to be less than 20. This removes cases where there is faulty data.
# Eliminating some offset points this way is considered less bad than having a questionable entry.
#
# We next make a dictionary with the offset_data point information, and stick that dictionary in a list. The list is then converted
# to a DataFrame.
#

offset_data = []
for i in range(len(current.loc[select_zero_start].index)):
    idx = current.loc[select_zero_start].index[i]
    if idx>len(current)-2:
        break
    data_range = current_scalerS2b.loc[(current_scalerS2b.ms>current.loc[idx].ms)&(current_scalerS2b.ms<(current.loc[idx+1].ms-2500))]
    if data_range.value.count()>30 and data_range.value.std()<20:
        offset_data.append(
            {
                "index":idx,
                "time":data_range.iloc[0].time, 
                "ms":data_range.iloc[0].ms,
                "dt":(data_range.iloc[-1].ms-data_range.iloc[0].ms), 
                "mean":data_range.value.mean(), 
                "std":data_range.value.std(), 
                "count":data_range.value.count(), 
                "min":data_range.value.min(), 
                "max":data_range.value.max()})

offset_df= pd.DataFrame(offset_data)
#
# We make an additional cut on the standard deviation. I don't think this does anything, because we already cut at 20 above.
# Kept it anyway.
#
offset_df = offset_df.loc[offset_df["std"]<30]
#
# Change the index to be the time stamp if each offset.
#
offset_df.set_index(['time'], inplace=True)  # Make the time column the index for the offset_f
#
# We now make a forward fill indexer. This means that for each time entry in current_scalerS2b, the *previous* most recent 
# entry of the offset_df will be used. 
#
offset_df_indexes = offset_df.index.get_indexer(current_scalerS2b.time, method="ffill")
#
# If there was no previous entry in offset_df, the indexer will point to -1. This would give you the *last* entry of the offset table.
# Instead we would want to first entry, so set the -1 indexes to 0.
#
offset_df_indexes[offset_df_indexes==-1] = 0
#
# We now have the offsets properly spaced to be added into the current_scalerS2b dataframe, so add the offsets and the std.
#
current_scalerS2b["offset"] = offset_df.iloc[offset_df_indexes,3].to_list()
current_scalerS2b["offset_std"] = offset_df.iloc[offset_df_indexes,4].to_list()
#
# We *do not make the correcting* to the value of the current_scalerS2b, because we do not know the attenuation coefficients.
# The corrected values could be calculated with:
# current_scalerS2b["corr"]= (current_scalerS2b["value"]-current_scalerS2b["offset"])/906.2
#
# Finally write the data to an HDF5 file.
#
store = HDFStore(output_file_name)
store.put("IPM2C21A", current_IPM2C21A)
store.put("IPM2C24A", current_IPM2C24A)
store.put("IPM2H01", current_IPM2H01)
store.put("scalerS2b", current_scalerS2b)
store.put("scaler_calc1b", current_FCup)
store.put("computed_fcup_offset", offset_df)
store.put("fcup_offset", FCup_offset)
store.put("fcup_slope", FCup_slope)
store.put("beam_stop", FCup_beam_stop)
store.put("beam_stop_atten", FCup_beam_atten)
store.put("DAQ_livetime", LT_DAQ)
store.put("Trigger_livetime", LT_Trigger)
store.put("Pulser_livetime", LT_clock)
store.close()

