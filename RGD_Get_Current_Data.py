# Get the current data and livetime data from MYA at JLab and save to an HDF5 file.
# Author: Maurik Holtrop (UNH)
# Date: 2024-02-14

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from pandas import HDFStore
from RunData.RunData import RunData

data = RunData(sqlcache=False, i_am_at_jlab=False)
start_time = datetime(2023, 10, 4, 16, 0, 0)
end_time = datetime(2023, 12, 15, 5, 0, 0)

current_IPM2C21A = data.Mya.get("IPM2C21A", start_time, end_time,
                                do_not_clean=True, run_number=1)
current_IPM2C24A = data.Mya.get("IPM2C24A", start_time, end_time,
                                do_not_clean=True, run_number=1)
current_IPM2H01 = data.Mya.get("IPM2H01", start_time, end_time,
                               do_not_clean=True, run_number=1)

current_scalerS2b = data.Mya.get("scalerS2b", start_time, end_time,
                                 do_not_clean=True, run_number=1)
current_FCup = data.Mya.get("scaler_calc1b", start_time, end_time,
                            do_not_clean=True, run_number=1)
FCup_offset = data.Mya.get("fcup_offset", start_time, end_time,
                           do_not_clean=True, run_number=1)
FCup_slope = data.Mya.get("fcup_slope", start_time, end_time,
                          do_not_clean=True, run_number=1)
FCup_beam_stop = data.Mya.get("beam_stop", start_time, end_time,
                              do_not_clean=True, run_number=1)
FCup_beam_atten = data.Mya.get("beam_stop_atten", start_time, end_time,
                               do_not_clean=True, run_number=1)

#
LT_DAQ = data.Mya.get("B_DAQ:livetime", start_time, end_time,
                      do_not_clean=True, run_number=1)
LT_Trigger = data.Mya.get("B_DET_TRIG_DISC_00:cTrg:livetime", start_time, end_time,
                          do_not_clean=True, run_number=1)
LT_clock = data.Mya.get("B_DAQ:livetime_pulser", start_time, end_time,
                        do_not_clean=True, run_number=1)

for dat in [current_IPM2C21A, current_IPM2C24A, current_IPM2H01, current_scalerS2b, current_FCup, FCup_offset, FCup_slope, FCup_beam_stop, FCup_beam_atten, LT_DAQ, LT_Trigger, LT_clock]:
    if 'x' in dat.keys():
        dat.drop(columns=['x','t'], inplace=True)

current, name, color = (current_IPM2C21A,"IPM2C21A","red")
select_zero_start = (np.diff(current.value,prepend=0) < -0.2)& (current.value < .01)
# I do not think this can be done without looping through the start points where the current drops to zero.
offset_data = []
for i in range(len(current.loc[select_zero_start].index)):
    idx = current.loc[select_zero_start].index[i]
    if idx>len(current)-2:
        break
    data_range = current_scalerS2b.loc[(current_scalerS2b.ms>current.loc[idx].ms)&(current_scalerS2b.ms<(current.loc[idx+1].ms-2500))]
    if data_range.value.count()>30 and data_range.value.std()<20:
        offset_data.append({"index":idx,"time":data_range.iloc[0].time, "ms":data_range.iloc[0].ms,"dt":(data_range.iloc[-1].ms-data_range.iloc[0].ms), "mean":data_range.value.mean(), "std":data_range.value.std(), "count":data_range.value.count(), "min":data_range.value.min(), "max":data_range.value.max()})
offset_df= pd.DataFrame(offset_data)

offset_df = offset_df.loc[offset_df["std"]<30]
offset_df.set_index(['time'], inplace=True)  # Make the time column the index for the offset_f
offset_df_indexes = offset_df.index.get_indexer(current_scalerS2b.time, method="ffill")
offset_df_indexes[offset_df_indexes==-1] = 0
current_scalerS2b["offset"] = offset_df.iloc[offset_df_indexes,3].to_list()
current_scalerS2b["offset_std"] = offset_df.iloc[offset_df_indexes,4].to_list()


store = HDFStore("RGD_raw_mya_data.h5")
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

