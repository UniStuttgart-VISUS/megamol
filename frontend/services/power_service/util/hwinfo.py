import pandas as pd
import numpy as np
import sys
from datetime import timezone

def time_cvt(dt):
    utc_dt = dt.tz_localize('Europe/Berlin').astimezone(tz=timezone.utc).timestamp()
    return np.uint64(utc_dt * 10000000) + 116444736000000000

# convert the hwinfo file into a parquet file

input_file = sys.argv[1]
output_file = sys.argv[2]

hwinfo_data = pd.read_csv(input_file, encoding_errors="backslashreplace", skipinitialspace=True, parse_dates=[0], dayfirst=True, skipfooter=2)
hwinfo_data = hwinfo_data.rename(columns=lambda x: x.strip())
hwinfo_data = hwinfo_data.loc[:, ~hwinfo_data.columns.str.contains('^Unnamed')]
#print(hwinfo_data)

# convert datetime to UTC to filetime timestamp
hwinfo_data["Date"] = pd.to_datetime(hwinfo_data["Date"].astype(str).str.cat(hwinfo_data["Time"], sep=" "))
hwinfo_data.drop(columns="Time", inplace=True)
#print(hwinfo_data)
hwinfo_data["Timestamps"] = hwinfo_data["Date"].apply(time_cvt)
#print(hwinfo_data)
#print(int(hwinfo_data["Timestamps"][0]))
#print(list(hwinfo_data))

# store as parquet file
hwinfo_data.to_parquet(output_file, compression="brotli")