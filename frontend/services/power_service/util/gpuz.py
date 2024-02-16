import pandas as pd
import numpy as np
import sys
from datetime import timezone

def time_cvt(dt):
    utc_dt = dt.tz_localize('Europe/Berlin').astimezone(tz=timezone.utc).timestamp()
    return np.uint64(utc_dt * 10000000) + 116444736000000000

# convert the gpu-z file into a parquet file

input_file = sys.argv[1]
output_file = sys.argv[2]

gpuz_data = pd.read_csv(input_file, encoding_errors="backslashreplace", skipinitialspace=True, parse_dates=[0])
gpuz_data = gpuz_data.rename(columns=lambda x: x.strip().replace("\\xb0", "°"))
gpuz_data = gpuz_data.loc[:, ~gpuz_data.columns.str.contains('^Unnamed')]
#print(gpuz_data)

# convert datetime to UTC to filetime timestamp
gpuz_data["Timestamps"] = gpuz_data["Date"].apply(time_cvt)
#print(gpuz_data)
#print(int(gpuz_data["Timestamps"][0]))

# store as parquet file
gpuz_data.to_parquet(output_file, compression="brotli")
