import pandas as pd
import numpy as np
import scipy as sp
import os
import sys

def read_parquet_files_in_folder(folder_path):
    data_map = {}
    for file in os.listdir(folder_path):
        if file.endswith(".parquet"):
            current_path = os.path.join(folder_path, file)
            filename = os.path.splitext(file)
            data_map[filename[0]] = pd.read_parquet(current_path)
    return data_map

at_high = False

def is_first_high(val, thres):
    global at_high
    found_high = False
    if not(at_high):
        if val > thres:
            at_high = True
            found_high = True
    else:
        if val <= thres:
            at_high = False
    return found_high

def get_high_loc(row, ts_name, thres):
    found = is_first_high(row[ts_name], thres)
    return found

def get_framemarker_loc(df, ts_name):
    # iterate through array and output all first occurence of high value
    global at_high
    at_high = False
    df['high'] = df.apply(get_high_loc, args=(ts_name, 2.5), axis=1)

def get_frame_interval_and_count(df, ts_name):
    mask = df['high']==True
    vals = df['high'].where(mask)
    first_idx = vals.first_valid_index()
    last_idx = vals.last_valid_index()
    count = vals.count()
    return df[ts_name][first_idx], df[ts_name][last_idx], (count-1), first_idx, last_idx

def match_ts_on_df(df, ts_name, first_ts, last_ts):
    first_idx = df[ts_name].where(df[ts_name]>=first_ts).first_valid_index()
    last_idx = df[ts_name].where(df[ts_name]<last_ts).last_valid_index()
    return first_idx, last_idx

def integrate(x, y, first_idx, last_idx):
    return sp.integrate.trapz(y[first_idx:last_idx], x=x[first_idx:last_idx]) / (1000*1000*10)


def main():
    data_map = read_parquet_files_in_folder(sys.argv[1])
    get_framemarker_loc(data_map["rtb01_s0"], "rtb01_frame")
    first_ts, last_ts, num_frames, first_idx, last_idx = get_frame_interval_and_count(data_map["rtb01_s0"], "timestamps")
    # print(first_ts)
    # print(last_ts)
    print(f"Number of frames {num_frames}")
    print("RTX:")
    # print(first_idx)
    # print(last_idx)
    ws = integrate(data_map["rtb01_s0"]["timestamps"], data_map["rta01_s0"]["rta01_A_12VHPWR"]*data_map["rtb02_s0"]["rtb02_V_12VHPWR"], first_idx, last_idx)
    print(f"\tEnergy: {ws} Ws")
    print(f"\tEnergy per frame: {ws/num_frames} Ws")
    first_idx, last_idx = match_ts_on_df(data_map["nvml_s0"], "NVML/NVIDIA GeForce RTX 4090/0000:0B:00.0_ts", first_ts, last_ts)
    # print(first_idx)
    # print(last_idx)
    ws = integrate(data_map["nvml_s0"]["NVML/NVIDIA GeForce RTX 4090/0000:0B:00.0_ts"], data_map["nvml_s0"]["NVML/NVIDIA GeForce RTX 4090/0000:0B:00.0_samples"], first_idx, last_idx)
    print("NVML:")
    print(f"\tEnergy: {ws} Ws")
    print(f"\tEnergy per frame: {ws/num_frames} Ws")
    first_idx, last_idx = match_ts_on_df(data_map["HMC01_s0"], "Timestamp", first_ts, last_ts)
    # print(first_idx)
    # print(last_idx)
    ws = integrate(data_map["HMC01_s0"]["Timestamp"], data_map["HMC01_s0"]["P[W]"], first_idx, last_idx)
    print("HMC:")
    print(f"\tEnergy: {ws} Ws")
    print(f"\tEnergy per frame: {ws/num_frames} Ws")

if __name__=='__main__':
    main()