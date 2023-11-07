import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import scipy as sp
import os
import sys
import json

def read_parquet_files_in_folder(folder_path):
    data_map = {}
    metadata = {}
    for file in os.listdir(folder_path):
        if file.endswith(".parquet"):
            current_path = os.path.join(folder_path, file)
            filename = os.path.splitext(file)
            metadata = pq.read_metadata(current_path).metadata
            data_map[filename[0]] = pd.read_parquet(current_path)
    return data_map, metadata

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

def get_analysis_recipes(md):
    tinker_b = md[b"tinkerforge"]
    tinker_b = tinker_b.replace(b"\x00", b"")
    tinker = json.loads(tinker_b)
    rtx_b = md[b"rtx"]
    rtx_b = rtx_b.replace(b"\x00", b"")
    rtx = json.loads(rtx_b)
    return tinker, rtx

def compute_with_recipe(df, recipe):
    data = {}
    for key,value in recipe.items():
        passed = True
        ret = np.zeros(len(df.index))
        try:
            add = value["ADD"]
            if len(add) == 0:
                passed = False
            for n in add:
                # print(n)
                ret = ret + df[n]
                
        except:
            passed = False
            pass
        try:
            mul = value["MUL"]
            ret = ret*df[mul[1]]
        except:
            pass
        # print(key)
        if passed:
            data[key] = ret
    return data

def concat_rtx_df(map):
    rtb01_df = map["rtb01_s0"]
    rtb02_df = map["rtb02_s0"]
    rtb02_df.drop("timestamps", axis=1)
    rtb03_df = map["rtb03_s0"]
    rtb03_df.drop("timestamps", axis=1)
    rta01_df = map["rta01_s0"]
    rta01_df.drop("timestamps", axis=1)
    return pd.concat([rtb01_df, rtb02_df, rtb03_df, rta01_df], axis=1)

def get_stats(td, tl, first_idx, last_idx, num_frames):
    total_ws = 0
    for key in td:
        print(f"\t{key}:")
        ws = integrate(tl, td[key], first_idx, last_idx)
        total_ws = total_ws+ws
        print(f"\t\tEnergy: {ws} Ws")
        print(f"\t\tEnergy per frame: {ws/num_frames} Ws")
    print(f"\t\t\tSum: {total_ws}")
    print(f"\t\t\tSum per frame: {total_ws/num_frames}")




def main():
    data_map, metadata = read_parquet_files_in_folder(sys.argv[1])
    tinker_r, rtx_r = get_analysis_recipes(metadata)
    tinker_data = compute_with_recipe(data_map["tinker_s0"], tinker_r)
    rtx_df = concat_rtx_df(data_map)
    rtx_data = compute_with_recipe(rtx_df, rtx_r)

    get_framemarker_loc(data_map["rtb01_s0"], "frame")
    first_ts, last_ts, num_frames, first_idx, last_idx = get_frame_interval_and_count(data_map["rtb01_s0"], "timestamps")
    # print(first_ts)
    # print(last_ts)

    # rtx
    print("RTX:")
    get_stats(rtx_data, data_map["rtb01_s0"]["timestamps"], first_idx, last_idx, num_frames)

    # tinker
    first_idx, last_idx = match_ts_on_df(data_map["tinker_s0"], "12VHPWR0_timestamps", first_ts, last_ts)
    # print(first_idx)
    # print(last_idx)
    print("Tinker:")
    get_stats(tinker_data, data_map["tinker_s0"]["12VHPWR0_timestamps"], first_idx, last_idx, num_frames)

    # nvml
    print("NVML:")
    first_idx, last_idx = match_ts_on_df(data_map["nvml_s0"], "NVML_timestamps", first_ts, last_ts)
    ws = integrate(data_map["nvml_s0"]["NVML_timestamps"], data_map["nvml_s0"]["NVML_samples"], first_idx, last_idx)
    print(f"\tEnergy: {ws} Ws")
    print(f"\tEnergy per frame: {ws/num_frames} Ws")

    # hmc
    print("HMC:")
    first_idx, last_idx = match_ts_on_df(data_map["HMC01_s0"], "Timestamp", first_ts, last_ts)
    ws = integrate(data_map["HMC01_s0"]["Timestamp"], data_map["HMC01_s0"]["P[W]"], first_idx, last_idx)
    print(f"\tEnergy: {ws} Ws")
    print(f"\tEnergy per frame: {ws/num_frames} Ws")

    # # print(first_ts)
    # # print(last_ts)
    # print(f"Number of frames {num_frames}")
    # print("RTX:")
    # # print(first_idx)
    # # print(last_idx)
    # ws = integrate(data_map["rtb01_s0"]["timestamps"], data_map["rta01_s0"]["rta01_A_12VHPWR"]*data_map["rtb02_s0"]["rtb02_V_12VHPWR"], first_idx, last_idx)
    # print(f"\tEnergy: {ws} Ws")
    # print(f"\tEnergy per frame: {ws/num_frames} Ws")
    # first_idx, last_idx = match_ts_on_df(data_map["nvml_s0"], "NVML/NVIDIA GeForce RTX 4090/0000:0B:00.0_ts", first_ts, last_ts)
    # # print(first_idx)
    # # print(last_idx)
    # ws = integrate(data_map["nvml_s0"]["NVML/NVIDIA GeForce RTX 4090/0000:0B:00.0_ts"], data_map["nvml_s0"]["NVML/NVIDIA GeForce RTX 4090/0000:0B:00.0_samples"], first_idx, last_idx)
    # print("NVML:")
    # print(f"\tEnergy: {ws} Ws")
    # print(f"\tEnergy per frame: {ws/num_frames} Ws")
    # first_idx, last_idx = match_ts_on_df(data_map["HMC01_s0"], "Timestamp", first_ts, last_ts)
    # # print(first_idx)
    # # print(last_idx)
    # ws = integrate(data_map["HMC01_s0"]["Timestamp"], data_map["HMC01_s0"]["P[W]"], first_idx, last_idx)
    # print("HMC:")
    # print(f"\tEnergy: {ws} Ws")
    # print(f"\tEnergy per frame: {ws/num_frames} Ws")

if __name__=='__main__':
    main()