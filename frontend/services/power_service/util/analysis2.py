import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os
import json

def read_parquet_files_in_folder(folder_path: str)->"tuple[dict, dict]":
    """
    Reads all parquet files in a given folder.
    Their content is collected as separate pandas dataframes in data_map with their filename as key.

    Parameters
    ----------
    folder_path: str
        path to the folder containing the parquet files.

    Returns
    -------
    data_map: dict
        Dictionary of file contents with their original filenames as key.
    metadata: dict
        Metadata of the last read parquet file.
    """
    data_map = {}
    metadata = {}
    for file in os.listdir(folder_path):
        if file.endswith(".parquet"):
            current_path = os.path.join(folder_path, file)
            filename = os.path.splitext(file)
            metadata = pq.read_metadata(current_path).metadata
            metadata[b"trigger_ts"] = np.fromstring(metadata[b"trigger_ts"], dtype=np.int64, sep=";")
            data_map[filename[0]] = pd.read_parquet(current_path)
    return data_map, metadata

def get_analysis_recipes(md: dict):
    tinker_b = md[b"tinkerforge"]
    tinker_b = tinker_b.replace(b"\x00", b"")
    tinker = json.loads(tinker_b)
    rtx_b = md[b"rtx"]
    rtx_b = rtx_b.replace(b"\x00", b"")
    rtx = json.loads(rtx_b)
    return tinker, rtx

def concat_rtx_df(map: dict):
    rtb01_df = map["rtb01_s0"]
    rtb02_df = map["rtb02_s0"]
    rtb02_df = rtb02_df.drop("timestamps", axis=1)
    rtb03_df = map["rtb03_s0"]
    rtb03_df = rtb03_df.drop("timestamps", axis=1)
    rta01_df = map["rta01_s0"]
    rta01_df = rta01_df.drop("timestamps", axis=1)
    return pd.concat([rtb01_df, rtb02_df, rtb03_df, rta01_df], axis=1)

def match_trigger_signal(df: pd.DataFrame, trigger_ts: np.int64):
    """
    Shifts the timeline of a dataframe by the time difference between the rising edge of the trigger signal and
    the recorded timestamp of the sent trigger.

    Parameters
    ----------
    df: DataFrame
        Dataframe with the timeline that needs to be shifted.
        The timeline needs to be in filetime.
    tigger_ts: int64
        The timestamp of the trigger in filetime.

    Returns
    -------
        The shifted dataframe.
    """
    trigger_idx = df["trigger"].where(df["trigger"] >= 2.5).first_valid_index()
    rtx_ts = df["timestamps"][trigger_idx]
    diff = trigger_ts - rtx_ts
    df["timestamps"] = df["timestamps"].add(diff)
    return df

def get_frame_interval_and_count(df:pd.DataFrame, ts_name):
    df["high"] = df["frame"] > np.float32(2.5)
    df["high"] = df["high"].mask(df["high"].shift(1)==df["high"])
    mask = df['high']==True
    vals = df['high'].where(mask)
    i = df[~mask].index
    first_idx = vals.first_valid_index()
    last_idx = vals.last_valid_index()
    count = vals.count()
    return df[ts_name][first_idx], df[ts_name][last_idx], (count-1), first_idx, last_idx, df.drop(i)[ts_name]

def compute_with_recipe(df, recipe):
    data = []
    data.append(df["timestamps"])
    if "frame" in df:
        data.append(df["frame"])
    # ret_df = pd.DataFrame([df["timestamps"]])
    for key,value in recipe.items():
        passed = True
        ret = np.zeros(len(df.index))
        try:
            add = value["ADD"]
            if len(add) == 0:
                passed = False
            for n in add:
                # print(n)
                ret = np.add(ret, df[n])
                
        except:
            passed = False
            pass
        try:
            mul = value["MUL"]
            ret = np.multiply(ret, df[mul[1]])
        except:
            pass
        # print(key)
        if passed:
            # data[key] = ret
            # pd.concat([ret_df, pd.Series(data=ret, name=key)], axis=1)
            data.append(pd.Series(data=ret, name=key))
    return pd.concat(data, axis=1)

def moving_average(rtx_data, rtx_ts):
    diff_ms = (rtx_ts.iat[1]-rtx_ts.iat[0])/10000
    factor = np.int32(np.ceil(10/diff_ms))
    windows = rtx_data.rolling(factor)
    ma = windows.mean()
    ma_list = ma.tolist()
    return ma_list

def integrate(x, y, first_idx, last_idx):
    return sp.integrate.trapz(y[first_idx:last_idx], x=x[first_idx:last_idx]) / (1000*1000*10)

def get_stats(df, first_idx, last_idx, num_frames):
    total_ws = 0
    gpu_ws = 0
    for key in df:
        if key == "timestamps":
            continue
        if key == "frame":
            continue
        if key == "high":
            continue
        print(f"\t{key}:")
        ws = integrate(df["timestamps"], df[key], first_idx, last_idx)
        total_ws = total_ws+ws
        if key == "12VHPWR":
            gpu_ws = gpu_ws + ws
        if key == "12VPEG":
            gpu_ws = gpu_ws + ws
        if key == "PEG":
            gpu_ws = gpu_ws + ws
        print(f"\t\tEnergy: {np.round(ws, 2)} Ws")
        print(f"\t\tEnergy per frame: {np.round(ws/num_frames, 2)} Ws")
    print(f"\tGPU Sum: {np.round(gpu_ws, 2)} ({np.round(gpu_ws/num_frames, 2)})")
    # print(f"\tGPU Sum per frame: {np.round(gpu_ws/num_frames, 2)}")
    print(f"\t\t\tSum: {total_ws}")
    print(f"\t\t\tSum per frame: {total_ws/num_frames}")

def interpolate_tinker(df, first_ts, last_ts):
    series = {}
    num_el = np.int64((((last_ts - first_ts) / 10000)/10)-1)
    ts = np.linspace(first_ts, last_ts, num=num_el, dtype=np.int64)
    for key,value in df.items():
        if not("timestamps" in key):
            ts_name = key.split("_")[0]+"_timestamps"
            n_values = np.interp(ts, df[ts_name], value)
            series[key] = n_values
    series["timestamps"] = ts
    return pd.DataFrame(series)

def match_ts_on_df(df, ts_name, first_ts, last_ts):
    first_idx = df[ts_name].where(df[ts_name]>=first_ts).first_valid_index()
    last_idx = df[ts_name].where(df[ts_name]<last_ts).last_valid_index()
    if first_idx is None:
        print(f"first_idx is None for {ts_name}")
        first_idx = 0
    if last_idx is None:
        print(f"last_idx is None for {ts_name}")
        last_idx = len(df.index)
    return first_idx, last_idx

def interpolate(rtx_ts_series, df, ts_name):
    data = []
    for key,value in df.items():
        if not("timestamps" in key):
            ret = np.interp(rtx_ts_series, df[ts_name], value)
            data.append(pd.Series(data=ret, name=key))
    return pd.concat(data, axis=1)

def plot(rtx_data, rtx_ts, rtx_frame, rtx_data_conv, data, marked_ts, name, filename, path, show):
    fig = plt.figure(figsize=(16,9))
    plt.plot(rtx_ts, rtx_data, label="osci")
    plt.plot(rtx_ts, rtx_data_conv, label="osci ma", linewidth=2)
    plt.plot(rtx_ts, data, label=name, linewidth=3)
    max_val = rtx_data.max()
    rtx_frame.loc[rtx_frame < 2.5] = 0
    rtx_frame.loc[rtx_frame >= 2.5] = np.float32(max_val)
    plt.plot(rtx_ts, rtx_frame, label="frame")
    # plt.vlines(marked_ts, ymin=np.full(len(marked_ts),0), ymax=np.full(len(marked_ts),300))
    plt.legend(loc="upper left")
    if not show:
        plt.savefig(path+"\\"+filename+"_frame.pdf")
    else:
        plt.show(block=True)

    fig = plt.figure(figsize=(16,9))
    plt.plot(rtx_ts, rtx_data, label="osci")
    plt.plot(rtx_ts, rtx_data_conv, label="osci ma", linewidth=2)
    plt.plot(rtx_ts, data, label=name, linewidth=3)
    plt.legend(loc="upper left")
    if not show:
        plt.savefig(path+"\\"+filename+".pdf")
    else:
        plt.show(block=True)

def plot_bin(rtx_data, rtx_ts, rtx_frame, rtx_data_conv, data, marked_ts, name, filename, path, show):
    rtx_data_mean, bin_edges, bin_number = sp.stats.binned_statistic(rtx_ts, rtx_data, statistic="mean", bins=np.floor(len(rtx_ts)/10000))
    rtx_data_dev, _, _ = sp.stats.binned_statistic(rtx_ts, rtx_data, statistic="std", bins=np.floor(len(rtx_ts)/10000))
    rtx_data_min = rtx_data_mean-rtx_data_dev
    rtx_data_max = rtx_data_mean+rtx_data_dev
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2


    fig = plt.figure(figsize=(16,9))
    # plt.plot(rtx_ts, rtx_data, label="osci")
    plt.fill_between(bin_centers, rtx_data_min, rtx_data_max, alpha=0.4, label="osci")
    plt.plot(rtx_ts, rtx_data_conv, label="osci ma", linewidth=2)
    plt.plot(rtx_ts, data, label=name, linewidth=3)
    max_val = rtx_data.max()
    rtx_frame.loc[rtx_frame < 2.5] = 0
    rtx_frame.loc[rtx_frame >= 2.5] = np.float32(max_val)
    plt.plot(rtx_ts, rtx_frame, label="frame")
    # plt.vlines(marked_ts, ymin=np.full(len(marked_ts),0), ymax=np.full(len(marked_ts),300))
    plt.legend(loc="upper left")
    if not show:
        plt.savefig(path+"\\"+filename+"_bin_frame.pdf")
    else:
        plt.show(block=True)

    fig = plt.figure(figsize=(16,9))
    # plt.plot(rtx_ts, rtx_data, label="osci")
    plt.fill_between(bin_centers, rtx_data_min, rtx_data_max, alpha=0.2, label="osci")
    plt.plot(rtx_ts, rtx_data_conv, label="osci ma", linewidth=2)
    plt.plot(rtx_ts, data, label=name, linewidth=3)
    plt.legend(loc="upper left")
    if not show:
        plt.savefig(path+"\\"+filename+"_bin.pdf")
    else:
        plt.show(block=True)

def plot_time(A, B):
    # Ay = np.full(len(A), 1)
    By = np.full(len(B), 2)
    print(By)
    Ay = np.full(2, 1)
    Ax = np.array([A[0], A[len(A)-1]])
    fig = plt.figure()
    plt.scatter(Ax, Ay, label="rtx")
    plt.scatter(B, By, label="adl")
    plt.legend()
    plt.show(block=True)

def select_frames(df_ts, marked_ts, num_frames):
    if num_frames > len(marked_ts):
        num_frames = num_frames/10
    med = np.floor(len(marked_ts)/2)
    off = np.ceil(num_frames/2)
    first = med-off
    last = first+num_frames
    first_ts = marked_ts.iloc[np.int32(first)]
    last_ts = marked_ts.iloc[np.int32(last)]
    first_idx = df_ts[df_ts==first_ts].index.to_list()[0]
    last_idx = df_ts[df_ts==last_ts].index.to_list()[0]
    return first_idx, last_idx


def main():
    folder_path = sys.argv[1]
    show = False
    data_map, metadata = read_parquet_files_in_folder(folder_path)
    gpu = "amd"
    # amd_name = "ASIC/AMD Radeon RX 6900 XT"
    amd_name = "BOARD/AMD Radeon RX 7900 XT"

    # extract computation recipes
    tinker_r, rtx_r = get_analysis_recipes(metadata)

    # get dataframe containing all rtx series
    rtx_df = concat_rtx_df(data_map)
    # shift the data on the timeline
    rtx_df = match_trigger_signal(rtx_df, metadata[b"trigger_ts"][0])
    # compute the series for the rtx data
    rtx_df = compute_with_recipe(rtx_df, rtx_r)

    first_ts, last_ts, num_frames, rtx_first_idx, rtx_last_idx, marked_ts = get_frame_interval_and_count(rtx_df, "timestamps")
    # print(f"stats: {first_ts}, {last_ts}, {num_frames}, {rtx_first_idx}, {rtx_last_idx}")

    print(f"Stats: time {(last_ts-first_ts)/10000} ms fcount: {num_frames}")

    # rtx
    print("RTX:")
    get_stats(rtx_df, rtx_first_idx, rtx_last_idx, num_frames)

    rtx_gpu_conv = moving_average(rtx_df["12VHPWR"][rtx_first_idx:rtx_last_idx]+rtx_df["12VPEG"][rtx_first_idx:rtx_last_idx], rtx_df["timestamps"][rtx_first_idx:rtx_last_idx])

    # tinker
    print("Tinker:")
    tinker_df = interpolate_tinker(data_map["tinker_s0"], first_ts, last_ts)
    tinker_df = compute_with_recipe(tinker_df, tinker_r)
    first_idx, last_idx = match_ts_on_df(tinker_df, "timestamps", first_ts, last_ts)
    get_stats(tinker_df, first_idx, last_idx, num_frames)
    tinker_df = interpolate(rtx_df["timestamps"][rtx_first_idx:rtx_last_idx], tinker_df, "timestamps")
    start_idx_1, end_idx_1 = select_frames(rtx_df["timestamps"], marked_ts, 1)
    start_idx_3, end_idx_3 = select_frames(rtx_df["timestamps"], marked_ts, 3)
    start_idx_100, end_idx_100 = select_frames(rtx_df["timestamps"], marked_ts, 50)
    # print(tinker_df)
    if gpu == "nvidia":
        # plot_bin(rtx_df["12VHPWR"][rtx_first_idx:rtx_last_idx]+rtx_df["12VPEG"][rtx_first_idx:rtx_last_idx], rtx_df["timestamps"][rtx_first_idx:rtx_last_idx], rtx_df["frame"][rtx_first_idx:rtx_last_idx], rtx_gpu_conv, tinker_df["12VHPWR"]+tinker_df["12VPEG"], marked_ts, "tinker", "tinker.pdf")
        plot(rtx_df["12VHPWR"][start_idx_100:end_idx_100]+rtx_df["12VPEG"][start_idx_100:end_idx_100], rtx_df["timestamps"][start_idx_100:end_idx_100], rtx_df["frame"][start_idx_100:end_idx_100], rtx_gpu_conv[start_idx_100:end_idx_100], tinker_df["12VHPWR"][start_idx_100:end_idx_100]+tinker_df["12VPEG"][start_idx_100:end_idx_100], marked_ts, "tinker", "f50", folder_path, show)
        plot_bin(rtx_df["12VHPWR"][start_idx_100:end_idx_100]+rtx_df["12VPEG"][start_idx_100:end_idx_100], rtx_df["timestamps"][start_idx_100:end_idx_100], rtx_df["frame"][start_idx_100:end_idx_100], rtx_gpu_conv[start_idx_100:end_idx_100], tinker_df["12VHPWR"][start_idx_100:end_idx_100]+tinker_df["12VPEG"][start_idx_100:end_idx_100], marked_ts, "tinker", "f50", folder_path, show)
        plot(rtx_df["12VHPWR"][start_idx_1:end_idx_1]+rtx_df["12VPEG"][start_idx_1:end_idx_1], rtx_df["timestamps"][start_idx_1:end_idx_1], rtx_df["frame"][start_idx_1:end_idx_1], rtx_gpu_conv[start_idx_1:end_idx_1], tinker_df["12VHPWR"][start_idx_1:end_idx_1]+tinker_df["12VPEG"][start_idx_1:end_idx_1], marked_ts, "tinker", "f1", folder_path, show)
        plot(rtx_df["12VHPWR"][start_idx_3:end_idx_3]+rtx_df["12VPEG"][start_idx_3:end_idx_3], rtx_df["timestamps"][start_idx_3:end_idx_3], rtx_df["frame"][start_idx_3:end_idx_3], rtx_gpu_conv[start_idx_3:end_idx_3], tinker_df["12VHPWR"][start_idx_3:end_idx_3]+tinker_df["12VPEG"][start_idx_3:end_idx_3], marked_ts, "tinker", "f3", folder_path, show)
    else:
        plot(rtx_df["12VHPWR"][start_idx_100:end_idx_100]+rtx_df["12VPEG"][start_idx_100:end_idx_100], rtx_df["timestamps"][start_idx_100:end_idx_100], rtx_df["frame"][start_idx_100:end_idx_100], rtx_gpu_conv[start_idx_100:end_idx_100], tinker_df["PEG"][start_idx_100:end_idx_100]+tinker_df["12VPEG"][start_idx_100:end_idx_100], marked_ts, "tinker", "f50", folder_path, show)
        plot_bin(rtx_df["12VHPWR"][start_idx_100:end_idx_100]+rtx_df["12VPEG"][start_idx_100:end_idx_100], rtx_df["timestamps"][start_idx_100:end_idx_100], rtx_df["frame"][start_idx_100:end_idx_100], rtx_gpu_conv[start_idx_100:end_idx_100], tinker_df["PEG"][start_idx_100:end_idx_100]+tinker_df["12VPEG"][start_idx_100:end_idx_100], marked_ts, "tinker", "f50", folder_path, show)
        plot(rtx_df["12VHPWR"][start_idx_1:end_idx_1]+rtx_df["12VPEG"][start_idx_1:end_idx_1], rtx_df["timestamps"][start_idx_1:end_idx_1], rtx_df["frame"][start_idx_1:end_idx_1], rtx_gpu_conv[start_idx_1:end_idx_1], tinker_df["PEG"][start_idx_1:end_idx_1]+tinker_df["12VPEG"][start_idx_1:end_idx_1], marked_ts, "tinker", "f1", folder_path, show)
        plot(rtx_df["12VHPWR"][start_idx_3:end_idx_3]+rtx_df["12VPEG"][start_idx_3:end_idx_3], rtx_df["timestamps"][start_idx_3:end_idx_3], rtx_df["frame"][start_idx_3:end_idx_3], rtx_gpu_conv[start_idx_3:end_idx_3], tinker_df["PEG"][start_idx_3:end_idx_3]+tinker_df["12VPEG"][start_idx_3:end_idx_3], marked_ts, "tinker", "f3", folder_path, show)

    if gpu == "nvidia":
        # nvml
        print("NVML:")
        first_idx, last_idx = match_ts_on_df(data_map["nvml_s0"], "NVML_timestamps", first_ts, last_ts)
        ws = integrate(data_map["nvml_s0"]["NVML_timestamps"], data_map["nvml_s0"]["NVML_samples"], first_idx, last_idx)
        print(f"\tEnergy: {np.round(ws,2)} ({np.round(ws/num_frames,2)}) Ws")
        # print(f"\tEnergy per frame: {np.round(ws/num_frames,2)} Ws")

    if gpu == "amd":
        # nvml
        print("ADL:")
        first_idx, last_idx = match_ts_on_df(data_map["adl_s0"], f"ADL_timestamps", first_ts, last_ts)
        ws = integrate(data_map["adl_s0"][f"ADL_timestamps"], data_map["adl_s0"][f"ADL_samples"], first_idx, last_idx)
        print(f"\tEnergy: {np.round(ws,2)} ({np.round(ws/num_frames,2)}) Ws")
        print(f"\tEnergy per frame: {np.round(ws/num_frames,2)} Ws")
        plot_time(np.array(rtx_df["timestamps"][rtx_first_idx:rtx_last_idx]), data_map["adl_s0"][f"ADL_timestamps"][first_idx:last_idx])

    # hmc
    print("HMC:")
    first_idx, last_idx = match_ts_on_df(data_map["HMC01"], "Timestamp", first_ts, last_ts)
    ws = integrate(data_map["HMC01"]["Timestamp"], data_map["HMC01"]["P[W]"], first_idx, last_idx)
    print(f"\tEnergy: {np.round(ws,2)} Ws")
    print(f"\tEnergy per frame: {np.round(ws/num_frames,2)} Ws")


if __name__=="__main__":
    main()