import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import json
import io

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
    # print(diff)
    df["timestamps"] = df["timestamps"].add(diff)
    return df

def get_tight_valid_range(df:pd.DataFrame, ts_name, old_min_val, old_max_val):
    new_min_val = old_min_val
    new_max_val = old_max_val
    for key,value in df.items():
        if ts_name in key:
            min_val = value.min()
            max_val = value.max()
            new_min_val = max(new_min_val, min_val)
            new_max_val = min(new_max_val, max_val)
    return new_min_val, new_max_val

def get_frame_interval_and_count(df:pd.DataFrame, ts_name, offset, min_ts, max_ts):
    minv = df["timestamps"].ge(min_ts).idxmin()
    maxv = df["timestamps"].ge(max_ts).idxmax()
    f_series = df["frame"].to_numpy()
    f_series[maxv:] = 0
    f_series[:minv] = 0
    df["high"] = f_series > np.float32(2.5)
    df["high"] = df["high"].mask(df["high"].shift(1)==df["high"])
    if offset > 0:
        off_idx = df[df[ts_name].gt(offset)].index[0]
        df.loc[0:off_idx, "high"]=False
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
            sub = value["SUB"]
            for n in sub:
                ret = np.subtract(ret, df[n])
        except:
            pass
        try:
            mul = value["MUL"]
            ret = np.multiply(ret, df[mul[0]])
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

def get_stats(df, first_idx, last_idx, num_frames, gpu):
    total_ws = 0
    gpu_ws = 0
    cpu_ws = 0
    for key in df:
        if key == "timestamps":
            continue
        if key == "frame":
            continue
        if key == "high":
            continue
        print(f"\t{key}:")
        ws = integrate(df["timestamps"], df[key], first_idx, last_idx)
        if ws == np.inf:
            ws = 0.0
        total_ws = total_ws+ws
        if gpu == "nvidia":
            if key == "12VHPWR":
                gpu_ws = gpu_ws + ws
        if key == "12VPEG":
            gpu_ws = gpu_ws + ws
            #cpu_ws = cpu_ws - ws
            total_ws = total_ws - ws
        if gpu == "amd" or gpu == "intel":
            if key == "PEG":
                gpu_ws = gpu_ws + ws
        if key == "P8":
            cpu_ws = cpu_ws + ws
        if key == "P4":
            cpu_ws = cpu_ws + ws
        if key == "3VPEG":
            total_ws = total_ws - ws
        # if key == "12VATX":
        #     cpu_ws = cpu_ws + ws
        print(f"\t\tEnergy: {np.round(ws, 2)} Ws")
        print(f"\t\tEnergy per frame: {np.round(ws/num_frames, 2)} Ws")
    print(f"\tGPU Sum: {np.round(gpu_ws, 2)} ({np.round(gpu_ws/num_frames, 2)})")
    print(f"\tCPU Sum: {np.round(cpu_ws, 2)} ({np.round(cpu_ws/num_frames, 2)})")
    # print(f"\tGPU Sum per frame: {np.round(gpu_ws/num_frames, 2)}")
    print(f"\t\t\tSum: {total_ws}")
    print(f"\t\t\tSum per frame: {total_ws/num_frames}")
    return np.round(gpu_ws, 2), np.round(cpu_ws, 2), np.round(total_ws, 2)

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

def interpolate(df, target_ts):
    series = {}
    for key, value in df.items():
        if not("timestamps" in key):
            ts_name = key.split("_")[0]+"_timestamps"
            n_values = np.interp(target_ts, df[ts_name], value)
            series[key] = n_values
    series["timestamps"] = target_ts
    return pd.DataFrame(series)

def interpolate_hmc(df, ts_name, target_ts):
    series = {}
    for key, value in df.items():
        if not(ts_name in key):
            n_values = np.interp(target_ts, df[ts_name], value)
            series[key] = n_values
    series["timestamps"] = target_ts
    return pd.DataFrame(series)

def match_ts_on_df(df, ts_name, first_ts, last_ts):
    first_idx = df[ts_name].where(df[ts_name]>=first_ts).first_valid_index()
    last_idx = df[ts_name].where(df[ts_name]<last_ts).last_valid_index()
    if first_idx is None:
        print(f"first_idx is None for {ts_name}")
        #first_idx = 0
        raise Exception()
    if last_idx is None:
        print(f"last_idx is None for {ts_name}")
        #last_idx = len(df.index)
        raise Exception()
    return first_idx, last_idx

# def interpolate(rtx_ts_series, df, ts_name):
#     data = []
#     for key,value in df.items():
#         if not("timestamps" in key):
#             ret = np.interp(rtx_ts_series, df[ts_name], value)
#             data.append(pd.Series(data=ret, name=key))
#     return pd.concat(data, axis=1)

def relay_plot(start, end, rtx_data, rtx_ts, rtx_frame, rtx_data_conv, conv_name, data, name, filename, path, show):
    plot(rtx_data[start:end], rtx_ts[start:end], rtx_frame[start:end], rtx_data_conv[start:end], conv_name, data[start:end], name, filename, path, show)

def relay_plot_bin(start, end, rtx_data, rtx_ts, rtx_frame, rtx_data_conv, conv_name, data, name, filename, path, show):
    plot_bin(rtx_data[start:end], rtx_ts[start:end], rtx_frame[start:end], rtx_data_conv[start:end], conv_name, data[start:end], name, filename, path, show)

def plot(rtx_data, rtx_ts, rtx_frame, rtx_data_conv, conv_name, data, name, filename, path, show):
    fig = plt.figure(figsize=(16,9))
    plt.plot(rtx_ts, rtx_data, label="osci")
    plt.plot(rtx_ts, rtx_data_conv, label=conv_name, linewidth=2)
    plt.plot(rtx_ts, data, label=name, linewidth=3)
    max_val = rtx_data.max()
    rtx_frame.loc[rtx_frame < 2.5] = 0
    rtx_frame.loc[rtx_frame >= 2.5] = np.float32(max_val)
    plt.plot(rtx_ts, rtx_frame, label="frame")
    # plt.vlines(marked_ts, ymin=np.full(len(marked_ts),0), ymax=np.full(len(marked_ts),300))
    plt.legend(loc="upper left")
    if not show:
        plt.savefig(path+"\\"+filename+"_frame.pdf")
        plt.close()
    else:
        plt.show(block=True)

    fig = plt.figure(figsize=(16,9))
    plt.plot(rtx_ts, rtx_data, label="osci")
    plt.plot(rtx_ts, rtx_data_conv, label=conv_name, linewidth=2)
    plt.plot(rtx_ts, data, label=name, linewidth=3)
    plt.legend(loc="upper left")
    if not show:
        plt.savefig(path+"/"+filename+".pdf")
        plt.close()
    else:
        plt.show(block=True)

def plot_bin(rtx_data, rtx_ts, rtx_frame, rtx_data_conv, conv_name, data, name, filename, path, show):
    div_val = 5000
    if len(rtx_ts) < 2*div_val:
        div_val = 1000
    rtx_data_mean, bin_edges, bin_number = sp.stats.binned_statistic(rtx_ts, rtx_data, statistic="mean", bins=np.floor(len(rtx_ts)/div_val))
    rtx_data_dev, _, _ = sp.stats.binned_statistic(rtx_ts, rtx_data, statistic="std", bins=np.floor(len(rtx_ts)/div_val))
    rtx_data_min = rtx_data_mean-rtx_data_dev
    rtx_data_max = rtx_data_mean+rtx_data_dev
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2


    fig = plt.figure(figsize=(16,9))
    # plt.plot(rtx_ts, rtx_data, label="osci")
    plt.fill_between(bin_centers, rtx_data_min, rtx_data_max, alpha=0.4, label="osci")
    plt.plot(bin_centers, rtx_data_mean, label="osci mean")
    plt.plot(rtx_ts, rtx_data_conv, label=conv_name, linewidth=2)
    plt.plot(rtx_ts, data, label=name, linewidth=3)
    max_val = rtx_data.max()
    rtx_frame.loc[rtx_frame < 2.5] = 0
    rtx_frame.loc[rtx_frame >= 2.5] = np.float32(max_val)
    plt.plot(rtx_ts, rtx_frame, label="frame")
    # plt.vlines(marked_ts, ymin=np.full(len(marked_ts),0), ymax=np.full(len(marked_ts),300))
    plt.legend(loc="upper left")
    if not show:
        plt.savefig(path+"/"+filename+"_bin_frame.pdf")
        plt.close()
    else:
        plt.show(block=True)

    fig = plt.figure(figsize=(16,9))
    # plt.plot(rtx_ts, rtx_data, label="osci")
    plt.fill_between(bin_centers, rtx_data_min, rtx_data_max, alpha=0.2, label="osci")
    plt.plot(rtx_ts, rtx_data_conv, label=conv_name, linewidth=2)
    plt.plot(rtx_ts, data, label=name, linewidth=3)
    plt.legend(loc="upper left")
    if not show:
        plt.savefig(path+"\\"+filename+"_bin.pdf")
        plt.close()
    else:
        plt.show(block=True)

frame_color = "#9edb07"
msr_color = "#785bd2"
adl_color = "#07dbae"
nvml_color = "#07dbae"
osci_color ="#021e42"
osci_band_color ="#03326e"
osci_color_cpu ="#054699"
osci_band_color_cpu ="#065ac5"
tinker_color = "#720a19"
tinker_color_cpu = "#ce122e"
hmc_color="#E5A314"

def sep_plot(start, end, ts, frame, rtx, data, data_name, tinker, filename, path, show, bin):
    div_val = np.ceil((ts[end]-ts[start])/10000/10)
    if div_val < 3:
        bin = False

    min_ts = ts.min()
    ts = ts.subtract(min_ts).divide(10000)

    if bin:
        rtx_mean, bin_edges, _ = sp.stats.binned_statistic(ts[start:end], rtx[start:end], statistic="mean", bins=div_val)
        rtx_dev, _, _ = sp.stats.binned_statistic(ts[start:end], rtx[start:end], statistic="std", bins=div_val)
        rtx_min = rtx_mean-rtx_dev
        rtx_max = rtx_mean+rtx_dev
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2

    fig = plt.figure(figsize=(16,9))
    plt.xlabel("time[ms]")
    plt.ylabel("P[W]")
    # plt.title(filename)
    if bin:
        plt.fill_between(bin_centers, rtx_min, rtx_max, alpha=0.2, label="Osc.", color=osci_band_color)
        plt.plot(bin_centers, rtx_mean, label="Osc.", color=osci_color)
    else:
        plt.plot(ts[start:end], rtx[start:end], label="Osc.", color=osci_color)
    plt.plot(ts[start:end], tinker[start:end], label="Tinker", linewidth=3, color=tinker_color)
    if not("None" in data):
        plt.plot(ts[start:end], data[start:end], label=data_name, linewidth=2, color=adl_color)
    max_val = rtx.median()
    frame[start:end].loc[frame < 2.5] = 0
    frame[start:end].loc[frame >= 2.5] = np.float32(max_val)
    plt.plot(ts[start:end], frame[start:end], label="Frame", color=frame_color)
    # plt.legend(loc="top")
    plt.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1), handletextpad=0.1)
    if not show:
        plt.savefig(path+"/"+filename+"_frame.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show(block=True)

def combined_plot(start, end, ts, frame, rtx_gpu, rtx_cpu, data_gpu, data_gpu_name, data_cpu, data_cpu_name, tinker_gpu, tinker_cpu, filename, path, show, gpu_bin, cpu_bin):
    div_val = np.ceil((ts[end]-ts[start])/10000/10)
    if div_val < 3:
        gpu_bin = False
        cpu_bin = False
    # div_val = 5000
    # if len(ts[start:end]) < 2*div_val:
    #     div_val = 1000
    min_ts = ts.min()
    ts = ts.subtract(min_ts).divide(10000)

    # bins=np.floor(len(ts[start:end])/div_val)
    if gpu_bin:
        rtx_gpu_mean, bin_edges, _ = sp.stats.binned_statistic(ts[start:end], rtx_gpu[start:end], statistic="mean", bins=div_val)
        rtx_gpu_dev, _, _ = sp.stats.binned_statistic(ts[start:end], rtx_gpu[start:end], statistic="std", bins=div_val)
        rtx_gpu_min = rtx_gpu_mean-rtx_gpu_dev
        rtx_gpu_max = rtx_gpu_mean+rtx_gpu_dev
        bin_width = (bin_edges[1] - bin_edges[0])
        gpu_bin_centers = bin_edges[1:] - bin_width/2

    if cpu_bin:
        rtx_cpu_mean, bin_edges, _ = sp.stats.binned_statistic(ts[start:end], rtx_cpu[start:end], statistic="mean", bins=div_val)
        rtx_cpu_dev, _, _ = sp.stats.binned_statistic(ts[start:end], rtx_cpu[start:end], statistic="std", bins=div_val)
        rtx_cpu_min = rtx_cpu_mean-rtx_cpu_dev
        rtx_cpu_max = rtx_cpu_mean+rtx_cpu_dev
        bin_width = (bin_edges[1] - bin_edges[0])
        cpu_bin_centers = bin_edges[1:] - bin_width/2

    fig = plt.figure(figsize=(16,9))
    plt.xlabel("time[ms]")
    plt.ylabel("P[W]")
    #plt.title(filename)
    if gpu_bin:
        plt.fill_between(gpu_bin_centers, rtx_gpu_min, rtx_gpu_max, alpha=0.2, label="Osc. GPU", color=osci_band_color)
        plt.plot(gpu_bin_centers, rtx_gpu_mean, label="Osc. GPU", color=osci_band_color)
    else:
        plt.plot(ts[start:end], rtx_gpu[start:end], label="Osc. GPU", color=osci_band_color)
    if cpu_bin:
        plt.fill_between(cpu_bin_centers, rtx_cpu_min, rtx_cpu_max, alpha=0.2, label="Osc. CPU", color=osci_band_color_cpu)
        plt.plot(cpu_bin_centers, rtx_cpu_mean, label="Osc. CPU", color=osci_color_cpu)
    else:
        plt.plot(ts[start:end], rtx_cpu[start:end], label="Osc. CPU", color=osci_color_cpu)
    if not("None" in data_gpu):
        plt.plot(ts[start:end], data_gpu[start:end], label=data_gpu_name, linewidth=2, color=adl_color)
    plt.plot(ts[start:end], data_cpu[start:end], label=data_cpu_name, linewidth=2, color=msr_color)
    plt.plot(ts[start:end], tinker_gpu[start:end], label="Tinker GPU", linewidth=3, color=tinker_color)
    plt.plot(ts[start:end], tinker_cpu[start:end], label="Tinker CPU", linewidth=3, color=tinker_color_cpu)
    max_val = rtx_gpu.median()
    frame[start:end].loc[frame < 2.5] = 0
    frame[start:end].loc[frame >= 2.5] = np.float32(max_val)
    plt.plot(ts[start:end], frame[start:end], label="Frame", color=frame_color)
    # plt.legend(loc="top")
    plt.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1), handletextpad=0.1)
    if not show:
        plt.savefig(path+"/"+filename+"_frame.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show(block=True)

def combined_plot_hmc(start, end, ts, frame, rtx_gpu, rtx_cpu, data_gpu, data_gpu_name, data_cpu, data_cpu_name, data, data_name, tinker_gpu, tinker_cpu, filename, path, show, gpu_bin, cpu_bin):
    div_val = np.ceil((ts[end]-ts[start])/10000/10)
    if div_val < 3:
        gpu_bin = False
        cpu_bin = False
    # div_val = 5000
    # if len(ts[start:end]) < 2*div_val:
    #     div_val = 1000
    min_ts = ts.min()
    ts = ts.subtract(min_ts).divide(10000)

    # bins=np.floor(len(ts[start:end])/div_val)
    if gpu_bin:
        rtx_gpu_mean, bin_edges, _ = sp.stats.binned_statistic(ts[start:end], rtx_gpu[start:end], statistic="mean", bins=div_val)
        rtx_gpu_dev, _, _ = sp.stats.binned_statistic(ts[start:end], rtx_gpu[start:end], statistic="std", bins=div_val)
        rtx_gpu_min = rtx_gpu_mean-rtx_gpu_dev
        rtx_gpu_max = rtx_gpu_mean+rtx_gpu_dev
        bin_width = (bin_edges[1] - bin_edges[0])
        gpu_bin_centers = bin_edges[1:] - bin_width/2

    if cpu_bin:
        rtx_cpu_mean, bin_edges, _ = sp.stats.binned_statistic(ts[start:end], rtx_cpu[start:end], statistic="mean", bins=div_val)
        rtx_cpu_dev, _, _ = sp.stats.binned_statistic(ts[start:end], rtx_cpu[start:end], statistic="std", bins=div_val)
        rtx_cpu_min = rtx_cpu_mean-rtx_cpu_dev
        rtx_cpu_max = rtx_cpu_mean+rtx_cpu_dev
        bin_width = (bin_edges[1] - bin_edges[0])
        cpu_bin_centers = bin_edges[1:] - bin_width/2

    fig = plt.figure(figsize=(16,9))
    plt.xlabel("time[ms]")
    plt.ylabel("P[W]")
    # plt.title(filename)
    if gpu_bin:
        plt.fill_between(gpu_bin_centers, rtx_gpu_min, rtx_gpu_max, alpha=0.2, label="Osc. GPU", color=osci_band_color)
        plt.plot(gpu_bin_centers, rtx_gpu_mean, label="Osc. GPU", color=osci_color)
    else:
        plt.plot(ts[start:end], rtx_gpu[start:end], label="Osc. GPU", color=osci_color)
    if cpu_bin:
        plt.fill_between(cpu_bin_centers, rtx_cpu_min, rtx_cpu_max, alpha=0.2, label="osci cpu", color=osci_band_color_cpu)
        plt.plot(cpu_bin_centers, rtx_cpu_mean, label="Osc. CPU", color=osci_color_cpu)
    else:
        plt.plot(ts[start:end], rtx_cpu[start:end], label="Osc. CPU", color=osci_color_cpu)
    if not("None" in data_gpu):
        plt.plot(ts[start:end], data_gpu[start:end], label=data_gpu_name, linewidth=2, color=adl_color)
    plt.plot(ts[start:end], data_cpu[start:end], label=data_cpu_name, linewidth=2, color=msr_color)
    plt.plot(ts[start:end], data[start:end], label=data_name, linewidth=2, color=hmc_color)
    plt.plot(ts[start:end], tinker_gpu[start:end], label="Tinker GPU", linewidth=3, color=tinker_color)
    plt.plot(ts[start:end], tinker_cpu[start:end], label="Tinker CPU", linewidth=3, color=tinker_color_cpu)
    max_val = rtx_gpu.median()
    frame[start:end].loc[frame < 2.5] = 0
    frame[start:end].loc[frame >= 2.5] = np.float32(max_val)
    plt.plot(ts[start:end], frame[start:end], label="Frame", color=frame_color)
    #plt.legend(loc="top")
    plt.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1), handletextpad=0.1)
    if not show:
        plt.savefig(path+"/"+filename+"_frame.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show(block=True)

def plot_gpuz(start, end, gpuz_time, gpuz_gpu_temp, gpuz_hotspot_temp, gpuz_mem_temp, gpuz_cpu_temp, gpuz_perfcap, filename, path):
    min_ts = gpuz_time[start:end].min()
    ts = gpuz_time.subtract(min_ts).divide(10000)

    fig = plt.figure(figsize=(16,9))
    #plt.xlabel("time [ms]")
    #plt.ylabel("temp [°C]")

    ax1 = plt.subplot()
    ax1.set_xlabel("time [ms]")
    ax1.set_ylabel("temp [°C]")

    l1, = ax1.plot(ts[start:end], gpuz_gpu_temp[start:end], label="GPU")
    l2, = ax1.plot(ts[start:end], gpuz_hotspot_temp[start:end], label="Hotspot")
    l3, = ax1.plot(ts[start:end], gpuz_mem_temp[start:end], label="Memory")
    l4, = ax1.plot(ts[start:end], gpuz_cpu_temp[start:end], label="CPU")

    ax2 = ax1.twinx()
    ax2.set_ylabel("PerfCap")
    l5, = ax2.plot(ts[start:end], gpuz_perfcap[start:end], label="PerfCap")

    # plt.legend(loc="upper left")
    plt.legend([l1, l2, l3, l4, l5], ["GPU", "Hotspot", "Memory", "CPU", "PerfCap"])

    plt.savefig(path+"/"+filename+".pdf", bbox_inches="tight")
    plt.close()

def plot_hwinfo(start, end, data, temp_cols, perf_cols, percentage, filename, path):
    min_ts = data["Timestamps"][start:end].min()
    ts = data["Timestamps"].subtract(min_ts).divide(10000)

    # fig = plt.figure(figsize=(16,9))
    
    plt.rcParams['figure.constrained_layout.use'] = True
    fig, ax = plt.subplots(1+len(perf_cols.items()), 1, figsize=(16,16))
    # fig.tight_layout()

    ax_counter = 0

    ax[ax_counter].set_xlabel("time [ms]")
    ax[ax_counter].set_ylabel("temp [°C]")

    for key, val in temp_cols.items():
        ax[ax_counter].plot(ts[start:end], data[val][start:end], label=key)

    ax[ax_counter].legend(loc="upper left")
    ax_counter = ax_counter + 1

    if percentage:
        y_label_name = "perf cap [%]"
    else:
        y_label_name = "perf cap [yes/no]"

    for key, val in perf_cols.items():
        ax[ax_counter].set_title(key)
        ax[ax_counter].set_xlabel("time [ms]")
        ax[ax_counter].set_ylabel(y_label_name)
        ax[ax_counter].plot(ts[start:end], data[val][start:end])
        ax_counter = ax_counter + 1

    # plt.legend(loc="upper right")
    plt.savefig(path+"/"+filename+".pdf", bbox_inches="tight")
    plt.close()
    
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
    off = np.ceil(num_frames/2 + 3)
    first = med-off
    last = first+num_frames
    first_ts = marked_ts.iloc[np.int32(first)]
    last_ts = marked_ts.iloc[np.int32(last)]
    first_idx = df_ts[df_ts==first_ts].index.to_list()[0]
    last_idx = df_ts[df_ts==last_ts].index.to_list()[0]
    return first_idx, last_idx

# https://stackoverflow.com/questions/46245035/pandas-dataframe-remove-outliers
def remove_outlier_indices(df, target):
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    trueList = ~((df[target] < (Q1 - 1.5 * IQR)) |(df[target] > (Q3 + 1.5 * IQR)))
    return trueList

def main():
    font = {'family' : 'Calibri',
        # 'weight' : 'bold',
        'size'   : 20}

    matplotlib.rc('font', **font)

    base_path = sys.argv[1]
    gpu_id = sys.argv[2]
    show = False
    if sys.argv[3] == "true":
        show = True
    subfolders = sys.argv[4].split(";")
    csv_path = sys.argv[5]
    tex_path = sys.argv[6]
    gpuz_path = sys.argv[7]

    msr_field_name = sys.argv[8]
    gpu_field_name = sys.argv[9]

    percentage = True
    if (gpu_id == "4090" or gpu_id == "4080" or gpu_id == "3090Ti"):
        gpu = "nvidia"
        temperature_vals = {"GPU": 'GPU-Temperatur [°C]', "Memory": 'GPU-Speicher Sperrschicht-Temperatur [°C]', "Hotspot": 'GPU-Hot-Spot-Temperatur [°C]'}
        percentage = False
        if (gpu_id == "3090Ti"):
            perfcap_vals = {"Power Throttling (avg)": 'GPU Leistungsbegrenzer (avg) [Yes/No]', "Power Throttling": 'Leistungsgrenzwert - Leistung [Yes/No]', "Temp Throttling": 'Leistungsgrenzwert - Thermisch [Yes/No]', "Voltage Reliability Throttling": 'Leistungsgrenzwert - Spannungs-Zuverlässigkeit [Yes/No]', "Max Voltage Throttling": 'Leistungsgrenzwert - Max. Betriebsspannung [Yes/No]', "Utilization Throttling": 'Leistungsgrenzwert - Auslastung [Yes/No]', "SLI GPUBoost Sync Throttling": 'Leistungsgrenzwert - SLI GPUBoost Sync [Yes/No]'}
        if (gpu_id == "4090"):
            perfcap_vals = {"Power Throttling": 'Leistungsgrenzwert - Leistung [Yes/No]', "Temp Throttling": 'Leistungsgrenzwert - Thermisch [Yes/No]', "Voltage Reliability Throttling": 'Leistungsgrenzwert - Spannungs-Zuverlässigkeit [Yes/No]', "Max Voltage Throttling": 'Leistungsgrenzwert - Max. Betriebsspannung [Yes/No]', "Utilization Throttling": 'Leistungsgrenzwert - Auslastung [Yes/No]', "SLI GPUBoost Sync Throttling": 'Leistungsgrenzwert - SLI GPUBoost Sync [Yes/No]'}


    if (gpu_id == "6900XT" or gpu_id == "7900XT" or gpu_id == "7900XTX"):
        gpu = "amd"
        temperature_vals = {"GPU":  'GPU Temperature [°C]', "Memory": 'GPU Memory Junction Temperature [°C]', "Hotspot": 'GPU-Hot-Spot-Temperatur [°C]'}
        if (gpu_id == "6900XT"):
            perfcap_vals = {"PPT": 'GPU PPT-Limit [%]', "Core TDC": 'GPU-Kern TDC-Limit [%]', "SOC TDC": 'GPU-SOC-TDC-Limit [%]', "GPU Temp (avg)": 'Thermische Grenzen der GPU (avg) [%]', "GPU Edge Temp": 'Thermischer Grenzwert der GPU-Kante [%]', "GPU Hotspot Temp": 'Thermischer Grenzwert für GPU-Hotspot [%]', "Memory Temp": 'Thermischer Grenzwert des GPU-Speichers [%]', "GPU VR GFX Temp": 'GPU VR GFX thermischer Grenzwert [%]', "GPU VR SOC Temp": 'Thermischer Grenzwert des GPU VR-SOC [%]', "GPU VR Memory Temp": 'GPU VR MEM Thermischer Grenzwert [%]'}
        if (gpu_id == "7900XT"):
            perfcap_vals = {"PPT": 'GPU PPT-Limit (Anhaltend) [%]', "Core TDC": 'GPU-Kern TDC-Limit [%]', "SOC TDC": 'GPU-SOC-TDC-Limit [%]', "GPU Temp (avg)": 'Thermische Grenzen der GPU (avg) [%]', "GPU Hotspot Temp": 'Thermischer Grenzwert für GPU-Hotspot [%]', "Memory Temp": 'Thermischer Grenzwert des GPU-Speichers [%]', "GPU VR GFX Temp": 'GPU VR GFX thermischer Grenzwert [%]', "GPU VR Memory Temp": 'GPU VR MEM Thermischer Grenzwert [%]'} #"GPU VR SOC Temp": 'Thermischer Grenzwert des  GPU VR-SOC [%]', 
        if (gpu_id == "7900XTX"):
            perfcap_vals = {"PPT": 'GPU PPT-Limit (Anhaltend) [%]', "Core TDC": 'GPU-Kern TDC-Limit [%]', "SOC TDC": 'GPU-SOC-TDC-Limit [%]', "GPU Temp (avg)": 'Thermische Grenzen der GPU (avg) [%]', "GPU Hotspot Temp": 'Thermischer Grenzwert für GPU-Hotspot [%]', "Memory Temp": 'Thermischer Grenzwert des GPU-Speichers [%]'}

    if (gpu_id == "A770"):
        gpu = "intel"
        temperature_vals = {"GPU":  'Globale GPU-Temperatur [°C]', "Memory": 'GPU-Speichertemperatur [°C]', "Hotspot": 'GPU Core Temperatur [°C]'}
        percentage = False
        perfcap_vals = {"Throttling (avg)": 'Gründe für GPU-Drosselung (avg) [Yes/No]', "Power Throttling": 'Durchschnittliche Leistung (PL1) [Yes/No]', "Burst Power Throttling": 'Burst-Leistung (PL2) [Yes/No]', "Current Throttling": 'Current (PL4) [Yes/No]', "Temp Throttling": 'Thermisch [Yes/No]', "Power Supply Throttling": 'Energieversorgung [Yes/No]', "Software Throttling": 'Softwarelimit [Yes/No]', "Hardware Throttling": 'Hardwarelimit [Yes/No]', "Memory Throttling (avg)": 'Gründe für Speicherdrosselung (avg) [Yes/No]', "Power Throttling 1": 'Durchschnittliche Leistung (PL1) [Yes/No].1', "Burst Power Throttling 1": 'Burst-Leistung (PL2) [Yes/No].1', "Current Throttling 1": 'Current (PL4) [Yes/No].1', "Temp Throttling 1": 'Thermisch [Yes/No].1', "Power Supply Throttling 1": 'Energieversorgung [Yes/No].1', "Software Throttling 1": 'Softwarelimit [Yes/No].1', "Hardware Throttling 1": 'Hardwarelimit [Yes/No].1'}

    if (gpu == "nvidia"):
        nvml_name = gpu_field_name + "_samples" 
    if (gpu == "amd"):
        adl_name = gpu_field_name + "_samples"

    gpu_rtx_ident = "12VHPWR"
    if (gpu == "amd" or gpu == "intel"):
        gpu_rtx_ident = "PEG"

    print("Analysis script")
    print("Parameter:")
    print(f"\tbase_path: {base_path}")
    print(f"\tgpu: {gpu}")
    print(f"\tshow: {show}")
    print(f"\tsubfolders: {subfolders}")
    print(f"\tcsv_path: {csv_path}")
    print(f"\ttex_path: {tex_path}")

    # folder_path = R"\\techhouse\t$\gralkapk\dev\res\spheres"
    # show = False
    # gpu = "nvidia"
    # # amd_name = "ASIC/AMD Radeon RX 6900 XT"
    # amd_name = "BOARD/AMD Radeon RX 7900 XT"

    gpu_name = os.path.basename(os.path.normpath(base_path))

    header = True
    if os.path.isfile(csv_path):
        header = False

    csv_file = io.open(csv_path, "a", encoding="utf-8")
    tex_file = io.open(tex_path, "a", encoding="utf-8")

    # gpuz_data = pd.read_parquet(gpuz_path)
    # hwinfo_data = pd.read_parquet(gpuz_path)

    if header:
        csv_file.write("gpu, method, time, num_frames, rtx_gpu, tinker_gpu, soft_gpu, rtx_cpu, tinker_cpu, msr, hmc_p, hmc_s, hmc_p_int, total_rtx, total_tinker\n")

    tex_file.write("\\begin{table}\n")
    tex_file.write("\t\\centering\n")
    tex_file.write(f"\t\\caption{{{gpu} {gpu_name}}}\n")
    tex_file.write("\t\\rowcolors{2}{gray!25}{white}\n")
    tex_file.write("\t\\begin{tabular}{|c V{4} c|c|c|}\n")
    tex_file.write("\t\t\\hline \\rowcolor{gray!50}\n")
    tex_file.write("\t\tMode & RTX & Tinker & Soft \\\\ \\hlineB{4}\n")

    for sf in subfolders:
        folder_path = base_path + "/" + sf
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            continue

        data_map, metadata = read_parquet_files_in_folder(folder_path)

        # extract computation recipes
        tinker_r, rtx_r = get_analysis_recipes(metadata)

        # get dataframe containing all rtx series
        rtx_df = concat_rtx_df(data_map)
        # shift the data on the timeline
        # rtx_df = match_trigger_signal(rtx_df, metadata[b"trigger_ts"][0])
        # compute the series for the rtx data
        rtx_df = compute_with_recipe(rtx_df, rtx_r)


        min_ts, max_ts = get_tight_valid_range(data_map["tinker_s0"], "timestamps", -2**63, 2**63 - 1)
        if gpu == "nvidia":
            min_ts, max_ts = get_tight_valid_range(data_map["nvml_s0"], "timestamps", min_ts, max_ts)
        if gpu == "amd":
            min_ts, max_ts = get_tight_valid_range(data_map["adl_s0"], "timestamps", min_ts, max_ts)
        min_ts, max_ts = get_tight_valid_range(data_map["msr_s0"], "timestamps", min_ts, max_ts)
        min_ts, max_ts = get_tight_valid_range(data_map["HMC01"], "Timestamp", min_ts, max_ts)
        # offset = 10000000+rtx_df["timestamps"].loc[0]
        #first_ts, last_ts, num_frames, rtx_first_idx, rtx_last_idx, marked_ts = get_frame_interval_and_count(rtx_df, "timestamps", offset)
        # print(f"stats: {first_ts}, {last_ts}, {num_frames}, {rtx_first_idx}, {rtx_last_idx}")
        first_ts, last_ts, num_frames, rtx_first_idx, rtx_last_idx, marked_ts = get_frame_interval_and_count(rtx_df, "timestamps", 0, min_ts, max_ts)
        # print(f"stats: {first_ts}, {last_ts}, {num_frames}, {rtx_first_idx}, {rtx_last_idx}")

        total_time = (last_ts-first_ts)/10000
        print(f"Stats: time {total_time} ms fcount: {num_frames}")

        # rtx
        print("RTX:")
        rtx_gpu_ws, rtx_cpu_ws, rtx_total_ws = get_stats(rtx_df, rtx_first_idx, rtx_last_idx, num_frames, gpu)

        #rtx_gpu_conv = moving_average(rtx_df[gpu_rtx_ident][rtx_first_idx:rtx_last_idx]+rtx_df["12VPEG"][rtx_first_idx:rtx_last_idx], rtx_df["timestamps"][rtx_first_idx:rtx_last_idx])

        soft_gpu_ws = 0.0

        if gpu == "nvidia":
            # nvml
            print("NVML:")
            # nvml_id = nvml_name.split("_")[0]
            # first_idx, last_idx = match_ts_on_df(data_map["nvml_s0"], f"{nvml_id}_timestamps", first_ts, last_ts)
            soft_df = interpolate(data_map["nvml_s0"], rtx_df["timestamps"])
            first_idx, last_idx = match_ts_on_df(soft_df, "timestamps", first_ts, last_ts)
            ws = integrate(soft_df["timestamps"], soft_df[nvml_name], first_idx, last_idx)
            print(f"\tEnergy: {np.round(ws,2)} ({np.round(ws/num_frames,2)}) Ws")
            # print(f"\tEnergy per frame: {np.round(ws/num_frames,2)} Ws")
            soft_gpu_ws = np.round(ws,2)
            soft_sample_name = nvml_name
            soft_name = "NVML"

        if gpu == "amd":
            # adl
            print("ADL:")
            # adl_id = adl_name.split("_")[0]
            # first_idx, last_idx = match_ts_on_df(data_map["adl_s0"], f"{adl_id}_timestamps", first_ts, last_ts)
            soft_df = interpolate(data_map["adl_s0"], rtx_df["timestamps"])
            first_idx, last_idx = match_ts_on_df(soft_df, "timestamps", first_ts, last_ts)
            ws = integrate(soft_df["timestamps"], soft_df[adl_name], first_idx, last_idx)
            print(f"\tEnergy: {np.round(ws,2)} ({np.round(ws/num_frames,2)}) Ws")
            # print(f"\tEnergy per frame: {np.round(ws/num_frames,2)} Ws")
            # plot_time(np.array(rtx_df["timestamps"][rtx_first_idx:rtx_last_idx]), data_map["adl_s0"][f"ADL_timestamps"][first_idx:last_idx])
            soft_gpu_ws = np.round(ws,2)
            soft_sample_name = adl_name
            soft_name = "ADL"

        if gpu != "amd" and gpu != "nvidia":
            soft_df = None

        # tinker
        print("Tinker:")
        tinker_df = interpolate(data_map["tinker_s0"], rtx_df["timestamps"])
        tinker_df = compute_with_recipe(tinker_df, tinker_r)
        first_idx, last_idx = match_ts_on_df(tinker_df, "timestamps", first_ts, last_ts)
        tinker_gpu_ws, tinker_cpu_ws, tinker_total_ws = get_stats(tinker_df, first_idx, last_idx, num_frames, gpu)
        # tinker_df = interpolate(rtx_df["timestamps"][rtx_first_idx:rtx_last_idx], tinker_df, "timestamps")
        start_idx_1, end_idx_1 = select_frames(rtx_df["timestamps"], marked_ts, 1)
        start_idx_3, end_idx_3 = select_frames(rtx_df["timestamps"], marked_ts, 3)
        start_idx_100, end_idx_100 = select_frames(rtx_df["timestamps"], marked_ts, 50)
        # print(tinker_df)
        if gpu == "nvidia":
            # plot_bin(rtx_df["12VHPWR"][rtx_first_idx:rtx_last_idx]+rtx_df["12VPEG"][rtx_first_idx:rtx_last_idx], rtx_df["timestamps"][rtx_first_idx:rtx_last_idx], rtx_df["frame"][rtx_first_idx:rtx_last_idx], rtx_gpu_conv, tinker_df["12VHPWR"]+tinker_df["12VPEG"], marked_ts, "tinker", "tinker.pdf")
            
            # plot(rtx_df["12VHPWR"][start_idx_100:end_idx_100]+rtx_df["12VPEG"][start_idx_100:end_idx_100], rtx_df["timestamps"][start_idx_100:end_idx_100], rtx_df["frame"][start_idx_100:end_idx_100], rtx_gpu_conv[start_idx_100:end_idx_100], tinker_df["12VHPWR"][start_idx_100:end_idx_100]+tinker_df["12VPEG"][start_idx_100:end_idx_100], "tinker", "f50", folder_path, show)
            # plot_bin(rtx_df["12VHPWR"][start_idx_100:end_idx_100]+rtx_df["12VPEG"][start_idx_100:end_idx_100], rtx_df["timestamps"][start_idx_100:end_idx_100], rtx_df["frame"][start_idx_100:end_idx_100], rtx_gpu_conv[start_idx_100:end_idx_100], tinker_df["12VHPWR"][start_idx_100:end_idx_100]+tinker_df["12VPEG"][start_idx_100:end_idx_100], "tinker", "f50", folder_path, show)
            # plot(rtx_df["12VHPWR"][start_idx_1:end_idx_1]+rtx_df["12VPEG"][start_idx_1:end_idx_1], rtx_df["timestamps"][start_idx_1:end_idx_1], rtx_df["frame"][start_idx_1:end_idx_1], rtx_gpu_conv[start_idx_1:end_idx_1], tinker_df["12VHPWR"][start_idx_1:end_idx_1]+tinker_df["12VPEG"][start_idx_1:end_idx_1], "tinker", "f1", folder_path, show)
            # plot(rtx_df["12VHPWR"][start_idx_3:end_idx_3]+rtx_df["12VPEG"][start_idx_3:end_idx_3], rtx_df["timestamps"][start_idx_3:end_idx_3], rtx_df["frame"][start_idx_3:end_idx_3], rtx_gpu_conv[start_idx_3:end_idx_3], tinker_df["12VHPWR"][start_idx_3:end_idx_3]+tinker_df["12VPEG"][start_idx_3:end_idx_3], "tinker", "f3", folder_path, show)
            
            sep_plot(start_idx_100, end_idx_100, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], soft_df[nvml_name], "NVML", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f50", folder_path, show, False)
            # relay_plot_bin(start_idx_100, end_idx_100, rtx_df["12VHPWR"]+rtx_df["12VPEG"], rtx_df["timestamps"], rtx_df["frame"], rtx_gpu_conv, "osci ma", tinker_df["12VHPWR"]+tinker_df["12VPEG"], "tinker", "f50", folder_path, show)
            sep_plot(start_idx_1, end_idx_1, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], soft_df[nvml_name], "NVML", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f1", folder_path, show, False)
            sep_plot(start_idx_3, end_idx_3, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], soft_df[nvml_name], "NVML", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f3", folder_path, show, False)
            tinker_peg_name = gpu_rtx_ident
        if gpu == "amd":
            # plot(rtx_df["12VHPWR"][start_idx_100:end_idx_100]+rtx_df["12VPEG"][start_idx_100:end_idx_100], rtx_df["timestamps"][start_idx_100:end_idx_100], rtx_df["frame"][start_idx_100:end_idx_100], rtx_gpu_conv[start_idx_100:end_idx_100], tinker_df["PEG"][start_idx_100:end_idx_100]+tinker_df["12VPEG"][start_idx_100:end_idx_100], "tinker", "f50", folder_path, show)
            # plot_bin(rtx_df["12VHPWR"][start_idx_100:end_idx_100]+rtx_df["12VPEG"][start_idx_100:end_idx_100], rtx_df["timestamps"][start_idx_100:end_idx_100], rtx_df["frame"][start_idx_100:end_idx_100], rtx_gpu_conv[start_idx_100:end_idx_100], tinker_df["PEG"][start_idx_100:end_idx_100]+tinker_df["12VPEG"][start_idx_100:end_idx_100], "tinker", "f50", folder_path, show)
            # plot(rtx_df["12VHPWR"][start_idx_1:end_idx_1]+rtx_df["12VPEG"][start_idx_1:end_idx_1], rtx_df["timestamps"][start_idx_1:end_idx_1], rtx_df["frame"][start_idx_1:end_idx_1], rtx_gpu_conv[start_idx_1:end_idx_1], tinker_df["PEG"][start_idx_1:end_idx_1]+tinker_df["12VPEG"][start_idx_1:end_idx_1], "tinker", "f1", folder_path, show)
            # plot(rtx_df["12VHPWR"][start_idx_3:end_idx_3]+rtx_df["12VPEG"][start_idx_3:end_idx_3], rtx_df["timestamps"][start_idx_3:end_idx_3], rtx_df["frame"][start_idx_3:end_idx_3], rtx_gpu_conv[start_idx_3:end_idx_3], tinker_df["PEG"][start_idx_3:end_idx_3]+tinker_df["12VPEG"][start_idx_3:end_idx_3], "tinker", "f3", folder_path, show)

            sep_plot(start_idx_100, end_idx_100, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], soft_df[adl_name], "ADL", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f50", folder_path, show, not(show))
            # relay_plot_bin(start_idx_100, end_idx_100, rtx_df["12VHPWR"]+rtx_df["12VPEG"], rtx_df["timestamps"], rtx_df["frame"], rtx_gpu_conv, "osci ma", tinker_df["PEG"]+tinker_df["12VPEG"], "tinker", "f50", folder_path, show)
            sep_plot(start_idx_1, end_idx_1, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], soft_df[adl_name], "ADL", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f1", folder_path, show, False)
            sep_plot(start_idx_3, end_idx_3, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], soft_df[adl_name], "ADL", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f3", folder_path, show, False)
            tinker_peg_name = gpu_rtx_ident
        if gpu == "intel":
            sep_plot(start_idx_100, end_idx_100, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], "None", "Intel", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f50", folder_path, show, not(show))
            sep_plot(start_idx_1, end_idx_1, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], "None", "Intel", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f1", folder_path, show, False)
            sep_plot(start_idx_3, end_idx_3, rtx_df["timestamps"], rtx_df["frame"], rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], "None", "Intel", tinker_df[gpu_rtx_ident]+tinker_df["12VPEG"], f"{gpu_id}_f3", folder_path, show, False)
            tinker_peg_name = gpu_rtx_ident


        osci_cpu = rtx_df["P8"]+rtx_df["P4"] #+rtx_df["12VATX"] #-rtx_df["12VPEG"]
        tinker_cpu = tinker_df["P8"]+tinker_df["P4"] #+tinker_df["12VATX"] #-tinker_df["12VPEG"]+tinker_df["3VATX"]+tinker_df["5VATX"]-tinker_df["3VPEG"]
        #tinker_cpu = tinker_df["12VATX"]-tinker_df["12VPEG"]
        # msr
        print("MSR:")
        #indices = remove_outlier_indices(data_map["msr_s0"], msr_field_name + "_samples")
        #data_map["msr_s0"] = data_map["msr_s0"][indices]
        msr_df = interpolate(data_map["msr_s0"], rtx_df["timestamps"])
        # fig = plt.figure(figsize=(16,9))
        # plt.xlabel("time [ms]")
        # plt.ylabel("P[W]")
        # plt.plot(rtx_df["timestamps"], osci_cpu, label="osci")
        # plt.plot(rtx_df["timestamps"], tinker_cpu, label="tinker")
        # plt.plot(rtx_df["timestamps"], msr_df[msr_field_name + "_samples"], label="msr")
        # #plt.plot(data_map["msr_s0"][msr_field_name + "_timestamps"], data_map["msr_s0"][msr_field_name + "_samples"], label="msr_old")
        # plt.legend()
        # plt.show()
        first_idx, last_idx = match_ts_on_df(msr_df, f"timestamps", first_ts, last_ts)
        sep_plot(start_idx_3, end_idx_3, rtx_df["timestamps"], rtx_df["frame"], osci_cpu, msr_df[msr_field_name + "_samples"], "MSR", tinker_cpu, f"{gpu_id}_cpu_f3", folder_path, show, False)
        if gpu == "amd" or gpu == "nvidia":
            combined_plot(start_idx_3, end_idx_3, rtx_df["timestamps"], rtx_df["frame"],\
                      rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], \
                       osci_cpu, \
                       soft_df[soft_sample_name], soft_name, \
                        msr_df[msr_field_name + "_samples"], "MSR", \
                        tinker_df[tinker_peg_name]+tinker_df["12VPEG"], \
                             tinker_cpu, f"{gpu_id}_gpu_cpu_f3", folder_path, show, True, True)
        else:
            combined_plot(start_idx_3, end_idx_3, rtx_df["timestamps"], rtx_df["frame"],\
                      rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], \
                       osci_cpu, \
                       "None", "None", \
                        msr_df[msr_field_name + "_samples"], "MSR", \
                        tinker_df[tinker_peg_name]+tinker_df["12VPEG"], \
                             tinker_cpu, f"{gpu_id}_gpu_cpu_f3", folder_path, show, True, True)
        # relay_plot_bin(start_idx_3, end_idx_3, rtx_df["P8"]+rtx_df["P4"]+rtx_df["12VATX"], rtx_df["timestamps"], rtx_df["frame"], msr_df["MSR_samples"], "msr", tinker_df["P8"]+tinker_df["P4"]+tinker_df["12VATX"], "tinker", "cpu_f3", folder_path, show)
        ws = integrate(msr_df[f"timestamps"], msr_df[msr_field_name + "_samples"], first_idx, last_idx)
        print(f"\tEnergy: {np.round(ws,2)} ({np.round(ws/num_frames,2)}) Ws")
        print(f"\tEnergy per frame: {np.round(ws/num_frames,2)} Ws")
        msr_ws = np.round(ws,2)

        # hmc
        print("HMC:")
        hmc_df = interpolate_hmc(data_map["HMC01"], "Timestamp", rtx_df["timestamps"])
        first_idx, last_idx = match_ts_on_df(hmc_df, "timestamps", first_ts, last_ts)
        ws = integrate(hmc_df["timestamps"], hmc_df["S[VA]"], first_idx, last_idx)
        p_ws = integrate(hmc_df["timestamps"], hmc_df["P[W]"], first_idx, last_idx)
        hmc_total_ws = integrate(data_map["HMC01"]["Timestamp"], data_map["HMC01"]["P[W]"], data_map["HMC01"]["P[W]"].index[0], data_map["HMC01"]["P[W]"].index[-1])
        hmc_int_ws = data_map["HMC01"]["WH[Wh]"][data_map["HMC01"]["WH[Wh]"].index[-1]] * 3600.0
        print(f"\tEnergy: {np.round(ws,2)} Ws")
        print(f"\tEnergy per frame: {np.round(ws/num_frames,2)} Ws")
        print(f"\tTotal Energy: {np.round(hmc_total_ws,2)} Ws vs. Int Energy: {np.round(hmc_int_ws,2)} Ws")
        hmc_s_ws = np.round(ws,2)
        hmc_p_ws = np.round(p_ws,2)
        hmc_p_int_ws = np.round((hmc_df["WH[Wh]"].iloc[last_idx]-hmc_df["WH[Wh]"].iloc[first_idx])*3600, 2)

        if gpu == "amd" or gpu == "nvidia":
            combined_plot_hmc(start_idx_3, end_idx_3, rtx_df["timestamps"], rtx_df["frame"],\
                      rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], \
                       osci_cpu, \
                       soft_df[soft_sample_name], soft_name, \
                        msr_df[msr_field_name + "_samples"], "MSR", \
                        hmc_df["P[W]"], "hmc", \
                        tinker_df[tinker_peg_name]+tinker_df["12VPEG"], \
                             tinker_cpu, f"{gpu_id}_gpu_cpu_hmc_f3", folder_path, show, True, True)
        else:
            combined_plot_hmc(start_idx_3, end_idx_3, rtx_df["timestamps"], rtx_df["frame"],\
                      rtx_df[gpu_rtx_ident]+rtx_df["12VPEG"], \
                       osci_cpu, \
                       "None", "None", \
                        msr_df[msr_field_name + "_samples"], "MSR", \
                        hmc_df["P[W]"], "hmc", \
                        tinker_df[tinker_peg_name]+tinker_df["12VPEG"], \
                             tinker_cpu, f"{gpu_id}_gpu_cpu_hmc_f3", folder_path, show, True, True)
            
        # gpuz
        # plot temperature and perf cap over measurement time
        # first_idx, last_idx = match_ts_on_df(gpuz_data, "Timestamps", first_ts, last_ts)
        # plot_gpuz(first_idx, last_idx, gpuz_data["Timestamps"], gpuz_data["GPU Temperature [°C]"], gpuz_data["Hot Spot [°C]"], gpuz_data["Memory Temperature [°C]"], gpuz_data["CPU Temperature [°C]"], gpuz_data["PerfCap Reason []"], "gpuz", folder_path)

        # hwinfo
        #first_idx, last_idx = match_ts_on_df(hwinfo_data, "Timestamps", first_ts, last_ts)
        #plot_hwinfo(first_idx, last_idx, hwinfo_data, temperature_vals, perfcap_vals, percentage, "hwinfo", folder_path)

        tex_sf = sf.replace('_', '\\_')
        csv_text = f"{gpu} {gpu_name}, {sf}, {total_time}, {num_frames}, {rtx_gpu_ws}, {tinker_gpu_ws}, {soft_gpu_ws}, {rtx_cpu_ws}, {tinker_cpu_ws}, {msr_ws}, {hmc_p_ws}, {hmc_s_ws}, {hmc_p_int_ws}, {rtx_total_ws}, {tinker_total_ws}\n"
        tex_text = f"\t\t{tex_sf} ({num_frames}) & {rtx_gpu_ws} ({np.round(rtx_gpu_ws/num_frames,2)}) & {tinker_gpu_ws} ({np.round(tinker_gpu_ws/num_frames, 2)}) & {soft_gpu_ws} ({np.round(soft_gpu_ws/num_frames, 2)}) \\\\ \\hline\n"

        csv_file.write(csv_text)
        tex_file.write(tex_text)

    tex_file.write("\t\\end{tabular}\n")
    tex_file.write("\\end{table}\n")

    csv_file.close()
    tex_file.close()


if __name__=="__main__":
    main()
