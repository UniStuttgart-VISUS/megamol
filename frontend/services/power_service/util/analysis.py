import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import scipy as sp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
import matplotlib.pyplot as plt

def read_parquet_files_in_folder(folder_path):
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

def match_trigger_signal(df, trigger_ts):
    trigger_idx = df["trigger"].where(df["trigger"] >= 2.5).first_valid_index()
    rtx_ts = df["timestamps"][trigger_idx]
    diff = trigger_ts - rtx_ts
    df["timestamps"] = df["timestamps"].add(diff)

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
    if first_idx is None:
        print(f"first_idx is None for {ts_name}")
        first_idx = 0
    if last_idx is None:
        print(f"last_idx is None for {ts_name}")
        last_idx = len(df.index)
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
    return series

def compute_with_recipe_tinker(df, recipe, first_ts, last_ts):
    data = {}
    for key,value in recipe.items():
        passed = True
        ws = 0
        try:
            add = value["ADD"]
            if len(add) == 0:
                passed = False
            for n in add:
                ts_name = n.split("_")[0]+"_timestamps"
                first_idx, last_idx = match_ts_on_df(df, ts_name, first_ts, last_ts)
                val = integrate(df[ts_name], df[n], first_idx, last_idx)
                ws = ws + val
        except:
            passed = False
            pass
        if passed:
            data[key] = ws
    return data

def compute_with_recipe_tinker_2(df, recipe, first_ts, last_ts):
    data = {}
    series = {}
    for key,value in recipe.items():
        passed = True
        ws = 0
        s = np.zeros(0)
        try:
            first = True
            add = value["ADD"]
            if len(add) == 0:
                passed = False
            for n in add:
                ts_name = "timestamps"
                # first_idx, last_idx = match_ts_on_df(df, ts_name, first_ts, last_ts)
                val = integrate(df[ts_name], df[n], 0, len(df[n]))
                ws = ws + val
                if first:
                    s = df[n]
                    first = False
                else:
                    s = s+df[n]
        except:
            passed = False
            pass
        if passed:
            data[key] = ws
            series[key] = s
    return data, series

def concat_rtx_df(map):
    rtb01_df = map["rtb01_s0"]
    rtb02_df = map["rtb02_s0"]
    rtb02_df = rtb02_df.drop("timestamps", axis=1)
    rtb03_df = map["rtb03_s0"]
    rtb03_df = rtb03_df.drop("timestamps", axis=1)
    rta01_df = map["rta01_s0"]
    rta01_df = rta01_df.drop("timestamps", axis=1)
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

def get_stats_tinker(td, num_frames):
    total_ws = 0
    for key in td:
        print(f"\t{key}:")
        total_ws = total_ws + td[key]
        print(f"\t\tEnergy: {td[key]} Ws")
        print(f"\t\tEnergy per frame: {td[key]/num_frames} Ws")
    print(f"\t\t\tSum: {total_ws}")
    print(f"\t\t\tSum per frame: {total_ws/num_frames}")

def interpolate_non_rtx(rtx_ts_series, data, ts):
    # ret = np.zeros(len(rtx_ts_series))
    # interp = sp.interpolate.BarycentricInterpolator(ts[first_idx:last_idx], data[first_idx:last_idx])
    # for i in range(len(rtx_ts_series)):
    #     # ret[i] = sp.interpolate.barycentric_interpolate(ts[first_idx:last_idx], data[first_idx:last_idx], rtx_ts_series[i])
    #     ret[i] = interp(rtx_ts_series[i])
    ret = np.interp(rtx_ts_series, ts, data)
    return ret

def moving_average(rtx_data, rtx_ts):
    diff_ms = (rtx_ts.iat[1]-rtx_ts.iat[0])/10000
    factor = np.int32(np.ceil(10/diff_ms))
    windows = rtx_data.rolling(factor)
    ma = windows.mean()
    ma_list = ma.tolist()
    return ma_list
    # rtx_data_conv=ma_list[factor-1:]
    # return rtx_data_conv

def plot_on_rtx_ts(rtx_data, rtx_ts, rtx_frame, rtx_data_conv, data, path):
    # gos = []
    # gos.append(go.Scatter(name="rtx", x=np.array(rtx_ts)[::6], y=np.array(rtx_data)[::6]))
    # gos.append(go.Scatter(name="rtx_conv", x=np.array(rtx_ts)[::6], y=np.array(rtx_data_conv)[::6]))
    # gos.append(go.Scatter(name="data", x=np.array(rtx_ts)[::6], y=np.array(data)[::6]))
    # # gos.append(go.Scatter(name="diff", x=rtx_ts, y=rtx_data-data))
    # plot = go.Figure(gos)
    # # plot.write_html(path)
    # # plot.write_image(path)
    # # plot.show(renderer="svg")
    # plot.show()

    # df = pd.concat([pd.Series(rtx_ts), pd.Series(rtx_data), pd.Series(rtx_data_conv), pd.Series(data)], keys=["timestamps", "rtx", "rtx_conv", "data"], axis=1)
    # plot = df.hvplot(x="timestamps", y=["rtx", "rtx_conv", "data"])
    # # hvplot.show(plot)
    # hvplot.save(plot, path)

    fig = plt.figure()
    plt.plot(rtx_ts, rtx_data, label="rtx")
    plt.plot(rtx_ts, rtx_data_conv, label="rtx_conv")
    plt.plot(rtx_ts, data, label="data")
    plt.plot(rtx_ts, rtx_frame.multiply(100), label="frame")
    plt.legend(loc="upper left")
    plt.show(block=True)



def plot_timestamps(df, rtx_series, nvml_series, hmc_series, trigger_series, trigger_ts, tinker_inter, path):
    gos = []
    counter = 0
    min_val = np.int64(np.iinfo(np.int64).max)
    for name, values in df.items():
        if "timestamps" in name:
            if min_val > values[0]:
                min_val = values[0]
    for name, values in df.items():
        if "timestamps" in name:
            y = np.full(len(values), counter)
            counter = counter + 1
            values = values.subtract(min_val).divide(10000)
            gos.append(go.Scatter(name=name, x=values, y=y, mode="markers"))
    y = np.full(len(tinker_inter), counter)
    counter = counter + 1
    tinker_inter = (tinker_inter -min_val) / 10000
    gos.append(go.Scatter(name="tinker_inter", x=tinker_inter, y=y, mode="markers"))
    rtx = rtx_series.subtract(min_val).divide(10000)
    y = np.full(2, counter)
    x = np.array([rtx[0], rtx[len(rtx)-1]])
    gos.append(go.Scatter(name="rtx", x=x, y=y, mode="markers"))
    gos.append(go.Scatter(name="rtx_trigger", x=[(trigger_ts-min_val)/10000], y=[19], mode="markers"))
    # gos.append(go.Scatter(name="rtx_trigger", x=[(133441159695437840 - min_val)/10000], y=[19], mode="markers"))
    # gos.append(go.Scatter(name="rtx_trigger", x=[(133441133522502006 - min_val)/10000, (133441133522502622 - min_val)/10000, (133441133522503175 - min_val)/10000], y=[19, 19, 19], mode="markers"))
    # gos.append(go.Scatter(name="buffer_start", x=[(133441133517866257 - min_val)/10000], y=[20], mode="markers"))
    # gos.append(go.Scatter(name="buffer_stop", x=[(133441133552075957 - min_val)/10000], y=[21], mode="markers"))
    nvml = nvml_series.subtract(min_val).divide(10000)
    y = np.full(len(nvml), counter + 1)
    gos.append(go.Scatter(name="nvml", x=nvml, y=y, mode='markers'))

    gos.append(go.Scatter(name="Trigger", x=rtx, y=trigger_series))

    hmc = hmc_series.subtract(min_val).divide(10000)
    y = np.full(len(hmc), counter + 2)
    gos.append(go.Scatter(name="hmc", x=hmc, y=y, mode='markers'))
    plot = go.Figure(gos)
    # plot.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=10))
    # plot.show(renderer="svg")
    # plot.write_html(path)
    # plot.write_image(path)
    plot.show()

def main():
    data_map, metadata = read_parquet_files_in_folder(sys.argv[1])
    
    tinker_r, rtx_r = get_analysis_recipes(metadata)
    # tinker_data = compute_with_recipe(data_map["tinker_s0"], tinker_r)
    rtx_df = concat_rtx_df(data_map)
    match_trigger_signal(rtx_df, metadata[b"trigger_ts"][0])
    rtx_data = compute_with_recipe(rtx_df, rtx_r)

    get_framemarker_loc(rtx_df, "frame")
    first_ts, last_ts, num_frames, rtx_first_idx, rtx_last_idx = get_frame_interval_and_count(rtx_df, "timestamps")
    # print(first_ts)
    # print(last_ts)

    # rtx
    print("RTX:")
    get_stats(rtx_data, rtx_df["timestamps"], rtx_first_idx, rtx_last_idx, num_frames)

    rtx_gpu_conv = moving_average(rtx_data["12VHPWR"][rtx_first_idx:rtx_last_idx]+rtx_data["12VPEG"][rtx_first_idx:rtx_last_idx], rtx_df["timestamps"][rtx_first_idx:rtx_last_idx])

    # tinker
    # first_idx, last_idx = match_ts_on_df(data_map["tinker_s0"], "12VHPWR0_timestamps", first_ts, last_ts)
    # print(first_idx)
    # print(last_idx)
    print("Tinker:")
    tinker_inter = interpolate_tinker(data_map["tinker_s0"], first_ts, last_ts)
    # tinker_data = compute_with_recipe_tinker(data_map["tinker_s0"], tinker_r, first_ts, last_ts)
    tinker_data, tinker_series = compute_with_recipe_tinker_2(tinker_inter, tinker_r, first_ts, last_ts)
    get_stats_tinker(tinker_data, num_frames)
    data_12VHPWR = interpolate_non_rtx(np.array(rtx_df["timestamps"][rtx_first_idx:rtx_last_idx]), tinker_series["12VHPWR"], tinker_inter["timestamps"])
    data_12VPEG = interpolate_non_rtx(np.array(rtx_df["timestamps"][rtx_first_idx:rtx_last_idx]), tinker_series["12VPEG"], tinker_inter["timestamps"])
    plot_on_rtx_ts(rtx_data["12VHPWR"][rtx_first_idx:rtx_last_idx]+rtx_data["12VPEG"][rtx_first_idx:rtx_last_idx], rtx_df["timestamps"][rtx_first_idx:rtx_last_idx], rtx_df["frame"][rtx_first_idx:rtx_last_idx], rtx_gpu_conv, data_12VHPWR+data_12VPEG, "tinker.html")
    
    # nvml
    print("NVML:")
    first_idx, last_idx = match_ts_on_df(data_map["nvml_s0"], "NVML_timestamps", first_ts, last_ts)
    ws = integrate(data_map["nvml_s0"]["NVML_timestamps"], data_map["nvml_s0"]["NVML_samples"], first_idx, last_idx)
    print(f"\tEnergy: {ws} Ws")
    print(f"\tEnergy per frame: {ws/num_frames} Ws")

    # interpolate nvml on rtx timescale
    data = interpolate_non_rtx(np.array(rtx_df["timestamps"][rtx_first_idx:rtx_last_idx]), data_map["nvml_s0"]["NVML_samples"][first_idx:last_idx], data_map["nvml_s0"]["NVML_timestamps"][first_idx:last_idx])
    plot_on_rtx_ts(rtx_data["12VHPWR"][rtx_first_idx:rtx_last_idx]+rtx_data["12VPEG"][rtx_first_idx:rtx_last_idx], rtx_df["timestamps"][rtx_first_idx:rtx_last_idx], rtx_df["frame"][rtx_first_idx:rtx_last_idx], rtx_gpu_conv, data, "nvml.html")

    # hmc
    print("HMC:")
    first_idx, last_idx = match_ts_on_df(data_map["HMC01"], "Timestamp", first_ts, last_ts)
    ws = integrate(data_map["HMC01"]["Timestamp"], data_map["HMC01"]["P[W]"], first_idx, last_idx)
    print(f"\tEnergy: {ws} Ws")
    print(f"\tEnergy per frame: {ws/num_frames} Ws")

    plot_timestamps(data_map["tinker_s0"], rtx_df["timestamps"], data_map["nvml_s0"]["NVML_timestamps"], data_map["HMC01"]["Timestamp"], rtx_df["trigger"], metadata[b"trigger_ts"][0], tinker_inter["timestamps"], "timeline.png")

    



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
