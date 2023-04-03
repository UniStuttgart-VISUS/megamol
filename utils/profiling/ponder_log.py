import argparse
import pandas
import os.path
import pathlib
from joblib import Parallel, delayed, cpu_count

parser = argparse.ArgumentParser(usage="%(prog)s <FILE>", description="check profiling log for inconsistencies")
parser.add_argument('files', nargs="*")
parser.add_argument('-p', action='store_true', help="parallelize frame analysis", dest='parallel')
parser.add_argument('-v', action='store_true', help="show verbose output", dest='verbose')
parser.add_argument('-w', action='store_true', help="write result to file", dest='write_out')
parser.add_argument('-f', action='store_true', help="overwrite file if existing", dest='overwrite_output')
parser.add_argument('-s', type=int, help="override start frame", dest='start_frame')
parser.add_argument('-e', type=int, help="override end frame", dest='end_frame')
parser.add_argument('-a', type=str, help="filter by API (CPU or OpenGL)", dest='api', default="CPU")
args = parser.parse_args()

df: pandas.DataFrame
self_time_col: int

class Call_Info:
    def __init__(self, line, duration):
        self.line = line
        self.duration = duration

def extract_call_pieces(parent_string):
    items = parent_string.split("->")
    left = items[0].rsplit("::", 1)
    right = items[1].rsplit("::", 1)
    source_mod = left[0]
    source_slot = left[1]
    dest_mod = right[0]
    dest_slot = right[1]
    return source_mod, source_slot, dest_mod, dest_slot

def verbose_print(*stuff, **what):
    if args.verbose:
        print(*stuff, **what)

def show_stack_frame(df, start, end, descr):
    verbose_print(f"{descr} start at {start}:\n{df.iloc[start]}")
    verbose_print(f"{descr} ends at {end}:\n{df.iloc[end]}")

def process_stack(dataframe, stack_start, stack_end):
    verbose_print(f"process_stack {stack_start} - {stack_end}")
    row = dataframe.iloc[stack_start]
    dur = row["duration (ns)"]
    #dur = row["total_time"]
    parent = row["parent"]
    callback = row["name"]
    source_mod, source_slot, dest_mod, dest_slot = extract_call_pieces(parent)

    verbose_print(f"I am {parent} : {callback} and cost {dur}")
    valid_end = stack_end
    for row_it in range(stack_start + 1, stack_end + 1):
        r2 = dataframe.iloc[row_it]
        p2 = r2["parent"]
        c2 = r2["name"]
        d2 = r2["duration (ns)"]
        #d2 = r2["total_time"]
        sm, ss, dm, ds = extract_call_pieces(p2)
        if dm == dest_mod:
            # we got called again ourselves, stack was closed
            verbose_print(f"{parent} called again: {p2}, ", end='')
            valid_end = row_it - 1
            break
        if sm == dest_mod:
            # we called stuff
            dur = dur - d2
            verbose_print(f"I called {p2} : {c2}, subtracting, remaining cost = {dur}")
    verbose_print("stopping.\n")
    assert dur >= 0
    #show_stack_frame(dataframe, stack_start + 1, valid_end, "children")
    #print(f"name {row.name} vs. stack_start {stack_start}")
    df.iat[row.name, self_time_col] = dur
    # TODO print in source format?
    #verbose_print(row.to_csv(index=False, header=False, line_terminator=''))
    # recurse children
    if (stack_start + 1 <= valid_end):
        verbose_print("recurse children")
        process_stack(dataframe, stack_start + 1, valid_end)

    # process rest at same level
    if (valid_end + 1 <= stack_end):
        verbose_print("rest of level")
        process_stack(dataframe, valid_end + 1, stack_end)

def analyze_frame(curr_frame):
    print(f"Analyzing Frame {curr_frame} for api {args.api}")
    dataframe = df.loc[(df["frame"] == curr_frame) & (df["type"] == "Call") & (df["api"] == args.api)]

    stack_start = 0
    stack_end = len(dataframe.index) - 1

    #show_stack_frame(curr_df, stack_start, stack_end, "stack")
    process_stack(dataframe, stack_start, stack_end)

for file in args.files:
    df = pandas.read_csv(file, header=0, sep=';', quotechar="'")

    df["self cost (ns)"] = ""
    self_time_col = df.columns.get_loc("self cost (ns)")

    print("found columns:")
    for col in df.columns:
        print(col, end=";")
    print()

    start_frame = df["frame"].min()
    end_frame = df["frame"].max()

    if args.start_frame:
        start_frame = args.start_frame
    if args.end_frame:
        end_frame = args.end_frame

    # we don't need the setup frame
    start_frame = max(0, start_frame)
    print(f"Analyzing frames {start_frame}-{end_frame}")

    if (args.parallel):
        Parallel(n_jobs=cpu_count(),require='sharedmem')(delayed(analyze_frame)(i) for i in range(start_frame, end_frame + 1))
    else:
        for curr_frame in range(start_frame, end_frame + 1):
            analyze_frame(curr_frame)

    if args.write_out:
        outfile = pathlib.Path(file).absolute()
        outfile = os.path.splitext(outfile)[0] + "_self.csv"
        if not os.path.exists(outfile) or args.overwrite_output:
            print(f"writing {outfile}...")
            df.to_csv(outfile, index=False, sep=';', quotechar="'")
        else:
            print(f"warning: not overwriting {outfile}!")




