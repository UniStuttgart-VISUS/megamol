import pandas

df = pandas.read_csv("pvp.txt", header=0, sep=';')

df["self cost (ns)"] = ""

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

def show_stack_frame(df, start, end, descr):
    print(f"{descr} start at {start}:\n{df.iloc[start]}")
    print(f"{descr} ends at {end}:\n{df.iloc[end]}")

def process_stack(stack_start, stack_end):
    print(f"process_stack {stack_start} - {stack_end}")
    row = dataframe.iloc[stack_start]
    dur = row["duration (ns)"]
    parent = row["parent"]
    callback = row["name"]
    source_mod, source_slot, dest_mod, dest_slot = extract_call_pieces(parent)

    print(f"I am {parent} : {callback} and cost {dur}")
    valid_end = stack_end
    for row_it in range(stack_start + 1, stack_end + 1):
        r2 = dataframe.iloc[row_it]
        p2 = r2["parent"]
        c2 = r2["name"]
        d2 = r2["duration (ns)"]
        sm, ss, dm, ds = extract_call_pieces(p2)
        if dm == dest_mod:
            # we got called again ourselves, stack was closed
            print(f"{parent} called again: {p2}, ", end='')
            valid_end = row_it - 1
            break
        if sm == dest_mod:
            # we called stuff
            dur = dur - d2
            print(f"I called {p2} : {c2}, subtracting, remaining cost = {dur}")
    print("stopping.\n")
    assert dur >= 0
    #show_stack_frame(dataframe, stack_start + 1, valid_end, "children")

    # TODO write the crap back without getting the copy of slice crap
    #row["self cost (ns)"] = dur
    # TODO print in source format
    #print(row.to_csv(index=False, header=False, line_terminator=''))
    # recurse children
    if (stack_start + 1 <= valid_end):
        print("recurse children")
        process_stack(stack_start + 1, valid_end)

    # process rest at same level
    if (valid_end + 1 <= stack_end):
        print("rest of level")
        process_stack(valid_end + 1, stack_end)



#for col in df.columns:
#    print(col)

start_frame = df["frame"].min()
end_frame = df["frame"].max()

# debug
#start_frame = 3
#end_frame = 10

# we don't need the setup frame
start_frame = max(0, start_frame)

for curr_frame in range(start_frame, end_frame):
    print(f"Analyzing Frame {curr_frame}")
    stack = {}
    dataframe = df.loc[(df["frame"] == curr_frame) & (df["type"] == "Call") & (df["api"] == "CPU")]

    stack_start = 0
    stack_end = len(dataframe.index) - 1

    #show_stack_frame(curr_df, stack_start, stack_end, "stack")
    process_stack(stack_start, stack_end)





