#version 430
//#extension GL_GOOGLE_include_directive: require

#include "core/bitflags.inc.glsl"

// Compute coherent runs of bits in a flag storage to compact data for, e.g., serialization.
// Invoke once per flag, making sure that the FIRST invocation rebuilds the onoff buffer (update_onoff uniform).
// The onoff values can be kept around for the other flags.
// Precondition: length of EVERYTHING rounded up to multiple of k except for workgroup data (see below).
layout(std430, binding = 0) buffer Flags
{
    coherent uint flags[];
};
// carries the bits where flags turn on or off (see consts below)
layout(std430, binding = 1) buffer OnOffBits
{
    coherent uint on_off[];
};
// prefix sum on where bits turn on only
layout(std430, binding = 2) buffer PrefixSumOn
{
    coherent uint prefix_sum_on[];
};
// compacted lists of start-end pairs of respective bit runs
// results_starts holds aggregate per thread
// these need to be re-bound per invocation to hold one result per flag.
// https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
layout(std430, binding = 3) buffer ResultStarts
{
    coherent uint result_starts[];
};
layout(std430, binding = 4) buffer ResultEnds
{
    coherent uint result_ends[];
};

// workgroup data, length is number of workgroups
layout(std430, binding = 5) buffer WorkgroupState
{
    coherent uint workgroup_state[];
};
layout(std430, binding = 6) buffer WorkgroupAggregate
{
    coherent uint workgroup_aggregate[];
};
layout(std430, binding = 7) buffer WorkgroupInclusivePrefix
{
    coherent uint workgroup_inclusive_prefix[];
};

layout(local_size_x = 16) in;
uniform int current_flag;
uniform bool update_onoff;
uniform uint num_flags;

const uint FLAG_ENABLED_ON      = FLAG_ENABLED;
const uint FLAG_FILTERED_ON     = FLAG_FILTERED;
const uint FLAG_SELECTED_ON     = FLAG_SELECTED;
const uint FLAG_SOFTSELECTED_ON = FLAG_SOFTSELECTED;
const uint FLAG_MASK_ALLON = FLAG_ENABLED_ON + FLAG_FILTERED_ON + FLAG_SELECTED_ON + FLAG_SOFTSELECTED_ON;

const uint FLAG_ENABLED_OFF      = FLAG_ENABLED_ON << 16;
const uint FLAG_FILTERED_OFF     = FLAG_FILTERED << 16;
const uint FLAG_SELECTED_OFF     = FLAG_SELECTED << 16;
const uint FLAG_SOFTSELECTED_OFF = FLAG_SOFTSELECTED << 16;
const uint FLAG_MASK_ALLOFF = FLAG_ENABLED_OFF + FLAG_FILTERED_OFF + FLAG_SELECTED_OFF + FLAG_SOFTSELECTED_OFF;

const uint PREFIX_STATE_INVALID = 1;
const uint PREFIX_STATE_AGGREGATE = 1 << 1;
const uint PREFIX_STATE_DONE = 1 << 2;

const uint k = gl_WorkGroupSize.x;
uint block_size = flags.length() / k;

void on_to_aggregate(uint idx, uint flag) {
    result_starts[idx] = (on_off[idx] & flag) > 0 ? 1 : 0;
}

// add values from indices into onoff_idx_a
void add_on_to_aggregate(uint onoff_idx_a, uint onoff_idx_b, uint flag) {
    uint a = (on_off[onoff_idx_a] & flag) > 0 ? 1 : 0;
    uint b = (on_off[onoff_idx_b] & flag) > 0 ? 1 : 0;
    result_starts[onoff_idx_a] = a + b;
}

// add values from indices into pfx_idx_a
void add_aggregates(uint pfx_idx_a, uint pfx_idx_b, uint flag) {
    uint a = (on_off[pfx_idx_a] & flag) > 0 ? 1 : 0;
    uint b = (on_off[pfx_idx_b] & flag) > 0 ? 1 : 0;
    result_starts[pfx_idx_a] = a + b;
}

void build_onoff(uint idx) {
    workgroup_state[gl_WorkGroupID.x] = PREFIX_STATE_INVALID; // we have not even started
    uint val = 0;
    uint myflags = flags[idx] & FLAG_MASK_ALLON; // there are only "on" in flags, but just in case
    if (idx == 0) {
        // everything that is on switches on here since we are the first element
        //val = myflags;
        // paranoid version if constants values change
        val += ((myflags & FLAG_ENABLED) > 0) ? FLAG_ENABLED_ON : 0;
        val += ((myflags & FLAG_FILTERED) > 0) ? FLAG_FILTERED_ON : 0;
        val += ((myflags & FLAG_SELECTED) > 0) ? FLAG_SELECTED_ON : 0;
        val += ((myflags & FLAG_SOFTSELECTED) > 0) ? FLAG_SOFTSELECTED_ON : 0;
    } else {
        uint leftflags = flags[idx - 1] & FLAG_MASK_ALLON; // just in case
        // check where a 0 in left becomes a 1 in my
        val += ((leftflags & FLAG_ENABLED) == 0) && ((myflags & FLAG_ENABLED) > 0) ? FLAG_ENABLED_ON : 0;
        val += ((leftflags & FLAG_FILTERED) == 0) && ((myflags & FLAG_FILTERED) > 0) ? FLAG_FILTERED_ON : 0;
        val += ((leftflags & FLAG_SELECTED) == 0) && ((myflags & FLAG_SELECTED) > 0) ? FLAG_SELECTED_ON : 0;
        val += ((leftflags & FLAG_SOFTSELECTED) == 0) && ((myflags & FLAG_SOFTSELECTED) > 0) ? FLAG_SOFTSELECTED_ON : 0;
    }
    if (idx == flags.length() - 1) {
        // everything that is still on switches off as we "leave" the array
        //val += myflags << 16;
        // paranoid version if constants values change
        val += ((myflags & FLAG_ENABLED) > 0) ? FLAG_ENABLED_OFF : 0;
        val += ((myflags & FLAG_FILTERED) > 0) ? FLAG_FILTERED_OFF : 0;
        val += ((myflags & FLAG_SELECTED) > 0) ? FLAG_SELECTED_OFF : 0;
        val += ((myflags & FLAG_SOFTSELECTED) > 0) ? FLAG_SOFTSELECTED_OFF : 0;
    } else {
        uint rightflags = flags[idx + 1] & FLAG_MASK_ALLON; // just in case
        // check where a 1 in my becomes a 0 in right
        val += ((myflags & FLAG_ENABLED) > 0) && ((rightflags & FLAG_ENABLED) == 0) ? FLAG_ENABLED_OFF : 0;
        val += ((myflags & FLAG_FILTERED) > 0) && ((rightflags & FLAG_FILTERED) == 0) ? FLAG_FILTERED_OFF : 0;
        val += ((myflags & FLAG_SELECTED) > 0) && ((rightflags & FLAG_SELECTED) == 0) ? FLAG_SELECTED_OFF : 0;
        val += ((myflags & FLAG_SOFTSELECTED) > 0) && ((rightflags & FLAG_SOFTSELECTED) == 0) ? FLAG_SOFTSELECTED_OFF : 0;
    }
    on_off[idx] = val;
}

// idx is linearized thread id
void radix_sum_on(uint idx, uint flag) {
    uint my_block_start = idx / block_size;
    uint my_local_idx = idx % block_size;
    uint my_workgroup = gl_WorkGroupID.x;

    uint offset = 1;
    if (my_local_idx < offset) {
        on_to_aggregate(idx, flag);
    } else {
        uint left_neighbor = my_local_idx - offset + my_block_start;
        add_on_to_aggregate(idx, left_neighbor, flag);
    }
    groupMemoryBarrier();
    for (uint offset = 2; offset < k; offset <<= 1) {
        if (my_local_idx < offset) {
            // already done
        } else {
            // reach farther
            uint left_neighbor = my_local_idx - offset + my_block_start;
            add_aggregates(idx, left_neighbor, flag);
        }
        groupMemoryBarrier();
    }
    // result_starts holds the thread aggregates
    if (my_local_idx == block_size - 1) {
        //the last idx has the local workgroup aggregate, so we do the serial work here
        workgroup_aggregate[my_workgroup] = result_starts[idx];
        workgroup_state[my_workgroup] = PREFIX_STATE_AGGREGATE;
        if (my_workgroup == 0) {
            // the first workgroup is completely done per defintion
            workgroup_inclusive_prefix[my_workgroup] = workgroup_aggregate[my_workgroup];
            workgroup_state[my_workgroup] = PREFIX_STATE_DONE;
        } else {
            uint exclusive_prefix = 0;
            for (uint prev = my_workgroup - 1; prev >= 0; --prev) {
                barrier();
                if (workgroup_state[prev] == PREFIX_STATE_AGGREGATE) {
                    exclusive_prefix += workgroup_aggregate[prev];
                } else if (workgroup_state[prev] == PREFIX_STATE_DONE) {
                    exclusive_prefix += workgroup_inclusive_prefix[prev];
                    break;
                }
            }
            workgroup_inclusive_prefix[my_workgroup] = exclusive_prefix + workgroup_aggregate[my_workgroup];
            workgroup_state[my_workgroup] = PREFIX_STATE_DONE;
        }
    }
    // now add the inclusive prefix of the predecessor to all thread aggregates, done.
    if (my_workgroup != 0) {
        uint prev = my_workgroup - 1;
        prefix_sum_on[idx] = result_starts[idx] + workgroup_inclusive_prefix[prev];
    }
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    // TODO not sure whether this is safe
    if (idx >= num_flags) {
        return;
    }
    if (update_onoff) {
        build_onoff(idx);
    }
    barrier();
    radix_sum_on(idx, current_flag);
    barrier();
    uint run_idx = prefix_sum_on[idx] - 1;
    // TODO this could be fragile if the computation of on and off flags is changed.
    if ((on_off[idx] & current_flag) > 0) {
        result_starts[run_idx] = idx;
    }
    if ((on_off[idx] & (current_flag << 16)) > 0) {
        result_ends[run_idx] = idx;
    }
}
