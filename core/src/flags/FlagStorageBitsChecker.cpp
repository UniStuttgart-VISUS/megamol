/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "FlagStorageBitsChecker.h"

#include "mmcore/flags/FlagStorage.h"

using namespace megamol;
using namespace megamol::core;

void BitsChecker::operator()(const tbb::blocked_range<int32_t>& r) {
    FlagStorageTypes::index_type curr_enabled_start = -1, curr_filtered_start = -1, curr_selected_start = -1;

    for (int32_t i = r.begin(); i != r.end(); ++i) {
        check_bits(FlagStorageTypes::flag_bits::ENABLED, enabled_starts, enabled_ends, curr_enabled_start, i, flags);
        check_bits(
            FlagStorageTypes::flag_bits::FILTERED, filtered_starts, filtered_ends, curr_filtered_start, i, flags);
        check_bits(
            FlagStorageTypes::flag_bits::SELECTED, selected_starts, selected_ends, curr_selected_start, i, flags);
    }

    local_terminate_bit(r, enabled_ends, curr_enabled_start);
    local_terminate_bit(r, filtered_ends, curr_filtered_start);
    local_terminate_bit(r, selected_ends, curr_selected_start);

    ASSERT(enabled_starts.size() == enabled_ends.size());
    ASSERT(filtered_starts.size() == filtered_ends.size());
    ASSERT(selected_starts.size() == selected_ends.size());
}

void BitsChecker::join_ranges(const FlagStorageTypes::index_vector& one_starts,
    const FlagStorageTypes::index_vector& one_ends, const FlagStorageTypes::index_vector& other_starts,
    const FlagStorageTypes::index_vector& other_ends, FlagStorageTypes::index_vector& out_starts,
    FlagStorageTypes::index_vector& out_ends) {
    FlagStorageTypes::index_type my_pos = 0, other_pos = 0;
    FlagStorageTypes::index_vector result_starts, result_ends;

    if (one_starts.empty()) {
        out_starts = other_starts;
        out_ends = other_ends;
        return;
    }
    if (other_starts.empty()) {
        out_starts = one_starts;
        out_ends = one_ends;
        return;
    }
    while (my_pos < one_starts.size() && other_pos < other_starts.size()) {
        const auto mystart = one_starts[my_pos];
        const auto otherstart = other_starts[other_pos];
        const auto myend = one_ends[my_pos];
        const auto otherend = other_ends[other_pos];

        if (mystart < otherstart) {
            if (myend < otherstart - 1) {
                result_starts.push_back(mystart);
                result_ends.push_back(myend);
                my_pos++;
            } else {
                result_starts.push_back(mystart);
                result_ends.push_back(otherend);
                my_pos++;
                other_pos++;
            }
        } else {
            ASSERT(mystart != otherstart);
            if (otherend < mystart - 1) {
                result_starts.push_back(otherstart);
                result_ends.push_back(otherend);
                other_pos++;
            } else {
                result_starts.push_back(otherstart);
                result_ends.push_back(myend);
                my_pos++;
                other_pos++;
            }
        }
    }
    // push everything after *_pos in one go
    const auto total_elems = one_starts.size() + other_starts.size();
    if (my_pos < one_starts.size()) {
        result_starts.reserve(total_elems);
        result_starts.insert(result_starts.end(), one_starts.begin() + my_pos, one_starts.end());
        result_ends.reserve(total_elems);
        result_ends.insert(result_ends.end(), one_ends.begin() + my_pos, one_ends.end());
    }
    if (other_pos < other_starts.size()) {
        result_starts.reserve(total_elems);
        result_starts.insert(result_starts.end(), other_starts.begin() + other_pos, other_starts.end());
        result_ends.reserve(total_elems);
        result_ends.insert(result_ends.end(), other_ends.begin() + other_pos, other_ends.end());
    }
    out_starts = result_starts;
    out_ends = result_ends;
}

void BitsChecker::join(const BitsChecker& other) {

    join_ranges(this->enabled_starts, this->enabled_ends, other.enabled_starts, other.enabled_ends,
        this->enabled_starts, this->enabled_ends);
    join_ranges(this->filtered_starts, this->filtered_ends, other.filtered_starts, other.filtered_ends,
        this->filtered_starts, this->filtered_ends);
    join_ranges(this->selected_starts, this->selected_ends, other.selected_starts, other.selected_ends,
        this->selected_starts, this->selected_ends);
}

void BitsChecker::local_terminate_bit(const tbb::blocked_range<int32_t>& r, FlagStorageTypes::index_vector& bit_ends,
    FlagStorageTypes::index_type curr_bit_start) {
    if (curr_bit_start > -1) {
        bit_ends.push_back(r.end() - 1);
    }
}

void BitsChecker::check_bits(FlagStorageTypes::flag_bits flag_bit, FlagStorageTypes::index_vector& bit_starts,
    FlagStorageTypes::index_vector& bit_ends, FlagStorageTypes::index_type& curr_bit_start,
    FlagStorageTypes::index_type x, const std::shared_ptr<FlagStorageTypes::flag_vector_type>& flags) {
    auto& f = (*flags)[x];
    auto flag_val = FlagStorageTypes::to_integral(flag_bit);
    if ((f & flag_val) > 0) {
        if (curr_bit_start == -1) {
            curr_bit_start = x;
            bit_starts.push_back(x);
        }
    } else {
        if (curr_bit_start > -1) {
            bit_ends.push_back(x - 1);
            curr_bit_start = -1;
        }
    }
}

void BitsChecker::terminate_bit(const std::shared_ptr<FlagStorageTypes::flag_vector_type>& cdata,
    FlagStorageTypes::index_vector& bit_ends, FlagStorageTypes::index_type curr_bit_start) {
    if (curr_bit_start > -1) {
        bit_ends.push_back(cdata->size() - 1);
    }
}
