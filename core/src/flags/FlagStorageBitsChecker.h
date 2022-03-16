/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <tbb/tbb.h>

#include "mmcore/flags/FlagStorageTypes.h"
#include "vislib/assert.h"

namespace megamol::core {

class BitsChecker {
public:
    BitsChecker(const std::shared_ptr<FlagStorageTypes::flag_vector_type>& flags) : flags(flags) {}

    BitsChecker(BitsChecker& b, tbb::split) : flags(b.flags) {
        // what to do? nothing probably
    }

    // when done, copies the result into the out parameters, so they can be identical to one or other.
    void join_ranges(const FlagStorageTypes::index_vector& one_starts, const FlagStorageTypes::index_vector& one_ends,
        const FlagStorageTypes::index_vector& other_starts, const FlagStorageTypes::index_vector& other_ends,
        FlagStorageTypes::index_vector& out_starts, FlagStorageTypes::index_vector& out_ends);

    void join(const BitsChecker& other);

    void operator()(const tbb::blocked_range<int32_t>& r);

    void local_terminate_bit(const tbb::blocked_range<int32_t>& r, FlagStorageTypes::index_vector& bit_ends,
        FlagStorageTypes::index_type curr_bit_start);

    static void check_bits(FlagStorageTypes::flag_bits flag_bit, FlagStorageTypes::index_vector& bit_starts,
        FlagStorageTypes::index_vector& bit_ends, FlagStorageTypes::index_type& curr_bit_start,
        FlagStorageTypes::index_type x, const std::shared_ptr<FlagStorageTypes::flag_vector_type>& flags);

    static void terminate_bit(const std::shared_ptr<FlagStorageTypes::flag_vector_type>& cdata,
        FlagStorageTypes::index_vector& bit_ends, FlagStorageTypes::index_type curr_bit_start);

    FlagStorageTypes::index_vector enabled_starts, enabled_ends;
    FlagStorageTypes::index_vector filtered_starts, filtered_ends;
    FlagStorageTypes::index_vector selected_starts, selected_ends;
    const std::shared_ptr<FlagStorageTypes::flag_vector_type>& flags;
};

} // namespace megamol::core
