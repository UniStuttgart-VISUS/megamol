/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "FlagStorageTypes.h"

namespace megamol::core {

class FlagCollection_CPU {
public:
    std::shared_ptr<FlagStorageTypes::flag_vector_type> flags;

    void validateFlagCount(FlagStorageTypes::index_type num) {
        if (flags->size() < num) {
            flags->resize(num);
            std::fill(
                flags->begin(), flags->end(), FlagStorageTypes::to_integral(FlagStorageTypes::flag_bits::ENABLED));
        }
    }
};
} // namespace megamol::core
