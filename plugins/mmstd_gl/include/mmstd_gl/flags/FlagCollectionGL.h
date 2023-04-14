/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "mmstd/flags/FlagStorageTypes.h"

namespace megamol::mmstd_gl {

class FlagCollection_GL {
public:
    std::shared_ptr<glowl::ImmutableBufferObject> flags;

    void validateFlagCount(core::FlagStorageTypes::index_type num) {
        constexpr auto defaultFlag = core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED);

        if (flags == nullptr || flags->getByteSize() / sizeof(core::FlagStorageTypes::flag_item_type) < num) {
            core::FlagStorageTypes::flag_vector_type temp_data(num, defaultFlag);
            auto temp_buffer = std::make_shared<glowl::ImmutableBufferObject>(
                temp_data, GL_DYNAMIC_STORAGE_BIT | GL_CLIENT_STORAGE_BIT);

            if (flags != nullptr) {
                glowl::ImmutableBufferObject::copy(*flags, *temp_buffer, 0, 0, flags->getByteSize());
            }

            flags = temp_buffer;
        }
    }
};
} // namespace megamol::mmstd_gl
