/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/flags/FlagStorageTypes.h"

namespace megamol::core_gl {

class FlagCollection_GL {
public:
    std::shared_ptr<glowl::BufferObject> flags;

    void validateFlagCount(core::FlagStorageTypes::index_type num) {
        if (flags->getByteSize() / sizeof(core::FlagStorageTypes::flag_item_type) < num) {
            core::FlagStorageTypes::flag_vector_type temp_data(
                num, core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED));
            std::shared_ptr<glowl::BufferObject> temp_buffer =
                std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, temp_data, GL_DYNAMIC_COPY);
            glowl::BufferObject::copy(flags.get(), temp_buffer.get(), 0, 0, flags->getByteSize());
            flags = temp_buffer;
        }
    }
};
} // namespace megamol::core_gl
