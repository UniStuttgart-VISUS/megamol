/*
 * FlagCollection_GL.h
 *
 * Copyright (C) 2019-2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/BufferObject.hpp"

#include "FlagStorage.h"

namespace megamol {
namespace core {

    class FlagCollection_GL {
    public:
        std::shared_ptr<glowl::BufferObject> flags;

        void validateFlagCount(uint32_t num) {
            if (flags->getByteSize() / sizeof(uint32_t) < num) {
                std::vector<uint32_t> temp_data(num, FlagStorage::ENABLED);
                std::shared_ptr<glowl::BufferObject> temp_buffer =
                    std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, temp_data, GL_DYNAMIC_DRAW);
                glowl::BufferObject::copy(flags.get(), temp_buffer.get(), 0, 0, flags->getByteSize());
                flags = temp_buffer;
            }
        }
    };

    class FlagCollection_CPU {
    public:
        std::shared_ptr<FlagStorage::FlagVectorType> flags;

        void validateFlagCount(uint32_t num) {
            if (flags->size() < num) {
                flags->resize(num);
                std::fill(flags->begin(), flags->end(), FlagStorage::ENABLED);
            }
        }
    };

} // namespace core
} // namespace megamol
