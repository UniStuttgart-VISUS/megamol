/*
 * FlagCollection_GL.h
 *
 * Copyright (C) 2019-2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "FlagStorage.h"

namespace megamol {
namespace core {

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
