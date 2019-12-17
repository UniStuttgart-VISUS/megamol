/*
 * osputils.h
 *
 * Copyright (C) 2019 by MegaMol Team. Alle Rechte vorbehalten.
 */
// Make crappy clang-format f*** off:
// clang-format off

#pragma once

#include <ospray.h>

#include "mmcore/moldyn/SimpleSphericalParticles.h"


namespace megamol {
namespace ospray {

    /**
     * Converts MegaMol's direction type into the corresponding OSPRay type.
     */
    extern constexpr OSPDataType ToOspray(
        const core::moldyn::SimpleSphericalParticles::DirDataType type);

    /**
     * Converts MegaMol's vertex type into the corresponding OSPRay type.
     */
    extern constexpr OSPDataType ToOspray(
        const core::moldyn::SimpleSphericalParticles::VertexDataType type);

} /* end namespace ospray */
} /* end namespace megamol */
