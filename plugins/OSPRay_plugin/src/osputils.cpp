/*
 * osputils.cpp
 *
 * Copyright (C) 2019 by MegaMol Team. Alle Rechte vorbehalten.
 */
// Make crappy clang-format f*** off:
// clang-format off

#include "stdafx.h"
#include "osputils.h"

#include "vislib/sys/Log.h"


/*
 * megamol::ospray::ToOspray
 */
constexpr OSPDataType megamol::ospray::ToOspray(
        const core::moldyn::SimpleSphericalParticles::DirDataType type) {
    using core::moldyn::SimpleSphericalParticles;
    using vislib::sys::Log;

    switch (type) {
        case SimpleSphericalParticles::DirDataType::DIRDATA_FLOAT_XYZ:
            return OSPDataType::OSP_FLOAT3;

        default:
            Log::DefaultLog.WriteWarn(_T("Directional data type %d cannot ")
                _T("be mapped to an OSPRay type."), type);
            return OSPDataType::OSP_UNKNOWN;
    }
}


/*
 * megamol::ospray::ToOspray
 */
constexpr OSPDataType megamol::ospray::ToOspray(
        const core::moldyn::SimpleSphericalParticles::VertexDataType type) {
    using core::moldyn::SimpleSphericalParticles;
    using vislib::sys::Log;

    switch (type) {
        case SimpleSphericalParticles::VertexDataType::VERTDATA_FLOAT_XYZ:
            return OSPDataType::OSP_FLOAT3;

        case SimpleSphericalParticles::VertexDataType::VERTDATA_FLOAT_XYZR:
            return OSPDataType::OSP_FLOAT4;

        default:
            Log::DefaultLog.WriteWarn(_T("Vertex data type %d cannot ")
                _T("be mapped to an OSPRay type."), type);
            return OSPDataType::OSP_UNKNOWN;
    }
}
