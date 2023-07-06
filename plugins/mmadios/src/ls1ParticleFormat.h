/*
 * ls1ParticleFormat.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/SimpleSphericalParticles.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <variant>

namespace megamol::adios {

class ls1ParticleFormat : public core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ls1ParticleFormat";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Converts ADIOS-based IO into MegaMols MultiParticleDataCall.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    ls1ParticleFormat();

    /** Dtor. */
    ~ls1ParticleFormat() override;

    bool create() override;

protected:
    void release() override;

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& caller);

private:
    std::vector<std::string> splitElementString(std::string&);

    core::CalleeSlot mpSlot;
    core::CallerSlot adiosSlot;
    core::CallerSlot transferfunctionSlot;

    core::param::ParamSlot representationSlot;
    core::param::ParamSlot forceFloatSlot;

    std::vector<std::vector<float>> mix;
    std::vector<std::vector<float>> dirs;
    std::vector<std::vector<uint64_t>> ids;
    std::vector<uint64_t> plist_count;
    std::vector<float> bbox;
    int num_plists;
    std::vector<std::array<float, 4>> list_colors;
    std::vector<float> list_radii;

    size_t currentFrame = -1;
    size_t version = 0;
    size_t datahash;

    geocalls::SimpleSphericalParticles::ColourDataType colType = geocalls::SimpleSphericalParticles::COLDATA_NONE;
    geocalls::SimpleSphericalParticles::VertexDataType vertType = geocalls::SimpleSphericalParticles::VERTDATA_NONE;
    geocalls::SimpleSphericalParticles::DirDataType dirType = geocalls::SimpleSphericalParticles::DIRDATA_NONE;
    geocalls::SimpleSphericalParticles::IDDataType idType = geocalls::SimpleSphericalParticles::IDDATA_NONE;

    size_t stride = 0;

    bool representationChanged(core::param::ParamSlot& p);
    bool representationDirty = false;

    template<typename T, typename K>
    std::vector<T> calcAtomPos(T com_x, T com_y, T com_z, K a_x, K a_y, K a_z, K qw, K qx, K qy, K qz) {

        std::vector<T> result(3);

        // TODO: apply quaternion

        // https://github.com/ls1mardyn/ls1-mardyn
        // ls1-mardyn src/molecules/Quaternion.cpp, line 43 ...
        auto const ww = qw * qw;
        auto const xx = qx * qx;
        auto const yy = qy * qy;
        auto const zz = qz * qz;
        auto const wx = qw * qx;
        auto const wy = qw * qy;
        auto const wz = qw * qz;
        auto const xy = qx * qy;
        auto const xz = qx * qz;
        auto const yz = qy * qz;
        //          1-2*(yy+zz)
        result[0] =
            (ww + xx - yy - zz) * a_x + static_cast<T>(2.) * (xy - wz) * a_y + static_cast<T>(2.) * (wy + xz) * a_z;
        //                            1-2*(xx+zz)
        result[1] =
            static_cast<T>(2.) * (wz + xy) * a_x + (ww - xx + yy - zz) * a_y + static_cast<T>(2.) * (yz - wx) * a_z;
        //                                              1-2*(xx+yy)
        result[2] =
            static_cast<T>(2.) * (xz - wy) * a_x + static_cast<T>(2.) * (wx + yz) * a_y + (ww - xx - yy + zz) * a_z;

        result[0] += com_x;
        result[1] += com_y;
        result[2] += com_z;

        return result;
    }
};

} // namespace megamol::adios
