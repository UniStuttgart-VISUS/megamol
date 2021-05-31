/*
 * ls1ParticleFormat.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/moldyn/SimpleSphericalParticles.h"
#include "mmcore/param/ParamSlot.h"
#include <variant>

namespace megamol {
namespace adios {
    
class ls1ParticleFormat : public core::Module {
    
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ls1ParticleFormat"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Converts ADIOS-based IO into MegaMols MultiParticleDataCall."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ls1ParticleFormat(void);

    /** Dtor. */
    virtual ~ls1ParticleFormat(void);

    bool create(void);

protected:
    void release(void);

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

    core::CalleeSlot mpSlot;
    core::CallerSlot adiosSlot;
    core::CallerSlot transferfunctionSlot;

    core::param::ParamSlot representationSlot;

    std::vector<std::vector<unsigned char>> mix;
    std::vector<uint64_t> plist_count;
    std::vector<float> bbox;
    int num_plists;
    std::vector<std::array<float,4>> list_colors;

    size_t currentFrame = -1;

    core::moldyn::SimpleSphericalParticles::ColourDataType colType = core::moldyn::SimpleSphericalParticles::COLDATA_NONE;
    core::moldyn::SimpleSphericalParticles::VertexDataType vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_NONE;
    core::moldyn::SimpleSphericalParticles::IDDataType idType = core::moldyn::SimpleSphericalParticles::IDDATA_NONE;

    size_t stride = 0;

    bool representationChanged(core::param::ParamSlot& p);
    bool representationDirty = false;

    template<typename T, typename K>
    std::vector<T> calcAtomPos(T com_x, T com_y, T com_z, K a_x, K a_y, K a_z, K qw, K qx, K qy, K qz) {
        
        std::vector<T> result(3);
        result[0] = a_x + com_x;
        result[1] = a_y + com_y;
        result[2] = a_z + com_z;

        // TODO: apply quaternion

        return result;
    }
};

} // end namespace adios
} // end namespace megamol
