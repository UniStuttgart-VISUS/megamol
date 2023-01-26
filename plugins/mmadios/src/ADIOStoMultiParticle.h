/*
 * ADIOStoMultiParticle.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/SimpleSphericalParticles.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include <variant>

namespace megamol {
namespace adios {

class ADIOStoMultiParticle : public core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ADIOStoMultiParticle";
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
    ADIOStoMultiParticle();

    /** Dtor. */
    ~ADIOStoMultiParticle() override;

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
    core::CalleeSlot mpSlot;
    core::CallerSlot adiosSlot;

    std::vector<std::vector<unsigned char>> mix;

    size_t currentFrame = -1;

    geocalls::SimpleSphericalParticles::ColourDataType colType = geocalls::SimpleSphericalParticles::COLDATA_NONE;
    geocalls::SimpleSphericalParticles::VertexDataType vertType = geocalls::SimpleSphericalParticles::VERTDATA_NONE;
    geocalls::SimpleSphericalParticles::IDDataType idType = geocalls::SimpleSphericalParticles::IDDATA_NONE;

    size_t stride = 0;

    std::vector<uint64_t> plist_offset;
    std::vector<float> list_box;
    std::vector<uint64_t> plist_count;
};

} // end namespace adios
} // end namespace megamol
