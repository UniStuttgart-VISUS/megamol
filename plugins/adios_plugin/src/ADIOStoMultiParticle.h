/*
 * ADIOStoMultiParticle.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/moldyn/SimpleSphericalParticles.h"

namespace megamol {
namespace adios {
    
class ADIOStoMultiParticle : public core::Module {
    
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ADIOStoMultiParticle"; }

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
    ADIOStoMultiParticle(void);

    /** Dtor. */
    virtual ~ADIOStoMultiParticle(void);

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

    std::vector<std::vector<char>> mix;

    size_t currentFrame = -1;

    core::moldyn::SimpleSphericalParticles::ColourDataType colType = core::moldyn::SimpleSphericalParticles::COLDATA_NONE;
    core::moldyn::SimpleSphericalParticles::VertexDataType vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_NONE;
    core::moldyn::SimpleSphericalParticles::IDDataType idType = core::moldyn::SimpleSphericalParticles::IDDATA_NONE;

    size_t stride = 0;

    std::vector<unsigned long long int> plist_offset;

};

} // end namespace adios
} // end namespace megamol