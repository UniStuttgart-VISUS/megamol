/*
 * ADIOSFlexConvert.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/moldyn/SimpleSphericalParticles.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace adios {
    
class ADIOSFlexConvert : public core::Module {
    
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "ADIOSFlexConvert"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "Converts ADIOS-based IO into MegaMol's MultiParticleDataCall."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { return true; }

    /** Ctor. */
    ADIOSFlexConvert();

    /** Dtor. */
    virtual ~ADIOSFlexConvert();

    bool create();

protected:
    void release();

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
    bool paramChanged(core::param::ParamSlot &p);

private:

    core::CalleeSlot mpSlot;
    core::CallerSlot adiosSlot;

    core::param::ParamSlot flexPosSlot;
    core::param::ParamSlot flexColSlot;
    core::param::ParamSlot flexBoxSlot;
    core::param::ParamSlot flexXSlot;
    core::param::ParamSlot flexYSlot;
    core::param::ParamSlot flexZSlot;


    std::vector<float> mix;

    size_t currentFrame = -1;

    core::moldyn::SimpleSphericalParticles::ColourDataType colType = core::moldyn::SimpleSphericalParticles::COLDATA_NONE;
    core::moldyn::SimpleSphericalParticles::VertexDataType vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_NONE;
    core::moldyn::SimpleSphericalParticles::IDDataType idType = core::moldyn::SimpleSphericalParticles::IDDATA_NONE;

    size_t stride = 0;

    bool _trigger_recalc = false;

};

} // end namespace adios
} // end namespace megamol