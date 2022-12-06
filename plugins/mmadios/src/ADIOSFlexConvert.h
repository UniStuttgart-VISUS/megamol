/*
 * ADIOSFlexConvert.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/SimpleSphericalParticles.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
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
    static const char* ClassName() {
        return "ADIOSFlexConvert";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Converts ADIOS-based IO into MegaMol's MultiParticleDataCall.";
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

private:
    bool inquireDataVariables(CallADIOSData* cad);
    bool inquireMetaDataVariables(CallADIOSData* cad);

    core::CalleeSlot mpSlot;
    core::CallerSlot adiosSlot;

    core::param::ParamSlot flexPosSlot;
    core::param::ParamSlot flexColSlot;
    core::param::ParamSlot flexBoxSlot;
    core::param::ParamSlot flexXSlot;
    core::param::ParamSlot flexYSlot;
    core::param::ParamSlot flexZSlot;
    core::param::ParamSlot flexAlignedPosSlot;
    core::param::ParamSlot flexIDSlot;
    core::param::ParamSlot flexVXSlot;
    core::param::ParamSlot flexVYSlot;
    core::param::ParamSlot flexVZSlot;

    vislib::math::Cuboid<float> bbox;
    bool hasVel, hasID;

    std::vector<float> mix;

    size_t currentFrame = -1;

    geocalls::SimpleSphericalParticles::ColourDataType colType = geocalls::SimpleSphericalParticles::COLDATA_NONE;
    geocalls::SimpleSphericalParticles::VertexDataType vertType = geocalls::SimpleSphericalParticles::VERTDATA_NONE;
    geocalls::SimpleSphericalParticles::IDDataType idType = geocalls::SimpleSphericalParticles::IDDATA_NONE;
    geocalls::SimpleSphericalParticles::DirDataType dirType = geocalls::SimpleSphericalParticles::DIRDATA_NONE;

    size_t stride = 0;

    bool _trigger_recalc = false;
};

} // end namespace adios
} // end namespace megamol
