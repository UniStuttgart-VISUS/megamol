/*
 * ADIOSFlexVolume.h
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/VolumetricDataCall.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace adios {

class ADIOSFlexVolume : public core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ADIOSFlexVolume";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Converts ADIOS-based IO into MegaMol's VolumetricDataCall.";
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
    ADIOSFlexVolume();

    /** Dtor. */
    virtual ~ADIOSFlexVolume();

    bool create();

protected:
    void release();
    bool inquireDataVariables(CallADIOSData* cad);
    bool inquireMetaDataVariables(CallADIOSData* cad);
    bool assertData(geocalls::VolumetricDataCall* vdc, CallADIOSData* cad);

    bool onGetData(core::Call& call);
    bool onGetExtents(core::Call& call);
    //bool onGetMetadata(core::Call& call);
    bool onStartAsync(core::Call& call);
    bool onStopAsync(core::Call& call);
    bool onTryGetData(core::Call& call);

private:
    core::CalleeSlot volumeSlot;
    core::CallerSlot adiosSlot;

    core::param::ParamSlot flexVelocitySlot;
    core::param::ParamSlot flexBoxSlot;
    core::param::ParamSlot memoryLayoutSlot;

    geocalls::VolumetricDataCall::Metadata metadata;
    std::vector<double> mins, maxes;
    std::vector<float> VMAG;
    vislib::math::Cuboid<float> bbox;

    size_t currentFrame = -1;
};

} // end namespace adios
} // end namespace megamol
