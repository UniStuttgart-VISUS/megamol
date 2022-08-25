/*
 * OSPRaySphericalVolume.h
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/AbstractOSPRayStructure.h"

namespace megamol {
namespace ospray {

class OSPRaySphericalVolume : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "OSPRaySphericalVolume";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Creator for OSPRay spherical volumes.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Dtor. */
    virtual ~OSPRaySphericalVolume(void);

    /** Ctor. */
    OSPRaySphericalVolume(void);

protected:
    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call& call);
    virtual bool getExtends(core::Call& call);

    bool paramChanged(core::param::ParamSlot& p);

    bool _trigger_recalc = false;

    /** The call for data */
    core::CallerSlot getDataSlot;

    /** The call for Transfer function */
    core::CallerSlot getTFSlot;

    megamol::core::param::ParamSlot clippingBoxLower;
    megamol::core::param::ParamSlot clippingBoxUpper;
    megamol::core::param::ParamSlot clippingBoxActive;
    megamol::core::param::ParamSlot showBoundingBox;

    megamol::core::param::ParamSlot volumeDataStringSlot;

    megamol::core::param::ParamSlot repType;
    megamol::core::param::ParamSlot IsoValue;

    std::vector<float> _resorted_data;
};

} // namespace ospray
} // namespace megamol
