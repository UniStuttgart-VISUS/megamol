/*
 * OSPRayGlassMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayMaterial.h"

namespace megamol {
namespace ospray {

class OSPRayGlassMaterial : public AbstractOSPRayMaterial {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "OSPRayGlassMaterial";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Configuration module for an OSPRay glass material";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    OSPRayGlassMaterial(void);

    /** Dtor. */
    virtual ~OSPRayGlassMaterial(void);

private:
    // GLASS
    core::param::ParamSlot glassEtaInside;
    core::param::ParamSlot glassEtaOutside;
    core::param::ParamSlot glassAttenuationColorInside;
    core::param::ParamSlot glassAttenuationColorOutside;
    core::param::ParamSlot glassAttenuationDistance;

    virtual bool InterfaceIsDirty();
    virtual void readParams();
};


} // namespace ospray
} // namespace megamol
