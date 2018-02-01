/*
* OSPRayHDRILight.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "AbstractOSPRayLight.h"

namespace megamol {
namespace ospray {

class OSPRayHDRILight : public AbstractOSPRayLight {
public:
    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "OSPRayHDRILight";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Configuration module for an OSPRay HDRI light source.";
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
    OSPRayHDRILight(void);

    /** Dtor. */
    virtual ~OSPRayHDRILight(void);

private:

    core::param::ParamSlot hdri_up;
    core::param::ParamSlot hdri_direction;
    core::param::ParamSlot hdri_evnfile;

    virtual bool InterfaceIsDirty();
    virtual void readParams();

};


} // namespace ospray
} // namespace megamol