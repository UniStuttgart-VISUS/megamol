/*
 * OSPRayThinGlassMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayMaterial.h"

namespace megamol::ospray {

class OSPRayThinGlassMaterial : public AbstractOSPRayMaterial {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayThinGlassMaterial";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for an OSPRay thin glass material";
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
    OSPRayThinGlassMaterial();

    /** Dtor. */
    ~OSPRayThinGlassMaterial() override;

private:
    //THINGLASS
    core::param::ParamSlot thinglassTransmission;
    core::param::ParamSlot thinglassEta;
    core::param::ParamSlot thinglassThickness;


    bool InterfaceIsDirty() override;
    void readParams() override;
};


} // namespace megamol::ospray
