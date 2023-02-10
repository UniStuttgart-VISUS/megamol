/*
 * OSPRayMetalMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayMaterial.h"

namespace megamol::ospray {

class OSPRayMetalMaterial : public AbstractOSPRayMaterial {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayMetalMaterial";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for an OSPRay metal material";
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
    OSPRayMetalMaterial();

    /** Dtor. */
    ~OSPRayMetalMaterial() override;

private:
    // METAL
    core::param::ParamSlot metalReflectance;
    core::param::ParamSlot metalEta;
    core::param::ParamSlot metalK;
    core::param::ParamSlot metalRoughness;

    bool InterfaceIsDirty() override;
    void readParams() override;
};


} // namespace megamol::ospray
