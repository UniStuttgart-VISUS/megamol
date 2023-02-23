/*
 * OSPRayMatteMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayMaterial.h"

namespace megamol::ospray {

class OSPRayMatteMaterial : public AbstractOSPRayMaterial {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayMatteMaterial";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for an OSPRay matte material";
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
    OSPRayMatteMaterial();

    /** Dtor. */
    ~OSPRayMatteMaterial() override;

private:
    // MATTE
    core::param::ParamSlot matteReflectance;

    bool InterfaceIsDirty() override;
    void readParams() override;
};


} // namespace megamol::ospray
