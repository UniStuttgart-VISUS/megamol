/*
 * OSPRayOBJMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayMaterial.h"

namespace megamol::ospray {

class OSPRayOBJMaterial : public AbstractOSPRayMaterial {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayOBJMaterial";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for an OSPRay OBJ material";
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
    OSPRayOBJMaterial();

    /** Dtor. */
    ~OSPRayOBJMaterial() override;

private:
    // material
    // OBJMaterial/ScivisMaterial
    core::param::ParamSlot Kd;
    core::param::ParamSlot Ks;
    core::param::ParamSlot Ns;
    core::param::ParamSlot d;
    core::param::ParamSlot Tf;

    bool InterfaceIsDirty() override;
    void readParams() override;
};


} // namespace megamol::ospray
