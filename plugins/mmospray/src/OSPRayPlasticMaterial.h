/*
 * OSPRayPlasticMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayMaterial.h"

namespace megamol {
namespace ospray {

class OSPRayPlasticMaterial : public AbstractOSPRayMaterial {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "OSPRayPlasticMaterial";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Configuration module for an OSPRay plastic material";
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
    OSPRayPlasticMaterial(void);

    /** Dtor. */
    virtual ~OSPRayPlasticMaterial(void);

private:
    // PLASTIC
    core::param::ParamSlot plasticPigmentColor;
    core::param::ParamSlot plasticEta;
    core::param::ParamSlot plasticRoughness;
    core::param::ParamSlot plasticThickness;

    virtual bool InterfaceIsDirty();
    virtual void readParams();
};


} // namespace ospray
} // namespace megamol
