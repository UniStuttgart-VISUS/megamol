/*
 * OSPRayMetallicPaintMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayMaterial.h"

namespace megamol {
namespace ospray {

class OSPRayMetallicPaintMaterial : public AbstractOSPRayMaterial {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "OSPRayMetallicPaintMaterial";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Configuration module for an OSPRay metallic paint material";
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
    OSPRayMetallicPaintMaterial(void);

    /** Dtor. */
    virtual ~OSPRayMetallicPaintMaterial(void);

private:
    // METALLICPAINT
    core::param::ParamSlot metallicShadeColor;
    core::param::ParamSlot metallicGlitterColor;
    core::param::ParamSlot metallicGlitterSpread;
    core::param::ParamSlot metallicEta;

    virtual bool InterfaceIsDirty();
    virtual void readParams();
};


} // namespace ospray
} // namespace megamol
