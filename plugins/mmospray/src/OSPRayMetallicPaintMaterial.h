/*
 * OSPRayMetallicPaintMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayMaterial.h"

namespace megamol::ospray {

class OSPRayMetallicPaintMaterial : public AbstractOSPRayMaterial {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayMetallicPaintMaterial";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for an OSPRay metallic paint material";
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
    OSPRayMetallicPaintMaterial();

    /** Dtor. */
    ~OSPRayMetallicPaintMaterial() override;

private:
    // METALLICPAINT
    core::param::ParamSlot metallicShadeColor;
    core::param::ParamSlot metallicGlitterColor;
    core::param::ParamSlot metallicGlitterSpread;
    core::param::ParamSlot metallicEta;

    bool InterfaceIsDirty() override;
    void readParams() override;
};


} // namespace megamol::ospray
