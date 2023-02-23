/*
 * OSPRayStructuredVolume.h
 * Copyright (C) 2009-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/AbstractOSPRayStructure.h"

namespace megamol::ospray {

class OSPRayStructuredVolume : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayStructuredVolume";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Creator for OSPRay structured volumes.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Dtor. */
    ~OSPRayStructuredVolume() override;

    /** Ctor. */
    OSPRayStructuredVolume();

protected:
    bool create() override;
    void release() override;

    bool readData(core::Call& call) override;
    bool getExtends(core::Call& call) override;

    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot getDataSlot;

    /** The call for Transfer function */
    core::CallerSlot getTFSlot;

    megamol::core::param::ParamSlot clippingBoxLower;
    megamol::core::param::ParamSlot clippingBoxUpper;
    megamol::core::param::ParamSlot clippingBoxActive;
    megamol::core::param::ParamSlot showBoundingBox;

    megamol::core::param::ParamSlot repType;
    megamol::core::param::ParamSlot IsoValue;
};

} // namespace megamol::ospray
