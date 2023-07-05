/*
 * OSPRayLineGeometry.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/AbstractOSPRayStructure.h"

namespace megamol::ospray {

class OSPRayLineGeometry : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayLineGeometry";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Creator for OSPRay Line Geometry.";
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
    ~OSPRayLineGeometry() override;

    /** Ctor. */
    OSPRayLineGeometry();

protected:
    bool create() override;
    void release() override;

    bool readData(core::Call& call) override;
    bool getExtends(core::Call& call) override;


private:
    /** detects interface dirtyness */
    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot getDataSlot;
    core::CallerSlot getLineDataSlot;

    core::param::ParamSlot globalRadiusSlot;

    core::param::ParamSlot smoothSlot;
};

} // namespace megamol::ospray
