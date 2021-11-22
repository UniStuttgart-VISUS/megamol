/*
 * OSPRayAPIStructure.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/AbstractOSPRayStructure.h"

namespace megamol {
namespace ospray {

class OSPRayAPIStructure : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "OSPRayAPIStructure";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Creator for OSPRay API structures.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Dtor. */
    virtual ~OSPRayAPIStructure(void);

    /** Ctor. */
    OSPRayAPIStructure(void);

protected:
    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call& call);
    virtual bool getExtends(core::Call& call);


    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot getDataSlot;
};

} // namespace ospray
} // namespace megamol
