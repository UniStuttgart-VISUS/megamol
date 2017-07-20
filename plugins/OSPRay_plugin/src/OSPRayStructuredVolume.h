/*
* OSPRayStructuredVolume.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "OSPRay_plugin/AbstractOSPRayStructure.h"

namespace megamol {
namespace ospray {

class OSPRayStructuredVolume : public AbstractOSPRayStructure {

public:

    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "OSPRayStructuredVolume";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Creator for OSPRay structured volumes.";
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
    virtual ~OSPRayStructuredVolume(void);

    /** Ctor. */
    OSPRayStructuredVolume(void);

protected:

    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call &call);
    virtual bool getExtends(core::Call &call);

    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot getDataSlot;

    megamol::core::param::ParamSlot clippingBoxLower;
    megamol::core::param::ParamSlot clippingBoxUpper;
    megamol::core::param::ParamSlot clippingBoxActive;

    megamol::core::param::ParamSlot repType;

    megamol::core::param::ParamSlot IsoValue;

    megamol::core::param::ParamSlot sliceNormal;
    megamol::core::param::ParamSlot sliceDist;
};

} // namespace ospray
} // namespace megamol