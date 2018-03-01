/*
* OSPRayPBSGeometry.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "OSPRay_plugin/AbstractOSPRayStructure.h"

namespace megamol {
namespace ospray {

class OSPRayPBSGeometry : public AbstractOSPRayStructure {

public:

    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "OSPRayPBSGeometry";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Creator for OSPRay sphere PBS geometries.";
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
    virtual ~OSPRayPBSGeometry(void);

    /** Ctor. */
    OSPRayPBSGeometry(void);

protected:

    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call &call);
    virtual bool getExtends(core::Call &call);


    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot getDataSlot;

    core::param::ParamSlot radiusSlot;

private:


    //data objects
    std::vector<float> x_data_float;
    std::vector<float> y_data_float;
    std::vector<float> z_data_float;

};

} // namespace ospray
} // namespace megamol