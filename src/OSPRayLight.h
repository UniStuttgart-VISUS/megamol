/*
* OSPRayLight.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "CallOSPRayLight.h"
#include "ospray/ospray.h"
#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"



namespace megamol {
namespace ospray {


class OSPRayLight: public core::Module {
public:

    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "OSPRayLight";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Configuration module for an OSPRay light source.";
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
    OSPRayLight(void);

    /** Dtor. */
    virtual ~OSPRayLight(void);


protected:
    virtual bool create();
    virtual void release();
    bool getLightCallback(core::Call& call);

    // Interface variables
    core::param::ParamSlot lightIntensity;
    core::param::ParamSlot lightType;
    core::param::ParamSlot lightColor;

    core::param::ParamSlot dl_angularDiameter;
    core::param::ParamSlot dl_direction;
    core::param::ParamSlot dl_eye_direction;

    core::param::ParamSlot pl_position;
    core::param::ParamSlot pl_radius;

    core::param::ParamSlot sl_position;
    core::param::ParamSlot sl_direction;
    core::param::ParamSlot sl_openingAngle;
    core::param::ParamSlot sl_penumbraAngle;
    core::param::ParamSlot sl_radius;

    core::param::ParamSlot ql_position;
    core::param::ParamSlot ql_edgeOne;
    core::param::ParamSlot ql_edgeTwo;

    core::param::ParamSlot hdri_up;
    core::param::ParamSlot hdri_direction;
    core::param::ParamSlot hdri_evnfile;

private:
    core::CallerSlot getLightSlot;
    core::CalleeSlot deployLightSlot;

    OSPRayLightContainer lightContainer;

    /*
    * Reads the Interface parameter and puts them into a light container
    */
    void readParams();

};
}
}