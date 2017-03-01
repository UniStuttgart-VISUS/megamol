/*
* AbstractOSPRayLight.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "CallOSPRayLight.h"
#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"



namespace megamol {
namespace ospray {


class AbstractOSPRayLight: public core::Module {
protected:

    /** Ctor. */
    AbstractOSPRayLight(void);

    /** Dtor. */
    virtual ~AbstractOSPRayLight(void);

    AbstractOSPRayLight& operator=(const AbstractOSPRayLight &rhs);

    virtual bool create();
    virtual void release();
    bool getLightCallback(core::Call& call);
    virtual bool InterfaceIsDirty() { return true; };
    virtual void readParams() {};
    bool AbstractIsDirty();

    OSPRayLightContainer lightContainer;

    // Interface variables
    core::param::ParamSlot lightIntensity;
    core::param::ParamSlot lightColor;


private:
    core::CallerSlot getLightSlot;
    core::CalleeSlot deployLightSlot;

};
} // namespace ospray
} // namespace megamol