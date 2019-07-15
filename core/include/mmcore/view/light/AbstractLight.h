/*
 * AbstractLight.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/light/CallLight.h"
#include "mmcore/api/MegaMolCore.std.h"

namespace megamol {
namespace core {
namespace view {
namespace light {

class MEGAMOLCORE_API AbstractLight : public core::Module {
protected:
    /** Ctor. */
    AbstractLight(void);

    /** Dtor. */
    virtual ~AbstractLight(void);

    virtual bool create();
    virtual void release();
    bool getLightCallback(core::Call& call);
    virtual bool InterfaceIsDirty() { return false; };
    virtual void readParams(){};
    bool AbstractIsDirty();

    LightContainer lightContainer;

    // Interface variables
    core::param::ParamSlot lightIntensity;
    core::param::ParamSlot lightColor;

private:
    core::CallerSlot getLightSlot;
    core::CalleeSlot deployLightSlot;
};
} // namespace light
} // namespace view
} // namespace core
} // namespace megamol
