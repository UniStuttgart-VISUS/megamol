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
#include "mmstd/light/CallLight.h"

namespace megamol {
namespace core {
namespace view {
namespace light {

class AbstractLight : public core::Module {
protected:
    /** Ctor. */
    AbstractLight(void);

    /** Dtor. */
    virtual ~AbstractLight(void);

    virtual bool create();
    virtual void release();
    bool getLightCallback(core::Call& call);
    bool getMetaDataCallback(core::Call& call);
    virtual bool InterfaceIsDirty() {
        return false;
    };
    virtual void readParams(){};
    virtual void addLight(LightCollection& light_collection) = 0;
    bool AbstractIsDirty();

    uint32_t version;

    std::shared_ptr<BaseLightType> lightsource;

    // Interface variables
    core::param::ParamSlot lightIntensity;
    core::param::ParamSlot lightColor;

private:
    bool rhs_connected;

    core::CallerSlot getLightSlot;
    core::CalleeSlot deployLightSlot;
};
} // namespace light
} // namespace view
} // namespace core
} // namespace megamol
