/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/light/CallLight.h"

namespace megamol::core::view::light {

class AbstractLight : public core::Module {
protected:
    /** Ctor. */
    AbstractLight();

    /** Dtor. */
    ~AbstractLight() override;

    bool create() override;
    void release() override;
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

} // namespace megamol::core::view::light
