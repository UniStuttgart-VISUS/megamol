/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmstd/light/AbstractLight.h"

namespace megamol::core::view::light {

struct DistantLightType : public BaseLightType {
    std::array<float, 3> direction;
    float angularDiameter;
    bool eye_direction;
};

class DistantLight : public AbstractLight {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "DistantLight";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for a distant light source.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Add the lightsource of this module to a given collection
     */
    void addLight(LightCollection& light_collection);

    /** Ctor. */
    DistantLight();

    /** Dtor. */
    virtual ~DistantLight();

private:
    core::param::ParamSlot angularDiameter;
    core::param::ParamSlot direction;
    core::param::ParamSlot eye_direction;

    virtual bool InterfaceIsDirty();
    virtual void readParams();
};

} // namespace megamol::core::view::light
