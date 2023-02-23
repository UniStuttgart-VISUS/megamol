/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmstd/light/AbstractLight.h"

namespace megamol::core::view::light {

struct QuadLightType : public BaseLightType {
    std::array<float, 3> position;
    std::array<float, 3> edgeOne;
    std::array<float, 3> edgeTwo;
};

class QuadLight : public AbstractLight {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "QuadLight";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for a quad light source.";
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
    void addLight(LightCollection& light_collection) override;

    /** Ctor. */
    QuadLight();

    /** Dtor. */
    ~QuadLight() override;

private:
    core::param::ParamSlot position;
    core::param::ParamSlot edgeOne;
    core::param::ParamSlot edgeTwo;

    bool InterfaceIsDirty() override;
    void readParams() override;
};

} // namespace megamol::core::view::light
