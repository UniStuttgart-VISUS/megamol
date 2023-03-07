/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmstd/light/AbstractLight.h"

namespace megamol::core::view::light {

struct HDRILightType : public BaseLightType {
    std::array<float, 3> up;
    std::array<float, 3> direction;
    vislib::TString evnfile;
};


class HDRILight : public AbstractLight {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "HDRILight";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for an HDRI light source.";
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
    HDRILight();

    /** Dtor. */
    ~HDRILight() override;

private:
    core::param::ParamSlot up;
    core::param::ParamSlot direction;
    core::param::ParamSlot evnfile;

    bool InterfaceIsDirty() override;
    void readParams() override;
};

} // namespace megamol::core::view::light
