/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmstd/light/AbstractLight.h"

namespace megamol::core::view::light {

struct TriDirectionalLightType : public BaseLightType {
    std::array<float, 3> key_direction;
    std::array<float, 3> fill_direction;
    std::array<float, 3> back_direction;
    bool in_view_space;
};

class TriDirectionalLighting : public AbstractLight {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TriDirectionalLighting";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Configuration module for a lightingt setup with three directional lights, i.e. three point lighting "
               "without falloff.";
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
    TriDirectionalLighting();

    /** Dtor. */
    ~TriDirectionalLighting() override;

private:
    core::param::ParamSlot m_key_direction;
    core::param::ParamSlot m_fill_direction;
    core::param::ParamSlot m_back_direction;
    core::param::ParamSlot m_in_view_space;

    bool InterfaceIsDirty() override;
    void readParams() override;
};

} // namespace megamol::core::view::light
