/*
 * SpotLight.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/view/light/AbstractLight.h"

namespace megamol {
namespace core {
namespace view {
namespace light {

class MEGAMOLCORE_API SpotLight : public AbstractLight {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "SpotLight"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Configuration module for a spot light source."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    SpotLight(void);

    /** Dtor. */
    virtual ~SpotLight(void);

private:
    core::param::ParamSlot sl_position;
    core::param::ParamSlot sl_direction;
    core::param::ParamSlot sl_openingAngle;
    core::param::ParamSlot sl_penumbraAngle;
    core::param::ParamSlot sl_radius;

    virtual bool InterfaceIsDirty();
    virtual void readParams();
};

} // namespace light
} // namespace view
} // namespace core
} // namespace megamol
