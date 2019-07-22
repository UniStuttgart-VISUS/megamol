/*
 * HDRILight.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/view/light/AbstractLight.h"

namespace megamol {
namespace core {
namespace view {
namespace light {

class MEGAMOLCORE_API HDRILight : public AbstractLight {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "HDRILight"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Configuration module for an HDRI light source."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    HDRILight(void);

    /** Dtor. */
    virtual ~HDRILight(void);

private:
    core::param::ParamSlot hdri_up;
    core::param::ParamSlot hdri_direction;
    core::param::ParamSlot hdri_evnfile;

    virtual bool InterfaceIsDirty();
    virtual void readParams();
};

} // namespace light
} // namespace view
} // namespace core
} // namespace megamol
