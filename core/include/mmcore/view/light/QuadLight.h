/*
 * QuadLight.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/view/light/AbstractLight.h"

namespace megamol {
namespace core {
namespace view {
namespace light {

class MEGAMOLCORE_API QuadLight : public AbstractLight {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "QuadLight"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Configuration module for a quad light source."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    QuadLight(void);

    /** Dtor. */
    virtual ~QuadLight(void);

private:
    core::param::ParamSlot ql_position;
    core::param::ParamSlot ql_edgeOne;
    core::param::ParamSlot ql_edgeTwo;

    virtual bool InterfaceIsDirty();
    virtual void readParams();
};

} // namespace light
} // namespace view
} // namespace core
} // namespace megamol
