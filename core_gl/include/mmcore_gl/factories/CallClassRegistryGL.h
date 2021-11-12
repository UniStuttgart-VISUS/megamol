/*
 * CallClassRegistry.h
 * Copyright (C) 20021 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/CallDescriptionManager.h"

namespace megamol {
namespace core_gl {
namespace factories {

    /**
     * Registers all call classes of the core in the provided manager
     *
     * @param instance The Manager
     */
    void register_call_classes_gl(core::factories::CallDescriptionManager& instance);

} /* end namespace factories */
} /* end namespace core */
} /* end namespace megamol */

