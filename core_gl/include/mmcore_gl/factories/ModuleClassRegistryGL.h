/*
 * ModuleClassRegistry.h
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/ModuleDescriptionManager.h"

namespace megamol {
namespace core_gl {
namespace factories {

    /**
     * Registers all module classes of the core in the provided manager
     *
     * @param instance The Manager
     */
    void register_module_classes_gl(core::factories::ModuleDescriptionManager& instance);

} /* end namespace factories */
} /* end namespace core */
} /* end namespace megamol */

