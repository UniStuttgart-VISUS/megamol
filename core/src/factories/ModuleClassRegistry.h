/*
 * ModuleClassRegistry.h
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FACTORIES_MODULECLASSREGISTRY_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_MODULECLASSREGISTRY_H_INCLUDED
#pragma once

#include "mmcore/factories/ModuleDescriptionManager.h"

namespace megamol {
namespace core {
namespace factories {

/**
 * Registers all module classes of the core in the provided manager
 *
 * @param instance The Manager
 */
void register_module_classes(ModuleDescriptionManager& instance);

} /* end namespace factories */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FACTORIES_MODULECLASSREGISTRY_H_INCLUDED */
