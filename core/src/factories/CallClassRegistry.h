/*
 * CallClassRegistry.h
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FACTORIES_CALLCLASSREGISTRY_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_CALLCLASSREGISTRY_H_INCLUDED
#pragma once

#include "mmcore/factories/CallDescriptionManager.h"

namespace megamol {
namespace core {
namespace factories {

/**
 * Registers all call classes of the core in the provided manager
 *
 * @param instance The Manager
 */
void register_call_classes(CallDescriptionManager& instance);

} /* end namespace factories */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FACTORIES_CALLCLASSREGISTRY_H_INCLUDED */
