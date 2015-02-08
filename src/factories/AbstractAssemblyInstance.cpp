/*
 * AbstractAssemblyInstance.cpp
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/factories/AbstractAssemblyInstance.h"


using namespace megamol::core::factories;


/*
 * AbstractAssemblyInstance::GetCallDescriptionManager
 */
const CallDescriptionManager&
AbstractAssemblyInstance::GetCallDescriptionManager(void) const {
    return this->call_descriptions;
}


/*
 * AbstractAssemblyInstance::GetModuleDescriptionManager
 */
const ModuleDescriptionManager&
AbstractAssemblyInstance::GetModuleDescriptionManager(void) const {
    return this->module_descriptions;
}


/*
 * AbstractAssemblyInstance::AbstractAssemblyInstance
 */
AbstractAssemblyInstance::AbstractAssemblyInstance(void) 
        : call_descriptions(), module_descriptions() {
    // intentionally empty
}


/*
 * AbstractAssemblyInstance::~AbstractAssemblyInstance
 */
AbstractAssemblyInstance::~AbstractAssemblyInstance(void) {
    // intentionally empty
}
