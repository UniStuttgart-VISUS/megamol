/*
 * ModuleDescription.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ModuleDescription.h"
#include "CoreInstance.h"


using namespace megamol::core;


/*
 * ModuleDescription::ModuleDescription
 */
ModuleDescription::ModuleDescription(void) : ObjectDescription() {
}


/*
 * ModuleDescription::~ModuleDescription
 */
ModuleDescription::~ModuleDescription(void) {
}


/*
 * ModuleDescription::CreateModule
 */
Module *ModuleDescription::CreateModule(const vislib::StringA& name,
        class ::megamol::core::CoreInstance *instance) const {
    Module *m = this->createModuleImpl();
    if (m != NULL) {
        m->coreInst = instance;
        m->setModuleName(name);
    }
    return m;
}


/*
 * ModuleDescription::IsAvailable
 */
bool ModuleDescription::IsAvailable(void) const {
    return true;
}
