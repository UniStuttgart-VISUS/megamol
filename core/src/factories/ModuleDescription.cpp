/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/ModuleDescription.h"

using namespace megamol::core;

/*
 * factories::ModuleDescription::IsAvailable
 */
bool factories::ModuleDescription::IsAvailable() const {
    return true;
}

/*
 * factories::ModuleDescription::CreateModule
 */
Module::ptr_type factories::ModuleDescription::CreateModule(const std::string& name) const {
    Module::ptr_type m = this->createModuleImpl();
    if (m) {
        m->fixParentBackreferences();
        m->setModuleName(vislib::StringA(name.c_str()));
    }
    return m;
}
