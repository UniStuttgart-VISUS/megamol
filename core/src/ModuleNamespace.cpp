/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/ModuleNamespace.h"

using namespace megamol::core;


/*
 * ModuleNamespace::ModuleNamespace
 */
ModuleNamespace::ModuleNamespace(const vislib::StringA& name) : AbstractNamedObjectContainer() {
    this->setName(name);
}


/*
 * ModuleNamespace::~ModuleNamespace
 */
ModuleNamespace::~ModuleNamespace() {
    // intentionally empty ATM
}


/*
 * ModuleNamespace::ClearCleanupMark
 */
void ModuleNamespace::ClearCleanupMark() {
    if (!this->CleanupMark())
        return;
    AbstractNamedObject::ClearCleanupMark();
    if (this->Parent() != NULL) {
        this->Parent()->ClearCleanupMark();
    }
}
