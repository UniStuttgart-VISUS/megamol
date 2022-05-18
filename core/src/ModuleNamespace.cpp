/*
 * ModuleNamespace.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "mmcore/ModuleNamespace.h"
#include "stdafx.h"

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
ModuleNamespace::~ModuleNamespace(void) {
    // intentionally empty ATM
}


/*
 * ModuleNamespace::ClearCleanupMark
 */
void ModuleNamespace::ClearCleanupMark(void) {
    if (!this->CleanupMark())
        return;
    AbstractNamedObject::ClearCleanupMark();
    if (this->Parent() != NULL) {
        this->Parent()->ClearCleanupMark();
    }
}
