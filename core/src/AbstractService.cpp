/*
 * AbstractService.cpp
 *
 * Copyright (C) 2016 by MegaMol Team (S. Grottel)
 * Alle Rechte vorbehalten.
 */
#include "mmcore/AbstractService.h"

using namespace megamol;

core::AbstractService::~AbstractService() {
    // intentionally empty
}

bool core::AbstractService::Initalize(bool& autoEnable) {
    // intentionally empty
    return true;
}

bool core::AbstractService::Deinitialize() {
    this->Disable();
    return true;
}

core::AbstractService::AbstractService(core::CoreInstance& core) : core(core), enabled(false) {
    // intentionally empty
}
