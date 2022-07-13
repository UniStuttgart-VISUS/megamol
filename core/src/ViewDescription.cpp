/*
 * ViewDescription.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "mmcore/ViewDescription.h"


/*
 * megamol::core::ViewDescription::ViewDescription
 */
megamol::core::ViewDescription::ViewDescription(const char* classname) : InstanceDescription(classname), viewID("") {
    vislib::StringA d;
    d.Format("View %s", classname);
    this->SetDescription(d);
}


/*
 * megamol::core::ViewDescription::~ViewDescription
 */
megamol::core::ViewDescription::~ViewDescription(void) {
    this->viewID.Clear();
}
