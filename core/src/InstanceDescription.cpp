/*
 * InstanceDescription.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/InstanceDescription.h"
#include "stdafx.h"


/*
 * megamol::core::InstanceDescription::InstanceDescription
 */
megamol::core::InstanceDescription::InstanceDescription(const char* classname)
        : ObjectDescription()
        , ParamValueSetRequest()
        , classname(classname)
        , description()
        , modules()
        , calls() {
    this->description.Format("Instance %s", this->classname.PeekBuffer());
}


/*
 * megamol::core::InstanceDescription::~InstanceDescription
 */
megamol::core::InstanceDescription::~InstanceDescription(void) {
    this->modules.Clear();
    this->calls.Clear();
}
