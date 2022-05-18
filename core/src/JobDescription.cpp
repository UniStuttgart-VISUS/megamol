/*
 * JobDescription.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "mmcore/JobDescription.h"
#include "stdafx.h"


/*
 * megamol::core::JobDescription::JobDescription
 */
megamol::core::JobDescription::JobDescription(const char* classname) : InstanceDescription(classname), jobModID("") {
    vislib::StringA d;
    d.Format("Job %s", classname);
    this->SetDescription(d);
}


/*
 * megamol::core::JobDescription::~JobDescription
 */
megamol::core::JobDescription::~JobDescription(void) {
    this->jobModID.Clear();
}
