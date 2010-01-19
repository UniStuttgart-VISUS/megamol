/*
 * LogEchoTarget.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "utility/LogEchoTarget.h"


/*
 * megamol::core::utility::LogEchoTarget::LogEchoTarget
 */
megamol::core::utility::LogEchoTarget::LogEchoTarget(void) 
        : vislib::sys::Log::EchoTarget(), target(NULL) {
    // Intentionally empty
}


/*
 * megamol::core::utility::LogEchoTarget::LogEchoTarget
 */
megamol::core::utility::LogEchoTarget::LogEchoTarget(mmcLogEchoFunction target)
        : vislib::sys::Log::EchoTarget(), target(target) {
    // Intentionally empty
}


/*
 * megamol::core::utility::LogEchoTarget::~LogEchoTarget
 */
megamol::core::utility::LogEchoTarget::~LogEchoTarget(void) {
    this->target = NULL; // DO NOT DELETE
}


/*
 * megamol::core::utility::LogEchoTarget::SetTarget
 */
void megamol::core::utility::LogEchoTarget::SetTarget(
        mmcLogEchoFunction target) {
    this->target = target;
}


/*
 * megamol::core::utility::LogEchoTarget::Write
 */
void megamol::core::utility::LogEchoTarget::Write(UINT level,
        const char *message) const {
    if (this->target) {
        this->target(level, message);
    }
}
