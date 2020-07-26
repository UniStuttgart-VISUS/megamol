/*
 * LogEchoTarget.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/LogEchoTarget.h"


/*
 * megamol::core::utility::LogEchoTarget::LogEchoTarget
 */
megamol::core::utility::LogEchoTarget::LogEchoTarget(void) 
        : megamol::core::utility::log::Log::Target(), target(NULL) {
    // Intentionally empty
}


/*
 * megamol::core::utility::LogEchoTarget::LogEchoTarget
 */
megamol::core::utility::LogEchoTarget::LogEchoTarget(mmcLogEchoFunction target)
        : megamol::core::utility::log::Log::Target(), target(target) {
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
 * megamol::core::utility::LogEchoTarget::Msg
 */
void megamol::core::utility::LogEchoTarget::Msg(UINT level, megamol::core::utility::log::Log::TimeStamp time,
        megamol::core::utility::log::Log::SourceID sid, const char *msg) {
    if (this->target && level <= this->Level()) {
        this->target(level, msg);
    }
}


/*
 * megamol::core::utility::LogEchoTarget::Msg
 */
void megamol::core::utility::LogEchoTarget::Msg(UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, std::string const& msg) {
    Msg(level, time, sid, msg.c_str());
}
