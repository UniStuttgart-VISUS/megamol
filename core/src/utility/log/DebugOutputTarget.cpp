/*
 * DebugOutputTarget.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/log/DebugOutputTarget.h"

#ifdef _WIN32

/*
 * megamol::core::utility::log::DebugOutputTarget::DebugOutputTarget
 */
megamol::core::utility::log::DebugOutputTarget::DebugOutputTarget(Log::UINT level) : Target(level) {
    logger = spdlog::get("default_debug");
    if (logger == nullptr) {
        sink = std::make_shared<spdlog::sinks::msvc_sink_mt>();
        sink->set_pattern(Log::std_pattern);
        logger = std::make_shared<spdlog::logger>("default_debug", sink);
    }
}


/*
 * megamol::core::utility::log::DebugOutputTarget::~DebugOutputTarget
 */
megamol::core::utility::log::DebugOutputTarget::~DebugOutputTarget(void) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::DebugOutputTarget::Msg
 */
void megamol::core::utility::log::DebugOutputTarget::Msg(Log::UINT level,
    megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid, const char* msg) {
    Msg(level, time, sid, std::string(msg));
}


/*
 * megamol::core::utility::log::DebugOutputTarget::Msg
 */
void megamol::core::utility::log::DebugOutputTarget::Msg(
    Log::UINT level, Log::TimeStamp time, Log::SourceID sid, std::string const& msg) {
    if (level > this->Level())
        return;
    logger->info("{}|{}", level, msg);
}

#endif /* _WIN32 */
