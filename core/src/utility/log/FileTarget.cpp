/*
 * FileTarget.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/log/FileTarget.h"

/*
 * megamol::core::utility::log::FileTarget::FileTarget
 */
megamol::core::utility::log::FileTarget::FileTarget(std::string const& path, Log::UINT level) : Target(level) {
    logger = spdlog::get(std::string("default_file_") + path);
    if (logger == nullptr) {
        sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, true);
        sink->set_pattern(Log::std_pattern);
        logger = std::make_shared<spdlog::logger>(std::string("default_file_") + path, sink);
    }
    filename = path;
}


/*
 * megamol::core::utility::log::FileTarget::~FileTarget
 */
megamol::core::utility::log::FileTarget::~FileTarget(void) {
    logger->flush();
}


/*
 * megamol::core::utility::log::FileTarget::Flush
 */
void megamol::core::utility::log::FileTarget::Flush(void) {
    logger->flush();
}


/*
 * megamol::core::utility::log::FileTarget::Msg
 */
void megamol::core::utility::log::FileTarget::Msg(Log::UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, const char* msg) {
    Msg(level, time, sid, std::string(msg));
}


/*
 * megamol::core::utility::log::FileTarget::Msg
 */
void megamol::core::utility::log::FileTarget::Msg(Log::UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, std::string const& msg) {
    if (level > this->Level())
        return;

    struct tm* timeStamp;
#ifdef _WIN32
#if (_MSC_VER >= 1400)
    struct tm __tS;
    timeStamp = &__tS;
    if (localtime_s(timeStamp, &time) != 0) {
        // timestamp error
        __tS.tm_hour = __tS.tm_min = __tS.tm_sec = 0;
    }
#else  /* (_MSC_VER >= 1400) */
    timeStamp = localtime(&time);
#endif /* (_MSC_VER >= 1400) */
#else  /* _WIN32 */
    timeStamp = localtime(&time);
#endif /* _WIN32 */
    logger->info("{}:{}:{}|{}|{}|{}", timeStamp->tm_hour, timeStamp->tm_min, timeStamp->tm_sec, sid, level, msg);
}
