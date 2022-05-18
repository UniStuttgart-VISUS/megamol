/*
 * StreamTarget.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/log/StreamTarget.h"

/*
 * megamol::core::utility::log::StreamTarget::StdOut
 */
const std::shared_ptr<megamol::core::utility::log::Log::Target> megamol::core::utility::log::StreamTarget::StdOut =
    std::make_shared<megamol::core::utility::log::StreamTarget>(std::cout);


/*
 * megamol::core::utility::log::StreamTarget::StdErr
 */
const std::shared_ptr<megamol::core::utility::log::Log::Target> megamol::core::utility::log::StreamTarget::StdErr =
    std::make_shared<megamol::core::utility::log::StreamTarget>(std::cerr);


/*
 * megamol::core::utility::log::StreamTarget::StreamTarget
 */
megamol::core::utility::log::StreamTarget::StreamTarget(std::ostream& stream, Log::UINT level) : Target(level) {
    logger = spdlog::get("default_stream");
    if (logger == nullptr) {
        sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(stream);
        sink->set_pattern(Log::std_pattern);
        logger = std::make_shared<spdlog::logger>("default_stream", sink);
    }
}


/*
 * megamol::core::utility::log:::StreamTarget::~StreamTarget
 */
megamol::core::utility::log::StreamTarget::~StreamTarget(void) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::StreamTarget::Flush
 */
void megamol::core::utility::log::StreamTarget::Flush(void) {
    logger->flush();
}


/*
 * megamol::core::utility::log::StreamTarget::Msg
 */
void megamol::core::utility::log::StreamTarget::Msg(Log::UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, const char* msg) {
    Msg(level, time, sid, std::string(msg));
}


/*
 * megamol::core::utility::log::StreamTarget::Msg
 */
void megamol::core::utility::log::StreamTarget::Msg(Log::UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, std::string const& msg) {
    if ((level > this->Level()))
        return;
    logger->info("{}|{}", level, msg);
}
