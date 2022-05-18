/*
 * DebugOutputTarget.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/utility/log/Log.h"

#define SPDLOG_EOL ""
#include "spdlog/sinks/msvc_sink.h"
#include "spdlog/spdlog.h"

#ifdef _WIN32

namespace megamol::core::utility::log {

/**
 * Target class echoing the log messages into a stream
 */
class DebugOutputTarget : public Log::Target {
public:
    /**
     * Ctor
     *
     * @param level The log level used for this target
     */
    DebugOutputTarget(Log::UINT level = Log::LEVEL_ERROR);

    /** Dtor */
    virtual ~DebugOutputTarget(void);

    /**
     * Writes a message to the log target
     *
     * @param level The level of the message
     * @param time The time stamp of the message
     * @param sid The object id of the source of the message
     * @param msg The message text itself
     */
    void Msg(Log::UINT level, Log::TimeStamp time, Log::SourceID sid, const char* msg) override;

    /**
     * Writes a message to the log target
     *
     * @param level The level of the message
     * @param time The time stamp of the message
     * @param sid The object id of the source of the message
     * @param msg The message text itself
     */
    void Msg(Log::UINT level, Log::TimeStamp time, Log::SourceID sid, std::string const& msg) override;

private:
    std::shared_ptr<spdlog::sinks::msvc_sink_mt> sink;

    std::shared_ptr<spdlog::logger> logger;
};

} // namespace megamol::core::utility::log


#endif /* _WIN32 */
