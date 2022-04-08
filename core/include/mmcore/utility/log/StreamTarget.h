/*
 * StreamTarget.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/utility/log/Log.h"

#define SPDLOG_EOL ""
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/spdlog.h"

namespace megamol::core::utility::log {

/**
 * Target class echoing the log messages into a stream
 */
class StreamTarget : public Log::Target {
public:
    /** Stream target to stdout */
    static const std::shared_ptr<Target> StdOut;

    /** Stream target to stderr */
    static const std::shared_ptr<Target> StdErr;

    /**
     * Ctor
     *
     * @param stream The stream to write the log messages to
     * @param level The log level used for this target
     */
    StreamTarget(std::ostream& stream, Log::UINT level = Log::LEVEL_ERROR);

    /** Dtor */
    virtual ~StreamTarget(void);

    /** Flushes any buffer */
    void Flush(void) override;

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
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink;

    std::shared_ptr<spdlog::logger> logger;
};

} // namespace megamol::core::utility::log
