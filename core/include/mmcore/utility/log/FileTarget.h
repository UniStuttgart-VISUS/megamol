/*
 * FileTarget.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>

#include "mmcore/utility/log/Log.h"

#define SPDLOG_EOL ""
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

namespace megamol::core::utility::log {

/**
 * Target class safing message to a ASCII text file
 */
class FileTarget : public Log::Target {
public:
    /**
     * Opens a physical log file
     *
     * @param path The path to the physical log file
     * @param level The log level used for this target
     */
    FileTarget(std::string const& path, Log::UINT level = Log::LEVEL_ERROR);

    /** Dtor */
    virtual ~FileTarget(void);

    /** Flushes any buffer */
    void Flush(void) override;

    /**
     * Answer the path to the physical log file
     *
     * @return The path to the physical log file
     */
    inline const std::string& Filename(void) const {
        return this->filename;
    }

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
    /** The file name of the log file used */
    std::string filename;

    std::shared_ptr<spdlog::sinks::basic_file_sink_mt> sink;

    std::shared_ptr<spdlog::logger> logger;
};

} // namespace megamol::core::utility::log
