/*
 * Log.h
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS).
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <cstdio>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

// Enclose title of log message between start and and tag to get it pushed in GUI popup
#define LOGMESSAGE_GUI_POPUP_START_TAG "<<<<<"
#define LOGMESSAGE_GUI_POPUP_END_TAG ">>>>>"

namespace spdlog {
namespace sinks {
class sink;
}
} // namespace spdlog

namespace megamol {
namespace core {
namespace utility {
namespace log {


/**
 * This is a utility class for managing a log file.
 */
class Log {
public:
    /** Default log message pattern for spdlog */
    static const char std_pattern[12];

    /** Default file log message pattern for spdlog */
    static const char file_pattern[22];

    /** Name of default logger in spdlog */
    static const char logger_name[23];

    /** Name of echo logger in spdlog */
    static const char echo_logger_name[20];

    /** type for time stamps */
    using TimeStamp = time_t;

    /** type for the id of the source object */
    using SourceID = size_t;

    /** unsigned int alias */
    using UINT = unsigned int;

    enum class log_level : UINT { none, info, warn, error, all = info };

    static std::string print_log_level(log_level level) {
        switch (level) {
        case log_level::none:
            return "none";
        case log_level::warn:
            return "warn";
        case log_level::error:
            return "error";
        case log_level::all:
        default:
            return "info";
        }
    }

    /**
     * Parse accepted log level attributes from string to log_level
     */
    static megamol::core::utility::log::Log::log_level ParseLevelAttribute(const std::string attr,
        megamol::core::utility::log::Log::log_level def = megamol::core::utility::log::Log::log_level::error);

    /** The default log object. */
    static Log& DefaultLog;

    /**
     * Ctor. Constructs a new log file without a physical file.
     *
     * @param level Sets the current log level.
     * @param msgbufsize The number of messages that will be stored in
     *                   memory if no physical log file is available.
     */
    Log(log_level level = log_level::info, unsigned int msgbufsize = 10);

    /**
     * Ctor. Constructs a new log file with the specified physical file.
     *
     * @param level Sets the current log level.
     * @param filename The name of the physical log file.
     * @param addSuffix If true a automatically generated suffix is added
     *                  to the name of the physical log file, consisting of
     *                  the name of the computer name, the current date and
     *                  time.
     */
    Log(log_level level, const char* filename, bool addSuffix);

    /** Dtor. */
    ~Log(void);

    /** Flushes the physical log file. */
    void FlushLog(void);

    /**
     * Sets or clears the autoflush flag. If the autoflush flag is set
     * a flush of all data to the physical log file is performed after
     * each message. Autoflush is enabled by default.
     *
     * @param enable New value for the autoflush flag.
     */
    /*inline void SetAutoFlush(bool enable) {
        this->autoflush = enable;
    }*/

    /**
     * Set a new echo level. Messages above this level will be ignored,
     * while the other messages will be echoed to the echo output stream.
     *
     * @param level The new echo level.
     */
    void SetEchoLevel(log_level level);

    /**
     * Add new target to echo log
     *
     * @param A new echo log target
     *
     * @return Index of new target in echo log
     */
    std::size_t AddEchoTarget(std::shared_ptr<spdlog::sinks::sink> target);

    /**
     * Remove echo target at index
     */
    void RemoveEchoTarget(std::size_t idx);

    /**
     * Set a new log level. Messages above this level will be ignored.
     *
     * @param level The new log level.
     */
    void SetLevel(log_level level);

    /**
     * Adds file target to the loggger.
     *
     * @param filename The name of the physical log file. If this parameter
     *                 is NULL, the current physical log file is closed,
     *                 but no new file will be opened.
     * @param addSuffix If true a automatically generated suffix is added
     *                  to the name of the physical log file, consisting of
     *                  the name of the computer name, the current date and
     *                  time.
     * @param level Set log level of the file target.
     */
    void AddFileTarget(const char* filename, bool addSuffix, log_level level = log_level::info);

    /**
     * Writes a formatted error message to the log. The level will be
     * 'LEVEL_ERROR'.
     *
     * @param fmt The log message
     */
    void WriteError(const char* fmt, ...);

    /**
     * Writes a formatted error message to the log. The level will be
     * 'LEVEL_INFO'.
     *
     * @param fmt The log message
     */
    void WriteInfo(const char* fmt, ...);

    /**
     * Writes a formatted messages with the specified log level to the log
     * file. The format of the message is similar to the printf functions.
     * A new line character is automatically appended if the last
     * character of fmt is no new line character.
     *
     * @param level The log level of the message.
     * @param fmt The log message.
     */
    void WriteMsg(log_level level, const char* fmt, ...);

    /**
     * Writes a formatted error message to the log. The level will be
     * 'LEVEL_WARN'.
     *
     * @param fmt The log message
     */
    void WriteWarn(const char* fmt, ...);

    /**
     * Writes a formatted error message to the log. The level will be
     * 'LEVEL_WARN + lvlOff'. Not that a high level offset value might
     * downgrade the message to info level.
     *
     * @param fmt The log message
     * @param lvlOff The log level offset
     */
    //void WriteWarn(int lvlOff, const char* fmt, ...);

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand.
     *
     * @return Reference to this.
     */
    Log& operator=(const Log& rhs);

private:
    /**
     * Writes a pre-formatted message with specified log level, time stamp
     * and source id to the log.
     *
     * @param level The level of the message
     * @param msg The message text itself
     */
    void writeMessage(log_level level, const std::string& msg);

    /**
     * Writes a pre-formatted message with specified log level, time stamp
     * and source id to the log.
     *
     * @param level The level of the message
     * @param msg The message text itself
     */
    void writeMessageVaA(log_level level, const char* fmt, va_list argptr);

    /**
     * Answer a file name suffix for log files
     *
     * @return A file name suffix for log files
     */
    std::string getFileNameSuffix(void);
};

} // namespace log
} // namespace utility
} // namespace core
} // namespace megamol
