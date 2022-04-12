#pragma once

#include "mmcore/utility/log/Log.h"

#define SPDLOG_EOL ""
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace megamol::core::utility::log {

class DefaultTarget : public Log::Target {
public:
    DefaultTarget(Log::UINT level = Log::LEVEL_ERROR);

    virtual ~DefaultTarget();

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
    std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> _stdout_sink;

    std::shared_ptr<spdlog::sinks::stderr_color_sink_mt> _stderr_sink;

    std::shared_ptr<spdlog::logger> _logger;
}; // end class DefaultTarget

} // end namespace megamol::core::utility::log
