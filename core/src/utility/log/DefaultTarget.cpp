#include "mmcore/utility/log/DefaultTarget.h"


megamol::core::utility::log::DefaultTarget::DefaultTarget(Log::UINT level) : Target(level) {
    _logger = spdlog::get("default_megamol_logger");
    if (_logger == nullptr) {
        _stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        _stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        _stdout_sink->set_level(spdlog::level::level_enum::debug);
#ifdef _WIN32
        _stdout_sink->set_color(spdlog::level::level_enum::info, 15);
#else
        _stdout_sink->set_color(spdlog::level::level_enum::info, _stdout_sink->white);
#endif
        _stdout_sink->set_pattern(Log::std_pattern);
        _stderr_sink->set_level(spdlog::level::level_enum::err);
#ifdef _WIN32
        _stderr_sink->set_color(spdlog::level::level_enum::info, 15);
#else
        _stderr_sink->set_color(spdlog::level::level_enum::info, _stderr_sink->white);
#endif
        _stderr_sink->set_pattern(Log::std_pattern);
        std::array<spdlog::sink_ptr, 2> sinks = {_stdout_sink, _stderr_sink};

        _logger = std::make_shared<spdlog::logger>("default_megamol_logger", sinks.begin(), sinks.end());
    }
}


megamol::core::utility::log::DefaultTarget::~DefaultTarget() {}


void megamol::core::utility::log::DefaultTarget::Flush() {
    _logger->flush();
}


void megamol::core::utility::log::DefaultTarget::Msg(Log::UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, const char* msg) {
    Msg(level, time, sid, std::string(msg));
}


void megamol::core::utility::log::DefaultTarget::Msg(Log::UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, std::string const& msg) {
    if ((level > this->Level()))
        return;

    if (level >= 1 && level <= 99) {
        _logger->error("{}|{}", level, msg);
        return;
    }

    if (level >= 100 && level <= 199) {
        _logger->warn("{}|{}", level, msg);
        return;
    }

    _logger->info("{}|{}", level, msg);
}
