/**
 * MegaMol
 * Copyright (c) 2006, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/log/Log.h"

#include <iomanip>
#include <sstream>

#include <spdlog/details/os.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "vislib/sys/SystemInformation.h"

const char megamol::core::utility::log::Log::std_pattern[12] = "%^(%l)|%v%$";

const char megamol::core::utility::log::Log::file_pattern[22] = "[%Y-%m-%d %T] (%l)|%v";

const char megamol::core::utility::log::Log::logger_name[23] = "default_megamol_logger";

const char megamol::core::utility::log::Log::echo_logger_name[20] = "echo_megamol_logger";

/*****************************************************************************/

std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> create_default_sink() {
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    stdout_sink->set_level(spdlog::level::level_enum::debug);
#ifdef _WIN32
    stdout_sink->set_color(spdlog::level::level_enum::info, 15);
#else
    stdout_sink->set_color(spdlog::level::level_enum::info, stdout_sink->white);
#endif
    stdout_sink->set_pattern(megamol::core::utility::log::Log::std_pattern);
    return stdout_sink;
}

std::shared_ptr<spdlog::sinks::basic_file_sink_mt> create_file_sink(std::string const& path) {
    auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, true);
    sink->set_pattern(megamol::core::utility::log::Log::file_pattern);
    return sink;
}

/*****************************************************************************/


/*
 * __vl_log_defaultlog
 */
static megamol::core::utility::log::Log __vl_log_defaultlog;


/*
 * megamol::core::utility::log::Log::DefaultLog
 */
megamol::core::utility::log::Log& megamol::core::utility::log::Log::DefaultLog(__vl_log_defaultlog);


spdlog::level::level_enum translate_level(megamol::core::utility::log::Log::log_level level) {
    switch (level) {
    case megamol::core::utility::log::Log::log_level::none:
        return spdlog::level::off;
    case megamol::core::utility::log::Log::log_level::warn:
        return spdlog::level::warn;
    case megamol::core::utility::log::Log::log_level::error:
        return spdlog::level::err;
    case megamol::core::utility::log::Log::log_level::info:
    default:
        return spdlog::level::info;
    }
}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(log_level level, unsigned int msgbufsize) {
    // Intentionally empty
    auto logger = spdlog::get(logger_name);
    if (logger == nullptr) {
        auto sink = create_default_sink();
        sink->set_level(translate_level(level));
        logger = std::make_shared<spdlog::logger>(logger_name, sink);
        spdlog::register_logger(logger);
    }
}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(log_level level, const char* filename, bool addSuffix) {
    auto logger = spdlog::get(logger_name);
    if (logger == nullptr) {
        auto sink = create_default_sink();
        sink->set_level(translate_level(level));
        std::string path(filename);
        if (addSuffix) {
            path += this->getFileNameSuffix();
        }
        auto file_sink = create_file_sink(path);
        file_sink->set_level(translate_level(level));
        std::array<spdlog::sink_ptr, 2> sinks = {sink, file_sink};
        logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
        spdlog::register_logger(logger);
    }
}


/*
 * megamol::core::utility::log::Log::~Log
 */
megamol::core::utility::log::Log::~Log() {
    // Intentionally empty
}


/*
 * megamol::core::utility::log::Log::FlushLog
 */
void megamol::core::utility::log::Log::FlushLog() {
    auto logger = spdlog::get(logger_name);
    if (logger)
        logger->flush();
    logger = spdlog::get(echo_logger_name);
    if (logger)
        logger->flush();
}


/*
 * megamol::core::utility::log::Log::SetEchoLevel
 */
void megamol::core::utility::log::Log::SetEchoLevel(log_level level) {
    auto logger = spdlog::get(echo_logger_name);
    if (logger) {
        for (auto& sink : logger->sinks()) {
            sink->set_level(translate_level(level));
        }
    }
}


std::size_t megamol::core::utility::log::Log::AddEchoTarget(std::shared_ptr<spdlog::sinks::sink> target) {
    auto logger = spdlog::get(echo_logger_name);
    if (logger) {
        logger->sinks().push_back(target);
        return logger->sinks().size() - 1;
    } else {
        logger = std::make_shared<spdlog::logger>(echo_logger_name, target);
        spdlog::register_logger(logger);
        return 0;
    }
}


void megamol::core::utility::log::Log::RemoveEchoTarget(std::size_t idx) {
    auto logger = spdlog::get(echo_logger_name);
    if (logger) {
        auto& sinks = logger->sinks();
        if (idx < sinks.size()) {
            sinks[idx] = std::make_shared<spdlog::sinks::null_sink_mt>();
        }
    }
}


/*
 * megamol::core::utility::log::Log::SetLevel
 */
void megamol::core::utility::log::Log::SetLevel(log_level level) {
    auto logger = spdlog::get(logger_name);
    if (logger) {
        for (auto& sink : logger->sinks()) {
            sink->set_level(translate_level(level));
        }
    }
}


/*
 * megamol::core::utility::log::Log::AddFileTarget
 */
void megamol::core::utility::log::Log::AddFileTarget(const char* filename, bool addSuffix, log_level level) {
    if (filename != nullptr) {
        std::string path(filename);
        if (addSuffix) {
            path += this->getFileNameSuffix();
        }
        auto logger = spdlog::get(logger_name);
        if (logger) {
            auto sink = create_file_sink(path);
            sink->set_level(translate_level(level));
            logger->sinks().push_back(sink);
        }
    }
}


/*
 * megamol::core::utility::log::Log::operator=
 */
megamol::core::utility::log::Log& megamol::core::utility::log::Log::operator=(const Log& rhs) {
    return *this;
}


/*
 * megamol::core::utility::log::Log::getFileNameSuffix
 */
std::string megamol::core::utility::log::Log::getFileNameSuffix() {
    auto t = spdlog::details::os::localtime();

    t.tm_mon += 1;
    t.tm_year += 1900;

    std::stringstream buf;
    buf << "_" << vislib::sys::SystemInformation::ComputerNameA().PeekBuffer() << "."
        << spdlog::details::os::thread_id() << "_" << std::setprecision(2) << t.tm_mday << "." << t.tm_mon << "."
        << std::setprecision(4) << t.tm_year << "_" << std::setprecision(2) << t.tm_hour << "." << t.tm_min;

    return buf.str();
}


inline static bool iequals(const std::string& one, const std::string& other) {

    return ((one.size() == other.size()) &&
            std::equal(one.begin(), one.end(), other.begin(),
                [](const char& c1, const char& c2) { return (c1 == c2 || std::toupper(c1) == std::toupper(c2)); }));
}


megamol::core::utility::log::Log::log_level megamol::core::utility::log::Log::ParseLevelAttribute(
    const std::string attr, megamol::core::utility::log::Log::log_level def) {
    megamol::core::utility::log::Log::log_level retval = megamol::core::utility::log::Log::log_level::info;
    if (iequals(attr, "error")) {
        retval = megamol::core::utility::log::Log::log_level::error;
    } else if (iequals(attr, "(error)")) {
        retval = megamol::core::utility::log::Log::log_level::error;
    } else if (iequals(attr, "warn")) {
        retval = megamol::core::utility::log::Log::log_level::warn;
    } else if (iequals(attr, "(warn)")) {
        retval = megamol::core::utility::log::Log::log_level::warn;
    } else if (iequals(attr, "warning")) {
        retval = megamol::core::utility::log::Log::log_level::warn;
    } else if (iequals(attr, "(warning)")) {
        retval = megamol::core::utility::log::Log::log_level::warn;
    } else if (iequals(attr, "info")) {
        retval = megamol::core::utility::log::Log::log_level::info;
    } else if (iequals(attr, "(info)")) {
        retval = megamol::core::utility::log::Log::log_level::info;
    } else if (iequals(attr, "none")) {
        retval = megamol::core::utility::log::Log::log_level::none;
    } else if (iequals(attr, "null")) {
        retval = megamol::core::utility::log::Log::log_level::none;
    } else if (iequals(attr, "zero")) {
        retval = megamol::core::utility::log::Log::log_level::none;
    } else if (iequals(attr, "all")) {
        retval = megamol::core::utility::log::Log::log_level::all;
    } else if (iequals(attr, "*")) {
        retval = megamol::core::utility::log::Log::log_level::all;
    }
    return retval;
}
