/*
 * Log.cpp
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS).
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */


#include "mmcore/utility/log/Log.h"

#include "vislib/sys/SystemInformation.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#endif /* _WIN32 */

#include "mmcore/utility/log/DefaultTarget.h"
#include "mmcore/utility/log/FileTarget.h"

#include "spdlog/details/os.h"
#include "spdlog/spdlog.h"


const char megamol::core::utility::log::Log::std_pattern[12] = "%^(%l)|%v%$";

const char megamol::core::utility::log::Log::file_pattern[22] = "[%Y-%m-%d %T] (%l)|%v";

const char megamol::core::utility::log::Log::logger_name[23] = "default_megamol_logger";

/*****************************************************************************/

std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> create_default_sink() {
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    stdout_sink->set_level(spdlog::level::level_enum::debug);
#ifdef _WIN32
    stdout_sink->set_color(spdlog::level::level_enum::info, 15);
#else
    stdout_sink->set_color(spdlog::level::level_enum::info, _stdout_sink->white);
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
 * megamol::core::utility::log::Log::Target::~Target
 */
megamol::core::utility::log::Log::Target::~Target(void) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::Log::Target::Target
 */
megamol::core::utility::log::Log::Target::Target(UINT level) : level(level) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::Log::Target::Flush
 */
void megamol::core::utility::log::Log::Target::Flush(void) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::Log::LEVEL_ALL
 */
const UINT megamol::core::utility::log::Log::LEVEL_ALL = UINT_MAX;


/*
 * megamol::core::utility::log::Log::LEVEL_ERROR
 */
const UINT megamol::core::utility::log::Log::LEVEL_ERROR = 1;


/*
 * megamol::core::utility::log::Log::LEVEL_INFO
 */
const UINT megamol::core::utility::log::Log::LEVEL_INFO = 200;


/*
 * megamol::core::utility::log::Log::LEVEL_NONE
 */
const UINT megamol::core::utility::log::Log::LEVEL_NONE = 0;


/*
 * megamol::core::utility::log::Log::LEVEL_WARN
 */
const UINT megamol::core::utility::log::Log::LEVEL_WARN = 100;


/*
 * __vl_log_defaultlog
 */
static megamol::core::utility::log::Log __vl_log_defaultlog;


/*
 * megamol::core::utility::log::Log::DefaultLog
 */
megamol::core::utility::log::Log& megamol::core::utility::log::Log::DefaultLog(__vl_log_defaultlog);


///*
// * megamol::core::utility::log::Log::CurrentTimeStamp
// */
//megamol::core::utility::log::Log::TimeStamp megamol::core::utility::log::Log::CurrentTimeStamp(void) {
//    return time(nullptr);
//}
//
//
///*
// * megamol::core::utility::log::Log::CurrentSourceID
// */
//megamol::core::utility::log::Log::SourceID megamol::core::utility::log::Log::CurrentSourceID(void) {
//    return spdlog::details::os::thread_id();
//}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(UINT level, unsigned int msgbufsize)
        : mt_level_(level)
        , et_level_(level)
        , ft_level_(level)
        , autoflush(true) {
    // Intentionally empty
    auto logger = spdlog::get(logger_name);
    if (logger == nullptr) {
        auto sink = create_default_sink();
        logger = std::make_shared<spdlog::logger>(logger_name, sink);
        spdlog::register_logger(logger);
    }
}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(UINT level, const char* filename, bool addSuffix)
        : mt_level_(level)
        , et_level_(level)
        , ft_level_(level)
        , autoflush(true) {
    auto logger = spdlog::get(logger_name);
    if (logger == nullptr) {
        auto sink = create_default_sink();
        std::string path(filename);
        if (addSuffix) {
            path += this->getFileNameSuffix();
        }
        auto file_sink = create_file_sink(path);
        std::array<spdlog::sink_ptr, 2> sinks = {sink, file_sink};
        logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
        spdlog::register_logger(logger);
    }
}


/*
 * megamol::core::utility::log::Log::~Log
 */
megamol::core::utility::log::Log::~Log(void) {
    // Intentionally empty
}


/*
 * megamol::core::utility::log::Log::FlushLog
 */
void megamol::core::utility::log::Log::FlushLog(void) {
    auto logger = spdlog::get(logger_name);
    if (logger)
        logger->flush();
}


/*
 * megamol::core::utility::log::Log::GetEchoLevel
 */
UINT megamol::core::utility::log::Log::GetEchoLevel(void) const {
    return et_level_;
}


/*
 * megamol::core::utility::log::Log::GetLevel
 */
UINT megamol::core::utility::log::Log::GetLevel(void) const {
    return mt_level_;
}


/*
 * megamol::core::utility::log::Log::GetFileLevel
 */
UINT megamol::core::utility::log::Log::GetFileLevel(void) const {
    return ft_level_;
}


/*
 * megamol::core::utility::log::Log::GetLogFileNameA
 */
std::string megamol::core::utility::log::Log::GetLogFileNameA(void) const {
    /*const std::shared_ptr<FileTarget> ft = std::dynamic_pointer_cast<FileTarget>(this->mainTarget);
    return (ft != nullptr) ? ft->Filename() : std::string();*/

    return std::string();
}


/*
 * megamol::core::utility::log::Log::SetEchoLevel
 */
void megamol::core::utility::log::Log::SetEchoLevel(UINT level) {
    et_level_ = level;
}


/*
 * megamol::core::utility::log::Log::SetEchoTarget
 */
void megamol::core::utility::log::Log::SetEchoTarget(std::shared_ptr<megamol::core::utility::log::Log::Target> target) {
    /*std::shared_ptr<Target> oet = this->echoTarget;

    this->echoTarget = target;
    if (this->echoTarget != nullptr) {
        if (oet != nullptr) {
            this->echoTarget->SetLevel(oet->Level());
        }
    }*/
}


/*
 * megamol::core::utility::log::Log::SetLevel
 */
void megamol::core::utility::log::Log::SetLevel(UINT level) {
    mt_level_ = level;
}


/*
 * megamol::core::utility::log::Log::SetFileLevel
 */
void megamol::core::utility::log::Log::SetFileLevel(UINT level) {
    ft_level_ = level;
}


/*
 * megamol::core::utility::log::Log::SetLogFileName
 */
bool megamol::core::utility::log::Log::SetLogFileName(const char* filename, bool addSuffix) {
    if (filename != nullptr) {
        std::string path(filename);
        if (addSuffix) {
            path += this->getFileNameSuffix();
        }
        auto logger = spdlog::get(logger_name);
        if (logger) {
            auto sink = create_file_sink(path);
            logger->sinks().push_back(sink);
        }
    }

    return true;
}


/*
 * megamol::core::utility::log::Log::SetMainTarget
 */
void megamol::core::utility::log::Log::SetMainTarget(std::shared_ptr<megamol::core::utility::log::Log::Target> target) {
    //std::shared_ptr<Target> omt = this->mainTarget;

    //this->mainTarget = target;
    //this->mainTarget->SetLevel(omt->Level());
}


/*
 * megamol::core::utility::log::Log::WriteError
 */
void megamol::core::utility::log::Log::WriteError(const char* fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->writeMessageVaA(log_level::error, fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteError
 */
void megamol::core::utility::log::Log::WriteError(int lvlOff, const char* fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->writeMessageVaA(log_level::error, fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteInfo
 */
void megamol::core::utility::log::Log::WriteInfo(const char* fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->writeMessageVaA(log_level::info, fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteInfo
 */
void megamol::core::utility::log::Log::WriteInfo(int lvlOff, const char* fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->writeMessageVaA(log_level::info, fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::writeMessage
 */
void megamol::core::utility::log::Log::writeMessage(log_level level, const std::string& msg) {
    auto logger = spdlog::get(logger_name);
    if (!logger)
        return;
    if (msg.empty())
        return;
    // remove newline at end because spdlog and other log targets already add newlines
    if (msg.back() == '\n') {
        this->writeMessage(level, msg.substr(0, msg.size() - 1));
        return;
    }

    switch (level) {
    case log_level::error:
        logger->error("{}", msg);
        break;
    case log_level::warn:
        logger->warn("{}", msg);
        break;
    case log_level::info:
    default:
        logger->info("{}", msg);
    }
}


/*
 * megamol::core::utility::log::Log::writeMessage
 */
void megamol::core::utility::log::Log::writeMessageVaA(log_level level, const char* fmt, va_list argptr) {
    std::string msg;
    if (fmt != nullptr) {
        va_list tmp;
        va_copy(tmp, argptr);
        msg.resize(1ull + std::vsnprintf(nullptr, 0, fmt, argptr));
        std::vsnprintf(msg.data(), msg.size(), fmt, tmp);
        va_end(tmp);
        msg.resize(msg.size() - 1);
    } else {
        msg = "Empty log message\n";
    }
    this->writeMessage(level, msg);
}


/*
 * megamol::core::utility::log::Log::WriteMsg
 */
void megamol::core::utility::log::Log::WriteMsg(const UINT level, const char* fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->writeMessageVaA(log_level::info, fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteWarn
 */
void megamol::core::utility::log::Log::WriteWarn(const char* fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->writeMessageVaA(log_level::warn, fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteWarn
 */
void megamol::core::utility::log::Log::WriteWarn(int lvlOff, const char* fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->writeMessageVaA(log_level::warn, fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::operator=
 */
megamol::core::utility::log::Log& megamol::core::utility::log::Log::operator=(const Log& rhs) {
    this->mt_level_ = rhs.mt_level_;
    this->et_level_ = rhs.et_level_;
    this->ft_level_ = rhs.ft_level_;
    this->autoflush = rhs.autoflush;
    return *this;
}


/*
 * megamol::core::utility::log::Log::getFileNameSuffix
 */
std::string megamol::core::utility::log::Log::getFileNameSuffix(void) {
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


UINT megamol::core::utility::log::Log::ParseLevelAttribute(const std::string attr) {

    UINT retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::error);
    if (iequals(attr, "error")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::error);
    } else if (iequals(attr, "warn")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::warn);
    } else if (iequals(attr, "warning")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::warn);
    } else if (iequals(attr, "info")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::info);
    } else if (iequals(attr, "none")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::info);
    } else if (iequals(attr, "null")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::info);
    } else if (iequals(attr, "zero")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::info);
    } else if (iequals(attr, "all")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::info);
    } else if (iequals(attr, "*")) {
        retval = static_cast<UINT>(megamol::core::utility::log::Log::log_level::info);
    } else {
        retval = std::stoi(attr);
        // dont catch stoi exceptions
        // let exception be handled by the one who called me
    }
    return retval;
}
