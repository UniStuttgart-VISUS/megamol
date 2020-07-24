/*
 * Log.cpp
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS).
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */


#include "mmcore/utility/log/Log.h"

#include "mmcore/utility/sys/SystemInformation.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#include <sstream>
#include <iomanip>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#endif /* _WIN32 */


const char megamol::core::utility::log::Log::std_pattern[3] = "%v";


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

/*****************************************************************************/
#ifdef _WIN32

/*
 * megamol::core::utility::log::Log::DebugOutputTarget::DebugOutputTarget
 */
megamol::core::utility::log::Log::DebugOutputTarget::DebugOutputTarget(UINT level)
        : Target(level) {
    logger = spdlog::get("default_debug");
    if (logger == nullptr) {
        sink = std::make_shared<spdlog::sinks::msvc_sink_mt>();
        sink->set_pattern(std_pattern);
        logger = std::make_shared<spdlog::logger>("default_debug", sink);
    }
}


/*
 * megamol::core::utility::log::Log::DebugOutputTarget::~DebugOutputTarget
 */
megamol::core::utility::log::Log::DebugOutputTarget::~DebugOutputTarget(void) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::Log::DebugOutputTarget::Msg
 */
void megamol::core::utility::log::Log::DebugOutputTarget::Msg(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const char *msg) {
    Msg(level, time, sid, std::string(msg));
}


/*
 * megamol::core::utility::log::Log::DebugOutputTarget::Msg
 */
void megamol::core::utility::log::Log::DebugOutputTarget::Msg(
    UINT level, TimeStamp time, SourceID sid, std::string const& msg) {
    if (level > this->Level()) return;
    logger->info("{}|{}", level, msg);
}

#endif /* _WIN32 */
/*****************************************************************************/

/*
 * megamol::core::utility::log::Log::FileTarget::FileTarget
 */
megamol::core::utility::log::Log::FileTarget::FileTarget(std::string const& path, UINT level)
        : Target(level) {
    logger = spdlog::get(std::string("default_file_") + path);
    if (logger == nullptr) {
        sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, true);
        sink->set_pattern(std_pattern);
        logger = std::make_shared<spdlog::logger>(std::string("default_file_") + path, sink);
    }
    filename = path;
}


/*
 * megamol::core::utility::log::Log::FileTarget::~FileTarget
 */
megamol::core::utility::log::Log::FileTarget::~FileTarget(void) {
    logger->flush();
}


/*
 * megamol::core::utility::log::Log::FileTarget::Flush
 */
void megamol::core::utility::log::Log::FileTarget::Flush(void) {
    logger->flush();
}


/*
 * megamol::core::utility::log::Log::FileTarget::Msg
 */
void megamol::core::utility::log::Log::FileTarget::Msg(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const char *msg) {
    Msg(level, time, sid, std::string(msg));
}


/*
 * megamol::core::utility::log::Log::FileTarget::Msg
 */
void megamol::core::utility::log::Log::FileTarget::Msg(UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, std::string const& msg) {
    if (level > this->Level()) return;

    struct tm* timeStamp;
#ifdef _WIN32
#    if (_MSC_VER >= 1400)
    struct tm __tS;
    timeStamp = &__tS;
    if (localtime_s(timeStamp, &time) != 0) {
        // timestamp error
        __tS.tm_hour = __tS.tm_min = __tS.tm_sec = 0;
    }
#    else  /* (_MSC_VER >= 1400) */
    timeStamp = localtime(&time);
#    endif /* (_MSC_VER >= 1400) */
#else      /* _WIN32 */
    timeStamp = localtime(&time);
#endif     /* _WIN32 */
    logger->info("{}:{}:{}|{}|{}|{}", timeStamp->tm_hour, timeStamp->tm_min, timeStamp->tm_sec, sid, level, msg);
}

/*****************************************************************************/

/*
 * megamol::core::utility::log::Log::OfflineTarget::OfflineTarget
 */
megamol::core::utility::log::Log::OfflineTarget::OfflineTarget(unsigned int bufferSize,
        UINT level) : Target(level), bufSize(bufferSize), msgCnt(0),
        msgs(new OfflineMessage[bufferSize]), omittedCnt(0) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::Log::OfflineTarget::~OfflineTarget
 */
megamol::core::utility::log::Log::OfflineTarget::~OfflineTarget(void) {
    ARY_SAFE_DELETE(this->msgs);
    this->bufSize = 0;
    this->msgCnt = 0;
}


/*
 * megamol::core::utility::log::Log::OfflineTarget::Msg
 */
void megamol::core::utility::log::Log::OfflineTarget::Msg(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const char *msg) {
    Msg(level, time, sid, std::string(msg));
}


/*
 * megamol::core::utility::log::Log::OfflineTarget::Msg
 */
void megamol::core::utility::log::Log::OfflineTarget::Msg(UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, std::string const& msg) {
    // Do not check the level. We store ALL messages
    if (this->msgCnt < this->bufSize) {
        this->msgs[this->msgCnt].level = level;
        this->msgs[this->msgCnt].time = time;
        this->msgs[this->msgCnt].sid = sid;
        this->msgs[this->msgCnt].msg = msg;
        this->msgCnt++;
    } else {
        this->omittedCnt++;
    }
}


/*
 * megamol::core::utility::log::Log::OfflineTarget::Reecho
 */
void megamol::core::utility::log::Log::OfflineTarget::Reecho(megamol::core::utility::log::Log::Target &target,
        bool remove) {
    for (unsigned int i = 0; i < this->msgCnt; i++) {
        target.Msg(this->msgs[i].level, this->msgs[i].time, this->msgs[i].sid,
            this->msgs[i].msg);
    }
    if (remove) this->msgCnt = 0;
    if (this->omittedCnt > 0) {
        std::stringstream omg;
        omg << this->omittedCnt << " offline log message" << ((this->omittedCnt == 1) ? "" : "s") << " omitted\n";
        target.Msg(Log::LEVEL_WARN, Log::CurrentTimeStamp(),
            Log::CurrentSourceID(), omg.str());
        if (remove) this->omittedCnt = 0;
    }
}


/*
 * megamol::core::utility::log::Log::OfflineTarget::SetBufferSize
 */
void megamol::core::utility::log::Log::OfflineTarget::SetBufferSize(unsigned int bufferSize) {
    OfflineMessage *om = this->msgs;
    this->msgs = new OfflineMessage[bufferSize];
    unsigned int cnt = std::min(this->msgCnt, bufferSize);
    this->omittedCnt += (this->msgCnt - cnt);
    this->bufSize = bufferSize;
    for (unsigned int i = 0; i < cnt; i++) {
        this->msgs[i].level = om[i].level;
        this->msgs[i].time = om[i].time;
        this->msgs[i].sid = om[i].sid;
        this->msgs[i].msg = om[i].msg;
    }
    delete[] om;
}

/*****************************************************************************/

/*
 * megamol::core::utility::log::Log::StreamTarget::StdOut
 */
const std::shared_ptr<megamol::core::utility::log::Log::Target>
megamol::core::utility::log::Log::StreamTarget::StdOut
    = std::make_shared<megamol::core::utility::log::Log::StreamTarget>(std::cout);


/*
 * megamol::core::utility::log::Log::StreamTarget::StdErr
 */
const std::shared_ptr<megamol::core::utility::log::Log::Target>
megamol::core::utility::log::Log::StreamTarget::StdErr
    = std::make_shared<megamol::core::utility::log::Log::StreamTarget>(std::cerr);


/*
 * megamol::core::utility::log::Log::StreamTarget::StreamTarget
 */
megamol::core::utility::log::Log::StreamTarget::StreamTarget(std::ostream& stream, UINT level)
        : Target(level) {
    logger = spdlog::get("default_stream");
    if (logger == nullptr) {
        sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(stream);
        sink->set_pattern(std_pattern);
        logger = std::make_shared<spdlog::logger>("default_stream", sink);
    }
}


/*
 * megamol::core::utility::log::Log::StreamTarget::~StreamTarget
 */
megamol::core::utility::log::Log::StreamTarget::~StreamTarget(void) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::Log::StreamTarget::Flush
 */
void megamol::core::utility::log::Log::StreamTarget::Flush(void) {
    logger->flush();
}


/*
 * megamol::core::utility::log::Log::StreamTarget::Msg
 */
void megamol::core::utility::log::Log::StreamTarget::Msg(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const char *msg) {
    Msg(level, time, sid, std::string(msg));
}


/*
 * megamol::core::utility::log::Log::StreamTarget::Msg
 */
void megamol::core::utility::log::Log::StreamTarget::Msg(UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, std::string const& msg) {
    if ((level > this->Level())) return;
    logger->info("{}|{}", level, msg);
}

/*****************************************************************************/

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


/*
 * megamol::core::utility::log::Log::CurrentTimeStamp
 */
megamol::core::utility::log::Log::TimeStamp megamol::core::utility::log::Log::CurrentTimeStamp(void) {
    return time(nullptr);
}


/*
 * megamol::core::utility::log::Log::CurrentSourceID
 */
megamol::core::utility::log::Log::SourceID megamol::core::utility::log::Log::CurrentSourceID(void) {
    return spdlog::details::os::thread_id();
}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(UINT level, unsigned int msgbufsize)
    : mainTarget(std::make_shared<OfflineTarget>(msgbufsize, level))
    , echoTarget(std::make_shared<OfflineTarget>(msgbufsize, level))
    , autoflush(true) {
    // Intentionally empty
}

/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(UINT level, const char *filename, bool addSuffix)
        : mainTarget(nullptr), echoTarget(nullptr), autoflush(true) {
    this->SetLogFileName(filename, addSuffix);
}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(const Log& source) : mainTarget(nullptr),
        echoTarget(nullptr), autoflush(true) {
    *this = source;
}


/*
 * megamol::core::utility::log::Log::~Log
 */
megamol::core::utility::log::Log::~Log(void) {
    // Intentionally empty
}


/*
 * megamol::core::utility::log::Log::EchoOfflineMessages
 */
void megamol::core::utility::log::Log::EchoOfflineMessages(bool remove) {
    std::shared_ptr<OfflineTarget> mot = std::dynamic_pointer_cast<OfflineTarget>(this->mainTarget);
    std::shared_ptr<OfflineTarget> eot = std::dynamic_pointer_cast<OfflineTarget>(this->echoTarget);

    if ((mot == nullptr) && (eot != nullptr) && this->mainTarget != nullptr) {
        eot->Reecho(*this->mainTarget, remove);
    } else if ((mot != nullptr) && (eot == nullptr) && this->echoTarget != nullptr) {
        mot->Reecho(*this->echoTarget, remove);
    }
}


/*
 * megamol::core::utility::log::Log::FlushLog
 */
void megamol::core::utility::log::Log::FlushLog(void) {
    if (this->mainTarget != nullptr) {
        this->mainTarget->Flush();
    }
    if (this->echoTarget != nullptr) {
        this->echoTarget->Flush();
    }
}


/*
 * megamol::core::utility::log::Log::GetEchoLevel
 */
UINT megamol::core::utility::log::Log::GetEchoLevel(void) const {
    if (this->echoTarget != nullptr) {
        return this->echoTarget->Level();
    }
    return 0;
}


/*
 * megamol::core::utility::log::Log::GetLevel
 */
UINT megamol::core::utility::log::Log::GetLevel(void) const {
    if (this->mainTarget != nullptr) {
        return this->mainTarget->Level();
    }
    return 0;
}


/*
 * megamol::core::utility::log::Log::GetLogFileNameA
 */
std::string megamol::core::utility::log::Log::GetLogFileNameA(void) const {
    const std::shared_ptr<FileTarget> ft = std::dynamic_pointer_cast<FileTarget>(this->mainTarget);
    return (ft != nullptr) ? ft->Filename() : std::string();
}


/*
 * megamol::core::utility::log::Log::GetOfflineMessageBufferSize
 */
unsigned int megamol::core::utility::log::Log::GetOfflineMessageBufferSize(void) const {
    const std::shared_ptr<OfflineTarget> mot = std::dynamic_pointer_cast<OfflineTarget>(this->mainTarget);
    const std::shared_ptr<OfflineTarget> eot = std::dynamic_pointer_cast<OfflineTarget>(this->echoTarget);

    if (mot != nullptr) {
        return mot->BufferSize();
    } else if (eot != nullptr) {
        return eot->BufferSize();
    }

    return 0;
}


/*
 * megamol::core::utility::log::Log::SetEchoLevel
 */
void megamol::core::utility::log::Log::SetEchoLevel(UINT level) {
    if (this->echoTarget != nullptr) {
        this->echoTarget->SetLevel(level);
    }
}


/*
 * megamol::core::utility::log::Log::SetEchoTarget
 */
void megamol::core::utility::log::Log::SetEchoTarget(
        std::shared_ptr<megamol::core::utility::log::Log::Target> target) {
    std::shared_ptr<Target> oet = this->echoTarget;
    std::shared_ptr<OfflineTarget> ot = std::dynamic_pointer_cast<OfflineTarget>(oet);

    this->echoTarget = target;
    if (this->echoTarget != nullptr) {
        if (oet != nullptr) {
            this->echoTarget->SetLevel(oet->Level());
        }
        if (ot != nullptr) {
            ot->Reecho(*this->echoTarget);
        }
    }
}


/*
 * megamol::core::utility::log::Log::SetLevel
 */
void megamol::core::utility::log::Log::SetLevel(UINT level) {
    if (this->mainTarget != nullptr) {
        this->mainTarget->SetLevel(level);
    }
}


/*
 * megamol::core::utility::log::Log::SetLogFileName
 */
bool megamol::core::utility::log::Log::SetLogFileName(const char *filename, bool addSuffix) {
    std::shared_ptr<Target> omt = this->mainTarget;
    std::shared_ptr<OfflineTarget> ot = std::dynamic_pointer_cast<OfflineTarget>(omt);

    if (filename == nullptr) {
        if (ot == nullptr) {
            this->mainTarget = std::make_shared<OfflineTarget>(20U, omt->Level());
        }
    } else {
        std::string path(filename);
        if (addSuffix) {
            path += this->getFileNameSuffix();
        }
        this->mainTarget = std::make_shared<FileTarget>(path, omt->Level());
        if (ot != nullptr) {
            ot->Reecho(*this->mainTarget);
        }
    }
    // ot will be deleted by SFX of omt

    return true;
}


/*
 * megamol::core::utility::log::Log::SetMainTarget
 */
void megamol::core::utility::log::Log::SetMainTarget(
        std::shared_ptr<megamol::core::utility::log::Log::Target> target) {
    std::shared_ptr<Target> omt = this->mainTarget;
    std::shared_ptr<OfflineTarget> ot = std::dynamic_pointer_cast<OfflineTarget>(omt);

    this->mainTarget = target;
    this->mainTarget->SetLevel(omt->Level());
    if (ot != nullptr) {
        ot->Reecho(*this->mainTarget);
    }
}


/*
 * megamol::core::utility::log::Log::SetOfflineMessageBufferSize
 */
void megamol::core::utility::log::Log::SetOfflineMessageBufferSize(unsigned int msgbufsize) {
    std::shared_ptr<OfflineTarget> mot = std::dynamic_pointer_cast<OfflineTarget>(this->mainTarget);
    std::shared_ptr<OfflineTarget> eot = std::dynamic_pointer_cast<OfflineTarget>(this->echoTarget);

    if (mot != nullptr) {
        mot->SetBufferSize(msgbufsize);
    }
    if (eot != nullptr) {
        eot->SetBufferSize(msgbufsize);
    }
}


/*
 * megamol::core::utility::log::Log::ShareTargetStorage
 */
void megamol::core::utility::log::Log::ShareTargetStorage(const megamol::core::utility::log::Log& master) {
    this->mainTarget = master.mainTarget;
    this->echoTarget = master.echoTarget;
}


/*
 * megamol::core::utility::log::Log::WriteError
 */
void megamol::core::utility::log::Log::WriteError(const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(LEVEL_ERROR, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteError
 */
void megamol::core::utility::log::Log::WriteError(int lvlOff, const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(
        static_cast<UINT>(static_cast<int>(LEVEL_ERROR) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteInfo
 */
void megamol::core::utility::log::Log::WriteInfo(const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(LEVEL_INFO, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteInfo
 */
void megamol::core::utility::log::Log::WriteInfo(int lvlOff, const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(
        static_cast<UINT>(static_cast<int>(LEVEL_INFO) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteMessage
 */
void megamol::core::utility::log::Log::WriteMessage(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const std::string& msg) {
    if (msg.empty()) return;
    if (msg.back() != '\n') {
        this->WriteMessage(level, time, sid, msg + "\n");
        return;
    }
    if (this->mainTarget != nullptr) {
        this->mainTarget->Msg(level, time, sid, msg);
        if (this->autoflush) {
            this->mainTarget->Flush();
        }
    }
    if (this->echoTarget != nullptr) {
        this->echoTarget->Msg(level, time, sid, msg);
        if (this->autoflush) {
            this->echoTarget->Flush();
        }
    }
}


/*
 * megamol::core::utility::log::Log::WriteMessage
 */
void megamol::core::utility::log::Log::WriteMessageVaA(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const char *fmt, va_list argptr) {
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
    this->WriteMessage(level, time, sid, msg);
}


/*
 * megamol::core::utility::log::Log::WriteMsg
 */
void megamol::core::utility::log::Log::WriteMsg(const UINT level, const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(level, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteWarn
 */
void megamol::core::utility::log::Log::WriteWarn(const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(LEVEL_WARN, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteWarn
 */
void megamol::core::utility::log::Log::WriteWarn(int lvlOff, const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(
        static_cast<UINT>(static_cast<int>(LEVEL_WARN) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::operator=
 */
megamol::core::utility::log::Log& megamol::core::utility::log::Log::operator=(const Log& rhs) {
    this->mainTarget = rhs.mainTarget;
    this->echoTarget = rhs.echoTarget;
    this->autoflush = rhs.autoflush;
    return *this;
}


/*
 * megamol::core::utility::log::Log::getFileNameSuffix
 */
std::string megamol::core::utility::log::Log::getFileNameSuffix(void) {
    TimeStamp timestamp = megamol::core::utility::log::Log::CurrentTimeStamp();
    struct tm *t;
#ifdef _WIN32
#if (_MSC_VER >= 1400)
    struct tm __tS;
    t = &__tS;
    if (localtime_s(t, &timestamp) != 0) {
        // timestamp error *** argh ***
        __tS.tm_hour = __tS.tm_min = __tS.tm_sec = 0;
    }
#else /* (_MSC_VER >= 1400) */
    t = localtime(&timestamp);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
    t = localtime(&timestamp);
#endif /* _WIN32 */

    t->tm_mon += 1;
    t->tm_year += 1900;

    std::stringstream buf;
    buf << "_" << vislib::sys::SystemInformation::ComputerNameA().PeekBuffer() << "." << CurrentSourceID() << "_"
        << std::setprecision(2) << t->tm_mday << "." << t->tm_mon << "." << std::setprecision(4) << t->tm_year << "_"
        << std::setprecision(2) << t->tm_hour << "." << t->tm_min;

    return buf.str();
}
