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

#include "mmcore/utility/log/OfflineTarget.h"
#include "mmcore/utility/log/FileTarget.h"

#include "spdlog/details/os.h"
#include "spdlog/spdlog.h"


const char megamol::core::utility::log::Log::std_pattern[7] = "%^%v%$";


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
    , fileTarget(std::make_shared<OfflineTarget>(msgbufsize, level))
    , autoflush(true) {
    // Intentionally empty
}

/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(UINT level, const char *filename, bool addSuffix)
        : mainTarget(nullptr), echoTarget(nullptr), fileTarget(nullptr), autoflush(true) {
    this->SetLogFileName(filename, addSuffix);
}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(const Log& source) : mainTarget(nullptr),
        echoTarget(nullptr), fileTarget(nullptr), autoflush(true) {
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
    if (this->fileTarget != nullptr) {
        this->fileTarget->Flush();
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
 * megamol::core::utility::log::Log::GetFileLevel
 */
UINT megamol::core::utility::log::Log::GetFileLevel(void) const {
    if (this->fileTarget != nullptr) {
        return this->fileTarget->Level();
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
    const std::shared_ptr<OfflineTarget> fot = std::dynamic_pointer_cast<OfflineTarget>(this->fileTarget);

    if (mot != nullptr) {
        return mot->BufferSize();
    } else if (eot != nullptr) {
        return eot->BufferSize();
    } else if (fot != nullptr) {
        return fot->BufferSize();
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
 * megamol::core::utility::log::Log::SetFileLevel
 */
void megamol::core::utility::log::Log::SetFileLevel(UINT level) {
    if (this->fileTarget != nullptr) {
        this->fileTarget->SetLevel(level);
    }
}



/*
 * megamol::core::utility::log::Log::SetLogFileName
 */
bool megamol::core::utility::log::Log::SetLogFileName(const char *filename, bool addSuffix) {
    std::shared_ptr<Target> oft = this->fileTarget;
    std::shared_ptr<OfflineTarget> ot = std::dynamic_pointer_cast<OfflineTarget>(oft);

    if (filename != nullptr) {
        std::string path(filename);
        if (addSuffix) {
            path += this->getFileNameSuffix();
        }
        this->fileTarget = std::make_shared<FileTarget>(path, oft->Level());
        if (ot != nullptr) {
            ot->Reecho(*this->fileTarget);
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
    std::shared_ptr<OfflineTarget> fot = std::dynamic_pointer_cast<OfflineTarget>(this->fileTarget);

    if (mot != nullptr) {
        mot->SetBufferSize(msgbufsize);
    }
    if (eot != nullptr) {
        eot->SetBufferSize(msgbufsize);
    }
    if (fot != nullptr) {
        fot->SetBufferSize(msgbufsize);
    }
}


/*
 * megamol::core::utility::log::Log::ShareTargetStorage
 */
void megamol::core::utility::log::Log::ShareTargetStorage(const megamol::core::utility::log::Log& master) {
    this->mainTarget = master.mainTarget;
    this->echoTarget = master.echoTarget;
    this->fileTarget = master.fileTarget;
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
    // remove newline at end because spdlog and other log targets already add newlines
    if (msg.back() == '\n') {
        this->WriteMessage(level, time, sid, msg.substr(0, msg.size()-1));
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
    if (this->fileTarget != nullptr) {
        this->fileTarget->Msg(level, time, sid, msg);
        if (this->autoflush) {
            this->fileTarget->Flush();
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
    this->fileTarget = rhs.fileTarget;
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

inline static bool iequals(const std::string& one, const std::string& other) {

    return ((one.size() == other.size()) &&
            std::equal(one.begin(), one.end(), other.begin(),
                [](const char& c1, const char& c2) { return (c1 == c2 || std::toupper(c1) == std::toupper(c2)); }));
}

UINT megamol::core::utility::log::Log::ParseLevelAttribute(const std::string attr) {

    UINT retval = megamol::core::utility::log::Log::LEVEL_ERROR;
    if (iequals(attr, "error")) {
        retval = megamol::core::utility::log::Log::LEVEL_ERROR;
    }
    else if (iequals(attr, "warn")) {
        retval = megamol::core::utility::log::Log::LEVEL_WARN;
    }
    else if (iequals(attr, "warning")) {
        retval = megamol::core::utility::log::Log::LEVEL_WARN;
    }
    else if (iequals(attr, "info")) {
        retval = megamol::core::utility::log::Log::LEVEL_INFO;
    }
    else if (iequals(attr, "none")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    }
    else if (iequals(attr, "null")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    }
    else if (iequals(attr, "zero")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    }
    else if (iequals(attr, "all")) {
        retval = megamol::core::utility::log::Log::LEVEL_ALL;
    }
    else if (iequals(attr, "*")) {
        retval = megamol::core::utility::log::Log::LEVEL_ALL;
    }
    else {
        retval = std::stoi(attr);
        // dont catch stoi exceptions
        // let exception be handled by the one who called me
    }
    return retval;
}
