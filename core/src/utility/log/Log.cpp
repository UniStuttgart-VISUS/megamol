/*
 * Log.cpp
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS).
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */


#include "mmcore/utility/log/Log.h"

#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "mmcore/utility/sys/SystemInformation.h"
#include "mmcore/utility/sys/Thread.h"
#include "vislib/Trace.h"
#include <climits>
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#endif /* _WIN32 */


#define TRACE_LVL vislib::Trace::LEVEL_INFO

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
    if (level > this->Level()) return;
    logger->info("{}|{}", level, msg);
}

#endif /* _WIN32 */
/*****************************************************************************/

/*
 * megamol::core::utility::log::Log::FileTarget::FileTarget
 */
megamol::core::utility::log::Log::FileTarget::FileTarget(const char *path, UINT level)
        : Target(level)/*, stream(NULL)*/ {
    logger = spdlog::get(std::string("default_file_") + std::string(path));
    if (logger == nullptr) {
        sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, true);
        sink->set_pattern(std_pattern);
        logger = std::make_shared<spdlog::logger>(std::string("default_file_") + std::string(path), sink);
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
    if (level > this->Level()) return;

    struct tm *timeStamp;
#ifdef _WIN32
#if (_MSC_VER >= 1400)
    struct tm __tS;
    timeStamp = &__tS;
    if (localtime_s(timeStamp, &time) != 0) {
        // timestamp error
        __tS.tm_hour = __tS.tm_min = __tS.tm_sec = 0;
    }
#else /* (_MSC_VER >= 1400) */
    timeStamp = localtime(&time);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
    timeStamp = localtime(&time);
#endif /* _WIN32 */

    std::string str(msg);
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    logger->info("{}:{}:{}|{}|{}|{}", timeStamp->tm_hour, timeStamp->tm_min, timeStamp->tm_sec,
        static_cast<unsigned int>(sid), level, str);
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
        vislib::StringA omg;
        omg.Format("%u offline log message%s omitted\n", this->omittedCnt,
            (this->omittedCnt == 1) ? "" : "s");
        target.Msg(Log::LEVEL_WARN, Log::CurrentTimeStamp(),
            Log::CurrentSourceID(), omg);
        if (remove) this->omittedCnt = 0;
    }
}


/*
 * megamol::core::utility::log::Log::OfflineTarget::SetBufferSize
 */
void megamol::core::utility::log::Log::OfflineTarget::SetBufferSize(unsigned int bufferSize) {
    OfflineMessage *om = this->msgs;
    this->msgs = new OfflineMessage[bufferSize];
    unsigned int cnt = vislib::math::Min(this->msgCnt, bufferSize);
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
const vislib::SmartPtr<megamol::core::utility::log::Log::Target>
megamol::core::utility::log::Log::StreamTarget::StdOut
    = new megamol::core::utility::log::Log::StreamTarget(std::cout);


/*
 * megamol::core::utility::log::Log::StreamTarget::StdErr
 */
const vislib::SmartPtr<megamol::core::utility::log::Log::Target>
megamol::core::utility::log::Log::StreamTarget::StdErr
    = new megamol::core::utility::log::Log::StreamTarget(std::cerr);


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
    return time(NULL);
}


/*
 * megamol::core::utility::log::Log::CurrentSourceID
 */
megamol::core::utility::log::Log::SourceID megamol::core::utility::log::Log::CurrentSourceID(void) {
    return static_cast<SourceID>(vislib::sys::Thread::CurrentID());
}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(UINT level, unsigned int msgbufsize)
        : mainTarget(new vislib::SmartPtr<Target>(
            new OfflineTarget(msgbufsize, level))),
        echoTarget(new vislib::SmartPtr<Target>(
            new OfflineTarget(msgbufsize, level))), autoflush(true) {
    VLTRACE(TRACE_LVL, "Log[%lu]::Log[%d]()\n",
        reinterpret_cast<unsigned long>(this), __LINE__);
    // Intentionally empty
}

/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(UINT level, const char *filename, bool addSuffix)
        : mainTarget(NULL), echoTarget(NULL), autoflush(true) {
    VLTRACE(TRACE_LVL, "Log[%lu]::Log[%d]()\n",
        reinterpret_cast<unsigned long>(this), __LINE__);
    this->SetLogFileName(filename, addSuffix);
}


/*
 * megamol::core::utility::log::Log::Log
 */
megamol::core::utility::log::Log::Log(const Log& source) : mainTarget(NULL),
        echoTarget(NULL), autoflush(true) {
    VLTRACE(TRACE_LVL, "Log[%lu]::Log[%d]()\n",
        reinterpret_cast<unsigned long>(this), __LINE__);
    *this = source;
}


/*
 * megamol::core::utility::log::Log::~Log
 */
megamol::core::utility::log::Log::~Log(void) {
    VLTRACE(TRACE_LVL, "Log[%lu]::~Log()\n",
        reinterpret_cast<unsigned long>(this));
    // Intentionally empty
}


/*
 * megamol::core::utility::log::Log::EchoOfflineMessages
 */
void megamol::core::utility::log::Log::EchoOfflineMessages(bool remove) {
    OfflineTarget *mot = this->mainTarget->DynamicCast<OfflineTarget>();
    OfflineTarget *eot = this->echoTarget->DynamicCast<OfflineTarget>();

    if ((mot == NULL) && (eot != NULL) && !this->mainTarget.IsNull()) {
        eot->Reecho(**this->mainTarget, remove);
    } else if ((mot != NULL) && (eot == NULL) && !this->echoTarget.IsNull()) {
        mot->Reecho(**this->echoTarget, remove);
    }
}


/*
 * megamol::core::utility::log::Log::FlushLog
 */
void megamol::core::utility::log::Log::FlushLog(void) {
    if (!this->mainTarget.IsNull() && !this->mainTarget->IsNull()) {
        this->mainTarget->operator->()->Flush();
    }
    if (!this->echoTarget.IsNull() && !this->echoTarget->IsNull()) {
        this->echoTarget->operator->()->Flush();
    }
}


/*
 * megamol::core::utility::log::Log::GetEchoLevel
 */
UINT megamol::core::utility::log::Log::GetEchoLevel(void) const {
    if (!this->echoTarget.IsNull() && !this->echoTarget->IsNull()) {
        return this->echoTarget->operator->()->Level();
    }
    return 0;
}


/*
 * megamol::core::utility::log::Log::GetLevel
 */
UINT megamol::core::utility::log::Log::GetLevel(void) const {
    if (!this->mainTarget.IsNull() && !this->mainTarget->IsNull()) {
        return this->mainTarget->operator->()->Level();
    }
    return 0;
}


/*
 * megamol::core::utility::log::Log::GetLogFileNameA
 */
vislib::StringA megamol::core::utility::log::Log::GetLogFileNameA(void) const {
    const FileTarget *ft = this->mainTarget->DynamicCast<FileTarget>();
    return (ft != NULL) ? vislib::StringA(ft->Filename()) : vislib::StringA::EMPTY;
}


/*
 * megamol::core::utility::log::Log::GetLogFileNameW
 */
vislib::StringW megamol::core::utility::log::Log::GetLogFileNameW(void) const {
    const FileTarget *ft = this->mainTarget->DynamicCast<FileTarget>();
    return (ft != NULL) ? ft->Filename() : vislib::StringW::EMPTY;
}


/*
 * megamol::core::utility::log::Log::GetOfflineMessageBufferSize
 */
unsigned int megamol::core::utility::log::Log::GetOfflineMessageBufferSize(void) const {
    const OfflineTarget *mot = this->mainTarget->DynamicCast<OfflineTarget>();
    const OfflineTarget *eot = this->echoTarget->DynamicCast<OfflineTarget>();

    if (mot != NULL) {
        return mot->BufferSize();
    } else if (eot != NULL) {
        return eot->BufferSize();
    }

    return 0;
}


/*
 * megamol::core::utility::log::Log::SetEchoLevel
 */
void megamol::core::utility::log::Log::SetEchoLevel(UINT level) {
    if (!this->echoTarget.IsNull() && !this->echoTarget->IsNull()) {
        this->echoTarget->operator->()->SetLevel(level);
    }
}


/*
 * megamol::core::utility::log::Log::SetEchoTarget
 */
void megamol::core::utility::log::Log::SetEchoTarget(
        vislib::SmartPtr<megamol::core::utility::log::Log::Target> target) {
    vislib::SmartPtr<Target> oet = *this->echoTarget;
    OfflineTarget *ot = oet.DynamicCast<OfflineTarget>();

    *this->echoTarget = target;
    if (!this->echoTarget->IsNull()) {
        if (!oet.IsNull()) {
            (*this->echoTarget)->SetLevel(oet->Level());
        }
        if (ot != NULL) {
            ot->Reecho(**this->echoTarget);
        }
    }
}


/*
 * megamol::core::utility::log::Log::SetLevel
 */
void megamol::core::utility::log::Log::SetLevel(UINT level) {
    if (!this->mainTarget.IsNull() && !this->mainTarget->IsNull()) {
        this->mainTarget->operator->()->SetLevel(level);
    }
}


/*
 * megamol::core::utility::log::Log::SetLogFileName
 */
bool megamol::core::utility::log::Log::SetLogFileName(const char *filename, bool addSuffix) {
    vislib::SmartPtr<Target> omt = *this->mainTarget;
    OfflineTarget *ot = omt.DynamicCast<OfflineTarget>();

    if (filename == NULL) {
        if (ot == NULL) {
            *this->mainTarget = new OfflineTarget(20U, omt->Level());
        }
    } else {
        vislib::StringA path(filename);
        if (addSuffix) {
            path += this->getFileNameSuffix();
        }
        *this->mainTarget = new FileTarget(path.PeekBuffer(), omt->Level());
        if (ot != NULL) {
            ot->Reecho(**this->mainTarget);
        }
    }
    // ot will be deleted by SFX of omt

    return true;
}


/*
 * megamol::core::utility::log::Log::SetMainTarget
 */
void megamol::core::utility::log::Log::SetMainTarget(
        vislib::SmartPtr<megamol::core::utility::log::Log::Target> target) {
    vislib::SmartPtr<Target> omt = *this->mainTarget;
    OfflineTarget *ot = omt.DynamicCast<OfflineTarget>();

    *this->mainTarget = target;
    (*this->mainTarget)->SetLevel(omt->Level());
    if (ot != NULL) {
        ot->Reecho(**this->mainTarget);
    }
}


/*
 * megamol::core::utility::log::Log::SetOfflineMessageBufferSize
 */
void megamol::core::utility::log::Log::SetOfflineMessageBufferSize(unsigned int msgbufsize) {
    OfflineTarget *mot = this->mainTarget->DynamicCast<OfflineTarget>();
    OfflineTarget *eot = this->echoTarget->DynamicCast<OfflineTarget>();

    if (mot != NULL) {
        mot->SetBufferSize(msgbufsize);
    }
    if (eot != NULL) {
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
void megamol::core::utility::log::Log::WriteError(const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(LEVEL_ERROR, Log::CurrentTimeStamp(),
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
 * megamol::core::utility::log::Log::WriteError
 */
void megamol::core::utility::log::Log::WriteError(int lvlOff, const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(
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
void megamol::core::utility::log::Log::WriteInfo(const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(LEVEL_INFO, Log::CurrentTimeStamp(),
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
 * megamol::core::utility::log::Log::WriteInfo
 */
void megamol::core::utility::log::Log::WriteInfo(int lvlOff, const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(
        static_cast<UINT>(static_cast<int>(LEVEL_INFO) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * megamol::core::utility::log::Log::WriteMessage
 */
void megamol::core::utility::log::Log::WriteMessage(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const vislib::StringA& msg) {
    if (!msg.EndsWith('\n')) {
        this->WriteMessage(level, time, sid, msg + "\n");
        return;
    }
    if (!this->mainTarget.IsNull() && !this->mainTarget->IsNull()) {
        (*this->mainTarget)->Msg(level, time, sid, msg);
        if (this->autoflush) {
            (*this->mainTarget)->Flush();
        }
    }
    if (!this->echoTarget.IsNull() && !this->echoTarget->IsNull()) {
        (*this->echoTarget)->Msg(level, time, sid, msg);
        if (this->autoflush) {
            (*this->echoTarget)->Flush();
        }
    }
}


/*
 * megamol::core::utility::log::Log::WriteMessage
 */
void megamol::core::utility::log::Log::WriteMessageVaA(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const char *fmt, va_list argptr) {
    vislib::StringA msg;
    if (fmt != NULL) {
        msg.FormatVa(fmt, argptr);
    } else {
        msg = "Empty log message\n";
    }
    this->WriteMessage(level, time, sid, msg);
}


/*
 * megamol::core::utility::log::Log::WriteMessage
 */
void megamol::core::utility::log::Log::WriteMessageVaW(UINT level,
        megamol::core::utility::log::Log::TimeStamp time, megamol::core::utility::log::Log::SourceID sid,
        const wchar_t *fmt, va_list argptr) {
    vislib::StringW msg;
    if (fmt != NULL) {
        msg.FormatVa(fmt, argptr);
    } else {
        msg = L"Empty log message\n";
    }
    // UTF8-Encoding may be better, but this is ok for now
    this->WriteMessage(level, time, sid, W2A(msg));
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
 * megamol::core::utility::log::Log::WriteMsg
 */
void megamol::core::utility::log::Log::WriteMsg(const UINT level, const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(level, Log::CurrentTimeStamp(),
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
void megamol::core::utility::log::Log::WriteWarn(const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(LEVEL_WARN, Log::CurrentTimeStamp(),
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
 * megamol::core::utility::log::Log::WriteWarn
 */
void megamol::core::utility::log::Log::WriteWarn(int lvlOff, const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(
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
vislib::StringA megamol::core::utility::log::Log::getFileNameSuffix(void) {
    vislib::StringA suffix;

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
    suffix.Format("_%s.%.8x_%.2d.%.2d.%.4d_%.2d.%.2d", 
        vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(),
        vislib::sys::Thread::CurrentID(), t->tm_mday, t->tm_mon, t->tm_year, 
        t->tm_hour, t->tm_min);

    return suffix;
}
