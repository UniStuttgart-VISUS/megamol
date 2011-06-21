/*
 * Log.cpp
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/Log.h"

#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemInformation.h"
#include "vislib/Thread.h"
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
#include "vislib/vislibsymbolimportexport.inl"


#define TRACE_LVL Trace::LEVEL_INFO


/*****************************************************************************/

/*
 * vislib::sys::Log::Target::~Target
 */
vislib::sys::Log::Target::~Target(void) {
    // intentionally empty
}


/*
 * vislib::sys::Log::Target::Target
 */
vislib::sys::Log::Target::Target(UINT level) : level(level) {
    // intentionally empty
}


/*
 * vislib::sys::Log::Target::Flush
 */
void vislib::sys::Log::Target::Flush(void) {
    // intentionally empty
}

/*****************************************************************************/
#ifdef _WIN32

/*
 * vislib::sys::Log::DebugOutputTarget::DebugOutputTarget
 */
vislib::sys::Log::DebugOutputTarget::DebugOutputTarget(UINT level)
        : Target(level) {
    // intentionally empty
}


/*
 * vislib::sys::Log::DebugOutputTarget::~DebugOutputTarget
 */
vislib::sys::Log::DebugOutputTarget::~DebugOutputTarget(void) {
    // intentionally empty
}


/*
 * vislib::sys::Log::DebugOutputTarget::Msg
 */
void vislib::sys::Log::DebugOutputTarget::Msg(UINT level,
        vislib::sys::Log::TimeStamp time, vislib::sys::Log::SourceID sid,
        const char *msg) {
    if (level > this->Level()) return;
    char tmp[21];
    tmp[0] = tmp[20] = 0;
    _snprintf_s(tmp, 20, 20, "%.4d|", level);
    ::OutputDebugStringA(tmp);
    ::OutputDebugStringA(msg);
}

#endif /* _WIN32 */
/*****************************************************************************/

/*
 * vislib::sys::Log::FileTarget::FileTarget
 */
vislib::sys::Log::FileTarget::FileTarget(const char *path, UINT level)
        : Target(level), stream(NULL) {

    int newFile = -1;
#ifdef _WIN32
#if (_MSC_VER >= 1400)
    if (_sopen_s(&newFile, path, _O_APPEND | _O_CREAT | _O_TEXT 
            | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE) != 0) {
        newFile = -1;
    }
#else /* (_MSC_VER >= 1400) */
    newFile = _sopen(path, _O_APPEND | _O_CREAT | _O_TEXT 
        | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
    newFile = open(path, O_APPEND | O_CREAT | O_LARGEFILE 
        | O_WRONLY, S_IRWXU | S_IRWXG);
#endif /* _WIN32 */
    this->stream = (newFile == -1) ? NULL : 
#ifdef _WIN32
        _fdopen(newFile, "ac");
#else /* _WIN32 */
        fdopen(newFile, "a");
#endif /* _WIN32 */

    if (this->stream != NULL) {
        this->filename = path;
    } else {
        this->filename.Clear();
    }

}


/*
 * vislib::sys::Log::FileTarget::FileTarget
 */
vislib::sys::Log::FileTarget::FileTarget(const wchar_t *path, UINT level)
        : Target(level), stream(NULL) {

    int newFile = -1;
#ifdef _WIN32
#if (_MSC_VER >= 1400)
    if (_wsopen_s(&newFile, path, _O_APPEND | _O_CREAT | _O_TEXT 
            | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE) != 0) {
        newFile = -1;
    }
#else /* (_MSC_VER >= 1400) */
    newFile = _wsopen(path, _O_APPEND | _O_CREAT | _O_TEXT 
        | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
    newFile = open(W2A(path), O_APPEND | O_CREAT | O_LARGEFILE 
        | O_WRONLY, S_IRWXU | S_IRWXG);
#endif /* _WIN32 */

    this->stream = (newFile == -1) ? NULL : 
#ifdef _WIN32
        _fdopen(newFile, "ac");
#else /* _WIN32 */
        fdopen(newFile, "a");
#endif /* _WIN32 */

    if (this->stream != NULL) {
        this->filename = path;
    } else {
        this->filename.Clear();
    }

}


/*
 * vislib::sys::Log::FileTarget::~FileTarget
 */
vislib::sys::Log::FileTarget::~FileTarget(void) {
    if (this->stream != NULL) {
        fflush(this->stream);
        fclose(this->stream);
        this->stream = NULL;
    }
}


/*
 * vislib::sys::Log::FileTarget::Flush
 */
void vislib::sys::Log::FileTarget::Flush(void) {
    if (this->stream != NULL) {
        fflush(this->stream);
    }
}


/*
 * vislib::sys::Log::FileTarget::Msg
 */
void vislib::sys::Log::FileTarget::Msg(UINT level,
        vislib::sys::Log::TimeStamp time, vislib::sys::Log::SourceID sid,
        const char *msg) {
    if ((this->stream == NULL) || (level > this->Level())) return;

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

    fprintf(this->stream, "%2d:%.2d:%.2d|%8x|%4u|%s", 
        timeStamp->tm_hour, timeStamp->tm_min, timeStamp->tm_sec, 
        static_cast<unsigned int>(sid), level, msg);
}

/*****************************************************************************/

/*
 * vislib::sys::Log::OfflineTarget::OfflineTarget
 */
vislib::sys::Log::OfflineTarget::OfflineTarget(unsigned int bufferSize,
        UINT level) : Target(level), bufSize(bufferSize), msgCnt(0),
        msgs(new OfflineMessage[bufferSize]), omittedCnt(0) {
    // intentionally empty
}


/*
 * vislib::sys::Log::OfflineTarget::~OfflineTarget
 */
vislib::sys::Log::OfflineTarget::~OfflineTarget(void) {
    ARY_SAFE_DELETE(this->msgs);
    this->bufSize = 0;
    this->msgCnt = 0;
}


/*
 * vislib::sys::Log::OfflineTarget::Msg
 */
void vislib::sys::Log::OfflineTarget::Msg(UINT level,
        vislib::sys::Log::TimeStamp time, vislib::sys::Log::SourceID sid,
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
 * vislib::sys::Log::OfflineTarget::Reecho
 */
void vislib::sys::Log::OfflineTarget::Reecho(vislib::sys::Log::Target &target,
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
 * vislib::sys::Log::OfflineTarget::SetBufferSize
 */
void vislib::sys::Log::OfflineTarget::SetBufferSize(unsigned int bufferSize) {
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
 * vislib::sys::Log::RedirectTarget::RedirectTarget
 */
vislib::sys::Log::RedirectTarget::RedirectTarget(vislib::sys::Log *log,
        UINT level) : Target(level), log(log) {
    // intentionally empty
}


/*
 * vislib::sys::Log::RedirectTarget::~RedirectTarget
 */
vislib::sys::Log::RedirectTarget::~RedirectTarget(void) {
    // intentionally empty
}


/*
 * vislib::sys::Log::RedirectTarget::Msg
 */
void vislib::sys::Log::RedirectTarget::Msg(UINT level,
        vislib::sys::Log::TimeStamp time, vislib::sys::Log::SourceID sid,
        const char *msg) {
    // Do not check the level. We redirect ALL messages
    if (this->log == NULL) return;
    this->log->WriteMessage(level, time, sid, msg);
}

/*****************************************************************************/

/*
 * vislib::sys::Log::StreamTarget::StdOut
 */
const vislib::SmartPtr<vislib::sys::Log::Target>
vislib::sys::Log::StreamTarget::StdOut
    = new vislib::sys::Log::StreamTarget(stdout);


/*
 * vislib::sys::Log::StreamTarget::StdErr
 */
const vislib::SmartPtr<vislib::sys::Log::Target>
vislib::sys::Log::StreamTarget::StdErr
    = new vislib::sys::Log::StreamTarget(stderr);


/*
 * vislib::sys::Log::StreamTarget::StreamTarget
 */
vislib::sys::Log::StreamTarget::StreamTarget(FILE *stream, UINT level)
        : Target(level), stream(stream) {
    // intentionally empty
}


/*
 * vislib::sys::Log::StreamTarget::~StreamTarget
 */
vislib::sys::Log::StreamTarget::~StreamTarget(void) {
    // intentionally empty
}


/*
 * vislib::sys::Log::StreamTarget::Flush
 */
void vislib::sys::Log::StreamTarget::Flush(void) {
    if (this->stream != NULL) {
        fflush(this->stream);
    }
}


/*
 * vislib::sys::Log::StreamTarget::Msg
 */
void vislib::sys::Log::StreamTarget::Msg(UINT level,
        vislib::sys::Log::TimeStamp time, vislib::sys::Log::SourceID sid,
        const char *msg) {
    if ((this->stream == NULL) || (level > this->Level())) return;
    fprintf(this->stream, "%.4u|%s", level, msg);
}

/*****************************************************************************/

/*
 * vislib::sys::Log::LEVEL_ALL
 */
const UINT vislib::sys::Log::LEVEL_ALL = UINT_MAX;


/*
 * vislib::sys::Log::LEVEL_ERROR
 */
const UINT vislib::sys::Log::LEVEL_ERROR = 1;


/*
 * vislib::sys::Log::LEVEL_INFO
 */
const UINT vislib::sys::Log::LEVEL_INFO = 200;


/*
 * vislib::sys::Log::LEVEL_NONE
 */
const UINT vislib::sys::Log::LEVEL_NONE = 0;


/*
 * vislib::sys::Log::LEVEL_WARN
 */
const UINT vislib::sys::Log::LEVEL_WARN = 100;


/*
 * __vl_log_defaultlog
 */
VISLIB_STATICSYMBOL vislib::sys::Log __vl_log_defaultlog;


/*
 * vislib::sys::Log::DefaultLog
 */
vislib::sys::Log& vislib::sys::Log::DefaultLog(__vl_log_defaultlog);


/*
 * vislib::sys::Log::CurrentTimeStamp
 */
vislib::sys::Log::TimeStamp vislib::sys::Log::CurrentTimeStamp(void) {
    return time(NULL);
}


/*
 * vislib::sys::Log::CurrentSourceID
 */
vislib::sys::Log::SourceID vislib::sys::Log::CurrentSourceID(void) {
    return static_cast<SourceID>(vislib::sys::Thread::CurrentID());
}


/*
 * vislib::sys::Log::Log
 */
vislib::sys::Log::Log(UINT level, unsigned int msgbufsize)
        : mainTarget(new vislib::SmartPtr<Target>(
            new OfflineTarget(msgbufsize, level))),
        echoTarget(new vislib::SmartPtr<Target>(
            new OfflineTarget(msgbufsize, level))), autoflush(true) {
    VLTRACE(TRACE_LVL, "Log[%lu]::Log[%d]()\n",
        reinterpret_cast<unsigned long>(this), __LINE__);
    // Intentionally empty
}

/*
 * vislib::sys::Log::Log
 */
vislib::sys::Log::Log(UINT level, const char *filename, bool addSuffix)
        : mainTarget(NULL), echoTarget(NULL), autoflush(true) {
    VLTRACE(TRACE_LVL, "Log[%lu]::Log[%d]()\n",
        reinterpret_cast<unsigned long>(this), __LINE__);
    this->SetLogFileName(filename, addSuffix);
}


/*
 * vislib::sys::Log::Log
 */
vislib::sys::Log::Log(UINT level, const wchar_t *filename, bool addSuffix)
        : mainTarget(NULL), echoTarget(NULL), autoflush(true) {
    VLTRACE(TRACE_LVL, "Log[%lu]::Log[%d]()\n",
        reinterpret_cast<unsigned long>(this), __LINE__);
    this->SetLogFileName(filename, addSuffix);
}


/*
 * vislib::sys::Log::Log
 */
vislib::sys::Log::Log(const Log& source) : mainTarget(NULL),
        echoTarget(NULL), autoflush(true) {
    VLTRACE(TRACE_LVL, "Log[%lu]::Log[%d]()\n",
        reinterpret_cast<unsigned long>(this), __LINE__);
    *this = source;
}


/*
 * vislib::sys::Log::~Log
 */
vislib::sys::Log::~Log(void) {
    VLTRACE(TRACE_LVL, "Log[%lu]::~Log()\n",
        reinterpret_cast<unsigned long>(this));
    // Intentionally empty
}


/*
 * vislib::sys::Log::EchoOfflineMessages
 */
void vislib::sys::Log::EchoOfflineMessages(bool remove) {
    OfflineTarget *mot = this->mainTarget->DynamicCast<OfflineTarget>();
    OfflineTarget *eot = this->echoTarget->DynamicCast<OfflineTarget>();

    if ((mot == NULL) && (eot != NULL) && !this->mainTarget.IsNull()) {
        eot->Reecho(**this->mainTarget, remove);
    } else if ((mot != NULL) && (eot == NULL) && !this->echoTarget.IsNull()) {
        mot->Reecho(**this->echoTarget, remove);
    }
}


/*
 * vislib::sys::Log::FlushLog
 */
void vislib::sys::Log::FlushLog(void) {
    if (!this->mainTarget.IsNull() && !this->mainTarget->IsNull()) {
        this->mainTarget->operator->()->Flush();
    }
    if (!this->echoTarget.IsNull() && !this->echoTarget->IsNull()) {
        this->echoTarget->operator->()->Flush();
    }
}


/*
 * vislib::sys::Log::GetEchoLevel
 */
UINT vislib::sys::Log::GetEchoLevel(void) const {
    if (!this->echoTarget.IsNull() && !this->echoTarget->IsNull()) {
        return this->echoTarget->operator->()->Level();
    }
    return 0;
}


/*
 * vislib::sys::Log::GetLevel
 */
UINT vislib::sys::Log::GetLevel(void) const {
    if (!this->mainTarget.IsNull() && !this->mainTarget->IsNull()) {
        return this->mainTarget->operator->()->Level();
    }
    return 0;
}


/*
 * vislib::sys::Log::GetLogFileNameA
 */
vislib::StringA vislib::sys::Log::GetLogFileNameA(void) const {
    const FileTarget *ft = this->mainTarget->DynamicCast<FileTarget>();
    return (ft != NULL) ? StringA(ft->Filename()) : StringA::EMPTY;
}


/*
 * vislib::sys::Log::GetLogFileNameW
 */
vislib::StringW vislib::sys::Log::GetLogFileNameW(void) const {
    const FileTarget *ft = this->mainTarget->DynamicCast<FileTarget>();
    return (ft != NULL) ? ft->Filename() : StringW::EMPTY;
}


/*
 * vislib::sys::Log::GetOfflineMessageBufferSize
 */
unsigned int vislib::sys::Log::GetOfflineMessageBufferSize(void) const {
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
 * vislib::sys::Log::SetEchoLevel
 */
void vislib::sys::Log::SetEchoLevel(UINT level) {
    if (!this->echoTarget.IsNull() && !this->echoTarget->IsNull()) {
        this->echoTarget->operator->()->SetLevel(level);
    }
}


/*
 * vislib::sys::Log::SetEchoTarget
 */
void vislib::sys::Log::SetEchoTarget(
        vislib::SmartPtr<vislib::sys::Log::Target> target) {
    SmartPtr<Target> oet = *this->echoTarget;
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
 * vislib::sys::Log::SetLevel
 */
void vislib::sys::Log::SetLevel(UINT level) {
    if (!this->mainTarget.IsNull() && !this->mainTarget->IsNull()) {
        this->mainTarget->operator->()->SetLevel(level);
    }
}


/*
 * vislib::sys::Log::SetLogFileName
 */
bool vislib::sys::Log::SetLogFileName(const char *filename, bool addSuffix) {
    SmartPtr<Target> omt = *this->mainTarget;
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
 * vislib::sys::Log::SetLogFileName
 */
bool vislib::sys::Log::SetLogFileName(const wchar_t *filename, bool addSuffix) {
    SmartPtr<Target> omt = *this->mainTarget;
    OfflineTarget *ot = omt.DynamicCast<OfflineTarget>();

    if (filename == NULL) {
        if (ot == NULL) {
            *this->mainTarget = new OfflineTarget(20U, omt->Level());
        }
    } else {
        vislib::StringW path(filename);
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
 * vislib::sys::Log::SetMainTarget
 */
void vislib::sys::Log::SetMainTarget(
        vislib::SmartPtr<vislib::sys::Log::Target> target) {
    SmartPtr<Target> omt = *this->mainTarget;
    OfflineTarget *ot = omt.DynamicCast<OfflineTarget>();

    *this->mainTarget = target;
    (*this->mainTarget)->SetLevel(omt->Level());
    if (ot != NULL) {
        ot->Reecho(**this->mainTarget);
    }
}


/*
 * vislib::sys::Log::SetOfflineMessageBufferSize
 */
void vislib::sys::Log::SetOfflineMessageBufferSize(unsigned int msgbufsize) {
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
 * vislib::sys::Log::ShareTargetStorage
 */
void vislib::sys::Log::ShareTargetStorage(const vislib::sys::Log& master) {
    this->mainTarget = master.mainTarget;
    this->echoTarget = master.echoTarget;
}


/*
 * vislib::sys::Log::WriteError
 */
void vislib::sys::Log::WriteError(const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(LEVEL_ERROR, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteError
 */
void vislib::sys::Log::WriteError(const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(LEVEL_ERROR, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteError
 */
void vislib::sys::Log::WriteError(int lvlOff, const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(
        static_cast<UINT>(static_cast<int>(LEVEL_ERROR) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteError
 */
void vislib::sys::Log::WriteError(int lvlOff, const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(
        static_cast<UINT>(static_cast<int>(LEVEL_ERROR) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteInfo
 */
void vislib::sys::Log::WriteInfo(const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(LEVEL_INFO, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteInfo
 */
void vislib::sys::Log::WriteInfo(const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(LEVEL_INFO, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteInfo
 */
void vislib::sys::Log::WriteInfo(int lvlOff, const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(
        static_cast<UINT>(static_cast<int>(LEVEL_INFO) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteInfo
 */
void vislib::sys::Log::WriteInfo(int lvlOff, const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(
        static_cast<UINT>(static_cast<int>(LEVEL_INFO) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteMessage
 */
void vislib::sys::Log::WriteMessage(UINT level,
        vislib::sys::Log::TimeStamp time, vislib::sys::Log::SourceID sid,
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
 * vislib::sys::Log::WriteMessage
 */
void vislib::sys::Log::WriteMessageVaA(UINT level,
        vislib::sys::Log::TimeStamp time, vislib::sys::Log::SourceID sid,
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
 * vislib::sys::Log::WriteMessage
 */
void vislib::sys::Log::WriteMessageVaW(UINT level,
        vislib::sys::Log::TimeStamp time, vislib::sys::Log::SourceID sid,
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
 * vislib::sys::Log::WriteMsg
 */
void vislib::sys::Log::WriteMsg(const UINT level, const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(level, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteMsg
 */
void vislib::sys::Log::WriteMsg(const UINT level, const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(level, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteWarn
 */
void vislib::sys::Log::WriteWarn(const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(LEVEL_WARN, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteWarn
 */
void vislib::sys::Log::WriteWarn(const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(LEVEL_WARN, Log::CurrentTimeStamp(),
        Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteWarn
 */
void vislib::sys::Log::WriteWarn(int lvlOff, const char *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaA(
        static_cast<UINT>(static_cast<int>(LEVEL_WARN) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Log::WriteWarn
 */
void vislib::sys::Log::WriteWarn(int lvlOff, const wchar_t *fmt, ...) {
    va_list argptr;
    va_start(argptr, fmt);
    this->WriteMessageVaW(
        static_cast<UINT>(static_cast<int>(LEVEL_WARN) + lvlOff),
        Log::CurrentTimeStamp(), Log::CurrentSourceID(), fmt, argptr);
    va_end(argptr);
}



/*
 * vislib::sys::Log::operator=
 */
vislib::sys::Log& vislib::sys::Log::operator=(const Log& rhs) {
    this->mainTarget = rhs.mainTarget;
    this->echoTarget = rhs.echoTarget;
    this->autoflush = rhs.autoflush;
    return *this;
}


/*
 * vislib::sys::Log::getFileNameSuffix
 */
vislib::StringA vislib::sys::Log::getFileNameSuffix(void) {
    vislib::StringA suffix;

    TimeStamp timestamp = vislib::sys::Log::CurrentTimeStamp();
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
