/*
 * Log.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/Log.h"

#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/mathfunctions.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemInformation.h"
#include "vislib/Thread.h"
#include <climits>
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#endif /* _WIN32 */


/*****************************************************************************/

/*
 * vislib::sys::Log::EchoTargetStream::StdOut
 */
const vislib::sys::Log::EchoTargetStream 
vislib::sys::Log::EchoTargetStream::StdOut(stdout);


/*
 * vislib::sys::Log::EchoTargetStream::StdErr
 */
const vislib::sys::Log::EchoTargetStream 
vislib::sys::Log::EchoTargetStream::StdErr(stderr);


/*
 * vislib::sys::Log::EchoTargetStream::EchoTargetStream
 */
vislib::sys::Log::EchoTargetStream::EchoTargetStream(FILE *stream) 
        : stream(stream) {
}


/*
 * vislib::sys::Log::EchoTargetStream::Write
 */
void vislib::sys::Log::EchoTargetStream::Write(UINT level, 
        const char *message) const {
    fprintf(this->stream, "%.4d|%s", level, message);
}


/*
 * vislib::sys::Log::EchoTargetStream::~EchoTargetStream
 */
vislib::sys::Log::EchoTargetStream::~EchoTargetStream() {
    // DO NOT CLOSE THE STREAM
}


/*****************************************************************************/

#ifdef _WIN32

/*
 * vislib::sys::Log::EchoTargetDebugOutput::Write
 */
void vislib::sys::Log::EchoTargetDebugOutput::Write(UINT level, 
        const char *message) const {
    char tmp[21];
    tmp[0] = tmp[20] = 0;
    _snprintf_s(tmp, 20, 20, "%.4d|", level);
    ::OutputDebugStringA(tmp);
    ::OutputDebugStringA(message);
}

#endif /* _WIN32 */

/*****************************************************************************/


/*
 * vislib::sys::Log::EchoTargetRedirect::EchoTargetRedirect
 */
vislib::sys::Log::EchoTargetRedirect::EchoTargetRedirect(void) : EchoTarget(),
        target(NULL) {
}


/*
 * vislib::sys::Log::EchoTargetRedirect::EchoTargetRedirect
 */
vislib::sys::Log::EchoTargetRedirect::EchoTargetRedirect(
        vislib::sys::Log *target) : EchoTarget(), target(target) {
}


/*
 * vislib::sys::Log::EchoTargetRedirect::~EchoTargetRedirect
 */
vislib::sys::Log::EchoTargetRedirect::~EchoTargetRedirect(void) {
    this->target = NULL; // DO NOT DELETE. Because we don't own the memory
}


/*
 * vislib::sys::Log::EchoTargetRedirect::SetTarget
 */
void vislib::sys::Log::EchoTargetRedirect::SetTarget(
        vislib::sys::Log *target) {
    this->target = target;
}


/*
 * vislib::sys::Log::EchoTargetRedirect::Write
 */
void vislib::sys::Log::EchoTargetRedirect::Write(UINT level, 
        const char *message) const {
    if (this->target) {
        this->target->WriteMsg(level, message);
    }
}

/*****************************************************************************/


/*
 * vislib::sys::Log::OfflineMessage::OfflineMessage
 */
vislib::sys::Log::OfflineMessage::OfflineMessage(void) : level(LEVEL_NONE) {
    // only declared for paranoia purpose
}


/*
 * vislib::sys::Log::OfflineMessage::~OfflineMessage
 */
vislib::sys::Log::OfflineMessage::~OfflineMessage(void) {
    // only declared for paranoia purpose
}


/*
 * vislib::sys::Log::OfflineMessage::operator=
 */
vislib::sys::Log::OfflineMessage& vislib::sys::Log::OfflineMessage::operator=(
        const OfflineMessage& rhs) {
    this->level = rhs.level;
    this->time = rhs.time;
    this->message = rhs.message;
    return *this;
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
 * vislib::sys::Log::DefaultLog
 */
vislib::sys::Log vislib::sys::Log::DefaultLog;


/*
 * vislib::sys::Log::emptyLogMsg
 */
const char *vislib::sys::Log::emptyLogMsg = "Empty log message\n";

/*
 * vislib::sys::Log::emptyLogMsg
 */
const char *vislib::sys::Log::omittedLogMsgs 
    = "%d further offline log message omitted\n";


/*
 * vislib::sys::Log::Log
 */
vislib::sys::Log::Log(UINT level, unsigned int msgbufsize) 
        : level(level), filename(), logfile(NULL), msgbufsize(0), 
        omittedMsgs(0), offlineMsgs(NULL), autoflush(true), 
        echoLevel(LEVEL_ERROR), echoOut(NULL) {
    this->SetOfflineMessageBufferSize(msgbufsize);
}

/*
 * vislib::sys::Log::Log
 */
vislib::sys::Log::Log(UINT level, const char *filename, bool addSuffix) 
        : level(level), filename(), logfile(NULL), msgbufsize(0), 
        omittedMsgs(0), offlineMsgs(NULL), autoflush(true), 
        echoLevel(LEVEL_ERROR), echoOut(NULL) {
    this->SetLogFileName(filename, addSuffix);
}


/*
 * vislib::sys::Log::Log
 */
vislib::sys::Log::Log(UINT level, const wchar_t *filename, bool addSuffix) 
        : level(level), filename(), logfile(NULL), msgbufsize(0), 
        omittedMsgs(0), offlineMsgs(NULL), autoflush(true), 
        echoLevel(LEVEL_ERROR), echoOut(NULL) {
    this->SetLogFileName(filename, addSuffix);
}


/*
 * vislib::sys::Log::Log
 */
vislib::sys::Log::Log(const Log& source) 
        : level(LEVEL_ERROR), filename(), logfile(NULL), msgbufsize(0), 
        omittedMsgs(0), offlineMsgs(NULL), autoflush(true), 
        echoLevel(LEVEL_ERROR), echoOut(NULL) {
    *this = source;
}


/*
 * vislib::sys::Log::~Log
 */
vislib::sys::Log::~Log(void) {
    if (this->logfile) {
        fclose(this->logfile);
        this->logfile = NULL;
    }
    if (this->offlineMsgs) {
        ARY_SAFE_DELETE(this->offlineMsgs);
    }
    // DO NOT fclose(echoOut), because we don't own it!
}


/*
 * vislib::sys::Log::EchoOfflineMessages
 */
void vislib::sys::Log::EchoOfflineMessages(bool remove) {
    if (this->offlineMsgs == NULL) {
        return;
    }

    if (this->echoOut != NULL) {
        for (unsigned int i = 0; i < this->msgbufsize; i++) {
            if ((this->offlineMsgs[i].level > 0) 
                    && (this->offlineMsgs[i].level <= this->echoLevel)) {
                this->echoOut->Write(this->offlineMsgs[i].level,
                    this->offlineMsgs[i].message.PeekBuffer());
            }
        }

        if (this->omittedMsgs > 0) {
            // write a special log message for the summary of the omitted 
            // messages
            vislib::StringA str;
            str.Format(vislib::sys::Log::omittedLogMsgs, this->omittedMsgs);
            this->echoOut->Write(LEVEL_INFO, str.PeekBuffer());
        }
    }

    if (remove) {
        for (unsigned int i = 0; i < this->msgbufsize; i++) {
            this->offlineMsgs[i].level = 0;
        }
        this->omittedMsgs = 0;
    }

}


/*
 * vislib::sys::Log::SetLogFileName
 */
bool vislib::sys::Log::SetLogFileName(const char *filename, bool addSuffix) {
    int newFile = -1;
    vislib::StringA newFileName = filename;

    if (filename != NULL) {
        // open the new file

        if (addSuffix) {
            newFileName += this->getFileNameSuffix();
        }

#ifdef _WIN32
#if (_MSC_VER >= 1400)
        if (_sopen_s(&newFile, newFileName, _O_APPEND | _O_CREAT | _O_TEXT 
                | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE) != 0) {
            newFile = -1;
        }
#else /* (_MSC_VER >= 1400) */
        newFile = _sopen(newFileName, _O_APPEND | _O_CREAT | _O_TEXT 
            | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
        newFile = open(newFileName, O_APPEND | O_CREAT | O_LARGEFILE 
            | O_WRONLY, S_IRWXU | S_IRWXG);
#endif /* _WIN32 */

        if (newFile == -1) {
            // unable to open the new log file.
            // we keep the old one or keep offline mode
            return false;
        }
    }

    // successfully opened the new log file
    if (this->logfile) {
        fclose(this->logfile);
    }
    this->logfile = (newFile == -1) ? NULL : 
#ifdef _WIN32
        _fdopen(newFile, "ac");
#else /* _WIN32 */
        fdopen(newFile, "a");
#endif /* _WIN32 */

    if (this->logfile) {
        this->filename = 
#ifdef _WIN32
            A2W(newFileName);
#else /* _WIN32 */
            newFileName;
#endif /* _WIN32 */

        // flush offline messages if any available
        // Do not echo offline messages, because they already had been echoed,
        // if they are of the right level
        if (this->offlineMsgs) {

            for (unsigned int i = 0; i < this->msgbufsize; i++) {
                if (this->offlineMsgs[i].level > 0) {
                    this->writeMsgPrefix(this->offlineMsgs[i].level, 
                        this->offlineMsgs[i].time);
                    fprintf(this->logfile, "%s", 
                        this->offlineMsgs[i].message.PeekBuffer());
                }
            }
        
            if (this->omittedMsgs > 0) {
                // write a special log message for the summary of the omitted 
                // messages
                this->writeMsgPrefix(LEVEL_INFO, 
                    vislib::sys::Log::currentTimeStamp());
                fprintf(this->logfile, vislib::sys::Log::omittedLogMsgs, 
                    this->omittedMsgs);
                this->omittedMsgs = 0;
            }

            if (this->autoflush) {
                fflush(this->logfile);
            }

            ARY_SAFE_DELETE(this->offlineMsgs);
        }
    } else {
        this->filename.Clear();

        if (this->offlineMsgs == NULL) {
            this->offlineMsgs = new OfflineMessage[this->msgbufsize];
            this->omittedMsgs = 0;
        }
    }

    return true;
}


/*
 * vislib::sys::Log::SetLogFileName
 */
bool vislib::sys::Log::SetLogFileName(const wchar_t *filename, bool addSuffix) {
    int newFile = -1;
    vislib::StringW newFileName = filename;

    if (filename != NULL) {
        // open the new file

        if (addSuffix) {
            newFileName += A2W(this->getFileNameSuffix());
        }

#ifdef _WIN32
#if (_MSC_VER >= 1400)
        if (_wsopen_s(&newFile, newFileName, _O_APPEND | _O_CREAT | _O_TEXT 
                | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE) != 0) {
            newFile = -1;
        }
#else /* (_MSC_VER >= 1400) */
        newFile = _wsopen(newFileName, _O_APPEND | _O_CREAT | _O_TEXT 
            | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
        newFile = open(W2A(newFileName), O_APPEND | O_CREAT | O_LARGEFILE 
            | O_WRONLY, S_IRWXU | S_IRWXG);
#endif /* _WIN32 */

        if (newFile == -1) {
            // unable to open the new log file.
            // we keep the old one or keep offline mode
            return false;
        }
    }

    // successfully opened the new log file
    if (this->logfile) {
        fclose(this->logfile);
    }
    this->logfile = (newFile == -1) ? NULL : 
#ifdef _WIN32
        _fdopen(newFile, "ac");
#else /* _WIN32 */
        fdopen(newFile, "a");
#endif /* _WIN32 */

    if (this->logfile) {
        this->filename = 
#ifdef _WIN32
            newFileName;
#else /* _WIN32 */
            W2A(newFileName);
#endif /* _WIN32 */

        // flush offline messages if any available
        // Do not echo offline messages, because they already had been echoed,
        // if they are of the right level
        if (this->offlineMsgs) {

            for (unsigned int i = 0; i < this->msgbufsize; i++) {
                if (this->offlineMsgs[i].level > 0) {
                    this->writeMsgPrefix(this->offlineMsgs[i].level, 
                        this->offlineMsgs[i].time);
                    fprintf(this->logfile, "%s", 
                        this->offlineMsgs[i].message.PeekBuffer());
                }
            }
        
            if (this->omittedMsgs > 0) {
                // write a special log message for the summary of the omitted 
                // messages
                this->writeMsgPrefix(LEVEL_INFO, 
                    vislib::sys::Log::currentTimeStamp());
                fprintf(this->logfile, vislib::sys::Log::omittedLogMsgs, 
                    this->omittedMsgs);
                this->omittedMsgs = 0;
            }

            if (this->autoflush) {
                fflush(this->logfile);
            }

            ARY_SAFE_DELETE(this->offlineMsgs);
        }
    } else {
        ASSERT(this->offlineMsgs == NULL);
        this->filename.Clear();

        this->offlineMsgs = new OfflineMessage[this->msgbufsize];
        this->omittedMsgs = 0;
    }

    return true;
}


/*
 * vislib::sys::Log::SetOfflineMessageBufferSize
 */
void vislib::sys::Log::SetOfflineMessageBufferSize(unsigned int msgbufsize) {
    if (this->logfile) {
        // if logfile is valid, we do not use the offline message buffer
        ASSERT(this->offlineMsgs == NULL);

    } else {
        OfflineMessage *nm = new OfflineMessage[msgbufsize];
        if (this->offlineMsgs) {
            // keep old messages
            unsigned int m = vislib::math::Min(this->msgbufsize, msgbufsize);
            for (unsigned int i = 0; i < m; i++) {
                nm[i] = this->offlineMsgs[i];
            }

            // keep record of discarded messages
            if (m < this->msgbufsize) {
                for (unsigned int i = m; i < this->msgbufsize; i++) {
                    if (this->offlineMsgs[i].level > 0) {
                        this->omittedMsgs++;
                    }
                }
            }

            // free memory of old offline messages
            delete[] this->offlineMsgs;

        }

        // accept new array
        this->offlineMsgs = nm;

    }

    this->msgbufsize = msgbufsize;
}


/*
 * vislib::sys::Log::FlushLog
 */
void vislib::sys::Log::FlushLog(void) {
    if (this->logfile) {
        ::fflush(this->logfile);
    }
}


/*
 * vislib::sys::Log::WriteMsg
 */
void vislib::sys::Log::WriteMsg(const UINT level, const char *fmt, ...) {

    // write echo message
    if ((this->echoOut != NULL) && (level <= this->echoLevel) && (level > 0)) {

        vislib::StringA txt;

        va_list argptr;
        va_start(argptr, fmt);
        txt.FormatVa(fmt, argptr);
        va_end(argptr);
        if (!txt.EndsWith('\n')) txt += "\n";

        this->echoOut->Write(level, txt.PeekBuffer());
    }

    // write log message
    if ((level <= this->level) && (level > 0)) {
        va_list argptr;
        if (this->logfile) {
            // physical log file available

            this->writeMsgPrefix(level, vislib::sys::Log::currentTimeStamp());

            if (fmt != NULL) {
                // write directly to the file
                va_start(argptr, fmt);
#ifdef _WIN32
#if (_MSC_VER >= 1400)
                ::vfprintf_s(this->logfile, fmt, argptr);
#else /* (_MSC_VER >= 1400) */
                ::vfprintf(this->logfile, fmt, argptr);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
                ::vfprintf(this->logfile, fmt, argptr);
#endif /* _WIN32 */
                va_end(argptr);

                // check if fmt ends with '\n', otherwise append one
                const char *p = fmt;
                while ((*p != 0) && (*(p + 1) != 0)) p++;
                if (*p != '\n') fprintf(this->logfile, "\n");

            } else {
                fprintf(this->logfile, vislib::sys::Log::emptyLogMsg);
            }

            if (this->autoflush) {
                ::fflush(this->logfile);
            }

        } else {
            // no physical log file
            OfflineMessage *om = this->nextOfflineMessage();
            if (om == NULL) {
                this->omittedMsgs++;

            } else {
                vislib::StringA::Size size = 0;

                // offline message
                om->level = level;
                om->time = vislib::sys::Log::currentTimeStamp();

                // Determine required buffer size. 
                va_start(argptr, fmt);
                size = vislib::CharTraitsA::Format(NULL, 0, fmt, argptr);
                va_end(argptr);

                // Allocate memory. 
                ASSERT(size >= 0);
                char *data = om->message.AllocateBuffer(size);
                
                // Write the actual output.
                va_start(argptr, fmt);
                vislib::CharTraitsA::Format(data, size + 1, fmt, argptr);
                va_end(argptr);

                // ensure linebreak
                if (!om->message.EndsWith('\n')) {
                    om->message += "\n";
                }
            }

        }
    } /* if ((level <= this->level) && (level > 0) && (fmt != NULL)) */
}


/*
 * vislib::sys::Log::WriteMsg
 */
void vislib::sys::Log::WriteMsg(const UINT level, const wchar_t *fmt, ...) {

    // write echo message
    if ((this->echoOut != NULL) && (level <= this->echoLevel) && (level > 0)) {
        vislib::StringW txt;

        va_list argptr;
        va_start(argptr, fmt);
        txt.FormatVa(fmt, argptr);
        va_end(argptr);
        if (!txt.EndsWith(L'\n')) txt += L"\n";

        this->echoOut->Write(level, W2A(txt));
    }

    // write log message
    if ((level <= this->level) && (level > 0)) {
        va_list argptr;
        if (this->logfile) {
            // physical log file available

            this->writeMsgPrefix(level, vislib::sys::Log::currentTimeStamp());

            if (fmt != NULL) {
                // write directly to the file
                va_start(argptr, fmt);
#ifdef _WIN32
#if (_MSC_VER >= 1400)
                ::vfwprintf_s(this->logfile, fmt, argptr);
#else /* (_MSC_VER >= 1400) */
                ::vfwprintf(this->logfile, fmt, argptr);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
                ::vfwprintf(this->logfile, fmt, argptr);
#endif /* _WIN32 */
                va_end(argptr);

                // check if fmt ends with '\n', otherwise append one
                const wchar_t *p = fmt;
                while ((*p != 0) && (*(p + 1) != 0)) p++;
                if (*p != L'\n') fprintf(this->logfile, "\n");

            } else {
                fprintf(this->logfile, vislib::sys::Log::emptyLogMsg);
            }

            if (this->autoflush) {
                ::fflush(this->logfile);
            }

        } else {
            // no physical log file
            OfflineMessage *om = this->nextOfflineMessage();
            if (om == NULL) {
                this->omittedMsgs++;

            } else {
                vislib::StringW tmp;
                vislib::StringW::Size size = 0;

                // offline message
                om->level = level;
                om->time = vislib::sys::Log::currentTimeStamp();

                // Determine required buffer size. 
                va_start(argptr, fmt);
                size = vislib::CharTraitsW::Format(NULL, 0, fmt, argptr);
                va_end(argptr);

                // Allocate memory. 
                ASSERT(size >= 0);
                wchar_t *data = tmp.AllocateBuffer(size);
                
                // Write the actual output.
                va_start(argptr, fmt);
                vislib::CharTraitsW::Format(data, size + 1, fmt, argptr);
                va_end(argptr);
                om->message = W2A(tmp);

                // ensure linebreak
                if (!om->message.EndsWith('\n')) {
                    om->message += "\n";
                }
            }

        }
    } /* if ((level <= this->level) && (level > 0) && (fmt != NULL)) */
}


/*
 * vislib::sys::Log::operator=
 */
vislib::sys::Log& vislib::sys::Log::operator=(const Log& rhs) {
    this->level = rhs.level;
    this->SetLevel(rhs.level);
    this->SetLogFileName((rhs.logfile) ? rhs.filename : NULL, false);
    this->SetOfflineMessageBufferSize(rhs.msgbufsize);
    this->SetAutoFlush(rhs.autoflush);
    this->omittedMsgs = rhs.omittedMsgs;
    if (rhs.logfile == NULL) {
        for (unsigned int i = 0; i < this->msgbufsize; i++) {
            this->offlineMsgs[i] = rhs.offlineMsgs[i];
        }
    }

    return *this;
}


/*
 * vislib::sys::Log::currentTimeStamp
 */
vislib::sys::Log::TimeStamp vislib::sys::Log::currentTimeStamp(void) {
    return time(NULL);
}


/*
 * vislib::sys::Log::writeMsgPrefix
 */
void vislib::sys::Log::writeMsgPrefix(UINT level, const TimeStamp& timestamp) {
    // Maybe this should be configurable
    ASSERT(this->logfile != NULL);

    struct tm *timeStamp;
#ifdef _WIN32
#if (_MSC_VER >= 1400)
    struct tm __tS;
    timeStamp = &__tS;
    if (localtime_s(timeStamp, &timestamp) != 0) {
        // timestamp error *** argh ***
        __tS.tm_hour = __tS.tm_min = __tS.tm_sec = 0;
    }
#else /* (_MSC_VER >= 1400) */
    timeStamp = localtime(&timestamp);
#endif /* (_MSC_VER >= 1400) */
#else /* _WIN32 */
    timeStamp = localtime(&timestamp);
#endif /* _WIN32 */

    fprintf(this->logfile, "%2d:%.2d:%.2d|%8x|%4d|", 
        timeStamp->tm_hour, timeStamp->tm_min, timeStamp->tm_sec, 
        vislib::sys::Thread::CurrentID(), level);
}


/*
 * vislib::sys::Log::nextOfflineMessage
 */
vislib::sys::Log::OfflineMessage *vislib::sys::Log::nextOfflineMessage(void) {
    ASSERT(this->logfile == NULL);
    for (unsigned int i = 0; i < this->msgbufsize; i++) {
        if (this->offlineMsgs[i].level == LEVEL_NONE) {
            return &this->offlineMsgs[i];
        }
    }
    return NULL;
}


/*
 * vislib::sys::Log::getFileNameSuffix
 */
vislib::StringA vislib::sys::Log::getFileNameSuffix(void) {
    vislib::StringA suffix;

    TimeStamp timestamp = vislib::sys::Log::currentTimeStamp();
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
