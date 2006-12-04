/*
 * Process.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Process.h"

#include <cstdarg>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/ImpersonationContext.h"
#include "vislib/Path.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::Process::Environment::Environment
 */
vislib::sys::Process::Environment::Environment(void) : data(NULL) {
}


/*
 * vislib::sys::Process::Environment::Environment
 */
vislib::sys::Process::Environment::Environment(const char *variable, ...) 
        : data(NULL) {
    va_list argptr;
    SIZE_T dataSize = 0;
    const char *arg;
    
    if (variable != NULL) {
#ifdef _WIN32
        char *insPos = NULL;

        /* Determine the required buffer size. */
        dataSize = ::strlen(variable) + 2;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            dataSize += (::strlen(arg) + 1);
        }
        va_end(argptr);

        /* Allocate buffer. */
        this->data = insPos = new char[dataSize];

        /* Copy the input. */
        dataSize = ::strlen(variable) + 1;
        ::memcpy(insPos, variable, dataSize * sizeof(char));
        insPos += dataSize;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            dataSize = ::strlen(arg) + 1;
            ::memcpy(insPos, arg, dataSize);
            insPos += dataSize;
        }
        va_end(argptr);

        /* Insert terminating double zero. */
        *insPos = '0';

#else /* _WIN32 */
        char **insPos = NULL;

        /* Count parameters. */
        dataSize = 1;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            dataSize++;
        }
        va_end(argptr);

        /* Allocate parameter array. */
        this->data = insPos = new char *[dataSize];

        /* Allocate variable memory and copy data. */
        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            *insPos = new char[::strlen(arg) + 1];
            ::strcpy(*insPos, arg);
            insPos++;
        }
        va_end(argptr);

        /* Last array element must be a NULL pointer. */
        *insPos = NULL;

#endif /* _WIN32 */
    }
}


/*
 * vislib::sys::Process::Environment::~Environment
 */
vislib::sys::Process::Environment::~Environment(void) {
#ifndef _WIN32
    /* Linux must delete the dynamically allocated strings. */
    if (this->data != NULL) {
        char **cursor = this->data;

        while (*cursor != NULL) {
            ARY_SAFE_DELETE(*cursor);
            cursor++;
        }
    }
#endif /* !_WIN32 */

    ARY_SAFE_DELETE(this->data);
}


/*
 * vislib::sys::Process::Environment::Environment
 */
vislib::sys::Process::Environment::Environment(const Environment& rhs) {
    throw UnsupportedOperationException("Environment", __FILE__, __LINE__);
}


/*
 * vislib::sys::Process::Environment::operator =
 */
vislib::sys::Process::Environment& 
vislib::sys::Process::Environment::operator =(const Environment& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}


/*
 * vislib::sys::Process::EMPTY_ENVIRONMENT
 */
const vislib::sys::Process::Environment 
vislib::sys::Process::EMPTY_ENVIRONMENT(NULL);


/*
 * vislib::sys::Process::Process
 */
vislib::sys::Process::Process(void) {
#ifdef _WIN32
    this->hProcess = NULL;
#else /* _WIN32 */
    this->pid = -1;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::~Process
 */
vislib::sys::Process::~Process(void) {
    // TODO: Implement
}


/*
 * vislib::sys::Process::Process
 */
vislib::sys::Process::Process(const Process& rhs) {
    throw UnsupportedOperationException("Process", __FILE__, __LINE__);
}


/*
 * vislib::sys::Process::create
 */
void vislib::sys::Process::create(const char *command, const char *arguments[],
        const char *user, const char *domain, const char *password,
        const Environment& environment, const char *currentDirectory) {
    ASSERT(command != NULL);

#ifdef _WIN32
    const char **arg = arguments;
    HANDLE hToken = NULL;
    StringA cmdLine("\"");
    cmdLine += command;
    cmdLine += "\"";
    if (arg != NULL) {
        cmdLine += " ";

        while (*arg != NULL) {
            cmdLine += *arg;
            arg++;
        }
    }

    /* A process must not be started twice. */
    if (this->hProcess != NULL) {
        throw IllegalStateException("This process was already created.",
            __FILE__, __LINE__);
    }

    STARTUPINFOA si;
    ::ZeroMemory(&si, sizeof(STARTUPINFOA));
    si.cb = sizeof(STARTUPINFOA);

    PROCESS_INFORMATION pi;
    ::ZeroMemory(&pi, sizeof(PROCESS_INFORMATION));

    if ((user != NULL) && (password != NULL)) {
        if (::LogonUserA(user, domain, password, LOGON32_LOGON_INTERACTIVE, 
                LOGON32_PROVIDER_DEFAULT, &hToken) == FALSE) {
            throw SystemException(__FILE__, __LINE__);
        }

        // TODO: hToken has insufficient permissions to spawn process.
        if (::CreateProcessAsUserA(hToken, NULL, 
                const_cast<char *>(cmdLine.PeekBuffer()), NULL, NULL, FALSE, 0,
                static_cast<void *>(environment), currentDirectory, &si, &pi)
                == FALSE) {
            throw SystemException(__FILE__, __LINE__);
        }
    } else {
        if (::CreateProcessA(NULL, const_cast<char *>(cmdLine.PeekBuffer()), 
                NULL, NULL, FALSE, 0, static_cast<void *>(environment),
                currentDirectory, &si, &pi) == FALSE) {
            throw SystemException(__FILE__, __LINE__);
        }
    }

    ::CloseHandle(pi.hThread);
    this->hProcess = pi.hProcess;

#else /* _WIN32 */
    StringA cmd = Path::Resolve(command);
    ImpersonationContext ic;
    pid_t pid;

    /* A process must not be started twice. */
    if (this->pid >= 0) {
        throw IllegalStateException("This process was already created.",
            __FILE__, __LINE__);
    }

    pid = ::fork();
    if (pid < 0) {
        /* Process creating failed. */
        throw SystemException(__FILE__, __LINE__);

    } else if (pid > 0) {
        /*
         * We are in the new process, impersonate as new user and spawn 
         * process. 
         */
        if ((user != NULL) && (password != NULL)) {
            ic.Impersonate(user, NULL, password);
        }

        /* Change to working directory, if specified. */
        if (currentDirectory != NULL) {
            if (::chdir(currentDirectory) != 0) {
                throw SystemException(__FILE__, __LINE__);
            }
        }

        if (environment.IsEmpty()) {
            char *tmp[] = { const_cast<char *>(cmd.PeekBuffer()), NULL };
            if (::execv(cmd.PeekBuffer(), tmp) != 0) {
                throw SystemException(__FILE__, __LINE__);
            }
                
        } else {
            assert(false);
            //if (::execve(cmd.PeekBuffer(), arguments, 
            //        static_cast<char * const*>(environment)) != 0) {
            //    throw SystemException(__FILE__, __LINE__);
            //}
        }
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::operator =
 */
vislib::sys::Process& vislib::sys::Process::operator =(const Process& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
