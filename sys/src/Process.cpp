/*
 * Process.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Process.h"

#include <cstdarg>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
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
    char **cursor = this->data;

    while (cursor != NULL) {
        ARY_SAFE_DELETE(*cursor);
        cursor++;
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
    // TODO: Implement
}


/*
 * vislib::sys::Process::~Process
 */
vislib::sys::Process::~Process(void) {
    // TODO: Implement
}


/*
 * vislib::sys::Process::Create
 */
void vislib::sys::Process::Create(const char *command, const char *arguments, 
        const Environment& environment, const char *currentDirectory) {
#ifdef _WIN32
    StringA cmdLine("\"");
    cmdLine += command;
    cmdLine += "\" ";
    cmdLine += arguments;

    STARTUPINFOA si;
    ::ZeroMemory(&si, sizeof(STARTUPINFOA));
    si.cb = sizeof(STARTUPINFOA);

    PROCESS_INFORMATION pi;
    ::ZeroMemory(&pi, sizeof(PROCESS_INFORMATION));

    // TODO: Delete old process?

    // TODO: must cast away constness of string for passing it to CreateProcess.
    //if (::CreateProcessA(NULL, cmdLine.PeekBuffer(), NULL, NULL, FALSE, 0, 
    //        static_cast<const void *>(environment), currentDirectory, &si, &pi) 
    //        == FALSE) {
    //    throw SystemException(__FILE__, __LINE__);
    //}

    this->hProcess = pi.hProcess;

#else /* _WIN32 */
    // TODO impl. missing
    assert(false);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::Create
 */
void vislib::sys::Process::Create(const char *command, const char *arguments,
        const char *user, const char *password,
        const Environment& environment, const char *currentDirectory) {
#ifdef _WIN32
    // TODO impl. missing
    assert(false);
#else /* _WIN32 */
    StringA idQuery;
    FILE *fp = NULL;
    uid_t uid;

    /* Retrieve the UID of the requested user. */
    idQuery.Format("id -u %s");
    if ((fp = ::popen(idQuery.PeekBuffer(), "r")) == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (::fscanf(fp, "%d", &uid) != 1) {
        ::pclose(fp);
        throw SystemException(__FILE__, __LINE__);
    }
    ::pclose(fp);
    



    // TODO impl. missing
    assert(false);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::Process
 */
vislib::sys::Process::Process(const Process& rhs) {
    throw UnsupportedOperationException("Process", __FILE__, __LINE__);
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
