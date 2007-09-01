/*
 * Process.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Process.h"

#include <cstdarg>

#include "vislib/assert.h"
#include "vislib/Console.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#ifdef _WIN32 // TODO: PAM disabled
#include "vislib/ImpersonationContext.h"
#endif // TODO
#include "vislib/Path.h"
#include "vislib/RawStorage.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::sys::Process::EMPTY_ENVIRONMENT
 */
const vislib::sys::Environment::Snapshot 
vislib::sys::Process::EMPTY_ENVIRONMENT;


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
        const Environment::Snapshot& environment, 
        const char *currentDirectory) {
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
                const_cast<void *>(static_cast<const void *>(environment)), 
                currentDirectory, &si, &pi)
                == FALSE) {
            throw SystemException(__FILE__, __LINE__);
        }
    } else {
        if (::CreateProcessA(NULL, const_cast<char *>(cmdLine.PeekBuffer()), 
                NULL, NULL, FALSE, 0, 
                const_cast<void *>(static_cast<const void *>(environment)),
                currentDirectory, &si, &pi) == FALSE) {
            throw SystemException(__FILE__, __LINE__);
        }
    }

    ::CloseHandle(pi.hThread);
    this->hProcess = pi.hProcess;

#else /* _WIN32 */
   // TODO: PAM disabled
    StringA query;  // Query to which to expand shell commands.
    StringA cmd;    // The command actually run.
    int pipe[2];    // A pipe to get the error code of exec.

    /* A process must not be started twice. */
    if (this->pid >= 0) {
        throw IllegalStateException("This process was already created.",
            __FILE__, __LINE__);
    }

    /* Detect and expand shell commands first first. */
    query.Format("which %s", command);
    if (Console::Run(query.PeekBuffer(), &cmd) == 0) {
        cmd.TrimEnd("\r\n");
    } else {
        cmd = Path::Resolve(command);
    }
    ASSERT(Path::IsAbsolute(cmd));

    /* Open a pipe to get the exec error codes. */
    if (::pipe(pipe) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    /* Make the system close the pipe if exec is called successfully. */
    if (::fcntl(pipe[1], F_SETFD, FD_CLOEXEC)) {
        ::close(pipe[0]);
        ::close(pipe[1]);
        throw SystemException(__FILE__, __LINE__);
    }

    /* Create new process. */
    this->pid = ::fork();
    if (this->pid == 0) {
        /* We are in the child process now. */
        ::close(pipe[0]);       // We do not need the read end any more.

        /** Impersonate as new user. */
        if ((user != NULL) && (password != NULL)) {
            //ic.Impersonate(user, NULL, password);
            ASSERT(false);
        }

        /* Change to working directory, if specified. */
        if (currentDirectory != NULL) {
            if (::chdir(currentDirectory) != 0) {
                ::write(pipe[1], &errno, sizeof(errno));
            }
        }

        if (environment.IsEmpty() && (arguments == NULL)) {
            char *tmp[] = { const_cast<char *>(cmd.PeekBuffer()), NULL };
            ::execv(cmd.PeekBuffer(), tmp);
        } else {
            int cntArgs = 1;
            if (arguments != NULL) {
                while (arguments[i] != NULL) {
                    cntArgs++;
                }
            }
            vislib::RawStorage args((cntArgs + 1) * sizeof(char *));
            args.As<char *>()[0] = const_cast<char *>(cmd.PeekBuffer());
            args.As<char *>()[cntArgs] = NULL;
            for (int i = 1; i < cntArgs; i++) {
                args.As<char *>()[i] = arguments[i - 1];
            }
            
            ::execve(cmd.PeekBuffer(), args, reinterpret_cast<char * const*>(
                static_cast<const void *>((environment)));
        }

        /* exec failed at this point, so report error to parent. */
        ::write(pipe[1], &errno, sizeof(errno));
        ::close(pipe[1]);
        ::_exit(errno);
    
    } else if (this->pid < 0) {
        /* Process creating failed. */
        ::close(pipe[0]);       // There is no child which could write.
        ::close(pipe[1]);       // We do not need the write end any more.
        throw SystemException(__FILE__, __LINE__);

    } else {
        /* Process was spawned, we are in parent. */
        ASSERT(this->pid > 0);
        ::close(pipe[1]);       // We do not need the write end any more.

        /* Try to read error from child process if e. g. exec failed. */
        if (read(pipe[0], &errno, sizeof(errno)) != -1) {
            /* Child process wrote an error code, so report it. */
            ASSERT(::GetLastError() == errno);  // Exception will use GLE.
            this->pid = -1;
            ::close(pipe[0]);
            throw SystemException(__FILE__, __LINE__);
        } else {
            /* Reading error failed, so the child already closed the pipe. */
            ::close(pipe[0]);
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
