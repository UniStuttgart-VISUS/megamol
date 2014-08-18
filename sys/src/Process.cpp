/*
 * Process.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2007 by Christoph Mueller. Alle Rechte vorbehalten.
 */


#include "vislib/Process.h"

#include <climits>
#include <cstdarg>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#endif /* !_WIN32 */

#include "vislib/assert.h"
#include "vislib/AutoHandle.h"
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
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"

#include "vislib/MissingImplementationException.h"


/*
 * vislib::sys::Process::Exists
 */
bool vislib::sys::Process::Exists(const PID processID) {
#ifdef _WIN32
    HANDLE hProcess;
    DWORD exitCode;

    // TODO: Es wäre sicherer die Existenz darüber festzustellen, daß der
    // Prozess nicht in signaled state ist. Darüber nachdenken, ob das 
    // irgendwelche Probleme machen könnte.
    // Ggf. dann auch bei den Threads den state anstatt des exit code verwenden.
    if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION, TRUE, processID))
            != NULL) {
        if (!::GetExitCodeProcess(hProcess, &exitCode)) {
            throw SystemException(__FILE__, __LINE__);
        }

        ::CloseHandle(hProcess);
        return (exitCode == STILL_ACTIVE);
    } 

    return false;

#else /* _WIN32 */
    return (!((::kill(processID, 0) == -1) && (errno == ESRCH)));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::Exit
 */
void vislib::sys::Process::Exit(const DWORD exitCode) {
#ifdef _WIN32
    ::ExitProcess(exitCode);
#else /* _WIN32 */
    ::exit(exitCode);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::ModuleFileNameA
 */
vislib::StringA vislib::sys::Process::ModuleFileNameA(const PID processID) {
#ifdef _WIN32
    AutoHandle hProcess(true);
    StringA retval;

    if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION, TRUE, processID))
            == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (!GetModuleFileNameA(static_cast<HMODULE>(static_cast<HANDLE>(hProcess)),
            retval.AllocateBuffer(MAX_PATH), MAX_PATH)) {
        throw SystemException(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    StringA procAddr;
    StringA retval;
    char *retvalPtr = NULL;
    int len = -1;

    procAddr.Format("/proc/%d/exe", processID);
    retvalPtr = retval.AllocateBuffer(PATH_MAX + 1);
    len = ::readlink(procAddr.PeekBuffer(), retvalPtr, PATH_MAX + 1);
    if (len == -1) {
        throw SystemException(__FILE__, __LINE__);
    }
    retvalPtr[len] = 0;

    return retval;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::ModuleFileNameA
 */
vislib::StringA vislib::sys::Process::ModuleFileNameA(void) {
#ifdef _WIN32
    StringA retval;

    if (!GetModuleFileNameA(NULL, retval.AllocateBuffer(MAX_PATH), MAX_PATH)) {
        throw SystemException(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    return Process::ModuleFileNameA(Process::CurrentID());
#endif /* _WIN32 */
}




/*
 * vislib::sys::Process::ModuleFileNameW
 */
vislib::StringW vislib::sys::Process::ModuleFileNameW(const PID processID) {
#ifdef _WIN32
    AutoHandle hProcess(true);
    StringW retval;

    if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION, TRUE, processID))
            == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (!GetModuleFileNameW(static_cast<HMODULE>(static_cast<HANDLE>(hProcess)),
            retval.AllocateBuffer(MAX_PATH), MAX_PATH)) {
        throw SystemException(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    return StringW(Process::ModuleFileNameA(processID));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::ModuleFileNameW
 */
vislib::StringW vislib::sys::Process::ModuleFileNameW(void) {
#ifdef _WIN32
    StringW retval;

    if (!GetModuleFileNameW(NULL, retval.AllocateBuffer(MAX_PATH), MAX_PATH)) {
        throw SystemException(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    return Process::ModuleFileNameA(Process::CurrentID());
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::Owner
 */
void vislib::sys::Process::Owner(const PID processID, vislib::StringA& outUser,
        vislib::StringA *outDomain) {
#ifdef _WIN32
    DWORD domainLen = 0;            // Length of domain name in characters.
    DWORD error = NO_ERROR;         // Last system error code.
    DWORD userInfoLen = 0;          // Size of 'userInfo' in bytes.
    DWORD userLen = 0;              // Length of user name in characters.
    AutoHandle hProcess(true);      // Process handle.
    AutoHandle hToken(true);        // Process security token.
    PSID sid = NULL;                // User SID.
    SID_NAME_USE snu;               // Type of retrieved SID.
    RawStorage tmpDomain;           // Storage for domain if discarded.
    RawStorage userInfo;            // Receives user info of security token.
    StringA::Char *user = NULL;     // Receives user name.
    StringA::Char *domain = NULL;   // Receives domain name.

    /* Clear return values. */
    outUser.Clear();
    if (outDomain != NULL) {
        outDomain->Clear();
    }

    /* Acquire a process handle. */
    if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, processID))
            != NULL) {

        /* Get security token of process. */
        if (::OpenProcessToken(hProcess, TOKEN_QUERY, hToken)) {

            /* Get user information of security token. */
            if (::GetTokenInformation(hToken, TokenUser, NULL, 0, &userInfoLen)
                    || ((error = ::GetLastError()) 
                    == ERROR_INSUFFICIENT_BUFFER)) {

                userInfo.AssertSize(userInfoLen);
                if (::GetTokenInformation(hToken, TokenUser, userInfo, 
                        userInfoLen, &userInfoLen)) {
                    sid = userInfo.As<TOKEN_USER>()->User.Sid;
                
                    /* Lookup the user name of the SID. */
                    if (::LookupAccountSidA(NULL, sid, NULL, &userLen, NULL, 
                            &domainLen, &snu) || ((error = ::GetLastError())
                            == ERROR_INSUFFICIENT_BUFFER)) {
                        user = outUser.AllocateBuffer(userLen);
                        if (outDomain != NULL) {
                            domain = outDomain->AllocateBuffer(domainLen);
                        } else {
                            tmpDomain.AssertSize(domainLen);
                            domain = tmpDomain.As<StringA::Char>();
                        }

                        if (!::LookupAccountSidA(NULL, sid, user, &userLen,
                                domain, &domainLen, &snu)) {
                            error = ::GetLastError();
                            outUser.Clear();
                            if (outDomain != NULL) {
                                outDomain->Clear();
                            }
                            ::CloseHandle(hToken);
                            ::CloseHandle(hProcess);
                            throw SystemException(error, __FILE__, __LINE__);
                        }

                    } else {
                        ::CloseHandle(hToken);
                        ::CloseHandle(hProcess);
                        throw SystemException(error, __FILE__, __LINE__);
                    } /* end if (::LookupAccountSidA(NULL, sid, NULL, ... */

                } else {
                    error = ::GetLastError();
                    throw SystemException(error, __FILE__, __LINE__);
                } /* end if (::GetTokenInformation(hToken, TokenUser, ...  */

            } else {
                error = ::GetLastError();
                throw SystemException(error, __FILE__, __LINE__);
            } /* end if (::GetTokenInformation(hToken, TokenUser, ...  */

        } else {
            error = ::GetLastError();
            throw SystemException(error, __FILE__, __LINE__);
        } /* end if (::OpenProcessToken(hProcess, TOKEN_QUERY, &hToken)) */

    } else {
        error = ::GetLastError();
        throw SystemException(error, __FILE__, __LINE__);
    } /* end if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION ... */

#else /* _WIN32 */
    // TODO: Try something like ps -p <pid> -o user
    throw MissingImplementationException("vislib::sys::Process::Owner", __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::Owner
 */
void vislib::sys::Process::Owner(const PID processID, vislib::StringW& outUser,
        vislib::StringW *outDomain) {
#ifdef _WIN32
    DWORD domainLen = 0;            // Length of domain name in characters.
    DWORD error = NO_ERROR;         // Last system error code.
    DWORD userInfoLen = 0;          // Size of 'userInfo' in bytes.
    DWORD userLen = 0;              // Length of user name in characters.
    AutoHandle hProcess(true);      // Process handle.
    AutoHandle hToken(true);        // Process security token.
    PSID sid = NULL;                // User SID.
    SID_NAME_USE snu;               // Type of retrieved SID.
    RawStorage tmpDomain;           // Storage for domain if discarded.
    RawStorage userInfo;            // Receives user info of security token.
    StringW::Char *user = NULL;     // Receives user name.
    StringW::Char *domain = NULL;   // Receives domain name.

    /* Clear return values. */
    outUser.Clear();
    if (outDomain != NULL) {
        outDomain->Clear();
    }

    /* Acquire a process handle. */
    if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, processID))
            != NULL) {

        /* Get security token of process. */
        if (::OpenProcessToken(hProcess, TOKEN_QUERY, hToken)) {

            /* Get user information of security token. */
            if (::GetTokenInformation(hToken, TokenUser, NULL, 0, &userInfoLen)
                    || ((error = ::GetLastError()) 
                    == ERROR_INSUFFICIENT_BUFFER)) {

                userInfo.AssertSize(userInfoLen);
                if (::GetTokenInformation(hToken, TokenUser, userInfo, 
                        userInfoLen, &userInfoLen)) {
                    sid = userInfo.As<TOKEN_USER>()->User.Sid;
                
                    /* Lookup the user name of the SID. */
                    if (::LookupAccountSidW(NULL, sid, NULL, &userLen, NULL, 
                            &domainLen, &snu) || ((error = ::GetLastError())
                            == ERROR_INSUFFICIENT_BUFFER)) {
                        user = outUser.AllocateBuffer(userLen);
                        if (outDomain != NULL) {
                            domain = outDomain->AllocateBuffer(domainLen);
                        } else {
                            tmpDomain.AssertSize(domainLen);
                            domain = tmpDomain.As<StringW::Char>();
                        }

                        if (!::LookupAccountSidW(NULL, sid, user, &userLen,
                                domain, &domainLen, &snu)) {
                            error = ::GetLastError();
                            outUser.Clear();
                            if (outDomain != NULL) {
                                outDomain->Clear();
                            }
                            ::CloseHandle(hToken);
                            ::CloseHandle(hProcess);
                            throw SystemException(error, __FILE__, __LINE__);
                        }

                    } else {
                        ::CloseHandle(hToken);
                        ::CloseHandle(hProcess);
                        throw SystemException(error, __FILE__, __LINE__);
                    } /* end if (::LookupAccountSidW(NULL, sid, NULL, ... */

                } else {
                    error = ::GetLastError();
                    throw SystemException(error, __FILE__, __LINE__);
                } /* end if (::GetTokenInformation(hToken, TokenUser, ...  */

            } else {
                error = ::GetLastError();
                throw SystemException(error, __FILE__, __LINE__);
            } /* end if (::GetTokenInformation(hToken, TokenUser, ...  */

        } else {
            error = ::GetLastError();
            throw SystemException(error, __FILE__, __LINE__);
        } /* end if (::OpenProcessToken(hProcess, TOKEN_QUERY, &hToken)) */

    } else {
        error = ::GetLastError();
        throw SystemException(error, __FILE__, __LINE__);
    } /* end if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION ... */

#else /* _WIN32 */
    StringA u;
    StringA d;

    Process::Owner(processID, u, &d);
    if (outDomain != NULL) {
        *outDomain = d;
    }
#endif /* _WIN32 */
}


/* 
 * vislib::sys::Process::OwnerA
 */
vislib::StringA vislib::sys::Process::OwnerA(const PID processID, 
            const bool includeDomain, const bool isLenient) {
    StringA retval;
    
    try {
#ifdef _WIN32
        if (includeDomain) {
#else /* _WIN32 */
        if (false) {
#endif /* _WIN32 */
            StringA user;
            Process::Owner(processID, user, &retval);
            retval += "\\";
            retval += user;
        } else {
            Process::Owner(processID, retval, NULL);
        }
    } catch (SystemException) {
        if (!isLenient) {
            throw;
        }
    }

    return retval;
}


/*
 * vislib::sys::Process::OwnerW
 */
vislib::StringW vislib::sys::Process::OwnerW(const PID processID, 
            const bool includeDomain, const bool isLenient) {
    StringW retval;
    
    try {
#ifdef _WIN32
        if (includeDomain) {
#else /* _WIN32 */
        if (false) {
#endif /* _WIN32 */
            StringW user;
            Process::Owner(processID, user, &retval);
            retval += L"\\";
            retval += user;
        } else {
            Process::Owner(processID, retval, NULL);
        }
    } catch (SystemException) {
        if (!isLenient) {
            throw;
        }
    }

    return retval;
}


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
 * vislib::sys::Process::GetID
 */
vislib::sys::Process::PID vislib::sys::Process::GetID(void) const {
#ifdef _WIN32
    DWORD retval = ::GetProcessId(this->hProcess);
    if (retval == 0) {
        throw SystemException(__FILE__, __LINE__);
    } else {
        return retval;
    }
#else /* _WIN32 */
    if (this->pid == -1) {
        throw SystemException(ESRCH, __FILE__, __LINE__);
    } else {
        return this->pid;
    }
#endif /* _WIN32 */
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
    DWORD error;    // Exception error code.
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
    VLTRACE(Trace::LEVEL_VL_INFO, "CreateProcess: command is \"%s\"\n",
        cmd.PeekBuffer());

    /* Open a pipe to get the exec error codes. */
    if (::pipe(pipe) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    /* Make the system close the pipe if exec is called successfully. */
    if (::fcntl(pipe[1], F_SETFD, FD_CLOEXEC)) {
        error = errno;
        ::close(pipe[0]);
        ::close(pipe[1]);
        throw SystemException(error, __FILE__, __LINE__);
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
                error = errno;
                ::write(pipe[1], &errno, sizeof(errno));
                ::close(pipe[1]);
                ::_exit(error);
            }
        }

        if (environment.IsEmpty() && (arguments == NULL)) {
            char *tmp[] = { const_cast<char *>(cmd.PeekBuffer()), NULL };
            ::execv(cmd.PeekBuffer(), tmp);
        } else {
            int cntArgs = 1;
            if (arguments != NULL) {
                for (int i = 0; (arguments[i] != NULL); i++) {
                    cntArgs++;
                }
            }
            vislib::RawStorage args((cntArgs + 1) * sizeof(char *));
            args.As<char *>()[0] = const_cast<char *>(cmd.PeekBuffer());
            args.As<char *>()[cntArgs] = NULL;
            for (int i = 1; i < cntArgs; i++) {
                args.As<char *>()[i] = const_cast<char *>(arguments[i - 1]);
            }
            
            ::execve(cmd.PeekBuffer(), args.As<char *>(), 
                reinterpret_cast<char **>(const_cast<void *>(
                static_cast<const void *>(environment))));
        }

        /* exec failed at this point, so report error to parent. */
        ::write(pipe[1], &errno, sizeof(errno));
        error = errno;
        ::close(pipe[1]);
        ::_exit(error);
    
    } else if (this->pid < 0) {
        /* Process creating failed. */
        error = errno;
        ::close(pipe[0]);       // There is no child which could write.
        ::close(pipe[1]);       // We do not need the write end any more.
        throw SystemException(error, __FILE__, __LINE__);

    } else {
        /* Process was spawned, we are in parent. */
        ASSERT(this->pid > 0);
        ::close(pipe[1]);       // We do not need the write end any more.

        /* Try to read error from child process if e. g. exec failed. */
        if (::read(pipe[0], &errno, sizeof(errno)) > 0) {
            /* Child process wrote an error code, so report it. */
            // Preserve errno for exception.
            error = errno;
            this->pid = -1;
            ::close(pipe[0]);
            throw SystemException(error, __FILE__, __LINE__);
        } else {
            /* Reading error failed, so the child already closed the pipe. */
            ::close(pipe[0]);
        }
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::Terminate
 */
void vislib::sys::Process::Terminate(const DWORD exitCode) {
#ifdef _WIN32
    if (!::TerminateProcess(this->hProcess, exitCode)) {
        throw SystemException(__FILE__, __LINE__);
    }
    
    ::CloseHandle(this->hProcess);
    this->hProcess = NULL;
#else /* _WIN32 */
    if (::kill(this->pid, SIGKILL) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    this->pid = -1;
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
