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

#include "the/assert.h"
#include "vislib/AutoHandle.h"
#include "vislib/Console.h"
#include "vislib/error.h"
#include "the/argument_exception.h"
#include "the/invalid_operation_exception.h"
#ifdef _WIN32 // TODO: PAM disabled
#include "vislib/ImpersonationContext.h"
#endif // TODO
#include "vislib/Path.h"
#include "vislib/RawStorage.h"
#include "the/system/system_exception.h"
#include "the/trace.h"
#include "the/not_supported_exception.h"
#include "the/not_implemented_exception.h"
#include "the/string.h"
#include "the/text/string_builder.h"


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
            throw the::system::system_exception(__FILE__, __LINE__);
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
void vislib::sys::Process::Exit(const unsigned int exitCode) {
#ifdef _WIN32
    ::ExitProcess(exitCode);
#else /* _WIN32 */
    ::exit(exitCode);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::ModuleFileNameA
 */
the::astring vislib::sys::Process::ModuleFileNameA(const PID processID) {
#ifdef _WIN32
    AutoHandle hProcess(true);
    the::astring retval(MAX_PATH, ' ');

    if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION, TRUE, processID))
            == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if (!GetModuleFileNameA(static_cast<HMODULE>(static_cast<HANDLE>(hProcess)),
            const_cast<char*>(retval.c_str()), MAX_PATH)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    the::astring procAddr;
    the::astring retval;
    char *retvalPtr = NULL;
    int len = -1;

    the::text::astring_builder::format_to(procAddr, "/proc/%d/exe", processID);
    retval = the::astring(1024, ' ');
    retvalPtr = const_cast<char*>(retval.c_str());
    len = ::readlink(procAddr.c_str(), retvalPtr, 1024 + 1);
    if (len == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
    retvalPtr[len] = 0;

    return retval;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::ModuleFileNameA
 */
the::astring vislib::sys::Process::ModuleFileNameA(void) {
#ifdef _WIN32
    the::astring retval(MAX_PATH, ' ');

    if (!GetModuleFileNameA(NULL, const_cast<char*>(retval.c_str()), MAX_PATH)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    return Process::ModuleFileNameA(Process::CurrentID());
#endif /* _WIN32 */
}




/*
 * vislib::sys::Process::ModuleFileNameW
 */
the::wstring vislib::sys::Process::ModuleFileNameW(const PID processID) {
#ifdef _WIN32
    AutoHandle hProcess(true);
    the::wstring retval(MAX_PATH, ' ');

    if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION, TRUE, processID))
            == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if (!GetModuleFileNameW(static_cast<HMODULE>(static_cast<HANDLE>(hProcess)),
            const_cast<wchar_t*>(retval.c_str()), MAX_PATH)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    return the::text::string_converter::to_w(Process::ModuleFileNameA(processID));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::ModuleFileNameW
 */
the::wstring vislib::sys::Process::ModuleFileNameW(void) {
#ifdef _WIN32
    the::wstring retval(MAX_PATH, ' ');

    if (!GetModuleFileNameW(NULL,
            const_cast<wchar_t*>(retval.c_str()), MAX_PATH)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    return the::text::string_converter::to_w(Process::ModuleFileNameA(Process::CurrentID()));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::Owner
 */
void vislib::sys::Process::Owner(const PID processID, the::astring& outUser,
        the::astring *outDomain) {
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
    the::astring::value_type *user = NULL;     // Receives user name.
    the::astring::value_type *domain = NULL;   // Receives domain name.

    /* Clear return values. */
    outUser.clear();
    if (outDomain != NULL) {
        outDomain->clear();
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
                        outUser = the::astring(userLen, ' ');
                        user = const_cast<char*>(outUser.c_str());
                        if (outDomain != NULL) {
                            (*outDomain) = the::astring(domainLen, ' ');
                            domain = const_cast<char*>(outDomain->c_str());
                        } else {
                            tmpDomain.AssertSize(domainLen);
                            domain = tmpDomain.As<the::astring::value_type>();
                        }

                        if (!::LookupAccountSidA(NULL, sid, user, &userLen,
                                domain, &domainLen, &snu)) {
                            error = ::GetLastError();
                            outUser.clear();
                            if (outDomain != NULL) {
                                outDomain->clear();
                            }
                            ::CloseHandle(hToken);
                            ::CloseHandle(hProcess);
                            throw the::system::system_exception(error, __FILE__, __LINE__);
                        }

                    } else {
                        ::CloseHandle(hToken);
                        ::CloseHandle(hProcess);
                        throw the::system::system_exception(error, __FILE__, __LINE__);
                    } /* end if (::LookupAccountSidA(NULL, sid, NULL, ... */

                } else {
                    error = ::GetLastError();
                    throw the::system::system_exception(error, __FILE__, __LINE__);
                } /* end if (::GetTokenInformation(hToken, TokenUser, ...  */

            } else {
                error = ::GetLastError();
                throw the::system::system_exception(error, __FILE__, __LINE__);
            } /* end if (::GetTokenInformation(hToken, TokenUser, ...  */

        } else {
            error = ::GetLastError();
            throw the::system::system_exception(error, __FILE__, __LINE__);
        } /* end if (::OpenProcessToken(hProcess, TOKEN_QUERY, &hToken)) */

    } else {
        error = ::GetLastError();
        throw the::system::system_exception(error, __FILE__, __LINE__);
    } /* end if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION ... */

#else /* _WIN32 */
    // TODO: Try something like ps -p <pid> -o user
    throw the::not_implemented_exception("vislib::sys::Process::Owner", __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::Owner
 */
void vislib::sys::Process::Owner(const PID processID, the::wstring& outUser,
        the::wstring *outDomain) {
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
    the::wstring::value_type *user = NULL;     // Receives user name.
    the::wstring::value_type *domain = NULL;   // Receives domain name.

    /* Clear return values. */
    outUser.clear();
    if (outDomain != NULL) {
        outDomain->clear();
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
                        outUser = the::wstring(userLen, L' ');
                        user = const_cast<wchar_t*>(outUser.c_str());
                        if (outDomain != NULL) {
                            (*outDomain) = the::wstring(domainLen, L' ');
                            domain = const_cast<wchar_t*>(outDomain->c_str());
                        } else {
                            tmpDomain.AssertSize(domainLen);
                            domain = tmpDomain.As<the::wstring::value_type>();
                        }

                        if (!::LookupAccountSidW(NULL, sid, user, &userLen,
                                domain, &domainLen, &snu)) {
                            error = ::GetLastError();
                            outUser.clear();
                            if (outDomain != NULL) {
                                outDomain->clear();
                            }
                            ::CloseHandle(hToken);
                            ::CloseHandle(hProcess);
                            throw the::system::system_exception(error, __FILE__, __LINE__);
                        }

                    } else {
                        ::CloseHandle(hToken);
                        ::CloseHandle(hProcess);
                        throw the::system::system_exception(error, __FILE__, __LINE__);
                    } /* end if (::LookupAccountSidW(NULL, sid, NULL, ... */

                } else {
                    error = ::GetLastError();
                    throw the::system::system_exception(error, __FILE__, __LINE__);
                } /* end if (::GetTokenInformation(hToken, TokenUser, ...  */

            } else {
                error = ::GetLastError();
                throw the::system::system_exception(error, __FILE__, __LINE__);
            } /* end if (::GetTokenInformation(hToken, TokenUser, ...  */

        } else {
            error = ::GetLastError();
            throw the::system::system_exception(error, __FILE__, __LINE__);
        } /* end if (::OpenProcessToken(hProcess, TOKEN_QUERY, &hToken)) */

    } else {
        error = ::GetLastError();
        throw the::system::system_exception(error, __FILE__, __LINE__);
    } /* end if ((hProcess = ::OpenProcess(PROCESS_QUERY_INFORMATION ... */

#else /* _WIN32 */
    the::astring u;
    the::astring d;

    Process::Owner(processID, u, &d);
    if (outDomain != NULL) {
        the::text::string_converter::convert(*outDomain, d);
    }
#endif /* _WIN32 */
}


/* 
 * vislib::sys::Process::OwnerA
 */
the::astring vislib::sys::Process::OwnerA(const PID processID, 
            const bool includeDomain, const bool isLenient) {
    the::astring retval;
    
    try {
#ifdef _WIN32
        if (includeDomain) {
#else /* _WIN32 */
        if (false) {
#endif /* _WIN32 */
            the::astring user;
            Process::Owner(processID, user, &retval);
            retval += "\\";
            retval += user;
        } else {
            Process::Owner(processID, retval, NULL);
        }
    } catch (the::system::system_exception) {
        if (!isLenient) {
            throw;
        }
    }

    return retval;
}


/*
 * vislib::sys::Process::OwnerW
 */
the::wstring vislib::sys::Process::OwnerW(const PID processID, 
            const bool includeDomain, const bool isLenient) {
    the::wstring retval;
    
    try {
#ifdef _WIN32
        if (includeDomain) {
#else /* _WIN32 */
        if (false) {
#endif /* _WIN32 */
            the::wstring user;
            Process::Owner(processID, user, &retval);
            retval += L"\\";
            retval += user;
        } else {
            Process::Owner(processID, retval, NULL);
        }
    } catch (the::system::system_exception) {
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
    throw the::not_supported_exception("Process", __FILE__, __LINE__);
}


/*
 * vislib::sys::Process::GetID
 */
vislib::sys::Process::PID vislib::sys::Process::GetID(void) const {
#ifdef _WIN32
    DWORD retval = ::GetProcessId(this->hProcess);
    if (retval == 0) {
        throw the::system::system_exception(__FILE__, __LINE__);
    } else {
        return retval;
    }
#else /* _WIN32 */
    if (this->pid == -1) {
        throw the::system::system_exception(ESRCH, __FILE__, __LINE__);
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
    THE_ASSERT(command != NULL);

#ifdef _WIN32
    const char **arg = arguments;
    HANDLE hToken = NULL;
    the::astring cmdLine("\"");
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
        throw the::invalid_operation_exception("This process was already created.",
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
            throw the::system::system_exception(__FILE__, __LINE__);
        }

        // TODO: hToken has insufficient permissions to spawn process.
        if (::CreateProcessAsUserA(hToken, NULL, 
                const_cast<char *>(cmdLine.c_str()), NULL, NULL, FALSE, 0,
                const_cast<void *>(static_cast<const void *>(environment)), 
                currentDirectory, &si, &pi)
                == FALSE) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
    } else {
        if (::CreateProcessA(NULL, const_cast<char *>(cmdLine.c_str()), 
                NULL, NULL, FALSE, 0, 
                const_cast<void *>(static_cast<const void *>(environment)),
                currentDirectory, &si, &pi) == FALSE) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
    }

    ::CloseHandle(pi.hThread);
    this->hProcess = pi.hProcess;

#else /* _WIN32 */
   // TODO: PAM disabled
    the::astring query;  // Query to which to expand shell commands.
    the::astring cmd;    // The command actually run.
    unsigned int error;    // Exception error code.
    int pipe[2];    // A pipe to get the error code of exec.

    /* A process must not be started twice. */
    if (this->pid >= 0) {
        throw the::invalid_operation_exception("This process was already created.",
            __FILE__, __LINE__);
    }

    /* Detect and expand shell commands first first. */
    the::text::astring_builder::format_to(query, "which %s", command);
    if (Console::Run(query.c_str(), &cmd) == 0) {
        the::text::string_utility::trim_end(cmd, "\r\n");
    } else {
        cmd = Path::Resolve(command);
    }
    THE_ASSERT(Path::IsAbsolute(cmd));
    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CreateProcess: command is \"%s\"\n",
        cmd.c_str());

    /* Open a pipe to get the exec error codes. */
    if (::pipe(pipe) == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    /* Make the system close the pipe if exec is called successfully. */
    if (::fcntl(pipe[1], F_SETFD, FD_CLOEXEC)) {
        error = errno;
        ::close(pipe[0]);
        ::close(pipe[1]);
        throw the::system::system_exception(error, __FILE__, __LINE__);
    }

    /* Create new process. */
    this->pid = ::fork();
    if (this->pid == 0) {
        /* We are in the child process now. */
        ::close(pipe[0]);       // We do not need the read end any more.

        /** Impersonate as new user. */
        if ((user != NULL) && (password != NULL)) {
            //ic.Impersonate(user, NULL, password);
            THE_ASSERT(false);
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

        if (environment.empty() && (arguments == NULL)) {
            char *tmp[] = { const_cast<char *>(cmd.c_str()), NULL };
            ::execv(cmd.c_str(), tmp);
        } else {
            int cntArgs = 1;
            if (arguments != NULL) {
                for (int i = 0; (arguments[i] != NULL); i++) {
                    cntArgs++;
                }
            }
            vislib::RawStorage args((cntArgs + 1) * sizeof(char *));
            args.As<char *>()[0] = const_cast<char *>(cmd.c_str());
            args.As<char *>()[cntArgs] = NULL;
            for (int i = 1; i < cntArgs; i++) {
                args.As<char *>()[i] = const_cast<char *>(arguments[i - 1]);
            }
            
            ::execve(cmd.c_str(), args.As<char *>(), 
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
        throw the::system::system_exception(error, __FILE__, __LINE__);

    } else {
        /* Process was spawned, we are in parent. */
        THE_ASSERT(this->pid > 0);
        ::close(pipe[1]);       // We do not need the write end any more.

        /* Try to read error from child process if e. g. exec failed. */
        if (::read(pipe[0], &errno, sizeof(errno)) > 0) {
            /* Child process wrote an error code, so report it. */
            // Preserve errno for exception.
            error = errno;
            this->pid = -1;
            ::close(pipe[0]);
            throw the::system::system_exception(error, __FILE__, __LINE__);
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
void vislib::sys::Process::Terminate(const unsigned int exitCode) {
#ifdef _WIN32
    if (!::TerminateProcess(this->hProcess, exitCode)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
    
    ::CloseHandle(this->hProcess);
    this->hProcess = NULL;
#else /* _WIN32 */
    if (::kill(this->pid, SIGKILL) == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    this->pid = -1;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Process::operator =
 */
vislib::sys::Process& vislib::sys::Process::operator =(const Process& rhs) {
    if (this != &rhs) {
        throw the::argument_exception("rhs", __FILE__, __LINE__);
    }

    return *this;
}
