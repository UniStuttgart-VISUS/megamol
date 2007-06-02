/*
 * LinuxDaemon.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/LinuxDaemon.h"

#include "vislib/SystemException.h"
#include "vislib/Trace.h"


#ifndef _WIN32


/*
 * vislib::sys::LinuxDaemon::~LinuxDaemon
 */
vislib::sys::LinuxDaemon::~LinuxDaemon(void) {
}


/*
 * vislib::sys::LinuxDaemon::Run
 */
bool vislib::sys::LinuxDaemon::Run(void) {
    pid_t pid;                  // The process ID of the daemon process.
    pid_t sid;                  // The session ID of the daemon process.
        
    /* Fork off the parent process */
    pid = ::fork();
    if (pid < 0) {
        /* Forking failed. */
        throw SystemException(__FILE__, __LINE__);

    } else if (pid > 0) {
        /* Forking succeeded, leave parent process. */
        return true;
    }
    /* We are in the forked child process now. */


    /* Change the file mode mask */
    ::umask(0);


    // Open logs
    

    /* Create a new SID for the child process. */
    if ((sid = ::setsid()) < 0) {
        // TODO error code, log
        return 1;
    }
        
    /* Change the current working directory. */
    if (::chdir("/") < 0) {
        // TODO error code, log
        return 1;
    }
        
    /* Close standard file descriptors. */
    ::close(STDIN_FILENO);
    ::close(STDOUT_FILENO);
    ::close(STDERR_FILENO);

    ::exit(this->OnRun(0, NULL));
        
    return false;
}


/*
 * vislib::sys::LinuxDaemon::LinuxDaemon
 */
vislib::sys::LinuxDaemon::LinuxDaemon(void) {
}

#endif /* !_WIN32 */
