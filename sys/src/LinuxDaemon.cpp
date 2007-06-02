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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <syslog.h>


/*
 * vislib::sys::LinuxDaemon<T>::~LinuxDaemon
 */
template<class T> vislib::sys::LinuxDaemon<T>::~LinuxDaemon(void) {
}


/*
 * vislib::sys::LinuxDaemon<T>::Run
 */
template<class T> bool vislib::sys::LinuxDaemon<T>::Run(void) {
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
 * vislib::sys::LinuxDaemon<T>::LinuxDaemon
 */
template<class T> vislib::sys::LinuxDaemon<T>::LinuxDaemon(void) {
}

#endif /* !_WIN32 */
