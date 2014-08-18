/*
 * LinuxDaemon.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_LINUXDAEMON_H_INCLUDED
#define VISLIB_LINUXDAEMON_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifndef _WIN32

#include "vislib/String.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <syslog.h>


namespace vislib {
namespace sys {


    /**
     * This is the superclass for classes that implement a Linux daemon. It 
     * faciliates the creation and running of daemons.
     *
     * Note: The template is intended for compatibility with Windows services
     * and possible future interface unification.
     */
    template<class T> class LinuxDaemon {

    public:

        /** Characters to use in this class. */
        typedef typename T::Char Char;

        /** String to use in this class. */
        typedef String<Char> String;

        /** Dtor. */
        ~LinuxDaemon(void);

        /**
         * Start the daemon. A new process will be forked which exectures the
         * OnRun method.
         */
        bool Run(void);

        /**
         * This method performs the daemons's task. It is called once the 
         * daemon has been initialised.
         *
         * @param argc Reserved. Must be 0.
         * @param argv Reserved. Must be NULL.
         *
         * @return The return code of the daemon.
         */
        virtual DWORD OnRun(const DWORD argc, const Char **argv) = 0;

    protected:

        /** Ctor. */
        LinuxDaemon(const String& name);

    };


    /** Instantiation for ANSI characters. */
    typedef LinuxDaemon<CharTraitsA> LinuxDaemonA;

    /** Instantiation for wide characters. */
    typedef LinuxDaemon<CharTraitsW> LinuxDaemonW;

} /* end namespace sys */
} /* end namespace vislib */


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
template<class T> vislib::sys::LinuxDaemon<T>::LinuxDaemon(const String& name) {
}

#endif /* !_WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_LINUXDAEMON_H_INCLUDED */
