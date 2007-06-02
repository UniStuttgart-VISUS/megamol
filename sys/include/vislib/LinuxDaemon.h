/*
 * LinuxDaemon.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_LINUXDAEMON_H_INCLUDED
#define VISLIB_LINUXDAEMON_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifndef _WIN32

#include "vislib/String.h"


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
        virtual DWORD OnRun(const DWORD argc, const Char *argv) = 0;

    protected:

        /** Ctor. */
        LinuxDaemon(void);

    };


    /** Instantiation for ANSI characters. */
    typedef LinuxDaemon<CharTraitsA> LinuxDaemonA;

    /** Instantiation for wide characters. */
    typedef LinuxDaemon<CharTraitsW> LinuxDaemonW;
    
} /* end namespace sys */
} /* end namespace vislib */

#endif /* !_WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_LINUXDAEMON_H_INCLUDED */
