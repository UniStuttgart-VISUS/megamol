/*
 * WorkItemCompletedListener.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_WORKITEMCOMPLETEDLISTENER_H_INCLUDED
#define VISLIB_WORKITEMCOMPLETEDLISTENER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Runnable.h"


namespace vislib {
namespace sys {


    /**
     * Classes implementing this interface can listen for work items being 
     * completed by a ThreadPool. This callback mechanism is also used to
     * return ownership of the Runnable object and its user data. See the
     * documentation of ThreadPool for further information.
     */
    class WorkItemCompletedListener {

    public:

        /** Ctor. */
        WorkItemCompletedListener(void);

        /** Dtor. */
        virtual ~WorkItemCompletedListener(void);

        /**
         * The thread pool calls this method once a work item (Runnable) has been 
         * completed.

         TODO documentation
         *
         * @param runnable
         * @param userData
         */
        virtual void OnWorkItemCompleted(Runnable *runnable, 
            void *userData, const DWORD exitCode) throw() = 0;
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_WORKITEMCOMPLETEDLISTENER_H_INCLUDED */

