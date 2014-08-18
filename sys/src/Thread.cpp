/*
 * Thread.cpp  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/Thread.h"

#ifndef _WIN32
#include <unistd.h>
#endif /* !_WIN32 */

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"

#include "DynamicFunctionPointer.h"

#include <cstdio>
#include <iostream>

#ifndef _WIN32
/**
 * Return code that marks a thread as still running. Make sure that the value 
 * is the same as on Windows.
 */
#define STILL_ACTIVE (259)
#endif /* !_WIN32 */


/*
 * vislib::sys::Thread::Sleep
 */
void vislib::sys::Thread::Sleep(const DWORD millis) {
#ifdef _WIN32
    ::Sleep(millis);
#else /* _WIN32 */
    if (millis >= 1000) {
        /* At least one second to sleep. Use ::sleep() for full seconds. */
        ::sleep(millis / 1000);
    }

    ::usleep((millis % 1000) * 1000);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Reschedule
 */
void vislib::sys::Thread::Reschedule(void) {
#ifdef _WIN32
#if (_WIN32_WINNT >= 0x0400)
    ::SwitchToThread();
#else
    DynamicFunctionPointer<BOOL (*)(void)> stt("kernel32", "SwitchToThread");
    if (stt.IsValid()) {
        stt();
    } else {
        ::Sleep(0);
    }
#endif
#else /* _WIN32 */
    if (::sched_yield() != 0) {
        throw SystemException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Thread
 */
vislib::sys::Thread::Thread(Runnable *runnable) 
        : id(0), runnable(runnable), runnableFunc(NULL) {
#ifdef _WIN32
    this->handle = NULL;

#else /* _WIN32 */
    ::pthread_attr_init(&this->attribs);
    ::pthread_attr_setscope(&this->attribs, PTHREAD_SCOPE_SYSTEM);
    ::pthread_attr_setdetachstate(&this->attribs, PTHREAD_CREATE_JOINABLE);

    this->exitCode = 0;

#endif /* _WIN32 */

    this->threadFuncParam.thread = this;
    this->threadFuncParam.userData = NULL;
}


/*
 * vislib::sys::Thread::Thread
 */
vislib::sys::Thread::Thread(Runnable::Function runnableFunc) 
        : id(0), runnable(NULL), runnableFunc(runnableFunc) {
#ifdef _WIN32
    this->handle = NULL;

#else /* _WIN32 */
    ::pthread_attr_init(&this->attribs);
    ::pthread_attr_setscope(&this->attribs, PTHREAD_SCOPE_SYSTEM);
    ::pthread_attr_setdetachstate(&this->attribs, PTHREAD_CREATE_JOINABLE);

    this->exitCode = 0;

#endif /* _WIN32 */

    this->threadFuncParam.thread = this;
    this->threadFuncParam.userData = NULL;
}


/*
 * vislib::sys::Thread::~Thread
 */
vislib::sys::Thread::~Thread(void) {
#ifdef _WIN32
    if (this->handle != NULL) {
        ::CloseHandle(this->handle);
    }

#else /* _WIIN32 */
    // TODO: Dirty hack, don't know whether this is always working.
    if (this->id != 0) {
        ::pthread_detach(this->id);
        ::pthread_attr_destroy(&this->attribs);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::GetExitCode
 */
DWORD vislib::sys::Thread::GetExitCode(void) const {
#ifdef _WIN32
    DWORD retval = 0;
    if (::GetExitCodeThread(this->handle, &retval) == FALSE) {
        throw SystemException(__FILE__, __LINE__);
    }

    return retval;

#else /* _WIN32 */
    return this->exitCode;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::IsRunning
 */
bool vislib::sys::Thread::IsRunning(void) const {
    try {
#ifdef _WIN32
        return ((this->handle != NULL)
#else /* _WIN32 */
        return ((this->id != 0)
#endif /* _WIN32 */
            && (this->GetExitCode() == STILL_ACTIVE));
    } catch (SystemException) {
        return false;
    }
}


/*
 * vislib::sys::Thread::Join
 */
void vislib::sys::Thread::Join(void) {
#ifdef _WIN32
    if (this->handle != NULL) {
        if (::WaitForSingleObject(this->handle, INFINITE) == WAIT_FAILED) {
            throw SystemException(__FILE__, __LINE__);
        }
    }

#else /* _WIN32 */
    if (this->id != 0) {
        if (::pthread_join(this->id, NULL) != 0) {
            throw SystemException(__FILE__, __LINE__);
        }
        this->id = 0;
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Start
 */
bool vislib::sys::Thread::Start(void *userData) {
    if (this->IsRunning()) {
        /*
         * The thread must not be started twice at the same time as this would
         * leave unclosed handles.
         */
        return false;
    }

    /* Set the user data. */
    this->threadFuncParam.userData = userData;

    /* Inform the runnable that we are about to start a thread. */
    if (this->runnable != NULL) {
        this->runnable->OnThreadStarting(userData);
    }

#ifdef _WIN32
    /* Close possible old handle. */
    if (this->handle != NULL) {
        ::CloseHandle(this->handle);
    }

    if ((this->handle = ::CreateThread(NULL, 0, Thread::ThreadFunc, 
            &this->threadFuncParam, 0, &this->id)) != NULL) {
        return true;

    } else {
        VLTRACE(Trace::LEVEL_VL_ERROR, "CreateThread() failed with error %d.\n", 
            ::GetLastError());
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    if (::pthread_create(&this->id, &this->attribs, Thread::ThreadFunc, 
            static_cast<void *>(&this->threadFuncParam)) == 0) {
        this->exitCode = STILL_ACTIVE;  // Mark thread as running.
        return true;

    } else {
        VLTRACE(Trace::LEVEL_VL_ERROR, "pthread_create() failed with error %d.\n", 
            ::GetLastError());
        throw SystemException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Terminate
 */
bool vislib::sys::Thread::Terminate(const bool forceTerminate, 
                                    const int exitCode) {
    ASSERT(exitCode != STILL_ACTIVE);   // User should never set this.

    if (forceTerminate) {
        /* Force immediate termination of the thread. */

#ifdef _WIN32
        if (::TerminateThread(this->handle, exitCode) == FALSE) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "TerminateThread() failed with error "
                "%d.\n", ::GetLastError());
            throw SystemException(__FILE__, __LINE__);
        }

        return true;

#else /* _WIN32 */
        this->exitCode = exitCode;

        if (::pthread_cancel(this->id) != 0) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "pthread_cancel() failed with error "
                "%d.\n", ::GetLastError());
            throw SystemException(__FILE__, __LINE__);
        }

        return true;
#endif /* _WIN32 */

    } else {
        return this->TryTerminate(true);
    } /* end if (forceTerminate) */
}


/*
 * vislib::sys::Thread::TryTerminate
 */
bool vislib::sys::Thread::TryTerminate(const bool doWait) {
    
    if (this->runnable == NULL) {
        throw IllegalStateException("TryTerminate can only be used, if the "
            "thread is using a Runnable.", __FILE__, __LINE__);
    }
    ASSERT(this->runnable != NULL); 

    if (this->runnable->Terminate()) {
        /*
         * Wait for thread to finish, if Runnable acknowledged and waiting was
         * requested.
         */
        if (doWait) {
            this->Join();
        } 
        
        return true;

    } else {
        /* Runnable did not acknowledge. */
        return false;
    }
}


#ifndef _WIN32
/*
 * vislib::sys::Thread::CleanupFunc
 */
void vislib::sys::Thread::CleanupFunc(void *param) {
    ASSERT(param != NULL);

    Thread *t = static_cast<Thread *>(param);

    /* 
     * In case the thread has still an exit code of STILL_ACTIVE, set a new one
     * to mark the thread as finished.
     */
    if (t->exitCode == STILL_ACTIVE) {
        VLTRACE(Trace::LEVEL_VL_WARN, "CleanupFunc called with exit code "
            "STILL_ACTIVE");
        t->exitCode = 0;
    }
}
#endif /* !_WIN32 */


/*
 * vislib::sys::Thread::ThreadFunc
 */
#ifdef _WIN32
DWORD WINAPI vislib::sys::Thread::ThreadFunc(void *param) {
#else /* _WIN32 */
void *vislib::sys::Thread::ThreadFunc(void *param) {
#endif /* _WIN32 */
    ASSERT(param != NULL);

    int retval = 0;
    ThreadFuncParam *tfp = static_cast<ThreadFuncParam *>(param);
    Thread *t = tfp->thread;
    ASSERT(t != NULL);

#ifndef _WIN32
    pthread_cleanup_push(Thread::CleanupFunc, t);
#endif /* !_WIN32 */

    if (t->runnable != NULL) {
        t->runnable->OnThreadStarted(tfp->userData);
        retval = t->runnable->Run(tfp->userData);
    } else {
        ASSERT(t->runnableFunc != NULL);
        retval = t->runnableFunc(tfp->userData);
    }
    ASSERT(retval != STILL_ACTIVE); // Thread should never use STILL_ACTIVE!

#ifndef _WIN32    
    t->exitCode = retval;
    pthread_cleanup_pop(1);
#endif /* !_WIN32 */

    VLTRACE(Trace::LEVEL_VL_INFO, "Thread [%u] has exited with code %d (0x%x).\n",
        t->id, retval, retval);

#ifdef _WIN32
    return static_cast<DWORD>(retval);
#else /* _WIN32 */
    return reinterpret_cast<void *>(retval);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Thread
 */
vislib::sys::Thread::Thread(const Thread& rhs) {
    throw UnsupportedOperationException("vislib::sys::Thread::Thread",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::Thread::operator =
 */
vislib::sys::Thread& vislib::sys::Thread::operator =(const Thread& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs_", __FILE__, __LINE__);
    }

    return *this;
}
