/*
 * Thread.cpp  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef _WIN32
#include <unistd.h>
#endif /* !_WIN32 */

#include "vislib/assert.h"
#include "vislib/Thread.h"
#include "vislib/Trace.h"


/*
 * vislib::sys::Thread::Sleep
 */
void vislib::sys::Thread::Sleep(const DWORD millis) {
#ifdef _WIN32
	::Sleep(millis);
#else /* _WIN32 */
	if (millis >= 1000) {
		/* At least one second to sleep. Use ::sleep() for full seconds. */
		for (DWORD i = 0; i < millis % 1000; i++) {
			::sleep(1000);
		}
	}

	::usleep((millis % 1000) * 1000);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Thread
 */
vislib::sys::Thread::Thread(Runnable& runnable) 
        : exitCode(static_cast<DWORD>(-1)), id(0), state(NEW) {
#ifdef _WIN32
	this->handle = NULL;

#else /* _WIN32 */
    ::pthread_attr_init(&this->attribs);
    ::pthread_attr_setscope(&this->attribs, PTHREAD_SCOPE_SYSTEM);
    ::pthread_attr_setdetachstate(&this->attribs, PTHREAD_CREATE_JOINABLE);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Thread
 */
vislib::sys::Thread::Thread(RunnableFunc threadFunc) 
        : exitCode(static_cast<DWORD>(-1)), id(0), state(NEW) {
#ifdef _WIN32
	this->handle = NULL;

#else /* _WIN32 */
    ::pthread_attr_init(&this->attribs);
    ::pthread_attr_setscope(&this->attribs, PTHREAD_SCOPE_SYSTEM);
    ::pthread_attr_setdetachstate(&this->attribs, PTHREAD_CREATE_JOINABLE);

#endif /* _WIN32 */
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
    ::pthread_attr_destroy(&this->attribs);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Start
 */
bool vislib::sys::Thread::Start(void) {
    State state = this->GetState();
    if ((state != NEW) && (state != FINISHED)) {
        return false;
    }

#ifdef _WIN32
	if ((this->handle = ::CreateThread(NULL, 0, Thread::ThreadFunc, this, 0,
            &this->id)) != NULL) {
		this->state = RUNNING;
        return true;

    } else {
		TRACE(_T("CreateThread() failed with error %d.\n"), ::GetLastError());
        return false;
    }

#else /* _WIN32 */
	if (::pthread_create(&this->id, &this->attribs, Thread::ThreadFunc, 
            static_cast<void *>(this)) == 0) {
        this->threadMutex.Lock();
		this->state = RUNNING;
		this->threadMutex.Unlock();
        return true;

    } else {
		TRACE("pthread_create() failed with error %d.\n", errno);
        return false;
    }
#endif /* _WIN32 */
}

/*
 * vislib::sys::Thread::Terminate
 */
bool vislib::sys::Thread::Terminate(const int exitCode) {
    State state = this->GetState();
    if ((state != RUNNING) && (state != SUSPENDED)) {
        return true;
    }

#ifdef _WIN32
    if (::TerminateThread(this->handle, exitCode) == TRUE) {
		this->exitCode = exitCode;
		this->state = FINISHED;
        return true;

    } else {
		TRACE(_T("TerminateThread() failed with error %d.\n"), ::GetLastError());
        return false;
    }

#else /* _WIN32 */
    if (::pthread_cancel(this->id) == 0) {
		this->exitCode = exitCode;
		this->state = FINISHED;
        return true;

    } else {
		TRACE(_T("pthread_cancel() failed with error %d.\n"), errno);
        return false;
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::Wait(void)
 */
bool vislib::sys::Thread::Wait(void) {
#ifdef _WIN32
    if (this->handle != NULL) {
        return (::WaitForSingleObject(this->handle, INFINITE) == WAIT_OBJECT_0);
    } else {
        return true;
    }

#else /* _WIN32 */
    if (this->id != 0) {
        return (::pthread_join(this->id, NULL) == 0);
    } else {
        return true;
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Thread::operator ()
 */
bool vislib::sys::Thread::operator ()(void) {
    State state = this->GetState();
    if ((state != NEW) && (state != FINISHED)) {
        return false;
    }

	this->state = RUNNING;
	this->exitCode = this->run();
	this->state = FINISHED;
    return true;
}


/*
 * vislib::sys::Thread::run
 */
int vislib::sys::Thread::run(void) {
	return 0;
}


/*
 * vislib::sys::Thread::ThreadFunc
 */
#ifdef _WIN32
DWORD WINAPI vislib::sys::Thread::ThreadFunc(void *param) {
#else /* _WIN32 */
void *vislib::sys::Thread::ThreadFunc(void *param) {
#endif /* _WIN32 */
	ASSERT(param != NULL);
	ASSERT(dynamic_cast<Thread *>(static_cast<Thread *>(param)) != NULL);

	int retval = 0;

	Thread *t = static_cast<Thread *>(param);
	retval = t->run();
    
	t->exitCode = retval;
	t->state = FINISHED;
	TRACE(_T("Thread [%u] finished with exit code %d.\n"), t->id, retval);

#ifdef _WIN32
	return static_cast<DWORD>(retval);
#else /* _WIN32 */
    return reinterpret_cast<void *>(retval);
#endif /* _WIN32 */
}



