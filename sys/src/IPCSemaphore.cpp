/*
 * IPCSemaphore.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/IPCSemaphore.h"


#ifndef _WIN32
#include <sys/ipc.h>
#include <sys/types.h>

#include "vislib/String.h"
#endif /* !_WIN32 */

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::IPCSemaphore::IPCSemaphore
 */
vislib::sys::IPCSemaphore::IPCSemaphore(const char name, 
        const long initialCount, const long maxCount) {
#ifdef _WIN32
    this->handle = NULL;    // Do not call superclass ctor!
#else /* _WIN32 */
    this->id = -1;
#endif /* _WIN32 */

    char n[] = { name, 0 };
    this->init(n, initialCount, maxCount);
}


/*
 * vislib::sys::IPCSemaphore::IPCSemaphore
 */
vislib::sys::IPCSemaphore::IPCSemaphore(const char *name, 
        const long initialCount, const long maxCount) {
#ifdef _WIN32
    this->handle = NULL;    // Do not call superclass ctor!
#else /* _WIN32 */
    this->id = -1;
#endif /* _WIN32 */

    this->init(name, initialCount, maxCount);
}


/*
 * vislib::sys::IPCSemaphore::~IPCSemaphore
 */
vislib::sys::IPCSemaphore::~IPCSemaphore(void) {
#ifndef _WIN32
	::semctl(this->id, MEMBER_IDX, IPC_RMID, 0);
#endif /* !_WIN32 */
}


#ifndef _WIN32

/*
 * vislib::sys::IPCSemaphore::Lock
 */
void vislib::sys::IPCSemaphore::Lock(void) {
    struct sembuf lock = { MEMBER_IDX, -1, 0 };

    lock.sem_num = MEMBER_IDX;
        
    if ((::semop(this->id, &lock, 1)) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }
}


/*
 * vislib::sys::IPCSemaphore::TryLock
 */
bool vislib::sys::IPCSemaphore::TryLock(void) {
    struct sembuf lock = { MEMBER_IDX, -1, IPC_NOWAIT };
    int error = 0;

    if (this->getCount() < 1) {
        return false;
    }

    lock.sem_num = MEMBER_IDX;
        
    if ((::semop(this->id, &lock, 1)) == -1) {
        if ((error = ::GetLastError()) == EAGAIN) {
            return false;
        } else {
            throw SystemException(__FILE__, __LINE__);
        }
    }

    return true;
}


/*
 * vislib::sys::IPCSemaphore::Unlock
 */
void vislib::sys::IPCSemaphore::Unlock(void) {
    struct sembuf unlock = { MEMBER_IDX, 1, IPC_NOWAIT };

    //TODO: This will not work
    //if (this->getCount() == this->maxCount) {
    //    /* Semaphore is not locked. */
    //    return;
    //}

    unlock.sem_num = MEMBER_IDX;

    if ((::semop(this->id, &unlock, 1)) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }
}


/*
 * vislib::sys::IPCSemaphore::DFT_PERMS 
 */
const int vislib::sys::IPCSemaphore::DFT_PERMS = 0666;


/*
 * vislib::sys::IPCSemaphore::MEMBER_IDX
 */
const int vislib::sys::IPCSemaphore::MEMBER_IDX = 0;



/*
 * vislib::sys::IPCSemaphore::getCount
 */
int vislib::sys::IPCSemaphore::getCount(void) {
    return ::semctl(this->id, MEMBER_IDX, GETVAL, 0);
}

#endif /* !_WIN32 */



/*
 * vislib::sys::IPCSemaphore::IPCSemaphore
 */
vislib::sys::IPCSemaphore::IPCSemaphore(const IPCSemaphore& rhs) {
    throw UnsupportedOperationException("IPCSemaphore::IPCSemaphore",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::IPCSemaphore::operator =
 */
vislib::sys::IPCSemaphore& vislib::sys::IPCSemaphore::operator =(
        const IPCSemaphore& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}


/*
 * vislib::sys::IPCSemaphore::init
 */
void vislib::sys::IPCSemaphore::init(const char *name, const long initialCount,
          const long maxCount) {
	long m = (maxCount > 0) ? maxCount : 1;
	long i = (initialCount < 0) ? 0 : ((initialCount > m) ? m : initialCount);

	ASSERT(m > 0);
	ASSERT(i >= 0);
	ASSERT(i <= m);

#ifdef _WIN32
    ASSERT(this->handle == NULL);

    /* Try to open existing semaphore. */
    if ((this->handle = ::OpenSemaphoreA(SYNCHRONIZE | SEMAPHORE_MODIFY_STATE, 
            FALSE, name)) == NULL) {
        this->handle = ::CreateSemaphoreA(NULL, i, m, name);
    }
    ASSERT(this->handle != NULL);

#else /* _WIN32 */

    this->maxCount = m;

    /* Remove Windows kernel namespaces from the name. */
    StringA n(name);
    if (n.StartsWithInsensitive("global\\") 
            || n.StartsWithInsensitive("local\\")) {
        n.Remove(0, n.Find('\\'));
    }
    ASSERT(n.Length() > 0);

    key_t key = ::ftok("/", n[0]);   // TODO: Ist das sinnvoll? Eher nicht ...

    /* Try to open existing semaphore. */
    if ((this->id = ::semget(key, MEMBER_IDX, DFT_PERMS)) == -1) {
        /* Semaphore does not exist, create a new one. */   

        this->id = ::semget(key, 1, IPC_CREAT | IPC_EXCL | DFT_PERMS);
        ASSERT(this->id != -1); // TODO: Throw exception here?

        /* Set initial count. */
        if (this->id != -1) {
            ::semctl(this->id, MEMBER_IDX, SETVAL, i);
        }
    }

#endif /* _WIN32 */
}
