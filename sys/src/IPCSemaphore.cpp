/*
 * IPCSemaphore.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/IPCSemaphore.h"

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/SystemException.h"


#ifndef _WIN32

/*
 * vislib::sys::IPCSemaphore::IPCSemaphore
 */
vislib::sys::IPCSemaphore::IPCSemaphore(const char name, 
        const long initialCount, const long maxCount) : id(-1) {
    this->init(name, initialCount, maxCount);
}


/*
 * vislib::sys::IPCSemaphore::IPCSemaphore
 */
vislib::sys::IPCSemaphore::IPCSemaphore(const char *name, 
        const long initialCount, const long maxCount) : id(-1) {
    this->init(name[0], initialCount, maxCount);
}


/*
 * vislib::sys::IPCSemaphore::~IPCSemaphore
 */
vislib::sys::IPCSemaphore::~IPCSemaphore(void) {
    ::semctl(this->id, MEMBER_IDX, IPC_RMID, 0);
}


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

    // TODO
    //if (this->getCount() == SEM_RESOURCE_MAX) {
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


/*
 * vislib::sys::IPCSemaphore::init
 */
void vislib::sys::IPCSemaphore::init(const char name, const long initialCount,
          const long maxCount) {
    // TODO maxCount is not handled!
    key_t key = ::ftok(".", name);   // TODO: Ist das sinnvoll? Eher nicht ...
    union semun options;

    /* Try to open existing semaphore. */
    if ((this->id = ::semget(key, MEMBER_IDX, DFT_PERMS)) == -1) {
        /* Semaphore does not exist, create a new one. */   

        this->id = ::semget(key, 1, IPC_CREAT | IPC_EXCL | DFT_PERMS));
        ASSERT(this->id != -1); // TODO: Throw exception here?

        /* Set initial count. */
        if (this->id != -1) {
            options.val = static_cast<int>(initialCount);
            ::semctl(this->id, MEMBER_IDX, SETVAL, options);
        }
    }
}

#endif /* !_WIN32 */
