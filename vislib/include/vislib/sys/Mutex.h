/*
 * Mutex.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 * Copyright 2019 MegaMol Dev Team
 */

#ifndef VISLIB_MUTEX_H_INCLUDED
#define VISLIB_MUTEX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <mutex>

#include "vislib/sys/Lockable.h"
#include "vislib/sys/SyncObject.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

/**
 * A platform independent mutex wrapper.
 *
 * Implementation notes: On Windows systems, this mutex can be used for
 * inter-process synchronization tasks. If you just need to synchronize
 * threads of a single process, consider using the critical section as it
 * is faster.
 *
 * @author Christoph Mueller
 */
class Mutex : public SyncObject {

public:
    /**
     * Create a new mutex, which is initially not locked.
     */
    Mutex() = default;

    /** Dtor. */
    ~Mutex() override = default;

    /**
     * Acquire a lock on the mutex for the calling thread. The method blocks
     * until the lock is acquired.
     *
     * @throws std::system_error when errors occur including OS errors
     */
    void Lock() override;

    /**
     * Try to acquire a lock on the mutex for the calling thread. If the
     * mutex is already locked by another thread, the method will return
     * after the specified timeout and the return value is false. The
     * method is non-blocking if the timeout is set zero.
     *
     * @param timeout The timeout for acquiring the mutex.
     *
     * @return true, if the lock was acquired, false, if not.
     */
    bool TryLock(const DWORD timeout = 0);

    /**
     * Release the mutex.
     */
    void Unlock() override;

private:
    /**
     * Forbidden copy ctor.
     *
     * @param rhs The object to be cloned.
     *
     * @throws UnsupportedOperationException Unconditionally.
     */
    Mutex(const Mutex& rhs);

    /**
     * Forbidden assignment.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     *
     * @throws IllegalParamException If (this != &rhs).
     */
    Mutex& operator=(const Mutex& rhs);

    std::recursive_timed_mutex mutex;
};


/** name typedef for Lockable with this SyncObject */
typedef Lockable<Mutex> MutexLockable;


} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MUTEX_H_INCLUDED */
