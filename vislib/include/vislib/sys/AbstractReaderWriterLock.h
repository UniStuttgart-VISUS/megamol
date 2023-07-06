/*
 * AbstractReaderWriterLock.h
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/sys/SyncObject.h"


namespace vislib::sys {


/**
 * Abstract base class for reader-writer locks, which allow for exclusive
 * and shared lock. A typical scenario is to lock a data object allowing
 * multiple concurrent read operations (shared lock) but synchronizes
 * individual write operations (exclusive lock).
 *
 * The behaviour derived from SyncObject uses to the exclusive lock.
 */
class AbstractReaderWriterLock : public SyncObject {

public:
    /** Ctor. */
    AbstractReaderWriterLock();

    /** Dtor. */
    ~AbstractReaderWriterLock() override;

    /**
     * Acquire an exclusive lock.
     */
    void Lock() override;

    /**
     * Acquires an exclusive lock
     */
    virtual void LockExclusive() = 0;

    /**
     * Tries to acquire the lock
     */
    virtual bool TryLock(unsigned long const timeout = 0) {
        return false;
    };

    /**
     * Acquires a shared lock
     */
    virtual void LockShared() = 0;

    /**
     * Release an exclusive lock
     */
    void Unlock() override;

    /**
     * Release an exclusive lock
     */
    virtual void UnlockExclusive() = 0;

    /**
     * Release a shared lock
     */
    virtual void UnlockShared() = 0;

private:
    /** Forbidden copy ctor. */
    AbstractReaderWriterLock(const AbstractReaderWriterLock& src);

    /** Forbidden assignment operator */
    AbstractReaderWriterLock& operator=(const AbstractReaderWriterLock& rhs);
};

} // namespace vislib::sys

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
