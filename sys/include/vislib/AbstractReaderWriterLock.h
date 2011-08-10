/*
 * AbstractReaderWriterLock.h
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTREADERWRITERLOCK_H_INCLUDED
#define VISLIB_ABSTRACTREADERWRITERLOCK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/SyncObject.h"


namespace vislib {
namespace sys {


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
        AbstractReaderWriterLock(void);

        /** Dtor. */
        virtual ~AbstractReaderWriterLock(void);

        /**
         * Acquire an exclusive lock.
         */
        virtual void Lock(void);

        /**
         * Aquires an exclusive lock
         */
        virtual void LockExclusive(void) = 0;

        /**
         * Aquires a shared lock
         */
        virtual void LockShared(void) = 0;

        /**
         * Release an exclusive lock
         */
        virtual void Unlock(void);

        /**
         * Release an exclusive lock
         */
        virtual void UnlockExclusive(void) = 0;

        /**
         * Release a shared lock
         */
        virtual void UnlockShared(void) = 0;

    private:

        /** Forbidden copy ctor. */
        AbstractReaderWriterLock(const AbstractReaderWriterLock& src);

        /** Forbidden assignment operator */
        AbstractReaderWriterLock& operator=(const AbstractReaderWriterLock& rhs);

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTREADERWRITERLOCK_H_INCLUDED */

