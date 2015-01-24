/*
 * SyncObject.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYNCOBJECT_H_INCLUDED
#define VISLIB_SYNCOBJECT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {
namespace sys {

    /**
     * This superclass defines the interface for all synchronisation objects. 
     * Such a facade class allows the implementation of an autolock mechanism
     * that releases a lock when the active block is left.
     *
     * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
     */
    class SyncObject {

    public:

        /** Dtor. */
        virtual ~SyncObject(void);

        /**
         * Acquire the lock.
         *
         * @throws SystemException If the lock could not be acquired.
         */
        virtual void Lock(void) = 0;

        /**
         * Release the lock.
         *
         * @throw SystemException If the lock could not be released.
         */
        virtual void Unlock(void) = 0;

    protected:

        /** Ctor. */
        inline SyncObject(void) {}

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline SyncObject(const SyncObject& rhs) {}

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline SyncObject& operator =(const SyncObject& rhs) {
            return *this;
        }
    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SYNCOBJECT_H_INCLUDED */
