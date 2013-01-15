/*
 * RawStoragePool.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RAWSTORAGEPOOL_H_INCLUDED
#define VISLIB_RAWSTORAGEPOOL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/RawStorage.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/StackTrace.h"


namespace vislib {


    /**
     * This class implements a pool of RawStorage instances for reuse.
     *
     * The strategy of RaiseAtLeast is to minimise the number of required
     * reallocations, i.e. if possible, a RawStorage that can already fulfill
     * the request is searched. If such a RawStorage exists, the strategy is
     * best fit. If no such RawStorage exists, the strategy is random and the
     * returned object is resized. If no unused RawStorage exists at all, a
     * new one is created and added to the pool.
     *
     * Note that RawStoragePool is not thread-safe at all!
     */
    class RawStoragePool {

    public:

        /** Ctor. */
        RawStoragePool(void);

        /** Dtor. */
        ~RawStoragePool(void);

        /**
         * Remove all elements from the pool.
         */
        void Clear(void);

        /**
         * Return an unused RawStorage of at least 'size' bytes.
         *
         * The object returned remains owned by the RawStoragePool. The user
         * must ensure that the pool exists as long as the returned object is in
         * use.
         *
         * The returned RawStorage is guaranteed to provide at least 'size'
         * bytes. If the request cannot be fulfilled, an exception is thrown.
         * This is only the case in low-memory situations when the pool cannot
         * create new RawStorage objects.
         *
         * @param size The number of bytes the returned RawStorage must have at 
         *             least.
         *
         * @return Pointer to a RawStorage instance. The object remains owner of
         *         the memory designated by this pointer.
         *
         * @throws std::bad_alloc If the request cannot be fulfilled because
         *                        of too low memory.
         */
        RawStorage *RaiseAtLeast(const SIZE_T size);

        /**
         * Return 'storage' for reuse.
         *
         * @param storage The RawStorage object to be returned. This must have
         *                been acquired from the same RawStoragePool before.
         *
         * @throws IllegalParamException If 'storage' was not created by this 
         *                               RawStoragePool.
         */
        void Return(RawStorage *storage);

        /**
         * Return 'storage' for reuse if not NULL.
         *
         * Note: The method will, although it tests for NULL, throw an exception
         * if the storage was not raised from this pool. This indicates a 
         * serious problem in the application code, so we do not hide it!
         *
         * @param storage The RawStorage object to be returned. This must have
         *                been acquired from the same RawStoragePool before.
         *
         * @throws IllegalParamException If 'storage' was not created by this 
         *                               RawStoragePool.
         */
        void SafeReturn(RawStorage *storage);

    private:

        /** RawStorage instances with in-use marker. */
        typedef struct PooledRawStorage_t {
            RawStorage *storage;
            bool isInUse;

            bool operator ==(const struct PooledRawStorage_t& rhs) const {
                return (this->storage == rhs.storage);
            }
        } PooledRawStorage;

        /** List of pooled RawStorage objects. */
        typedef SingleLinkedList<PooledRawStorage> RawStorageList;

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        RawStoragePool(const RawStoragePool& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (&rhs != this).
         */
        RawStoragePool& operator =(const RawStoragePool& rhs);

        /** The pooled RawStorage instances. */
        RawStorageList storageList;

    };

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RAWSTORAGEPOOL_H_INCLUDED */
