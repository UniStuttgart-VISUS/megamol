/*
 * RawStoragePool.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/RawStoragePool.h"

#include <climits>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::RawStoragePool::RawStoragePool
 */
vislib::RawStoragePool::RawStoragePool(void) {
    // Nothing to do.
}


/*
 * vislib::RawStoragePool::~RawStoragePool
 */
vislib::RawStoragePool::~RawStoragePool(void) {
    // Nothing to do.
}


/*
 * vislib::RawStoragePool::RaiseAtLeast
 */
vislib::RawStorage *vislib::RawStoragePool::RaiseAtLeast(const SIZE_T size) {
    PooledRawStorage *bestFit = NULL;
    PooledRawStorage *firstUnused = NULL;
    SIZE_T bestDist = SIZE_MAX;
    RawStorageList::Iterator it = this->storage.GetIterator();

    while (it.HasNext()) {
        PooledRawStorage& n = it.Next();

        if (!n.isInUse && (firstUnused == NULL)) {
            firstUnused = &n;
        }

        if ((n.storage.GetSize() > size) && (n.storage.GetSize() < bestDist)) {
            bestFit = &n;
            bestDist = n.storage.GetSize() - size;

        } else if (n.storage.GetSize() == 0) {
            bestFit = &n;
            break;  // Cannot find any better fit.
        }
    }

    if (bestFit != NULL) {
        ASSERT(bestFit->storage.GetSize() >= size);
        bestFit->isInUse = true;
        return &(bestFit->storage);

    } else if (firstUnused != NULL) {
        ASSERT(firstUnused->storage.GetSize() < size);
        firstUnused->isInUse = true;
        firstUnused->storage.AssertSize(size);
        return &(firstUnused->storage);

    } else {
        PooledRawStorage n;
        this->storage.Append(n);
        ASSERT(this->storage.Last() == n);
        ASSERT(&(this->storage.Last().storage) == &n.storage);

        this->storage.Last().isInUse = true;
        this->storage.Last().storage.AssertSize(size);
        return &(this->storage.Last().storage);
    }

}


/*
 * vislib::RawStoragePool::Return
 */
void vislib::RawStoragePool::Return(RawStorage *storage) {
    RawStorageList::Iterator it = this->storage.GetIterator();

    while (it.HasNext()) {
        PooledRawStorage& n = it.Next();
        if (&(n.storage) == storage) {
            n.isInUse = false;
        }
    }
    /* 'storage' was not found at this point. */

    throw IllegalParamException("storage", __FILE__, __LINE__);
}


/*
 * vislib::RawStoragePool::RawStoragePool
 */
vislib::RawStoragePool::RawStoragePool(const RawStoragePool& rhs) {
    throw UnsupportedOperationException("RawStoragePool::RawStoragePool",
        __FILE__, __LINE__);
}


/*
 * vislib::RawStoragePool::operator =
 */
vislib::RawStoragePool& vislib::RawStoragePool::operator =(
        const RawStoragePool& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
