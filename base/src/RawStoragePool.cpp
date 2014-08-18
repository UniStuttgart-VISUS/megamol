/*
 * RawStoragePool.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/RawStoragePool.h"

#include <climits>

#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::RawStoragePool::RawStoragePool
 */
vislib::RawStoragePool::RawStoragePool(void) {
    VLAUTOSTACKTRACE;
    // Nothing to do.
}


/*
 * vislib::RawStoragePool::~RawStoragePool
 */
vislib::RawStoragePool::~RawStoragePool(void) {
    VLAUTOSTACKTRACE;
    this->Clear();
}


/*
 * vislib::RawStoragePool::Clear
 */
void vislib::RawStoragePool::Clear(void) {
    VLAUTOSTACKTRACE;
    while (!this->storageList.IsEmpty()) {
        SAFE_DELETE(this->storageList.First().storage);
        this->storageList.RemoveFirst();
    }
}


/*
 * vislib::RawStoragePool::RaiseAtLeast
 */
vislib::RawStorage *vislib::RawStoragePool::RaiseAtLeast(const SIZE_T size) {
    VLAUTOSTACKTRACE;
    PooledRawStorage *bestFit = NULL;
    PooledRawStorage *firstUnused = NULL;
    SIZE_T bestDist = SIZE_MAX;
    RawStorageList::Iterator it = this->storageList.GetIterator();

    while (it.HasNext()) {
        PooledRawStorage& n = it.Next();

        if (!n.isInUse && (firstUnused == NULL)) {
            firstUnused = &n;
        }

        if (!n.isInUse && (n.storage->GetSize() > size)
                && (n.storage->GetSize() < bestDist)) {
            bestFit = &n;
            bestDist = n.storage->GetSize() - size;

        } else if (n.storage->GetSize() == size) {
            bestFit = &n;
            break;  // Cannot find any better fit.
        }
    }

    if (bestFit != NULL) {
        ASSERT(bestFit->storage->GetSize() >= size);
        bestFit->isInUse = true;
        return bestFit->storage;

    } else if (firstUnused != NULL) {
        firstUnused->isInUse = true;
        firstUnused->storage->AssertSize(size);
        return firstUnused->storage;

    } else {
        PooledRawStorage n;
        n.isInUse = true;
        n.storage = new RawStorage(size);
        this->storageList.Append(n);

        ASSERT(this->storageList.Last().storage->GetSize() == size);
        return this->storageList.Last().storage;
    }

}


/*
 * vislib::RawStoragePool::Return
 */
void vislib::RawStoragePool::Return(RawStorage *storage) {
    VLAUTOSTACKTRACE;
    RawStorageList::Iterator it = this->storageList.GetIterator();

    while (it.HasNext()) {
        PooledRawStorage& n = it.Next();
        if (n.storage == storage) {
            n.isInUse = false;
            return;
        }
    }
    /* 'storage' was not found at this point. */

    throw IllegalParamException("storage", __FILE__, __LINE__);
}


/*
 *  vislib::RawStoragePool::SafeReturn
 */
void vislib::RawStoragePool::SafeReturn(RawStorage *storage) {
    VLAUTOSTACKTRACE;
    if (storage != NULL) {
        this->Return(storage);
    }
}


/*
 * vislib::RawStoragePool::RawStoragePool
 */
vislib::RawStoragePool::RawStoragePool(const RawStoragePool& rhs) {
    VLAUTOSTACKTRACE;
    throw UnsupportedOperationException("RawStoragePool::RawStoragePool",
        __FILE__, __LINE__);
}


/*
 * vislib::RawStoragePool::operator =
 */
vislib::RawStoragePool& vislib::RawStoragePool::operator =(
        const RawStoragePool& rhs) {
    VLAUTOSTACKTRACE;
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
