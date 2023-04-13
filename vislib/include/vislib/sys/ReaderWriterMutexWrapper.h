/*
 * ReaderWriterMutexWrapper.h
 *
 * Copyright 2019 MegaMol Dev Team
 */
#pragma once

#include "AbstractReaderWriterLock.h"
#include "Mutex.h"

namespace vislib::sys {

class ReaderWriterMutexWrapper : public AbstractReaderWriterLock {
public:
    ReaderWriterMutexWrapper() = default;

    ~ReaderWriterMutexWrapper() override {}

    void Lock() override {
        mutex.Lock();
    }

    void LockExclusive() override {
        Lock();
    };

    void LockShared() override {
        Lock();
    };

    bool TryLock(unsigned long const timeout = 0) override {
        return mutex.TryLock(timeout);
    };

    void Unlock() override {
        mutex.Unlock();
    }

    void UnlockExclusive() override {
        Unlock();
    };

    void UnlockShared() override {
        Unlock();
    };

private:
    Mutex mutex;
}; // end class ReaderWriterMutexWrapper

} // namespace vislib::sys
