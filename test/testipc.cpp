/*
 * testipc.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "testipc.h"
#include "testhelper.h"

#include "vislib/Console.h"
#include "vislib/IPCSemaphore.h"
#include "vislib/Process.h"
#include "vislib/Semaphore.h"
#include "vislib/SharedMemory.h"
#include "vislib/SystemInformation.h"
#include "vislib/SystemException.h"
#include "vislib/Thread.h"


#define TEST_IPC_SEM_NAME ("Local\\testipcsem")
#define TEST_IPC_END_SEM_NAME ("Local\\testipcendguard")
#define TEST_IPC_SHMEM_NAME ("Local\\testipcshmem")
#define TEST_IPC_SHMEM_SIZE (SystemInformation::AllocationGranularity())


void TestIpc(void) {
    using namespace vislib::sys;
    Semaphore sem(TEST_IPC_SEM_NAME);
    Semaphore endSem(TEST_IPC_END_SEM_NAME);
    SharedMemory shMem, shMemErr;
    Process ipc2;
#ifdef _WIN32   // TODO: This does not work on Linux ... somehow
    const char *ipc2Params[] = { "ipc2", NULL };
#endif /* _WIN32 */

    AssertException("Open non-existing shared memory.", shMemErr.Open(
        TEST_IPC_SHMEM_NAME, SharedMemory::READ_WRITE, 
        SharedMemory::OPEN_ONLY, TEST_IPC_SHMEM_SIZE),
        SystemException);
    AssertFalse("Failed open produced non-open memory.", shMemErr.IsOpen());

    AssertNoException("Create shared memory.", shMem.Open(TEST_IPC_SHMEM_NAME,
        SharedMemory::READ_WRITE, SharedMemory::OPEN_CREATE, 
        TEST_IPC_SHMEM_SIZE));

    AssertTrue("Shared memory is open.", shMem.IsOpen());

    AssertException("Create shared memory, no open.", shMemErr.Open(
        TEST_IPC_SHMEM_NAME, SharedMemory::READ_WRITE, 
        SharedMemory::CREATE_ONLY, TEST_IPC_SHMEM_SIZE),
        SystemException);
    AssertFalse("Failed create produced non-open memory.", shMemErr.IsOpen());

    // Write specific data to shared memory.
    sem.Lock();
    if (shMem.IsOpen()) {
        *shMem.As<char>() = 'v';
        AssertEqual("Data written to shared memory.", *shMem.As<char>(), 'v');
    }
    sem.Unlock();

#ifdef _WIN32   // TODO: This does not work on Linux ... somehow
    AssertNoException("Create child process.", ipc2.Create(
        Process::ModuleFileNameA(), ipc2Params));
#endif

    endSem.Lock();
    endSem.Lock();      // Wait for end to be signaled.
    
    sem.Lock();
    AssertEqual("Shared memory changed.", *shMem.As<char>(), '2');
    sem.Unlock();
}


void TestIpc2(void) {
    using namespace vislib::sys;
    Semaphore sem(TEST_IPC_SEM_NAME);
    Semaphore endSem(TEST_IPC_END_SEM_NAME);
    SharedMemory shMem;

    AssertNoException("Open shared memory.", shMem.Open(TEST_IPC_SHMEM_NAME,
        SharedMemory::READ_WRITE, SharedMemory::OPEN_ONLY, 
        TEST_IPC_SHMEM_SIZE));

    AssertTrue("Shared memory is open.", shMem.IsOpen());

    if (shMem.IsOpen()) {
        sem.Lock();
        AssertEqual("Shared memory content.", *shMem.As<char>(), 'v');
        *shMem.As<char>() = '2';
        sem.Unlock();
    }

    // Signal that process has finished.
    Console::Flush();
    endSem.Unlock();
}
