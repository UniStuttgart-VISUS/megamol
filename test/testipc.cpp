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
#include "vislib/SharedMemory.h"
#include "vislib/SystemInformation.h"
#include "vislib/Thread.h"


#define TEST_IPC_SEM_NAME ("Local\\testipcsem")
#define TEST_IPC_END_SEM_NAME ("Local\\testipcendguard")
#define TEST_IPC_SHMEM_NAME ("Local\\testipcshmem")
#define TEST_IPC_SHMEM_SIZE (SystemInformation::AllocationGranularity())


void TestIpc(void) {
    using namespace vislib::sys;
    IPCSemaphore sem(TEST_IPC_SEM_NAME);
    IPCSemaphore endSem(TEST_IPC_END_SEM_NAME);
    SharedMemory shMem;
    Process ipc2;
    const char *ipc2Params[] = { "ipc2", NULL };

    AssertNoException("Create shared memory.", shMem.Open(TEST_IPC_SHMEM_NAME,
        SharedMemory::READ_WRITE, SharedMemory::CREATE_ONLY, 
        TEST_IPC_SHMEM_SIZE));

    AssertTrue("Shared memory is open.", shMem.IsOpen());

    // Write specific data to shared memory.
    sem.Lock();
    if (shMem.IsOpen()) {
        *shMem.As<char>() = 'v';
    }
    sem.Unlock();

    AssertNoException("Create child process.", ipc2.Create(
        Process::ModuleFileNameA(), ipc2Params));

    endSem.Lock();
    endSem.Lock();      // Wait for end to be signaled.
    
    sem.Lock();
    AssertEqual("Shared memory changed.", *shMem.As<char>(), '2');
    sem.Unlock();
}


void TestIpc2(void) {
    using namespace vislib::sys;
    IPCSemaphore sem(TEST_IPC_SEM_NAME);
    IPCSemaphore endSem(TEST_IPC_END_SEM_NAME);
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
