/*
 * testnamedpipe.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testnamedpipe.h"
#include "testhelper.h"

#include <vislib/Mutex.h>
#include <vislib/NamedPipe.h>
#include <vislib/String.h>
#include <vislib/Thread.h>
#include <vislib/Trace.h>
#include <vislib/SystemException.h>

#define PIPE_NAME "DieHorstPipe"
#define COM_STR_01 "Hello World through a named pipe"
#define COM_STR_02 "A second message which will be sent in parts"
#define COM_STR_03 "A second message which will be received in parts"

vislib::sys::Mutex m1;
vislib::sys::Mutex m2;
vislib::sys::Mutex m3;

void BarrierT1(void) {
    m3.Lock();
    m1.Unlock();
    m2.Lock();
    m3.Unlock();
    printf("IER ===>\n");
    m1.Lock();
    m2.Unlock();
}

void BarrierT2(void) {
    m1.Lock();
    printf("<=== BARR");
    m2.Unlock();
    m3.Lock();
    m1.Unlock();
    m2.Lock();
    m3.Unlock();
}

#if (_MSC_VER >= 1400)
#pragma warning(disable: 4996)
#endif

#define SyncHere BarrierT2
DWORD TestNamedPipeSecondThread(void *param) {
    vislib::sys::NamedPipe pipe;
    char buf[256];
    buf[255] = 0;

    // initialze Barrier
    m2.Lock();

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    try {
        pipe.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_READ, 1000);
        SyncHere();
        AssertTrue("Open Pipe 2 for reading", true);
    } catch(...) {
        SyncHere();
        AssertTrue("Open Pipe 2 for reading", false);
    }

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    SyncHere();

    /* { // does not work under linux
        vislib::sys::NamedPipe pipe3;
        try {
            pipe3.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_READ);
            SyncHere();
            AssertTrue("Cannot open Pipe 3 for reading", false);
        } catch(...) {
            SyncHere();
            AssertTrue("Cannot open Pipe 3 for reading", true);
        }
    } // */

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    pipe.Read(buf, 256);
    AssertEqual("Communication #1 correct", strcmp(buf, COM_STR_01), 0);

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    pipe.Read(buf, 256);
    AssertEqual("Communication #2 correct", strcmp(buf, COM_STR_02), 0);

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    pipe.Read(buf, 16);
    pipe.Read(&buf[16], 240);
    AssertEqual("Communication #3 correct", strcmp(buf, COM_STR_03), 0);

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    pipe.Close();

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    SyncHere();

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    SyncHere();

    try {
        pipe.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_READ);
        SyncHere();
        AssertTrue("Open Pipe 2 for reading", true);
    } catch(...) {
        SyncHere();
        AssertTrue("Open Pipe 2 for reading", false);
    }

    pipe.Read(buf, 256);
    AssertEqual("Communication #4 correct", strcmp(buf, COM_STR_01), 0);

    SyncHere();

    AssertException("Read from closed pipe", pipe.Read(buf, 256), vislib::sys::SystemException);

    SyncHere();

    pipe.Close();

    SyncHere();

    try {
        pipe.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_WRITE);
        SyncHere();
        AssertTrue("Open Pipe 2 for writing", true);
    } catch(...) {
        SyncHere();
        AssertTrue("Open Pipe 2 for writing", false);
    }

    strncpy(buf, COM_STR_01, 255);
    pipe.Write(buf, 256);

    strncpy(buf, COM_STR_02, 255);
    pipe.Write(buf, 16);
    pipe.Write(&buf[16], 240);

    strncpy(buf, COM_STR_03, 255);
    pipe.Write(buf, 256);

    SyncHere();

    m2.Unlock();

    return 0;
}
#undef SyncHere

#define SyncHere BarrierT1
void TestNamedPipe(void) {
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_ALL);

    vislib::sys::Thread secondThread(TestNamedPipeSecondThread);
    vislib::sys::NamedPipe pipe;
    char buf[256];
    buf[255] = 0;


    // test named pip open timeout
    AssertFalse("Open Pipe times out", pipe.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_WRITE, 1000));

    AssertFalse("Open Pipe times out", pipe.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_READ, 1000));


    // initialize Barrier
    m1.Lock();
    m2.Lock();

    secondThread.Start();
    m2.Unlock();

    try {
        pipe.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_WRITE, 1000);
        AssertTrue("Open Pipe 1 for writing", true);
    } catch(...) {
        AssertTrue("Open Pipe 1 for writing", false);
    }
    // SyncHere();
    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);
    SyncHere();
    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);
    SyncHere();
    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    strncpy(buf, COM_STR_01, 255);
    pipe.Write(buf, 256);

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    strncpy(buf, COM_STR_02, 255);
    pipe.Write(buf, 16);
    pipe.Write(&buf[16], 240);

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    strncpy(buf, COM_STR_03, 255);
    pipe.Write(buf, 256);

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    SyncHere();

    //VLTRACE(VISLIB_TRCELVL_INFO, "Here: {%d}\n", __LINE__);

    AssertException("Write to closed pipe", pipe.Write(buf, 256), vislib::sys::SystemException);

    pipe.Close();

    SyncHere();

    try {
        pipe.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_WRITE);
        AssertTrue("Open Pipe 1 for writing", true);
    } catch(...) {
        AssertTrue("Open Pipe 1 for writing", false);
    }
    SyncHere();

    strncpy(buf, COM_STR_01, 255);
    pipe.Write(buf, 256);

    pipe.Close();

    SyncHere();
    SyncHere();
    SyncHere();

    try {
        pipe.Open(PIPE_NAME, vislib::sys::NamedPipe::PIPE_MODE_READ);
        AssertTrue("Open Pipe 1 for reading", true);
    } catch(...) {
        AssertTrue("Open Pipe 1 for reading", false);
    }
    SyncHere();

    pipe.Read(buf, 256);
    AssertEqual("Communication #5 correct", strcmp(buf, COM_STR_01), 0);

    pipe.Read(buf, 256);
    AssertEqual("Communication #6 correct", strcmp(buf, COM_STR_02), 0);

    pipe.Read(buf, 16);
    pipe.Read(&buf[16], 240);
    AssertEqual("Communication #7 correct", strcmp(buf, COM_STR_03), 0);

    SyncHere();

    secondThread.Join();

    m1.Unlock();
}
#undef SyncHere

#if (_MSC_VER >= 1400)
#pragma warning(default: 4996)
#endif
