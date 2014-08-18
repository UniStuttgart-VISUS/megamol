/*
 * testinterlocked.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "testinterlocked.h"
#include "testhelper.h"

#include "vislib/Interlocked.h"


void TestInterlocked(void) {
    using vislib::sys::Interlocked;
    INT32 hugo = 0;
    void *heinz = NULL;
    void *horst = &hugo;

    hugo = 5;
    AssertEqual("Interlocked::Increment", Interlocked::Increment(&hugo), 6);
    AssertEqual("Incremented", hugo, 6);

    AssertEqual("Interlocked::Decrement", Interlocked::Decrement(&hugo), 5);
    AssertEqual("Decremented", hugo, 5);

    AssertEqual("Interlocked::CompareExchange", Interlocked::CompareExchange(&hugo, 2, 0), 5);
    AssertEqual("Not exchanged", hugo, 5);

    AssertEqual("Interlocked::CompareExchange", Interlocked::CompareExchange(&hugo, 2, 5), 5);
    AssertEqual("Exchanged", hugo, 2);

    AssertEqual("Interlocked::ExchangeAdd", Interlocked::ExchangeAdd(&hugo, 2), 2);
    AssertEqual("Added", hugo, 4);

    AssertEqual("Interlocked::ExchangeSub", Interlocked::ExchangeSub(&hugo, 2), 4);
    AssertEqual("Subtracted", hugo, 2);

    AssertEqual("Interlocked::Exchange", Interlocked::Exchange(&hugo, 10), 2);
    AssertEqual("Exchanged", hugo, 10);

    AssertEqual("Interlocked::Exchange", Interlocked::Exchange(&heinz, horst), (void *) NULL);
    AssertEqual("Exchanged", heinz, horst);
}

