/*
 * testVIPCStrTab.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#include "testVIPCStrTab.h"

#include "testhelper.h"
#include "vislib/Thread.h"
#include "vislib/VolatileIPCStringTable.h"


const char *TestEntryName = "vislibtesttestentry";
const char *TestEntryValue = "hugo";


void TestVIPCStrTabSet(void) {
    // Sets a vipc string tab value and waits for several seconds before the programm is closed
    vislib::sys::VolatileIPCStringTable::Entry *heapEntry;

    {
        vislib::sys::VolatileIPCStringTable::Entry innerEntry 
            = vislib::sys::VolatileIPCStringTable::Create(TestEntryName, "Stuffeldibla");
        heapEntry = new vislib::sys::VolatileIPCStringTable::Entry(innerEntry);
    }

    heapEntry->SetValue("Zeug");

    vislib::sys::VolatileIPCStringTable::Entry entry(*heapEntry);
    delete heapEntry;

    entry.SetValue(TestEntryValue);

    // wait
    vislib::sys::Thread::Sleep(10000); // 10 seconds

}


void TestVIPCStrTabGet(void) {
    // Gets a vipc string tab value
    vislib::StringA value = vislib::sys::VolatileIPCStringTable::GetValue(TestEntryName);
    AssertEqual("VIPCStrTab Entry Value is correct", value, TestEntryValue);
}
