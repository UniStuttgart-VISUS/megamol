/*
 * testpointers.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testpointers.h"
#include "testhelper.h"

#ifndef DEBUG
#define DEBUG
#endif
#include "vislib/SmartPtr.h"


void TestSmartPtr(void) {
    using vislib::SmartPtr;

    SmartPtr<int> int0;
    SmartPtr<int> int1(new int(5));

    ::AssertTrue("Smart pointer initially NULL", int0.IsNull());
	::AssertEqual("NULL has no references.", int0._GetCnt(), UINT(0));
    ::AssertFalse("Smart pointer initialisation", int1.IsNull());
    ::AssertEqual("Smart pointer content check", *int1, 5);
	::AssertEqual("Refrence count of 1 is 1.", int1._GetCnt(), UINT(1));

    int0 = int1;
    ::AssertTrue("Smart pointer assignment", int0 == int1);
    ::AssertFalse("Smart pointer inequality", int0 != int1);
    ::AssertEqual("Assignment content check", *int0, *int1);
    ::AssertEqual("Assignment content check", *int0, 5);
	::AssertEqual("Refrence count after assignment.", int1._GetCnt(), UINT(2));
	::AssertEqual("Equal reference count.", int0._GetCnt(), int1._GetCnt());

    SmartPtr<int> int2(int1);
    ::AssertFalse("Smart pointer copy ctor", int2.IsNull());
	::AssertEqual("Refrence count after copy ctor.", int2._GetCnt(), UINT(3));
    ::AssertTrue("Smart pointer copy ctor", int0 == int1);
    ::AssertEqual("Copy ctor content check", *int0, *int1);
    ::AssertEqual("Copy ctor content check", *int0, 5);

    int0 = NULL;
    ::AssertTrue("Smart pointer NULL assignment", int0.IsNull());
    ::AssertFalse("No assignment cross effects.", int1.IsNull());
    ::AssertFalse("No assignment cross effects.", int2.IsNull());
	::AssertEqual("Refrence count decrement after NULL asignment.", int1._GetCnt(), UINT(2));

    int2 = int0;
    ::AssertTrue("NULL smart pointer assignment", int2.IsNull());
    ::AssertFalse("No assignment cross effects.", int1.IsNull());
	::AssertEqual("Refrence count decrement after smart pointer assignment.", int1._GetCnt(), UINT(1));

    int1 = NULL;
    ::AssertTrue("Last reference destroyed", int1.IsNull());
	::AssertEqual("Reference count 0 after object destroyed.", int1._GetCnt(), UINT(0));

    {
        SmartPtr<int> int3(new int(7));
    }
}
