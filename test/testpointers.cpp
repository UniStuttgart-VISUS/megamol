/*
 * testpointers.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testpointers.h"
#include "testhelper.h"

#include "vislib/SmartPtr.h"


void TestSmartPtr(void) {
    using vislib::SmartPtr;

    SmartPtr<int> int0;
    SmartPtr<int> int1(new int(5));
    SmartPtr<int> int2(int1);

    ::AssertTrue("Smart pointer initially NULL", int0.IsNull());
    ::AssertFalse("Smart pointer initialisation", int1.IsNull());
    ::AssertEqual("Smart pointer content check", *int1, 5);

    int0 = int1;
    ::AssertTrue("Smart pointer assignment", int0 == int1);
    ::AssertFalse("Smart pointer inequality", int0 != int1);
    ::AssertEqual("Assignment content check", *int0, *int1);
    ::AssertEqual("Assignment content check", *int0, 5);

    ::AssertFalse("Smart pointer copy ctor", int2.IsNull());
    ::AssertTrue("Smart pointer copy ctor", int0 == int1);
    ::AssertEqual("Copy ctor content check", *int0, *int1);
    ::AssertEqual("Copy ctor content check", *int0, 5);

    int0 = NULL;
    ::AssertTrue("Smart pointer NULL assignment", int0.IsNull());
    ::AssertFalse("No assignment cross effects.", int1.IsNull());
    ::AssertFalse("No assignment cross effects.", int2.IsNull());

    int2 = int0;
    ::AssertTrue("NULL smart pointer assignment", int2.IsNull());
    ::AssertFalse("No assignment cross effects.", int1.IsNull());

    int1 = NULL;
    ::AssertTrue("Last reference destroyed", int1.IsNull());

    {
        SmartPtr<int> int3(new int(7));
    }
}
