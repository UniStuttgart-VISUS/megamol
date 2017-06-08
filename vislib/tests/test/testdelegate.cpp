/*
 * testdelegate.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#include "testdelegate.h"
#include "testhelper.h"
#include "vislib/Delegate.h"
#include <cstdio>

//#define _DO_NOT_COMPILE 1

static int funcNum = 0;

static void func1(bool& b, int i) {
    printf("func1 called\n");
    funcNum = 1;
    b = (i != 0);
}

static void func1Ctxt(bool& b, int i, const int* j) {
    printf("func1Ctxt called\n");
    funcNum = 5;
    b = (i != *j);
}

#ifdef _DO_NOT_COMPILE
static void func2(int i1, int i2) {
    printf("func2 called\n");
    funcNum = 2;
}
#endif /* _DO_NOT_COMPILE */

class TestClass {
public:
    void Meth1(bool& b, int i) {
        printf("meth1 called\n");
        funcNum = -1;
        b = (i != 0);
    }
    void Meth1Ctxt(bool& b, int i, const int* j) {
        printf("meth1Ctxt called\n");
        funcNum = -3;
        b = (i != *j);
    }
#ifdef _DO_NOT_COMPILE
    void Meth2(int i1, int i2) {
        printf("meth2 called\n");
        funcNum = -2;
    }
#endif /* _DO_NOT_COMPILE */
};

static double funct3(bool b, int i, float f) {
    printf("funct3 called\n");
    funcNum = 3;
    return static_cast<double>(static_cast<float>(b ? i : 0) + f);
}

static void funct4(void) {
    printf("funct4 called\n");
    funcNum = 4;
}


/*
 * TestDelegate
 */
void TestDelegate(void) {
    TestClass testObj;
    vislib::Delegate<void, bool&, int> testNullDelegate;
    vislib::Delegate<void, bool&, int> testFuncDelegate(&func1);
    vislib::Delegate<void, bool&, int> testFuncDelegate2s(&func1);
#ifdef _DO_NOT_COMPILE
    vislib::Delegate<void, bool, int> testBadFuncDelegate(&func2);
#endif /* _DO_NOT_COMPILE */
    vislib::Delegate<void, bool&, int> testClassDelegate(testObj, &TestClass::Meth1);
#ifdef _DO_NOT_COMPILE
    vislib::Delegate<void, bool, int> testBadClassDelegate(testObj, &TestClass::Meth2);
#endif /* _DO_NOT_COMPILE */
    vislib::Delegate<double, bool, int, float> testFuncDelegate3(&funct3);
    vislib::Delegate<> testFuncDelegate0(&funct4);
    int j = 0;
    vislib::Delegate<void, bool&, int> testFuncCtxtDelegate(&func1Ctxt, &j);
    testFuncCtxtDelegate.Set(&func1Ctxt, &j);
    vislib::Delegate<void, bool&, int> testClassCtxtDelegate(testObj, &TestClass::Meth1Ctxt, &j);
    testClassCtxtDelegate.Set(testObj, &TestClass::Meth1Ctxt, &j);

    bool b = false;

    ::AssertEqual("b == false", b, false);
    ::AssertTrue("testFuncDelegate == testFuncDelegate2s", testFuncDelegate == testFuncDelegate2s);
    testFuncDelegate2s.Unset();
    ::AssertFalse("testFuncDelegate != testFuncDelegate2s", testFuncDelegate == testFuncDelegate2s);
    testFuncDelegate2s = testFuncDelegate;
    ::AssertTrue("testFuncDelegate == testFuncDelegate2s", testFuncDelegate == testFuncDelegate2s);
    ::AssertEqual("funcNum = 0", funcNum, 0);
    testFuncDelegate(b, 1);
    ::AssertEqual("b == true", b, true);
    ::AssertEqual("funcNum = 1", funcNum, 1);
    testClassDelegate(b, 0);
    ::AssertEqual("b == false", b, false);
    ::AssertEqual("funcNum = -1", funcNum, -1);
    testFuncCtxtDelegate(b, 1);
    ::AssertEqual("b == true", b, true);
    ::AssertEqual("funcNum = 5", funcNum, 5);
    testClassCtxtDelegate(b, 0);
    ::AssertEqual("b == false", b, false);
    ::AssertEqual("funcNum = -3", funcNum, -3);
    double d = testFuncDelegate3(true, 1, 1.0f);
    ::AssertEqual("funcNum = 3", funcNum, 3);
    ::AssertNearlyEqual("d = 2.0", d, 2.0);
    testFuncDelegate0();
    ::AssertEqual("funcNum = 4", funcNum, 4);

}

