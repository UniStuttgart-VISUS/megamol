/*
 * teststacktrace.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "teststacktrace.h"
#include "testhelper.h"

#include "vislib/Exception.h"
#include "vislib/StackTrace.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/ThreadSafeStackTrace.h"
#include "vislib/Thread.h"


/*
 * The manual test stack
 */
vislib::SingleLinkedList<vislib::StringA> manualStack;


/*
 * The manual test stack 2
 */
vislib::SingleLinkedList<vislib::StringA> manualStack2;


/*
 * The manual test stack 3
 */
vislib::SingleLinkedList<vislib::StringA> manualStack3;


/*
 * testStack
 */
void testStack(const char* s = NULL) {
    vislib::StringA soll;
    vislib::StringA ist;

    if (s == NULL) {
        char *buf = NULL;
        unsigned int len;
        vislib::StackTrace::GetStackString((char*)NULL, len);
        buf = new char[len];
        vislib::StackTrace::GetStackString(buf, len);
        ist = buf;
        delete[] buf;
    } else {
        ist = s;
    }

    vislib::SingleLinkedList<vislib::StringA>::Iterator iter = manualStack.GetIterator();
    while (iter.HasNext()) {
        vislib::StringA s = iter.Next();
        soll.Prepend("\n");
        soll.Prepend(s);
    }

    AssertEqual("Stack Trace as expected", soll, ist);
    printf("Stack:\nExpected: %sFound: %s\n", soll.PeekBuffer(), ist.PeekBuffer());
}


/*
 * testStack2
 */
void testStack2(const vislib::SingleLinkedList<vislib::StringA>& tehStack) {
    vislib::StringA soll;
    vislib::StringA ist;

    char *buf = NULL;
    unsigned int len;
    vislib::StackTrace::GetStackString((char*)NULL, len);
    buf = new char[len];
    vislib::StackTrace::GetStackString(buf, len);
    ist = buf;
    delete[] buf;

    vislib::ConstIterator<vislib::SingleLinkedList<vislib::StringA>::Iterator>
        iter = tehStack.GetConstIterator();
    while (iter.HasNext()) {
        vislib::StringA s = iter.Next();
        soll.Prepend("\n");
        soll.Prepend(s);
    }

    AssertEqual("Stack Trace as expected", soll, ist);
    printf("Stack:\nExpected: %sFound: %s\n", soll.PeekBuffer(), ist.PeekBuffer());
}


/*
 * function3
 */
void function3(void) {
    VLSTACKTRACE("function3", "horst", 1);
    manualStack.Append("function3 [horst:1]");

    testStack();

    manualStack.RemoveLast();
}

/*
 * functionEX
 */
void functionEX(void) {
    VLSTACKTRACE("functionEX", "THEND", 42);
    manualStack.Append("functionEX [THEND:42]");

    throw vislib::Exception("Here we go!", "HERE", 128);
}


/*
 * function2
 */
void function2(void) {
    VLSTACKTRACE("function2", "hugo", 10);
    manualStack.Append("function2 [hugo:10]");

    testStack();
    function3();
    testStack();

    manualStack.RemoveLast();
}


/*
 * function1
 */
void function1(void) {
    VLSTACKTRACE("function1", "", 99);
    manualStack.Append("function1 [:99]");

    testStack();
    function2();
    testStack();
    function3();
    testStack();
    functionEX();

    manualStack.RemoveLast();
}


/*
 * TestStackTrace
 */
void TestStackTrace(void) {
    vislib::StackTrace::Initialise();
    VLSTACKTRACE("TestStackTrace", "", 0);
    manualStack.Clear();
    manualStack.Append("TestStackTrace [:0]");

    try {

        testStack();
        function1();
        testStack();

        AssertFalse("No exception thrown", true);

    } catch(vislib::Exception e) {
        printf("vislib::Exception:\n");
        printf("    Msg: %s\n", e.GetMsgA());
        printf("    File: %s\n", e.GetFile());
        printf("    Line: %d\n", e.GetLine());
        printf("    Stack: %s\n", e.GetStack());

        testStack(e.GetStack());

    } catch(...) {
        printf("Unknown exception!\n");
        AssertFalse("Unknown exception", true);
    }

    manualStack.Clear();
    manualStack.Append("TestStackTrace [:0]");

    testStack();

    manualStack.RemoveLast();
}


/*
 * randomSleep
 */
void randomSleep(void) {
    vislib::sys::Thread::Sleep(50 + (rand() % 251));
}


/*
 * tfunction
 */
void tfunction(vislib::SingleLinkedList<vislib::StringA>& stack, int i) {
    VLSTACKTRACE("tfunction", "", i);
    vislib::StringA str;
    str.Format("tfunction [:%d]", i);
    stack.Append(str);

    randomSleep();
    testStack2(stack);
    randomSleep();

    if (i < 10) {
        tfunction(stack, i + 1 + (rand() % 2));
    }

    randomSleep();
    testStack2(stack);
    randomSleep();

    stack.RemoveLast();
}


/*
 * thread1run
 */
DWORD thread1run(void*) {
    VLSTACKTRACE("thread1run", "", 1);
    manualStack3.Append("thread1run [:1]");

    randomSleep();
    tfunction(manualStack3, 2);

    randomSleep();
    testStack2(manualStack3);
    randomSleep();

    manualStack3.RemoveLast();
    return 0;
}


/*
 * thread2run
 */
DWORD thread2run(void*) {
    VLSTACKTRACE("thread2run", "", 2);
    manualStack2.Append("thread2run [:2]");

    randomSleep();
    testStack2(manualStack2);
    randomSleep();

    tfunction(manualStack2, 1);

    randomSleep();
    testStack2(manualStack2);
    randomSleep();

    manualStack2.RemoveLast();
    return 0;
}


/*
 * thread3run
 */
DWORD thread3run(void*) {
    VLSTACKTRACE("thread3run", "horst", 3);
    manualStack.Append("thread3run [horst:3]");

    randomSleep();
    testStack();
    randomSleep();
    randomSleep();

    try {

        function1();

        AssertFalse("No exception thrown", true);

    } catch(vislib::Exception e) {
        printf("vislib::Exception:\n");
        printf("    Msg: %s\n", e.GetMsgA());
        printf("    File: %s\n", e.GetFile());
        printf("    Line: %d\n", e.GetLine());
        printf("    Stack: %s\n", e.GetStack());

        testStack(e.GetStack());

    } catch(...) {
        printf("Unknown exception!\n");
        AssertFalse("Unknown exception", true);
    }

    manualStack.Clear();
    manualStack.Append("thread3run [horst:3]");

    randomSleep();
    testStack();
    randomSleep();

    manualStack.RemoveLast();
    return 0;
}


/*
 * TestMTStackTrace
 */
void TestMTStackTrace(void) {
    vislib::sys::ThreadSafeStackTrace::Initialise(NULL, true); // <= Forces reinitialisation of the stack trace using the mulitthread implementation
    manualStack.Clear();
    manualStack2.Clear();
    manualStack3.Clear();

    vislib::sys::Thread t1(thread1run);
    vislib::sys::Thread t2(thread2run);
    vislib::sys::Thread t3(thread3run);

    t1.Start(NULL);
    t2.Start(NULL);
    t3.Start(NULL);

    t1.Join();
    t2.Join();
    t3.Join();
}
