/*
 * teststring.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "teststring.h"
#include "testhelper.h"

#include "vislib/String.h"


void TestString(void) {
    using namespace vislib;
    
    TestStringA();
    TestStringW();

    StringA a1("Horst");
    StringW w1(a1);
    StringW w2(L"Hugo");
    StringA a2(w2);

    AssertTrue("ANSI to wide character conversion constructor", !::wcscmp(w1, L"Horst"));
    AssertTrue("Wide to ANSI character conversion constructor", !::strcmp(a2, "Hugo"));
}


void TestStringA(void) {
    using namespace vislib;

    StringA s1;
    StringA s2("Horst");
    StringA s3(s2);
    StringA s4('h', 5);

    AssertTrue("Default Constructor creates empty string", !::strcmp(s1.PeekBuffer(), ""));
    AssertTrue("C string constructor", !::strcmp(s2.PeekBuffer(), "Horst"));
    AssertTrue("Copy constructor", !::strcmp(s2.PeekBuffer(), s3.PeekBuffer()));
    AssertTrue("Character constructor", !::strcmp(s4.PeekBuffer(), "hhhhh"));

    AssertTrue("Test for empty string", s1.IsEmpty());
    AssertFalse("Test for emtpy string", s2.IsEmpty());

    AssertEqual("\"Horst\" consists of 5 characters", s2.Length(), 5);

    AssertEqual("\"Horst\"[0] is 'H'", s2[0], 'H');
    AssertEqual("\"Horst\"[4] is 't'", s2[4], 't');

    try {
        s2[-1];
        AssertTrue("OutOfRangeException at begin", false);
    } catch (OutOfRangeException) {
        AssertTrue("OutOfRangeException at begin", true);
    }

    try {
        s2[s2.Length()];
        AssertTrue("OutOfRangeException at end", false);
    } catch (OutOfRangeException) {
        AssertTrue("OutOfRangeException at end", true);
    }

    AssertTrue("Test for inequality", s1 != s2);
    AssertTrue("Test for equality", s2 == s3);

    s1 = s2;
    AssertTrue("Assignment operator", !::strcmp(s1.PeekBuffer(), s2.PeekBuffer()));

    AssertTrue("\"Horst\" begins with \"H\"", s1.StartsWith("H"));
    AssertTrue("\"Horst\" begins with \"Ho\"", s1.StartsWith("Ho"));
    AssertFalse("\"Horst\" does not begin with \"Hu\"", s1.StartsWith("Hu"));

    AssertTrue("\"Horst\" ends with \"t\"", s1.EndsWith("t"));
    AssertTrue("\"Horst\" ends with \"st\"", s1.EndsWith("st"));
    AssertFalse("\"Horst\" does not ends with \"go\"", s1.EndsWith("go"));

    AssertEqual("First 'o' in \"Horst\"", s1.IndexOf('o'), 1);
    AssertEqual("First 'o' in \"Horst\" after 2", s1.IndexOf('o', 2), StringA::INVALID_POS);
    AssertEqual("Last 'o' in \"Horst\"", s1.LastIndexOf('o'), 1);

    s4 = s2 + " und Hugo";
    AssertTrue("Concatenation", !::strcmp(s4.PeekBuffer(), "Horst und Hugo"));

    s2 += " und Hugo";
    AssertTrue("Assignment concatenation", !::strcmp(s2.PeekBuffer(), "Horst und Hugo"));

    s2.Replace('H', 'h');
    AssertTrue("Character replacement", !::strcmp(s2.PeekBuffer(), "horst und hugo"));

    
}

void TestStringW(void) {
    using namespace vislib;

    StringW s1;
    StringW s2(L"Horst");
    StringW s3(s2);
    StringW s4(L'h', 5);

    AssertTrue("Default Constructor creates empty string", !::wcscmp(s1.PeekBuffer(), L""));
    AssertTrue("C string constructor", !::wcscmp(s2.PeekBuffer(), L"Horst"));
    AssertTrue("Copy constructor", !::wcscmp(s2.PeekBuffer(), s3.PeekBuffer()));
    AssertTrue("Character constructor", !::wcscmp(s4.PeekBuffer(), L"hhhhh"));

    AssertTrue("Test for empty string", s1.IsEmpty());
    AssertFalse("Test for emtpy string", s2.IsEmpty());

    AssertEqual("\"Horst\" consists of 5 characters", s2.Length(), 5);

    AssertEqual("\"Horst\"[0] is 'H'", s2[0], L'H');
    AssertEqual("\"Horst\"[4] is 't'", s2[4], L't');

    try {
        s2[-1];
        AssertTrue("OutOfRangeException at begin", false);
    } catch (OutOfRangeException) {
        AssertTrue("OutOfRangeException at begin", true);
    }

    try {
        s2[s2.Length()];
        AssertTrue("OutOfRangeException at end", false);
    } catch (OutOfRangeException) {
        AssertTrue("OutOfRangeException at end", true);
    }

    AssertTrue("Test for inequality", s1 != s2);
    AssertTrue("Test for equality", s2 == s3);

    s1 = s2;
    AssertTrue("Assignment operator", !::wcscmp(s1.PeekBuffer(), s2.PeekBuffer()));

    AssertTrue("\"Horst\" begins with \"H\"", s1.StartsWith(L"H"));
    AssertTrue("\"Horst\" begins with \"Ho\"", s1.StartsWith(L"Ho"));
    AssertFalse("\"Horst\" does not begin with \"Hu\"", s1.StartsWith(L"Hu"));

    AssertTrue("\"Horst\" ends with \"t\"", s1.EndsWith(L"t"));
    AssertTrue("\"Horst\" ends with \"st\"", s1.EndsWith(L"st"));
    AssertFalse("\"Horst\" does not ends with \"go\"", s1.EndsWith(L"go"));

    AssertEqual("First 'o' in \"Horst\"", s1.IndexOf(L'o'), 1);
    AssertEqual("First 'o' in \"Horst\" after 2", s1.IndexOf(L'o', 2), StringA::INVALID_POS);
    AssertEqual("Last 'o' in \"Horst\"", s1.LastIndexOf(L'o'), 1);

    s4 = s2 + L" und Hugo";
    AssertTrue("Concatenation", !::wcscmp(s4.PeekBuffer(), L"Horst und Hugo"));

    s2 += L" und Hugo";
    AssertTrue("Assignment concatenation", !::wcscmp(s2.PeekBuffer(), L"Horst und Hugo"));

    s2.Replace('H', 'h');
    AssertTrue("Character replacement", !::wcscmp(s2.PeekBuffer(), L"horst und hugo"));
}
