/*
 * testenvironment.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testenvironment.h"

#include <iostream>

#include "vislib/Environment.h"
#include "the/system/system_exception.h"

#include "testhelper.h"


void TestEnvironment(void) {
    using namespace std;
    using namespace vislib::sys;
    the::astring tmp;

    AssertNoException("Getting PATH.", tmp = Environment::GetVariable("PATH"));
    cout << tmp.c_str() << endl;

    AssertNoException("Setting environment variable.", Environment::SetVariable("CROWBAR", "27"));
    AssertNoException("Getting environment variable.", tmp = Environment::GetVariable("CROWBAR"));
    AssertTrue("Variable CORWBAR is set.", Environment::IsSet("CROWBAR"));
    AssertEqual("Correct environment variable set.", tmp.c_str(), "27");


    Environment::Snapshot snapshot;
    AssertTrue("Dft ctor creates empty snapshot.", snapshot.empty());
    AssertEqual("Empty snapshot has zero entries.", snapshot.Count(), size_t(0));
    snapshot.Clear();
    AssertEqual("Clearing empty snapshot has no effect.", snapshot.Count(), size_t(0));

    snapshot = Environment::Snapshot(static_cast<wchar_t *>(NULL));
    AssertTrue("Ctor with NULL parameter empty snapshot.", snapshot.empty());

    snapshot = Environment::Snapshot("CROWBAR=27", NULL);
    AssertFalse("Ctor with one parameter creates non-empty snapshot.", snapshot.empty());
    AssertEqual("Snapshot contains one element.", snapshot.Count(), size_t(1));
    AssertTrue("Variable CORWBAR is set.", snapshot.IsSet("CROWBAR"));

    snapshot = Environment::Snapshot(L"CROWBAR=27", NULL);
    AssertFalse("Ctor with one parameter creates non-empty snapshot.", snapshot.empty());
    AssertEqual("Snapshot contains one element.", snapshot.Count(), size_t(1));
    AssertTrue("Variable CORWBAR is set.", snapshot.IsSet("CROWBAR"));

    snapshot = Environment::Snapshot(L"CROWBAR=27", L"HORSTIFY=TRUE", NULL);
    AssertFalse("Ctor with two parameter creates non-empty snapshot.", snapshot.empty());
    AssertEqual("Snapshot contains two elements.", snapshot.Count(), size_t(2));
    AssertTrue("Variable CORWBAR is set.", snapshot.IsSet("CROWBAR"));
    AssertTrue("Variable HORSTIFY is set.", snapshot.IsSet("HORSTIFY"));
    AssertEqual("CROWBAR is 27 (Unicode).", snapshot.GetVariable(L"CROWBAR").c_str(), L"27");
    AssertEqual("CROWBAR is 27 (ANSI).", snapshot.GetVariable("CROWBAR").c_str(), "27");
    AssertEqual("HORSTIFY is TRUE (Unicode).", snapshot.GetVariable(L"HORSTIFY").c_str(), L"TRUE");
    AssertEqual("HORSTIFY is TRUE (ANSI).", snapshot.GetVariable("HORSTIFY").c_str(), "TRUE");
    AssertFalse("Variable HUGOFY is not set.", snapshot.IsSet("HUGOFY"));
    AssertEqual("HUGOFY is empty (Unicode).", snapshot.GetVariable(L"HUGOFY").c_str(), L"");
    AssertEqual("HUGOFY is empty (ANSI).", snapshot.GetVariable("HUGOFY").c_str(), "");

    snapshot = Environment::CreateSnapshot();
    cout << "Dump of complete environment snapshot:" << endl;
    size_t cntVariables = snapshot.Count();
    for (size_t i = 0; i < cntVariables; i++) {
        the::astring name, value;
        snapshot.GetAt(i, name, value);
        cout << name << " = \"" << value << "\"" << endl;
    }
}
