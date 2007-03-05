/*
 * testprocess.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testprocess.h"

#include <iostream>

#ifndef _WIN32
#include <unistd.h>
#endif /* !_WIN32 */

#include "testhelper.h"
//#include "vislib/ImpersonationContext.h"
//#include "vislib/PAMException.h"
#include "vislib/Process.h"
#include "vislib/SystemException.h"

#define IMPERSONATION_USER "mueller"
#define IMPERSONATION_DOMAIN "VISNEW"
#define IMPERSONATION_PASSWORD "mueller"


void TestProcess(void) {
    using namespace std;
    using namespace vislib::sys;

#ifdef _WIN32   // TODO PAM disabled

    ImpersonationContext i;
    try {
        i.Impersonate(IMPERSONATION_USER, IMPERSONATION_DOMAIN, IMPERSONATION_PASSWORD);
#ifndef _WIN32
        cout << "Impersonated as " << ::getuid() << endl;
        cout << "Impersonated as " << ::geteuid() << endl;
#endif /* !_WIN32 */

        i.Revert();
#ifndef _WIN32
        cout << "Impersonated as " << ::getuid() << endl;
        cout << "Impersonated as " << ::geteuid() << endl;
#endif /* !_WIN32 */
    } catch (SystemException se) {
        cout << se.GetMsgA() << endl;
    } catch (PAMException pe) {
        cout << pe.GetMsgA() << endl;
    }

    Process::Environment envi1;

    ::AssertTrue("New environment is empty", envi1.IsEmpty());
    ::AssertTrue("EMPTY_ENVIRONMENT is empty", Process::EMPTY_ENVIRONMENT.IsEmpty());
#ifdef _WIN32
    ::AssertEqual("Empty environment is NULL", static_cast<const void *>(envi1), static_cast<const void *>(NULL));
#else
    ::AssertEqual("Empty environment is NULL", static_cast<char *const *>(envi1), static_cast<char *const *>(NULL));
#endif 

    Process::Environment envi2("test=hugo", NULL);
    ::AssertFalse("Initialised environment is not empty", envi2.IsEmpty());

    Process::Environment envi3("test=hugo", "hugo=horst", NULL);
    ::AssertFalse("Initialised environment is not empty", envi3.IsEmpty());

    try {
        Process p1;
#ifdef _WIN32
        p1.Create("notepad.exe");
#else
        p1.Create("top");
#endif 

        //Process p2;
        //p2.Create("notepad.exe", NULL, IMPERSONATION_USER, IMPERSONATION_DOMAIN, IMPERSONATION_PASSWORD);
    } catch (SystemException se) {
        cout << se.GetMsgA() << endl;
    }

#endif /* _WIN32 */  // TODO PAM disabled
}
