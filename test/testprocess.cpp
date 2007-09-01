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
#ifdef _WIN32
#include "vislib/ImpersonationContext.h"
#include "vislib/PAMException.h"
#endif  // TODO PAM disabled
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

#endif /* _WIN32 */  // TODO PAM disabled

    Process p1;
#ifdef _WIN32
    AssertNoException("Process::Create", p1.Create("notepad.exe"));
    AssertNoException("Process::Terminate", p1.Terminate());
#else /* _WIN32 */
    AssertNoException("Process::Create", p1.Create("ps"));
#endif /* _WIN32 */

    Process p2;
#ifdef _WIN32
    AssertException("Process::Create (non-existing image)", p2.Create(".\\crowbar27.exe"), SystemException);
#else /* _WIN32 */
    AssertException("Process::Create (non-existing image)", p2.Create("./crowbar27"), SystemException);
#endif /* _WIN32 */

    //Process p2;
    //p2.Create("notepad.exe", NULL, IMPERSONATION_USER, IMPERSONATION_DOMAIN, IMPERSONATION_PASSWORD);
}
