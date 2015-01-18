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
#include "vislib/ImpersonationContext.h"
#include "vislib/Process.h"
#include "vislib/SystemException.h"


void TestProcess(void) {
    using namespace std;
    using namespace vislib::sys;
    char impersonationUser[256];
    char impersonationDomain[256];
    char impersonationPassword[256];

    ::printf("Domain: ");
    ::fgets(impersonationDomain, sizeof(impersonationDomain), stdin);
    for (int i = 0; impersonationDomain[i] != 0; i++) {
        if (impersonationDomain[i] == '\n') {
            impersonationDomain[i] = '\0';
        }
    }
    ::printf("User: ");
    ::fgets(impersonationUser, sizeof(impersonationUser), stdin);
    for (int i = 0; impersonationUser[i] != 0; i++) {
        if (impersonationUser[i] == '\n') {
            impersonationUser[i] = '\0';
        }
    }
    ::printf("Password: ");
    ::fgets(impersonationPassword, sizeof(impersonationPassword), stdin);
        for (int i = 0; impersonationPassword[i] != 0; i++) {
        if (impersonationPassword[i] == '\n') {
            impersonationPassword[i] = '\0';
        }
    }

    ImpersonationContext i;
    try {
        i.Impersonate(impersonationUser, impersonationDomain, impersonationPassword);
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
    }

    Process p1;
#ifdef _WIN32
    AssertNoException("Process::Create", p1.Create("notepad.exe"));
#else
    AssertNoException("Process::Create", p1.Create("ps"));
#endif 

    Process p2;
#ifdef _WIN32
    AssertException("Process::Create (non-existing image)", p2.Create(".\\crowbar27.exe"), SystemException);
#else /* _WIN32 */
    AssertException("Process::Create (non-existing image)", p2.Create("./crowbar27"), SystemException);
#endif /* _WIN32 */

    //Process p2;
    //p2.Create("notepad.exe", NULL, IMPERSONATION_USER, IMPERSONATION_DOMAIN, IMPERSONATION_PASSWORD);
}

