/*
 * testtrayicon.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testtrayicon.h"

#include <iostream>

#include "resource.h"

#include "vislib/TrayIcon.h"
#include "vislib/Thread.h"
#include "vislib/SystemException.h"


#ifdef _WIN32

class MyTrayIcon : public vislib::sys::TrayIcon {

public:
    inline MyTrayIcon(void) : vislib::sys::TrayIcon() {} 

    virtual ~MyTrayIcon(void) {}

    virtual LRESULT onNotify(WPARAM wParam, LPARAM lParam);
};


LRESULT MyTrayIcon::onNotify(WPARAM wParam, LPARAM lParam) {
    std::cout << "Tray icon was notified." << std::endl;
    return 0;
}

#endif /* _WIN32 */


/*
 * ::TestTrayIcon
 */
void TestTrayIcon(void) {
#ifdef _WIN32

    //MyTrayIcon trayIcon;
    MyTrayIcon myTrayIcon;
    vislib::sys::TrayIcon trayIcon;

    trayIcon.Create(NULL, WM_APP + 1, 42, ::GetModuleHandle(NULL), "ANSI-VISlib™",
        101, true); 
    std::cout << "Tray icon becomes visible in two seconds." << std::endl;
    vislib::sys::Thread::Sleep(2000);
    trayIcon.Show();
    trayIcon.ShowBalloonHelp("I can do balloon help, me.", "Achtung, Achtung!", 
        NIIF_INFO);

    // Sleep to see the balloon help.
    for (int i = 2; i > 0; i--) {
        std::cout << "Waiting for " << i << " seconds ..." << std::endl;
        vislib::sys::Thread::Sleep(1000);
    }

    trayIcon.ShowBalloonHelp(::GetModuleHandle(NULL), IDS_TEST);
    // Sleep to see the balloon help.
    for (int i = 2; i > 0; i--) {
        std::cout << "Waiting for " << i << " seconds ..." << std::endl;
        vislib::sys::Thread::Sleep(1000);
    }

    trayIcon.ShowBalloonHelp(L"");


    myTrayIcon.Create(NULL, WM_APP + 1, 43, ::GetModuleHandle(NULL), L"VISlib™", 
        101, false, L"Die VISlib kann schon richtig nerven. Man könnte fast "
        L"meinen, man mache sich zum Horst für VIS ... oder für VISUS. Jetzt "
        L"wollen wir doch mal sehen, ob wir nicht die Grenze für die maximale "
        L"Länge einer Balloon-Help erreichen können. Oder zumindest fast.",
        L"Die VISlib™ kann schon richtig nerven ...", NIIF_ERROR); 

    // Sleep to see the balloon help.
    for (int i = 5; i > 0; i--) {
        std::cout << "Waiting for " << i << " seconds ..." << std::endl;
        vislib::sys::Thread::Sleep(1000);
    }

#endif /* _WIN32 */
}
