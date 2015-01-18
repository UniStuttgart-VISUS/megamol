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

    std::cout << "Tray icon becomes transparent and greyscale and stuff in two seconds." << std::endl;
    vislib::sys::Thread::Sleep(2000);

	HDC hDC = ::GetDC(NULL);
	HBITMAP hAndBmp = CreateCompatibleBitmap(hDC, 16, 16);
	HBITMAP hXorBmp = CreateCompatibleBitmap(hDC, 16, 16);
	HDC hAndDC = CreateCompatibleDC(hDC); 
	HDC hXorDC = CreateCompatibleDC(hDC);
	HBITMAP hOldAndBmp  = reinterpret_cast<HBITMAP>(SelectObject(hAndDC, hAndBmp)); // think alpha
	HBITMAP hOldXorBmp  = reinterpret_cast<HBITMAP>(SelectObject(hXorDC, hXorBmp)); // think color
	for (unsigned int x = 0; x < 16; x++) {
		for (unsigned int y = 0; y < 16; y++) {
			if (y < 8) {
				SetPixel(hAndDC, x, y, RGB(255, 255, 255));
				//SetPixel(hXorDC, x, y, RGB(0, 0, 0)); // superfluous
			} else {
				SetPixel(hAndDC, x, y, RGB(0, 0, 0));
				SetPixel(hXorDC, x, y, RGB(x * 16, x * 16, x * 16));
			}
		}
	}
	SelectObject(hAndDC, hOldAndBmp);
	SelectObject(hXorDC, hOldXorBmp);

	ICONINFO myII;
	myII.fIcon = true;
	myII.xHotspot = 0;
	myII.yHotspot = 0;
	myII.hbmMask = hAndBmp;
	myII.hbmColor = hXorBmp;
	HICON hIco = CreateIconIndirect(&myII);
	if (hIco == NULL) {
		std::cout << "FAILED creating an icon from the bitmaps: " << GetLastError() << std::endl;
	}
	trayIcon.SetIcon(hIco);
	DestroyIcon(hIco);

	DeleteObject(hAndBmp);
	DeleteObject(hXorBmp);
	DeleteDC(hAndDC);
	DeleteDC(hXorDC);
	ReleaseDC(NULL, hDC);

    std::cout << "Tray icon becomes vis again in two seconds." << std::endl;
    vislib::sys::Thread::Sleep(2000);

	trayIcon.SetIcon(::GetModuleHandle(NULL), 101);

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

	trayIcon.Destroy();

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
