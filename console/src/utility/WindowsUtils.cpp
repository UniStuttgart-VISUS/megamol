/*
 * WindowsUtils.cpp
 *
 * Copyright (C) 2016 by MegaMol Team, TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#ifdef _WIN32
#include "utility/WindowsUtils.h"
#include "vislib/String.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/VersionNumber.h"
#include <strsafe.h>
#include <iostream>
#include <iomanip>
#include <mutex>

using namespace megamol;
using namespace megamol::console;

namespace {

    bool TestWindowsVersion() {
        const TCHAR* systemDll = _T("Kernel32.dll");
        DWORD size = GetFileVersionInfoSize(systemDll, nullptr);
        char *buf = new char[size];
        GetFileVersionInfo(systemDll, 0, size, buf);
        void* vidata = nullptr;
        unsigned int visize = 0;
        TCHAR subblock[50];
        subblock[0] = 0;
        if (VerQueryValue(buf, _T("\\VarFileInfo\\Translation"), &vidata, &visize)) {
            WORD * trans = reinterpret_cast<WORD*>(vidata);
            StringCchPrintf(subblock, 50, _T("\\StringFileInfo\\%04x%04x\\ProductVersion"), trans[0], trans[1]);
        }
        vidata = nullptr;
        visize = 0;
        vislib::VersionNumber ver;
        if (VerQueryValue(buf, subblock, &vidata, &visize)) {
            ver = vislib::VersionNumber(reinterpret_cast<char*>(vidata));
        }
        delete[] buf;
        return ver.GetMajorVersionNumber() >= 10;
    }

}

void utility::windowsConsoleWindowSetup(void* hCore) {
    // ask configuration if we want to move the console window
    int moveConWin = -1;
    mmcValueType valType;
    const void *val = ::mmcGetConfigurationValueA(hCore, MMC_CFGID_VARIABLE, "MoveConsoleWindow", &valType);
    try {
        switch (valType) {
        case MMC_TYPE_CSTR:
            moveConWin = vislib::CharTraitsA::ParseInt(static_cast<const char*>(val));
            break;
        case MMC_TYPE_WSTR:
            moveConWin = vislib::CharTraitsW::ParseInt(static_cast<const wchar_t*>(val));
            break;
        }
    } catch (...) {
    }
    if (moveConWin < 0) return;

    // The console window handle
    HWND hWnd = ::GetConsoleWindow();
    vislib::sys::SystemInformation::MonitorRectArray monitors;
    vislib::sys::SystemInformation::MonitorRects(monitors);
    moveConWin = vislib::math::Clamp<int>(moveConWin, 0, static_cast<int>(monitors.Count() - 1));

    // test operating system version for new console implementation
    if (TestWindowsVersion()) {
        // Windows 10 or newer
        UINT flags = SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOZORDER;

        RECT wr;
        ::GetWindowRect(hWnd, &wr);

        POINT p;
        p.x = monitors[moveConWin].Left() + 1;
        p.y = monitors[moveConWin].Top() + 1;
        HMONITOR hMon = ::MonitorFromPoint(p, MONITOR_DEFAULTTONEAREST);
        MONITORINFO monInfo;
        monInfo.cbSize = sizeof(MONITORINFO);
        if (::GetMonitorInfo(hMon, &monInfo) == 0) {
            monInfo.rcWork.top = 0;
            monInfo.rcWork.bottom = monitors[moveConWin].Height();
            flags |= SWP_NOSIZE;
        }

        ::SetWindowPos(hWnd, NULL, monitors[moveConWin].Left(), monitors[moveConWin].Top(), wr.right - wr.left, monInfo.rcWork.bottom - monInfo.rcWork.top, flags);

    } else {
        DWORD le = GetLastError();
        if (le != ERROR_OLD_WIN_VERSION) {
            // it is an error. I do not care
        }

        // pre Windows 10 console resize need to be performed by SCREEN_BUFFER size

        HANDLE hCO = ::GetStdHandle(STD_OUTPUT_HANDLE);
        COORD maxSize = ::GetLargestConsoleWindowSize(hCO);

        ::SetWindowPos(hWnd, NULL, monitors[moveConWin].Left(), monitors[moveConWin].Top(), 0, 0,
            SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOZORDER);

        RECT r;
        ::GetWindowRect(hWnd, &r);
        int h1 = r.bottom - r.top;

        CONSOLE_SCREEN_BUFFER_INFO csbi;
        ::GetConsoleScreenBufferInfo(hCO, &csbi);
        csbi.srWindow.Bottom++;
        ::SetConsoleWindowInfo(hCO, TRUE, &csbi.srWindow);
        ::Sleep(10);

        ::GetWindowRect(hWnd, &r);
        int h2 = r.bottom - r.top;

        if (h2 > h1) {
            POINT p;
            p.x = monitors[moveConWin].Left() + 1;
            p.y = monitors[moveConWin].Top() + 1;
            HMONITOR hMon = ::MonitorFromPoint(p, MONITOR_DEFAULTTONEAREST);
            MONITORINFO monInfo;
            monInfo.cbSize = sizeof(MONITORINFO);
            if (::GetMonitorInfo(hMon, &monInfo) == 0) {
                monInfo.rcWork.top = 0;
                monInfo.rcWork.bottom = monitors[moveConWin].Height();
            } else if (monInfo.rcWork.top != 0) {
                ::SetWindowPos(hWnd, NULL, monInfo.rcWork.left, monInfo.rcWork.top, 0, 0,
                    SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOZORDER);
            }

            h1 = static_cast<int>(floor(static_cast<double>(monInfo.rcWork.bottom - monInfo.rcWork.top - h2) / static_cast<double>(h2 - h1)));

            csbi.srWindow.Bottom += h1;

            if (::SetConsoleWindowInfo(hCO, TRUE, &csbi.srWindow) == 0) {
                csbi.srWindow.Bottom -= h1;
                while (h2 < (monInfo.rcWork.bottom - monInfo.rcWork.top)) {
                    csbi.srWindow.Bottom++;
                    if (::SetConsoleWindowInfo(hCO, TRUE, &csbi.srWindow) == 0) break;
                    ::GetWindowRect(hWnd, &r);
                    h2 = r.bottom - r.top;
                }
            }
        }

    }

}

namespace {
    std::mutex consoleEchoLock;
}

void MEGAMOLCORE_CALLBACK utility::windowsConsoleLogEcho(unsigned int level, const char* message) {
    std::lock_guard<std::mutex> lock(consoleEchoLock);

    HANDLE hCO = ::GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO sbi;
    if (level <= 200) {
        GetConsoleScreenBufferInfo(hCO, &sbi);
        if (level <= 1) {
            SetConsoleTextAttribute(hCO, FOREGROUND_RED | FOREGROUND_INTENSITY);
        } else if (level <= 100) {
            SetConsoleTextAttribute(hCO, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
        } else {
            SetConsoleTextAttribute(hCO, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
        }
    }
    std::cout << std::setw(4) << level;
    if (level <= 200) {
        SetConsoleTextAttribute(hCO, sbi.wAttributes);
    }
    std::cout << "|" << message;
}

#endif
