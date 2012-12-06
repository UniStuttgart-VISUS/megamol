/*
 * testsysinfo.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/SystemException.h"
#include "vislib/SystemInformation.h"

#include <iostream>

void TestSysInfo(void) {
    using namespace vislib::sys;

    try {
        vislib::TString userName;
        vislib::TString compName;
        vislib::sys::SystemInformation::UserName(userName);
        vislib::sys::SystemInformation::ComputerName(compName);
        ::_tprintf(_T("Running as %s@%s (%u Proc.)\n"), userName.PeekBuffer(), compName.PeekBuffer()
            , vislib::sys::SystemInformation::ProcessorCount());

        std::cout << "Page Size: " << vislib::sys::SystemInformation::PageSize() << " Bytes." << std::endl;
        std::cout << "Total Memory: " << vislib::sys::SystemInformation::PhysicalMemorySize() << " Bytes." << std::endl;
        std::cout << "Free Memory:  " << vislib::sys::SystemInformation::AvailableMemorySize() << " Bytes." << std::endl;

        std::cout << "System Type: " << vislib::sys::SystemInformation::SystemType() << std::endl;
        std::cout << "System Word Size: " << vislib::sys::SystemInformation::SystemWordSize() << std::endl;
        std::cout << "Self System Type: " << vislib::sys::SystemInformation::SelfSystemType() << std::endl;
        std::cout << "Self Word Size: " << vislib::sys::SystemInformation::SelfWordSize() << std::endl;

        DWORD verMajor;
        DWORD verMinor;
        vislib::sys::SystemInformation::SystemVersion(verMajor, verMinor);
        std::cout << "System version " << verMajor << "." << verMinor << std::endl;

        SystemInformation::MonitorRectArray monitorSizes;
        DWORD cntMonitors = SystemInformation::MonitorRects(monitorSizes);
        std::cout << "Found " << cntMonitors << " monitors" << std::endl;
        for (DWORD i = 0; i < cntMonitors; i++) {
            std::cout << "Monitor " << i << " has origin ("
                << monitorSizes[int(i)].Left() << ", " 
                << monitorSizes[int(i)].Top() << ") (top) and size [" 
                << monitorSizes[int(i)].Width() << ", " 
                << monitorSizes[int(i)].Height() << "]" 
                << std::endl;
        }

        SystemInformation::MonitorRect priMonSize = SystemInformation::PrimaryMonitorRect();
        std::cout << "Pimary monitor origin: (" 
            << priMonSize.Left() << ", "
            << priMonSize.Top() << ") (top)" << std::endl
            << "Primary monitor size: [" 
            << priMonSize.Width() << ", " 
            << priMonSize.Height() << "]" << std::endl;

        SystemInformation::MonitorRect virtScreen = SystemInformation::VirtualScreen();
        std::cout << "Virtual screen from ("
            << virtScreen.Left() << ", "
            << virtScreen.Top() << ") (top) to ("
            << virtScreen.Right() << ", "
            << virtScreen.Top() << ")" << std::endl
            << "Virtual screen size: ["
            << virtScreen.Width() << ", "
            << virtScreen.Height() << "]" << std::endl;


    } catch (SystemException se) {
        std::cout << "SystemException: " << se.GetErrorCode() << " " << se.GetMsgA() << std::endl;

    } catch (vislib::Exception e) {
        std::cout << "Exception: " << e.GetMsgA() << std::endl;

    } catch(...) {
        std::cout << "Unexpected exception catched." << std::endl;
    }

}
