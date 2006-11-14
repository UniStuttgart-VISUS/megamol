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

    } catch (SystemException e) {
        std::cout << "SystemException: " << e.GetErrorCode() << " " << e.GetMsg() << std::endl;

    } catch(...) {
        std::cout << "Unexpected exception catched." << std::endl;
    }

}
