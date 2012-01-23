/*
 * ibtest.cpp
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "vislib/IbvCommChannel.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/IbvInformation.h"
#include "vislib/Trace.h"


typedef vislib::SmartRef<vislib::net::ib::IbvCommChannel> IbChannel;


int _tmain(int argc, _TCHAR **argv) {
    using namespace vislib;
    using namespace vislib::net;
    using namespace vislib::net::ib;
    using namespace vislib::sys;

    vislib::Trace::GetInstance().SetLevel(Trace::LEVEL_ALL);

    IbvInformation::DeviceList devices;

    try {
        IbvInformation::GetInstance().GetDevices(devices);

        for (SIZE_T i = 0; i < devices.Count(); i++) {
            std::cout << "Device #" << i << ":" << std::endl;

            std::cout << "\tGUID: " 
                << devices[i].GetNodeGuidA().PeekBuffer() 
                << ": " << std::endl;

            std::cout << "\tSystem Image GUID: " 
                << devices[i].GetSystemImageGuidA().PeekBuffer() << std::endl;

            std::cout << "\tNumber of ports: " << devices[i].GetPortCount() 
                << std::endl;

            for (SIZE_T j = 0; j < devices[i].GetPortCount(); j++) {
                const IbvInformation::Port& port = devices[i].GetPort(j);
                std::cout << "\tPort #" << j << ": " << std::endl;

                std::cout << "\t\tGUID: "
                    << port.GetPortGuidA().PeekBuffer()
                    << std::endl;

                std::cout << "\t\tState: " << port.GetStateA().PeekBuffer()
                    << std::endl;

                std::cout << "\t\tPhysical State: " 
                    << port.GetPhysicalStateA().PeekBuffer()
                    << std::endl;
            }
        }

    } catch (COMException e) {
        std::cerr << "Retrieving IB devices failed: " << e.GetMsgA() 
            << std::endl;
    }


    try {
        IbChannel channel = IbvCommChannel::Create();

        SmartRef<AbstractCommEndPoint> ep = IPCommEndPoint::Create(IPEndPoint::FAMILY_INET, 12345);
        //SmartRef<AbstractCommEndPoint> ep = IPCommEndPoint::Create(IPEndPoint::FAMILY_INET, "169.254.197.153", 12345);

        channel->Bind(ep);
        channel->Listen();
    } catch (COMException e) {
        std::cerr << "Starting IB server failed: " << e.GetMsgA() 
            << std::endl;
    }

    ::_getch();
    return 0;
}
