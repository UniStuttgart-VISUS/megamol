/*
 * ibtest.cpp
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "vislib/IbvCommChannel.h"
#include "vislib/IPCommEndPoint.h"


typedef vislib::SmartRef<vislib::net::ib::IbvCommChannel> IbChannel;


int _tmain(int argc, _TCHAR **argv) {
    using namespace vislib;
    using namespace vislib::net;
    using namespace vislib::net::ib;


    IbChannel channel = IbvCommChannel::Create();

    SmartRef<AbstractCommEndPoint> ep = IPCommEndPoint::Create(IPEndPoint::FAMILY_INET, 12345);

    channel->Bind(ep);
    channel->Listen();
    return 0;
}
