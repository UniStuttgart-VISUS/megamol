/*
 * ibtest.cpp
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "vislib/IbvCommChannel.h"


typedef vislib::SmartRef<vislib::net::ib::IbvCommChannel> IbChannel;


int _tmain(int argc, _TCHAR **argv) {

    IbChannel channel = vislib::net::ib::IbvCommChannel::Create();
    return 0;
}

