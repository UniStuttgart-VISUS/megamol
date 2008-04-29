/*
 * testipv6.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "testipv6.h"

#include "vislib/DNS.h"
#include "vislib/IPAddress6.h"
#include "vislib/Socket.h"
#include "vislib/SocketException.h"
#include "testhelper.h"


void TestIPv6(void) {
    using namespace vislib::net;

    Socket::Startup();

    AssertTrue("LOOPBACK constant", IPAddress6::LOOPBACK.IsLoopback());
    AssertTrue("LOCALHOST constant", IPAddress6::LOCALHOST.IsLoopback());
    AssertTrue("ANY constant", IPAddress6::ANY.IsUnspecified());
    AssertTrue("UNSPECIFIED constant", IPAddress6::UNSPECIFIED.IsUnspecified());


    IPAddress6 ipAddr;
    AssertTrue("Default ctor creates loopback address", ipAddr.IsLoopback());

    IPAddress ipV4Addr;
    ipAddr.MapV4Address(ipV4Addr);
    AssertTrue("Mapping IPv4 address", ipAddr.IsV4Mapped());

    IPAddress6 ipAddr2(ipAddr);
    AssertTrue("Copy ctor", (IN6_ADDR_EQUAL((in6_addr *) ipAddr, (in6_addr *) ipAddr2) != 0));
    AssertTrue("operator ==", ipAddr == ipAddr2);
    AssertFalse("operator !=", ipAddr != ipAddr2);

    ipAddr2 = ipV4Addr;
    AssertTrue("IPv4 assignment", ipAddr2.IsV4Mapped());


    IPHostEntryA hostEntryA;
    AssertNoException("Looking up google.de", DNS::GetHostEntry(hostEntryA, "google.de"));

    Socket::Cleanup();
}
