/*
 * testipv6.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "testipv6.h"

#include "vislib/DNS.h"
#include "vislib/IllegalStateException.h"
#include "vislib/IPEndPoint.h"
#include "vislib/IPAddress.h"
#include "vislib/IPAddress6.h"
#include "vislib/Socket.h"
#include "vislib/SocketException.h"
#include "testhelper.h"


void TestIPv6(void) {
#ifndef _WIN32
#define IN6_ADDR_EQUAL IN6_ARE_ADDR_EQUAL
#endif /* !_WIN32 */
    using namespace vislib::net;

    Socket::Startup();

    //std::cout << IPAddress::LOCALHOST.ToStringA().PeekBuffer() << std::endl;

    AssertTrue("LOOPBACK constant", IPAddress6::LOOPBACK.IsLoopback());
    std::cout << "LOOPBACK constant is: " << IPAddress6::LOOPBACK.ToStringA().PeekBuffer() << std::endl;

    AssertTrue("LOCALHOST constant", IPAddress6::LOCALHOST.IsLoopback());
    std::cout << "LOCALHOST constant is: " << IPAddress6::LOCALHOST.ToStringA().PeekBuffer() << std::endl;

    AssertTrue("ANY constant", IPAddress6::ANY.IsUnspecified());
    std::cout << "ANY constant is: " << IPAddress6::ANY.ToStringA().PeekBuffer() << std::endl;

    AssertTrue("UNSPECIFIED constant", IPAddress6::UNSPECIFIED.IsUnspecified());
    std::cout << "UNSPECIFIED constant is: " << IPAddress6::UNSPECIFIED.ToStringA().PeekBuffer() << std::endl;


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

    AssertNoException("IPAddress6::Create(\"::1\")", ipAddr = IPAddress6::Create("::1"));
    AssertTrue("Created LOOPBACK address (IsLoopback).", ipAddr.IsLoopback());
    AssertEqual("Created LOOPBACK address (operator ==).", ipAddr, IPAddress6::LOOPBACK);

    IPHostEntryA hostEntryA;
    AssertNoException("Looking up 127.0.0.1", DNS::GetHostEntry(hostEntryA, "127.0.0.1"));
    AssertNoException("Looking up ::1", DNS::GetHostEntry(hostEntryA, "::1"));
    AssertNoException("Looking up ::0000:0001", DNS::GetHostEntry(hostEntryA, "::0000:0001"));
    AssertNoException("Looking up fe80::5efe:127.0.0.1", DNS::GetHostEntry(hostEntryA, "fe80::5efe:127.0.0.1"));
    AssertNoException("Looking up www.google.de", DNS::GetHostEntry(hostEntryA, "www.google.de"));

    IPHostEntryW hostEntryW;
    AssertNoException("Looking up 127.0.0.1", DNS::GetHostEntry(hostEntryW, L"127.0.0.1"));
    AssertNoException("Looking up ::1", DNS::GetHostEntry(hostEntryW, L"::1"));
    AssertNoException("Looking up ::0000:0001", DNS::GetHostEntry(hostEntryW, L"::0000:0001"));
    AssertNoException("Looking up fe80::5efe:127.0.0.1", DNS::GetHostEntry(hostEntryW, L"fe80::5efe:127.0.0.1"));
    AssertNoException("Looking up www.google.de", DNS::GetHostEntry(hostEntryW, L"www.google.de"));


    IPEndPoint endPointV4;
    std::cout << "IPv4 default end point: " << endPointV4.ToStringA().PeekBuffer() << std::endl;

    IPEndPoint endPointV6(IPAddress6::ANY, 0);
    std::cout << "IPv6 default end point: " << endPointV6.ToStringA().PeekBuffer() << std::endl;
    AssertException("GetIPAddress4 or IPv6 endpoint.", endPointV6.GetIPAddress4(), vislib::IllegalStateException);

    ipAddr2 = ipV4Addr;
    endPointV6.SetIPAddress(ipAddr2);
    AssertNoException("GetIPAddress4 or IPv6 mapped IPv4 endpoint.", endPointV6.GetIPAddress4());

    Socket::Cleanup();
}
