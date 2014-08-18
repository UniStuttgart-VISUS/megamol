/*
 * testnetinfo.cpp
 *
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <winsock2.h>
#endif /* _WIN32 */

#include "testnetinfo.h"
#include "testhelper.h"

#include "vislib/StringConverter.h"
#include "vislib/NetworkInformation.h"



bool PrintAdapter(const vislib::net::NetworkInformation::Adapter& a, void *ctx) {
    using namespace vislib;
    using namespace vislib::net;

    std::cout << std::endl;
    std::cout << "ID: " << a.GetID() << std::endl;
    std::cout << "\tFriendly Name: " << W2A(a.GetName()) << std::endl;
    std::cout << "\tDescription: " << W2A(a.GetDescription()) << std::endl;

    std::cout << "\tUnicast Addresses:" << std::endl;
    const NetworkInformation::UnicastAddressList& unicastAddresses = a.GetUnicastAddresses();
    for (SIZE_T i = 0; i < unicastAddresses.Count(); i++) {
        const NetworkInformation::UnicastAddressInformation& ai = unicastAddresses[i];
        std::cout << "\t\t" << ai.GetAddress().ToStringA().PeekBuffer() 
            << "/" << ai.GetPrefixLength() << std::endl;
        std::cout << "\t\t\tOrigin: ";
        try {
            std::cout << NetworkInformation::Stringise(ai.GetPrefixOrigin());
        } catch (...) {
            std::cout << "INVALID";
        }
        std::cout << ", ";
        try {
            std::cout << NetworkInformation::Stringise(ai.GetSuffixOrigin());
        } catch (...) {
            std::cout << "INVALID";
        } 
        std::cout << std::endl;

        if (ai.GetAddressFamily() == IPAgnosticAddress::FAMILY_INET) {
            std::cout << "\t\t\tNetmask: ";
            std::cout << NetworkInformation::PrefixToNetmask4(ai.GetPrefixLength()).ToStringA().PeekBuffer();
            std::cout << std::endl;
        }
    }
    std::cout << "\tMulticast Addresses:" << std::endl;
    try {
        const NetworkInformation::AddressList& multicastAddresses = a.GetMulticastAddresses();
        for (SIZE_T i = 0; i < multicastAddresses.Count(); i++) {
            std::cout << "\t\t" << (multicastAddresses[i]).ToStringA().PeekBuffer() << std::endl;
        }
    } catch (...) {
        std::cout << "\t\tINVALID";
    }
    std::cout << std::endl;

    std::cout << "\tAnycast Addresses:" << std::endl;
    try {
        const NetworkInformation::AddressList& anycastAddresses = a.GetAnycastAddresses();
        for (SIZE_T i = 0; i < anycastAddresses.Count(); i++) {
            std::cout << "\t\t" << (anycastAddresses[i]).ToStringA().PeekBuffer() << std::endl;
        }
    } catch (...) {
        std::cout << "\t\tINVALID";
    }
    std::cout << std::endl;

    std::cout << "\tIPv4 Broadcast Address: ";
    try {
        std::cout << a.GetBroadcastAddress().ToStringA().PeekBuffer();
    } catch (...) {
        std::cout << "INVALID";
    }
    std::cout << std::endl;

    std::cout << "\tMTU: ";
    try {
        std::cout << a.GetMTU();
    } catch (...) {
        std::cout << "INVALID";
    }
    std::cout << std::endl;

    std::cout << "\tType: ";
    try {
        std::cout << NetworkInformation::Stringise(a.GetType());
    } catch (...) {
        std::cout << "INVALID";
    }
    std::cout << std::endl;

    std::cout << "\tStatus: ";
    try {
        std::cout << NetworkInformation::Stringise(a.GetStatus());
    } catch (...) {
        std::cout << "INVALID";
    }
    std::cout << std::endl;

    std::cout << "\tMAC: ";
    try {
        std::cout << a.FormatPhysicalAddressA().PeekBuffer();
    } catch (...) {
        std::cout << "INVALID";
    }
    std::cout << std::endl;

    return true;
}


void PrintAdapterList(const vislib::net::NetworkInformation::AdapterList& a) {
    for (SIZE_T i = 0; i < a.Count(); i++) {
        std::cout << "\t\"" << a[i].GetID() << "\" \"" << W2A(a[i].GetName()) << "\"" << std::endl;
    }
}


void TestNetworkInformation(void) {
    using namespace vislib::net;

    NetworkInformation::Adapter guessedAdapter;
    IPEndPoint guessedEndPoint;
    const char *guessInput = NULL;
    float wildness = 0.0;


    Socket::Startup();

    SIZE_T cntAdapters = NetworkInformation::CountAdapters();
    std::cout << "Number of network adapters: " << cntAdapters << std::endl;
    NetworkInformation::EnumerateAdapters(::PrintAdapter);
    std::cout << std::endl;


    /* Get some specific stuff. */
    NetworkInformation::AdapterList loopbackAdapters;
    NetworkInformation::GetAdaptersForUnicastAddress(loopbackAdapters, IPAddress::LOCALHOST);
    std::cout << std::endl << "Adapters bound to loopback address: " << std::endl;
    ::PrintAdapterList(loopbackAdapters);

    NetworkInformation::AdapterList ethernetAdapters;
    NetworkInformation::GetAdaptersForType(ethernetAdapters, NetworkInformation::Adapter::TYPE_ETHERNET);
    std::cout << std::endl << "Ethernet adapters: " << std::endl;
    ::PrintAdapterList(ethernetAdapters);

    if (!ethernetAdapters.IsEmpty()) {
        NetworkInformation::UnicastAddressInformation ref = ethernetAdapters[0].GetUnicastAddresses()[0];
        NetworkInformation::AdapterList result;

        std::cout << std::endl << "Adapters in subnet of " << ref.GetAddress().ToStringA().PeekBuffer()
            << "/" << ref.GetPrefixLength() << ": " << std::endl;
        NetworkInformation::GetAdaptersForUnicastPrefix(result, ref.GetAddress(), ref.GetPrefixLength());
        ::PrintAdapterList(result);
    }



    /* Test wild guess. */
    guessInput = "LAN";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "LAN-Verbindun";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "eth0";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "localhost/255.255.255.0:12345";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "localhost/255.255.255.0";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "localhost:12345";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "localhost:";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "::1";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "[::1]";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "::1/128:12345";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "fe80::bd28:d109:20ac:8a21%9:12345";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "[::1]:12345";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "127.0.0.1/24:12345";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "127.0.0.1/24";
    wildness = NetworkInformation::GuessAdapter(guessedAdapter, guessInput);
    std::cout << "Guessed \"" << guessedAdapter.GetID().PeekBuffer() 
        << "\" (\"" << W2A(guessedAdapter.GetName()) << "\") "
        << "with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;


    /* Guess local end points. */
    guessInput = "127.0.0.1/24";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "salsa/24";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "salsa/255.255.255.0";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "klassik/255.255.255.0";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "klassik";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "localhost/128";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "/64";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "/255.255.255.0";
    wildness = NetworkInformation::GuessLocalEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;


    /* Guess remote endpoints. */
    guessInput = "salsa.informatik.uni-stuttgart.de";
    wildness = NetworkInformation::GuessRemoteEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "salsa.informatik.uni-stuttgart.de/255.255.255.0";
    wildness = NetworkInformation::GuessRemoteEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "salsa.informatik.uni-stuttgart.de/255.255.0.0";
    wildness = NetworkInformation::GuessRemoteEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "salsa.informatik.uni-stuttgart.de/48";
    wildness = NetworkInformation::GuessRemoteEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    guessInput = "/48";
    wildness = NetworkInformation::GuessRemoteEndPoint(guessedEndPoint, guessInput);
    std::cout << "Guessed \"" << guessedEndPoint.ToStringA().PeekBuffer()
        << "\" with wildness " << wildness 
        << " from \"" << guessInput << "\"" << std::endl;

    Socket::Cleanup();

    //NetworkInformation::DiscardCache();
    //_CrtDumpMemoryLeaks();
}
