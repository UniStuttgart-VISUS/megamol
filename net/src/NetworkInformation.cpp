/*
 * NetworkInformation.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/Socket.h"

#include "vislib/NetworkInformation.h"
#include "vislib/assert.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Socket.h"

#ifdef _WIN32
#include "iphlpapi.h"
#include "ws2tcpip.h"

#ifdef _MSC_VER
#pragma comment(lib, "Iphlpapi")
#endif /* _MSC_VER */

#else /* _WIN32 */
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h> 
#include <net/if_arp.h> 
#include <errno.h>

#endif /* _WIN32 */


/*
 * vislib::net::NetworkInformation::countNetAdapters
 */
unsigned int vislib::net::NetworkInformation::countNetAdapters = 0;


/*
 * vislib::net::NetworkInformation::netAdapters
 */
vislib::net::NetworkInformation::AdapterArray
    vislib::net::NetworkInformation::netAdapters;


/*
 * vislib::net::NetworkInformation::Adapter::Adapter
 */
vislib::net::NetworkInformation::Adapter::Adapter(void) 
        : name(), address(127, 0, 0, 1), netmask(0, 0, 0, 1), 
        broadcast(127, 0, 0, 1), mac(), nameValid(NOT_VALID), 
        addressValid(NOT_VALID), netmaskValid(NOT_VALID), 
        broadcastValid(NOT_VALID), macValid(NOT_VALID) {
}


/*
 * vislib::net::NetworkInformation::Adapter::~Adapter
 */
vislib::net::NetworkInformation::Adapter::~Adapter(void) {
}


/*
 * vislib::net::NetworkInformation::AdapterCount
 */
unsigned int vislib::net::NetworkInformation::AdapterCount(void) {
    if (netAdapters.IsNull()) {
        initAdapters();
    }

    return countNetAdapters;
}


/*
 * vislib::net::NetworkInformation::AdapterInformation
 */
const vislib::net::NetworkInformation::Adapter& vislib::net::NetworkInformation::AdapterInformation(unsigned int i) {
    if (netAdapters.IsNull()) {
        initAdapters();
    }

    if (i >= countNetAdapters) {
        throw vislib::OutOfRangeException(i, 0, countNetAdapters - 1, __FILE__, __LINE__);
    }

    return (&*netAdapters)[i];
}


/*
 * vislib::net::NetworkInformation::initAdapters
 */
void vislib::net::NetworkInformation::initAdapters(void) {
    ASSERT(netAdapters.IsNull());
    countNetAdapters = 0;

    try {
        // need to be initialized under windows.
        vislib::net::Socket::Startup(); 

#ifdef _WIN32
        PIP_ADAPTER_INFO adapterInfo;
        DWORD retval = 0;
        ULONG outbufLen = sizeof(IP_ADAPTER_INFO);

        // initially create some memory
        adapterInfo = (IP_ADAPTER_INFO *)malloc(sizeof(IP_ADAPTER_INFO));

        // Make an initial call to GetAdaptersInfo to get
        // the necessary size into the ulOutBufLen variable
        retval = GetAdaptersInfo(adapterInfo, &outbufLen);
        if (retval == ERROR_BUFFER_OVERFLOW) {
            free(adapterInfo);
            adapterInfo = (IP_ADAPTER_INFO *)malloc(outbufLen); 

            // recall with correct size.
            retval = GetAdaptersInfo(adapterInfo, &outbufLen);
        }

        if (retval == ERROR_NOT_SUPPORTED) { 
            // freaky operating system. Need crowbars!
            // Try 'gethostbyname'-crowbar to at least get the ip addresses
            const unsigned int hostnameSize = 1024; // crowbar
            char hostname[hostnameSize];
            ASSERT(countNetAdapters == 0);

            if (gethostname(hostname, hostnameSize) == 0) {
                struct addrinfo aiHints;
                struct addrinfo *aiList = NULL, *aiIt;
                int retVal;

                memset(&aiHints, 0, sizeof(aiHints));
                aiHints.ai_flags = AI_CANONNAME;
                aiHints.ai_family = AF_INET;

                if ((retVal = getaddrinfo(hostname, NULL, &aiHints, &aiList)) == 0) {
                    ASSERT(countNetAdapters == 0);
                    aiIt = aiList;
                    while (aiIt) {
                        countNetAdapters++;
                        aiIt = aiIt->ai_next;
                    }

                    netAdapters = new Adapter[countNetAdapters];
                    countNetAdapters = 0;

                    aiIt = aiList;
                    while (aiIt) {
                        (&*netAdapters)[countNetAdapters].name.Format("Adapter %u", countNetAdapters);
                        (&*netAdapters)[countNetAdapters].nameValid = Adapter::VALID_GENERATED;
                        (&*netAdapters)[countNetAdapters].address = IPAddress(reinterpret_cast<struct sockaddr_in*>(aiIt->ai_addr)->sin_addr);
                        (&*netAdapters)[countNetAdapters].addressValid = Adapter::VALID;
                        (&*netAdapters)[countNetAdapters].netmaskValid = Adapter::NOT_VALID;
                        (&*netAdapters)[countNetAdapters].broadcastValid = Adapter::NOT_VALID;
                        (&*netAdapters)[countNetAdapters].macValid = Adapter::NOT_VALID;

                        countNetAdapters++;
                        aiIt = aiIt->ai_next;
                    }
                }
            }

        } else if (retval == NO_ERROR) { // data has been received
            PIP_ADAPTER_INFO ai = adapterInfo;
            ASSERT(countNetAdapters == 0);
            while (ai) {
                countNetAdapters++;
                ai = ai->Next;
            }

            netAdapters = new Adapter[countNetAdapters];
            countNetAdapters = 0;

            ai = adapterInfo;
            vislib::StringA tmp;
            while (ai) {
                // netAdapters[countNetAdapters].name = ai->AdapterName;
                (&*netAdapters)[countNetAdapters].name = ai->Description;
                (&*netAdapters)[countNetAdapters].nameValid = Adapter::VALID;
                (&*netAdapters)[countNetAdapters].address = IPAddress(ai->IpAddressList.IpAddress.String);
                (&*netAdapters)[countNetAdapters].addressValid = Adapter::VALID;
                (&*netAdapters)[countNetAdapters].netmask = IPAddress(ai->IpAddressList.IpMask.String);
                (&*netAdapters)[countNetAdapters].netmaskValid = Adapter::VALID;
                
                // calculate broadcast address
                UINT32 mask = static_cast<UINT32>(
                    static_cast<struct in_addr>((&*netAdapters)[countNetAdapters].netmask).s_addr);
                if (mask == 0) {
                    (&*netAdapters)[countNetAdapters].broadcastValid = Adapter::NOT_VALID;
                } else {
                    UINT32 addr = static_cast<UINT32>(
                        static_cast<struct in_addr>((&*netAdapters)[countNetAdapters].address).s_addr);
                    struct in_addr broadcast;
                    broadcast.s_addr = addr | ~mask;
                    (&*netAdapters)[countNetAdapters].broadcast = broadcast;
                    (&*netAdapters)[countNetAdapters].broadcastValid = Adapter::VALID_GENERATED;
                }

                // store mac address
                (&*netAdapters)[countNetAdapters].mac = "";
                for (unsigned int i = 0; i < ai->AddressLength; i++) {
                    tmp.Format((i > 0) ? "-%.2X" : "%.2X", ai->Address[i]);
                    (&*netAdapters)[countNetAdapters].mac += tmp;
                }
                (&*netAdapters)[countNetAdapters].macValid = Adapter::VALID;

                countNetAdapters++;
                ai = ai->Next;
            }
        }

        free(adapterInfo);

#else  /* _WIN32 */
        int testSocket = socket(PF_INET, SOCK_DGRAM, 0);
        if (testSocket != -1) {
            // testSocket creating successful

            struct ifreq info;
            countNetAdapters = 0;

            // count the adapters
            for (int i = 1; i < 256; i++) {
                info.ifr_ifindex = i;
                if (ioctl(testSocket, SIOCGIFNAME, &info) == 0) {
                    // Only use ethernet interfaces
                    ioctl(testSocket, SIOCGIFHWADDR, &info);
                    if ((info.ifr_hwaddr.sa_family == ARPHRD_ETHER) 
                            || (info.ifr_hwaddr.sa_family == ARPHRD_INFINIBAND)) {
                        countNetAdapters++;
                    }
                }
            }

            netAdapters = new Adapter[countNetAdapters];
            countNetAdapters = 0;

            // collect adapters informations
            for (int i = 1; i < 256; i++) {
                info.ifr_ifindex = i;
                if (ioctl(testSocket, SIOCGIFNAME, &info) == 0) {
                    // Only use ethernet interfaces
                    ioctl(testSocket, SIOCGIFHWADDR, &info);
                    if ((info.ifr_hwaddr.sa_family == ARPHRD_ETHER) 
                            || (info.ifr_hwaddr.sa_family == ARPHRD_INFINIBAND)) {

                        // name             SIOCGIFNAME
                        info.ifr_ifindex = i;
                        if (ioctl(testSocket, SIOCGIFNAME, &info) == 0) {
                            (&*netAdapters)[countNetAdapters].name = info.ifr_name;
                            (&*netAdapters)[countNetAdapters].nameValid = Adapter::VALID;
                        } else {
                            (&*netAdapters)[countNetAdapters].name.Format("Adapter %u", countNetAdapters);
                            (&*netAdapters)[countNetAdapters].nameValid = Adapter::VALID_GENERATED;
                        }

                        // ip-address       SIOCGIFADDR
                        info.ifr_ifindex = i;
                        if (ioctl(testSocket, SIOCGIFADDR, &info) == 0) {
                            struct sockaddr_in *addr = reinterpret_cast<struct sockaddr_in*>(&info.ifr_addr);
                            (&*netAdapters)[countNetAdapters].address = IPAddress(addr->sin_addr);
                            (&*netAdapters)[countNetAdapters].addressValid = Adapter::VALID;
                        } else {
                            (&*netAdapters)[countNetAdapters].addressValid = Adapter::NOT_VALID;
                        }

                        // ip-subnet-mask   SIOCGIFNETMASK
                        info.ifr_ifindex = i;
                        if (ioctl(testSocket, SIOCGIFNETMASK, &info) == 0) {
                            struct sockaddr_in *mask = reinterpret_cast<struct sockaddr_in*>(&info.ifr_netmask);
                            (&*netAdapters)[countNetAdapters].netmask = IPAddress(mask->sin_addr);
                            (&*netAdapters)[countNetAdapters].netmaskValid = Adapter::VALID;
                        } else {
                            (&*netAdapters)[countNetAdapters].netmaskValid = Adapter::NOT_VALID;
                        }

                        // ip-broadcast     SIOCGIFBRDADDR
                        info.ifr_ifindex = i;
                        if (ioctl(testSocket, SIOCGIFBRDADDR, &info) == 0) {
                            struct sockaddr_in *bcaddr = reinterpret_cast<struct sockaddr_in*>(&info.ifr_broadaddr);
                            (&*netAdapters)[countNetAdapters].broadcast = IPAddress(bcaddr->sin_addr);
                            (&*netAdapters)[countNetAdapters].broadcastValid = Adapter::VALID;
                        } else {
                            if ((&*netAdapters)[countNetAdapters].addressValid &&
                                    (&*netAdapters)[countNetAdapters].netmaskValid) {
                                // calculate broadcast address
                                UINT32 mask = static_cast<UINT32>(
                                    static_cast<struct in_addr>((&*netAdapters)[countNetAdapters].netmask).s_addr);
                                if (mask == 0) {
                                    (&*netAdapters)[countNetAdapters].broadcastValid = Adapter::NOT_VALID;
                                } else {
                                    UINT32 addr = static_cast<UINT32>(
                                        static_cast<struct in_addr>((&*netAdapters)[countNetAdapters].address).s_addr);
                                    struct in_addr broadcast;
                                    broadcast.s_addr = addr | ~mask;
                                    (&*netAdapters)[countNetAdapters].broadcast = broadcast;
                                    (&*netAdapters)[countNetAdapters].broadcastValid = Adapter::VALID_GENERATED;
                                }
                            } else {
                                (&*netAdapters)[countNetAdapters].broadcastValid = Adapter::NOT_VALID;
                            }
                        }

                        // mac address      SIOCGIFHWADDR
                        info.ifr_ifindex = i;
                        if (ioctl(testSocket, SIOCGIFHWADDR, &info) == 0) {
                            unsigned char *uci = (unsigned char *)(&info.ifr_hwaddr.sa_data);
                            (&*netAdapters)[countNetAdapters].mac.Format("%.2X-%.2X-%.2X-%.2X-%.2X-%.2X", 
                                uci[0], uci[1], uci[2], uci[3], uci[4], uci[5]);
                            (&*netAdapters)[countNetAdapters].macValid = Adapter::VALID;
                        } else {
                            (&*netAdapters)[countNetAdapters].mac = "";
                            (&*netAdapters)[countNetAdapters].macValid = Adapter::NOT_VALID;
                        }

                        countNetAdapters++;
                    }
                }
            }

            close(testSocket);
        }

#endif /* _WIN32 */

        // cleanup because we are the good guys.
        vislib::net::Socket::Cleanup(); 

    } catch(...) {
        countNetAdapters = 0;

    }

    if (countNetAdapters == 0) {
        // generate a clean state
        netAdapters = new Adapter[1]; // create one dummy element to avoid 
                                      // calling 'InitAdapters' all the time.
    }

}


/*
 * vislib::net::NetworkInformation::NetworkInformation
 */
vislib::net::NetworkInformation::NetworkInformation(void) {
    throw vislib::UnsupportedOperationException("NetworkInformation ctor", __FILE__, __LINE__);
}


/*
 * vislib::net::NetworkInformation::NetworkInformation
 */
vislib::net::NetworkInformation::NetworkInformation(const NetworkInformation &rhs) {
    throw vislib::UnsupportedOperationException("NetworkInformation copy ctor", __FILE__, __LINE__);
}


/*
 * vislib::net::NetworkInformation::~NetworkInformation
 */
vislib::net::NetworkInformation::~NetworkInformation(void) {
    throw vislib::UnsupportedOperationException("NetworkInformation dtor", __FILE__, __LINE__);
}
