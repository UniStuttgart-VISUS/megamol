/*
 * DiscoveryTestApp.h
 *
 * Copyright (C) 2009 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_DISCOVERYTESTAPP_H_INCLUDED
#define VISLIBTEST_DISCOVERYTESTAPP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "vislib/Socket.h"
#include "vislib/CmdLineProvider.h"
#include "vislib/DiscoveryListener.h"
#include "vislib/DiscoveryService.h"


class DiscoveryTestApp : public vislib::net::cluster::DiscoveryListener {

public:

    static DiscoveryTestApp& GetInstance(void);

    virtual ~DiscoveryTestApp(void);

    virtual void Initialise(vislib::sys::CmdLineProviderA& inOutCmdLine);

    virtual void Initialise(vislib::sys::CmdLineProviderW& inOutCmdLine);

    virtual void OnNodeFound(vislib::net::cluster::DiscoveryService& src,
        const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer) throw();

    virtual void OnNodeLost(vislib::net::cluster::DiscoveryService& src,
        const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer,
        const vislib::net::cluster::DiscoveryListener::NodeLostReason reason) 
        throw();

    virtual void OnUserMessage(vislib::net::cluster::DiscoveryService& src,
        const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer, 
        const bool isClusterMember,
        const UINT32 msgType, const BYTE *msgBody) throw();

    virtual DWORD Run(void);

protected:

    DiscoveryTestApp(void);

    vislib::net::cluster::DiscoveryService cds;
};

#endif /* VISLIBTEST_DISCOVERYTESTAPP_H_INCLUDED */
