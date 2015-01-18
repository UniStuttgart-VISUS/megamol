/*
 * PlainServer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_PLAINSERVER_H_INCLUDED
#define VISLIBTEST_PLAINSERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/AbstractServerNode.h"


/**
 * This is an empty server node that just reports messages to the console.
 */
class PlainServer : public vislib::net::cluster::AbstractServerNode {

public:

    static PlainServer& GetInstance(void);

    virtual ~PlainServer(void);

    virtual void Initialise(vislib::sys::CmdLineProviderA& inOutCmdLine);

    virtual void Initialise(vislib::sys::CmdLineProviderW& inOutCmdLine);

    virtual DWORD Run(void);

protected:

    PlainServer(void);

    virtual bool onMessageReceived(const vislib::net::Socket& src, 
        const UINT msgId, const BYTE *body, const SIZE_T cntBody);
};

#endif /* VISLIBTEST_PLAINSERVER_H_INCLUDED */
