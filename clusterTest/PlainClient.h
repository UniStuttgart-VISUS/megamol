/*
 * PlainClient.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_PLAINCLIENT_H_INCLUDED
#define VISLIBTEST_PLAINCLIENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/AbstractClientNode.h"


class PlainClient : public vislib::net::cluster::AbstractClientNode {

public:

    static PlainClient& GetInstance(void);

    virtual ~PlainClient(void);

    virtual void Initialise(vislib::sys::CmdLineProviderA& inOutCmdLine);

    virtual void Initialise(vislib::sys::CmdLineProviderW& inOutCmdLine);

    virtual unsigned int Run(void);

protected:

    PlainClient(void);

    virtual bool onMessageReceived(const vislib::net::Socket& src, 
        const unsigned int msgId, const uint8_t *body, const size_t cntBody);
};

#endif /* VISLIBTEST_PLAINCLIENT_H_INCLUDED */
