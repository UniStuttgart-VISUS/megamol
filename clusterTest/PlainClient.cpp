/*
 * PlainClient.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "PlainClient.h"

#include <iostream>


/*
 * PlainClient::GetInstance
 */
PlainClient& PlainClient::GetInstance(void) {
    static PlainClient *instance = NULL;

    if (instance == NULL) {
        instance = new PlainClient();
    }

    return *instance;
}


/*
 * PlainClient::~PlainClient
 */
PlainClient::~PlainClient(void) {
}


/*
 * PlainClient::Initialise
 */
void PlainClient::Initialise(vislib::sys::CmdLineProviderA& inOutCmdLine) {
    vislib::net::cluster::AbstractClientNode::Initialise(inOutCmdLine);
}


/*
 * PlainClient::Initialise
 */
void PlainClient::Initialise(vislib::sys::CmdLineProviderW& inOutCmdLine) {
    vislib::net::cluster::AbstractClientNode::Initialise(inOutCmdLine);
}


/*
 * PlainClient::Run
 */
DWORD PlainClient::Run(void) {
    return vislib::net::cluster::AbstractClientNode::Run();
}


/*
 * PlainClient::PlainClient
 */
PlainClient::PlainClient(void) : vislib::net::cluster::AbstractClientNode() {
}


/*
 * PlainClient::onMessageReceived
 */
bool PlainClient::onMessageReceived(const vislib::net::Socket& src, 
        const UINT msgId, const BYTE *body, const SIZE_T cntBody) {
    std::cout << "PlainClient received message " << msgId << " with " 
        << cntBody << " Bytes of body data" << std::endl;
    return false;   
}
