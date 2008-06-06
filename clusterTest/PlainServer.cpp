/*
 * PlainServer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "PlainServer.h"

#include <iostream>


/*
 * PlainServer::GetInstance
 */
PlainServer& PlainServer::GetInstance(void) {
    static PlainServer *instance = NULL;

    if (instance == NULL) {
        instance = new PlainServer();
    }

    return *instance;
}


/*
 * PlainServer::~PlainServer
 */
PlainServer::~PlainServer(void) {
}


/*
 * PlainServer::Initialise
 */
void PlainServer::Initialise(vislib::sys::CmdLineProviderA& inOutCmdLine) {
    vislib::net::cluster::AbstractServerNode::Initialise(inOutCmdLine);
}


/*
 * PlainServer::Initialise
 */
void PlainServer::Initialise(vislib::sys::CmdLineProviderW& inOutCmdLine) {
    vislib::net::cluster::AbstractServerNode::Initialise(inOutCmdLine);
}


/*
 * PlainServer::Run
 */
DWORD PlainServer::Run(void) {
    DWORD retval = vislib::net::cluster::AbstractServerNode::Run();
    char dowel;
    
    std::cin >> dowel;
    return retval;
}


/*
 * PlainServer::PlainServer
 */
PlainServer::PlainServer(void) : vislib::net::cluster::AbstractServerNode() {
}


/*
 * PlainServer::onMessageReceived
 */
bool PlainServer::onMessageReceived(const vislib::net::Socket& src, 
        const UINT msgId, const BYTE *body, const SIZE_T cntBody) {
    std::cout << "PlainServer received message " << msgId << " with " 
        << cntBody << " Bytes of body data" << std::endl;
    return false;   
}
