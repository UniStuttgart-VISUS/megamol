/*
 * PlainClient.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "PlainClient.h"

#include <iostream>

#include "vislib/clustermessages.h"


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
    DWORD retval = vislib::net::cluster::AbstractClientNode::Run();
    char input[1024];

    std::cout << "Input some stuff and press enter to send. "
        << "Ctrl+C to exit." << std::endl;

    do {
        *input = 0;
        std::cin >> input;
        this->sendMessage(VLC1_USER_MSG_ID(1), reinterpret_cast<BYTE *>(input),
            static_cast<UINT32>(strlen(input) + 1));
    } while ((*input != 0) && (*input != '\r') && (*input != '\n'));

    return retval;
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
