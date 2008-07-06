/*
 * AsyncSocketSender.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AsyncSocketSender.h"

#include "vislib/Exception.h"
#include "vislib/Trace.h"


/*
 * vislib::net::AsyncSocketSender::AsyncSocketSender
 */
vislib::net::AsyncSocketSender::AsyncSocketSender(void) : socket(NULL) {
    // Nothing else to do.
}


/*
 * vislib::net::AsyncSocketSender::~AsyncSocketSender
 */
vislib::net::AsyncSocketSender::~AsyncSocketSender(void) {
    try {
        this->Terminate();
    } catch (Exception& e) {
        TRACE(Trace::LEVEL_VL_WARN, "Exception while destroying "
            "AsyncSocketSender: %s\n", e.GetMsgA());
    }
}


/*
 * vislib::net::AsyncSocketSender::Run
 */
DWORD vislib::net::AsyncSocketSender::Run(void *socket) {
    return this->Run(static_cast<Socket *>(socket));
}


/*
 * vislib::net::AsyncSocketSender::Run
 */
DWORD vislib::net::AsyncSocketSender::Run(Socket *socket) {
    // TODO
    return 0;
}


/*
 * vislib::net::AsyncSocketSender::Run
 */
bool vislib::net::AsyncSocketSender::Terminate(void) {
    // TODO
    return false;
}
