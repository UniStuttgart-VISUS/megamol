/*
 * messagereceiver.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MESSAGERECEIVER_H_INCLUDED
#define VISLIB_MESSAGERECEIVER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/Socket.h"      // Must be first.
#include "vislib/AbstractClusterNode.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This is the context structure that must be passed to the 
     * ReceiveMessages() runnable function. It defines the object for which
     * the receiver thread is working and the socket that it is using.
     */
    typedef struct ReceiveMessagesCtx_t {
        AbstractClusterNode *Receiver;
        Socket Socket;
    } ReceiveMessagesCtx;


    /**
     * This thread worker function receives messages for an AbstractClusterNode
     * and calls the onMessageReceived callback method on the owner every time
     * a message was received and recognised on the specified socket.
     *
     * @param receiveMessagesCtx A pointer to a ReceiveMessagesCtx structure.
     *
     * @return The socket error code that caused the function to end or -1 in 
     *         case of an unexpected error.
     */
    DWORD ReceiveMessages(void *receiveMessagesCtx); 

   
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_MESSAGERECEIVER_H_INCLUDED */

