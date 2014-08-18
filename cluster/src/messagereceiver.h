/*
 * messagereceiver.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MESSAGERECEIVER_H_INCLUDED
#define VISLIB_MESSAGERECEIVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "vislib/Socket.h"      // Must be first.
#include "vislib/AbstractClusterNode.h"
#include "vislib/Event.h"


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
        vislib::net::Socket *Socket;
    } ReceiveMessagesCtx;


    /**
     * Allocate a ReceiveMessagesCtx structure on the heap and initialse it
     * using the provided values.
     *
     * @param receiver The object that is to receive notifications.
     * @param socket   The socket to be used for communication.
     *
     * @return Pointer to a context structure on the heap that has been 
     *         initialsed with the provided values.
     *
     * @throws std::bad_alloc In case of insufficient heap memory.
     */
    ReceiveMessagesCtx *AllocateRecvMsgCtx(AbstractClusterNode *receiver,
        Socket *socket);


    /**
     * Release the memory 'ctx' and set the pointer NULL.
     *
     * @param ctx The context to release. This pointer should have been 
     *            allocated using AllocateRecvMsgCtx().
     */
    void FreeRecvMsgCtx(ReceiveMessagesCtx *& ctx);


    /**
     * This thread worker function receives messages for an AbstractClusterNode
     * and calls the onMessageReceived callback method on the owner every time
     * a message was received and recognised on the specified socket.
     *
     * The context passed to the function has the following lifetime (This is 
     * very similar to an IRP on Windows that can run in an arbitrary thread
     * context):
     * 1. The caller allocates it using AllocateRecvMsgCtx().
     * 2. The caller passes it to the thread that is running this function. The
     *    thread function then takes ownership of the memory.
     * 3. The thread calls the onMessageReceiverExiting() notification method of
     *    the receiver specified in the context structure just before it is
     *    exiting. The ownership of the context structure is passed to this 
     *    method.
     * 4. The onMessageReceiverExiting() releases the context stucture using
     *    FreeRecvMsgCtx() - that is what the default implementation in 
     *    AbstractClusterNode::onMessageReceiverExiting() does - or reuses it
     *    for another receiver thread.
     *
     * @param receiveMessagesCtx A pointer to a ReceiveMessagesCtx structure.
     *                           THIS STRUCTURE SHOULD HAVE BEEN ALLOCATED USING
     *                           AllocateRecvMsgCtx(). The function takes 
     *                           ownership of the memory.
     *
     * @return The socket error code that caused the function to end or -1 in 
     *         case of an unexpected error.
     */
    DWORD ReceiveMessages(void *receiveMessagesCtx); 

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_MESSAGERECEIVER_H_INCLUDED */
