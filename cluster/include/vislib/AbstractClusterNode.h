/*
 * AbstractNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCLUSTERNODE_H_INCLUDED
#define VISLIB_ABSTRACTCLUSTERNODE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IPEndPoint.h"          // Must be first.
#include "vislib/CmdLineProvider.h"
#include "vislib/Socket.h"
#include "vislib/SocketException.h"
#include "vislib/types.h"


namespace vislib {
namespace net {
namespace cluster {

    /** Forward declaration of an opaque, internal context structure. */
    typedef struct ReceiveMessagesCtx_t *PReceiveMessagesCtx;


    /**
     * This class defines the interface for all all specialised VISlib graphics
     * cluster application nodes.
     *
     * The class is required to provide a constistent networking interface to all
     * implementing classes, both server and client node implementations.
     *
     * The direct children server and client adapter classes are used to 
     * implement the "delegate to sister" pattern, i.e. they do the communication
     * work for all classes that inherit from AbstractClusterNode in order to use
     * the communication function (e.g. AbstractControllerNode uses the sending
     * methods, which are implemented by the server node implementation). Therefore,
     * all direct children of AbstractClusterNode must use virtual inheritance.
     */
    class AbstractClusterNode {

    public:

        /** 
         * This type is used as a unique identified for peer nodes. This 
         * identifier is the address of the peer node.
         */
        typedef IPEndPoint PeerIdentifier;

        /** 
         * The default port used by the default communication implementation, if
         * no other value is provided either via the command line or via the 
         * API.
         */
        static const SHORT DEFAULT_PORT;

        /** Dtor. */
        virtual ~AbstractClusterNode(void);

        /**
         * Initialise the node.
         *
         * Implementing subclasses should build an initialisation chain by 
         * calling their parent class implementation first.
         *
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws MissingImplementation If not overwritten by subclasses.
         *                               Calling this interface implementation
         *                               is a severe logic error.
         */
        virtual void Initialise(sys::CmdLineProviderA& inOutCmdLine) = 0;

        /**
         * Initialise the node.
         *
         * Implementing subclasses should build an initialisation chain by 
         * calling their parent class implementation first.
         *
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws MissingImplementation If not overwritten by subclasses.
         *                               Calling this interface implementation
         *                               is a severe logic error.
         */
        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine) = 0;

        /**
         * Run the node. Initialise must have been called before.
         *
         * @return The exit code that should be returned as exit code of the
         *         application.
         *
         * @throws MissingImplementation If not overwritten by subclasses.
         *                               Calling this interface implementation
         *                               is a severe logic error.
         */
        virtual DWORD Run(void) = 0;

    protected:

        /**
         * This function pointer type is used as a callback for the forEachPeer
         * method. Such a function is executed for each peer node that the
         * client or server node knowns.
         *
         * Functions that are used here may throw any exception. This exception
         * must be caught by the enumerator method. The enumerator method 
         * continues with the next peer node after an exception.
         *
         * @param thisPtr    The pointer to the node object calling the 
         *                   callback function, which allows the callback to 
         *                   access instance members.
         * @param peerId     The unique identifier of the peer node.
         * @param peerSocket The socket representing the peer node.
         * @param context    A user-defined value passed to forEachPeer().
         *
         * @return The function returns true in order to indicate that the 
         *         enumeration should continue, or false in order to stop after 
         *         the current node.
         */
        typedef bool (* ForeachPeerFunc)(AbstractClusterNode *thisPtr,
            const PeerIdentifier& peerId, Socket& peerSocket, void *context);

        /**
         * This enumeration defines possible sources of communication errors.
         *
         * RECEIVE_COMMUNICATION_ERROR indicates that a communication error 
         * occurred while receiving data from a peer node.
         *
         * SEND_COMMUNICATION_ERROR indicates that a communication error 
         * occurred while sending data to a peer node.
         */
        typedef enum {
            RECEIVE_COMMUNICATION_ERROR = 1,
            SEND_COMMUNICATION_ERROR
        } ComErrorSource;

        /** Ctor. */
        AbstractClusterNode(void);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        AbstractClusterNode(const AbstractClusterNode& rhs);

        /**
         * Answer the number of known peer nodes.
         *
         * @return The number of known peer nodes.
         */
        virtual SIZE_T countPeers(void) const = 0;

        /**
         * Call 'func' for each known peer node (socket).
         *
         * On server nodes, the function is usually called for all the client
         * nodes, on client nodes only once (for the server). However, 
         * implementations in subclasses may differ.
         *
         * @param func    The function to be executed for each peer node.
         * @param context This is an additional pointer that is passed to 
         *                'func'.
         *
         * @return The number of sucessful calls to 'func' that have been made.
         */
        virtual SIZE_T forEachPeer(ForeachPeerFunc func, void *context) = 0;

        /**
         * Call 'func' for the peer node that has the specified ID 'peerId'. If
         * such a peer node is not known, nothing should be done.
         *
         * On server nodes, the function should check for a client with the
         * specified ID; on client nodes the implementation should check whether
         * 'peerId' references the server node.
         *
         * @param peerId  The identifier of the node to run 'func' for.
         * @param func    The function to be execured for the specified peer 
         *                node.
         * @param context This is an additional pointer that is passed to 
         *                'func'.
         *
         * @return true if 'func' was executed, false otherwise.
         */
        virtual bool forPeer(const PeerIdentifier& peerId, ForeachPeerFunc func,
            void *context) = 0;

        /**
         * This method is called when a communication error occurs.
         *
         * This method and all overridded implementations must not throw any
         * exception.
         *
         * @param peerId The node that caused the communication error.
         * @param src    The type of communication that caused the error.
         * @param err    The exception that identifies the error.
         */
        virtual void onCommunicationError(const PeerIdentifier& peerId,
            const ComErrorSource src, const SocketException& err) throw();

        /**
         * This method is called when data have been received and a valid 
         * message has been found in the packet.
         *
         * @param src     The socket the message has been received from.
         * @param msgId   The message ID.
         * @param body    Pointer to the message body.
         * @param cntBody The number of bytes designated by 'body'.
         *
         * @return true in order to signal that the message has been processed,
         *         false if the implementation did ignore it.
         */
        virtual bool onMessageReceived(const Socket& src, const UINT msgId,
            const BYTE *body, const SIZE_T cntBody) = 0;

        /**
         * The message receiver thread calls this method once it exists.
         *
         * The default implementation here releases 'recvCtx' using 
         * FreeRecvMsgCtx(). You should call this implementation if you want to 
         * do this, or provide your own implementation if you want to reuse 
         * the context.
         *
         * @param socket The socket that was used from communication with the
         *               peer node.
         * @param rmc    The receive context that was used by the receiver 
         *               thread. The method takes ownership of the context and
         *               should release if not needed any more.
         */
        virtual void onMessageReceiverExiting(Socket& socket,
            PReceiveMessagesCtx rmc);

        /**
         * This method is called once a connection with a peer node was 
         * established.
         *
         * Subclasses must call this method once a new connection was 
         * established and all resources for the connection have been allocated,
         * i. e. the connection is ready to be used if the callback method is
         * executed.
         *
         * This method and all overriding implementations must not throw
         * an exception!
         *
         * @param peerId The identifier of the new peer node.
         */
        virtual void onPeerConnected(const PeerIdentifier& peerId) throw();

        /**
         * Send a message to all known peer nodes. 
         *
         * This method is similar to sendToEachPeer(), but also performs 
         * packaging of messages. The caller must only provide the message ID,
         * message body and size of the body.
         *
         * @param msgId   The message ID.
         * @param data    The message body data.
         * @param cntData The size of 'data'.
         *
         * @return The number of messages successfully delivered.
         */
        virtual SIZE_T sendMessage(const UINT32 msgId, const BYTE *data, 
            const UINT32 cntData);

        /**
         * Send a message to a specific peer node.
         *
         * This method is similar to sendToEachPeer(), but also performs 
         * packaging of messages. The caller must only provide the message ID,
         * message body and size of the body.
         *
         * @param msgId   The message ID.
         * @param data    The message body data.
         * @param cntData The size of 'data'.
         *
         * @return true if the message was delivered, false if the peer node was
         *         not found or a communication error occurred.
         */
        virtual bool sendMessage(const PeerIdentifier& peerId, 
            const UINT32 msgId, const BYTE *data, const UINT32 cntData);


        /**
         * Send 'cntData' bytes of data beginning at 'data' to each known peer
         * node.
         *
         * This is a blocking call, which returns after all messages have been
         * successfully delivered or the communication has eventually failed.
         *
         * @param data    Pointer to the data to be sent.
         * @param cntData The number of bytes to be sent.
         *
         * @return The number of messages successfully delivered.
         */
        virtual SIZE_T sendToEachPeer(const BYTE *data, const SIZE_T cntData);

        /**
         * Send 'cntData' bytes of data beginning at 'data' to the peer node
         * identified by 'peerId'. If such a node is not known, this is an 
         * error.
         *
         * The method makes best efforts to send the complete data packet and
         * blocks until everything has been sent.
         *
         * @param peerId
         * @param data
         * @param cntData
         *
         * @return true if the message was delivered, false if the peer node was
         *         not found or a communication error occurred.
         */
        virtual bool sendToPeer(const PeerIdentifier& peerId,
            const BYTE *data, const SIZE_T cntData);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractClusterNode& operator =(const AbstractClusterNode& rhs);

    private:

        /**
         * This is the context parameter for SendToPeerFunc().
         */
        typedef struct SendToPeerCtx_t {
            const BYTE *Data;
            SIZE_T CntData;
        } SendToPeerCtx;

        /**
         * This function is a callback that sends the data specified in
         * the 'context' structure, which must be of type SendToPeerCtx to
         * the specified peer node.
         *
         * The function blocks until all of the context->CntData bytes have
         * been sent.
         *
         * The function calls the onCommunicationError() callback method on
         * 'thisPtr' before throwing SocketExceptions.
         *
         * @param thisPtr    The pointer to the node object calling the 
         *                   callback function, which allows the callback to 
         *                   access instance members.
         * @param peerId     The unique identifier of the peer node.
         * @param peerSocket The socket representing the peer node.
         * @param context    Pointer to a SendToPeerCtx passed to forEachPeer().
         *
         * @return The function returns true in order to indicate that the 
         *         enumeration should continue, or false in order to stop after 
         *         the current node.
         *
         * @throws SocketException In case that sending the data failed.
         */
        static bool sendToPeerFunc(AbstractClusterNode *thisPtr,
            const PeerIdentifier& peerId, Socket& peerSocket, void *context);

        /* Grant access to message handler methods. */
        friend DWORD ReceiveMessages(void *receiveMessagesCtx);

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCLUSTERNODE_H_INCLUDED */
