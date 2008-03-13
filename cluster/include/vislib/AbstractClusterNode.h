/*
 * AbstractNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCLUSTERNODE_H_INCLUDED
#define VISLIB_ABSTRACTCLUSTERNODE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SocketAddress.h"       // Must be first.
#include "vislib/Socket.h"              // Must be first.
#include "vislib/CmdLineProvider.h"
#include "vislib/types.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class defines the interface for all all specialised VISlib graphics 
     * cluster application nodes.
     */
    class AbstractClusterNode {

    public:

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
         * @throws
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
         * @throws
         */
        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine) = 0;

        /**
         * Run the node. Initialise must have been called before.
         *
         * @return The exit code that should be returned as exit code of the
         *         application.
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
         * @param peerSocket The socket representing the peer node.
         * @param context    A user-defined value passed to forEachPeer().
         *
         * @return The function returns true in order to indicate that the 
         *         enumeration should continue, or false in order to stop after 
         *         the current node.
         */
        typedef bool (* ForeachPeerFunc)(AbstractClusterNode *thisPtr, 
            Socket& peerSocket, void *context);


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
         * @param context This is an additional pointer that is passed 'func'.
         *
         * @return The number of sucessful calls to 'func' that have been made.
         */
        virtual SIZE_T forEachPeer(ForeachPeerFunc func, void *context) = 0;

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
        virtual SIZE_T sendToEachPeer(const BYTE *data, const SIZE_T cntData) 
            = 0;

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractClusterNode& operator =(const AbstractClusterNode& rhs);

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCLUSTERNODE_H_INCLUDED */

