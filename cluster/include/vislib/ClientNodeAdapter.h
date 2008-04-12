/*
 * ClientNodeAdapter.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CLIENTNODEADAPTER_H_INCLUDED
#define VISLIB_CLIENTNODEADAPTER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractClientNode.h"
#include "vislib/Thread.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class implements the client node communication functionality.
     */
    class ClientNodeAdapter : public AbstractClientNode {

    public:

        /** Dtor. */
        ~ClientNodeAdapter(void);

        virtual const SocketAddress& GetServerAddress(void) const;

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
        virtual void Initialise(sys::CmdLineProviderA& inOutCmdLine);

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
        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine);

        /**
         * Connect to the server address specified before and starts the message
         * receiver thread. Afterwards, the method returns.
         *
         * You should call this Run() method first in subclasses as your own 
         * operations will probably not return immediately.
         *
         * @return 0 in case of success.
         *
         * @throws IllegalStateException If the client node is already 
         *                               connected.
         * @throws SocketException If it was not possible to connect to the 
         *                         server.
         * @throws SystemException If the message receiver thread could not be
         *                         started.
         */
        virtual DWORD Run(void);

        virtual void SetServerAddress(const SocketAddress& serverAddress);

    protected:

        /** Superclass typedef. */
        typedef AbstractClientNode Super;

        /** Ctor. */
        ClientNodeAdapter(void);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        ClientNodeAdapter(const ClientNodeAdapter& rhs);

        /**
         * Answer the number of known peer nodes.
         *
         * @return The number of known peer nodes.
         */
        virtual SIZE_T countPeers(void) const;

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
         *         This is at most 1 for a client node.
         */
        virtual SIZE_T forEachPeer(ForeachPeerFunc func, void *context);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        ClientNodeAdapter& operator =(const ClientNodeAdapter& rhs);

    private:

        /** The receiver thread that generates the message events. */
        sys::Thread *receiver;

        /** The address of the server node to connect to. */
        SocketAddress serverAddress;

        /** The socket for communicating with the server. */
        Socket socket;

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CLIENTNODEADAPTER_H_INCLUDED */

