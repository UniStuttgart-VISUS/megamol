/*
 * AbstractServerNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTSERVERNODE_H_INCLUDED
#define VISLIB_ABSTRACTSERVERNODE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractClusterNode.h"
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/RunnableThread.h"
#include "vislib/TcpServer.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This is a specialisation of the AbstractNode that runs a separate
     * server thread.
     */
    class AbstractServerNode : public AbstractClusterNode, 
            public TcpServer::Listener {

    public:

        /** Dtor. */
        ~AbstractServerNode(void);

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

        virtual bool OnNewConnection(Socket& socket, 
            const SocketAddress& addr) throw();

        virtual void OnServerStopped(void) throw();

    protected:

        /** Ctor. */
        AbstractServerNode(void);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        AbstractServerNode(const AbstractServerNode& rhs);

        /**
         * Call 'func' for each known client socket.
         *
         * @param func The function to be executed for each peer node.
         *
         * @return The number of calls to 'func' that have been made.
         */
        virtual SIZE_T forEachPeer(ForeachPeerFunc func, void *context = NULL);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractServerNode& operator =(const AbstractServerNode& rhs);

        /** The TCP server waiting for clients. */
        sys::RunnableThread<TcpServer> server;

        /** The client sockets. */
        Array<Socket> sockets;

        /** Lock for protecting the 'sockets' member. */
        sys::CriticalSection socketsLock;

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSERVERNODE_H_INCLUDED */

