/*
 * ServerNodeAdapter.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SERVERNODEADAPTER_H_INCLUDED
#define VISLIB_SERVERNODEADAPTER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractServerNode.h"
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/RunnableThread.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class implements the functionality of the graphics cluster server
     * node.
     *
     * Note that this class is not literally an adapter, because it does not 
     * implement all pure virtual methods of its parent classes. It therefore
     * cannot be instantiated.
     */
    class ServerNodeAdapter : public AbstractServerNode {

    public:

        /** Dtor. */
        ~ServerNodeAdapter(void);

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

        /** Superclass typedef. */
        typedef AbstractServerNode Super;

        /** Ctor. */
        ServerNodeAdapter(void);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        ServerNodeAdapter(const ServerNodeAdapter& rhs);

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
         */
        virtual SIZE_T forEachPeer(ForeachPeerFunc func, void *context);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        ServerNodeAdapter& operator =(const ServerNodeAdapter& rhs);

    private:

        /** The TCP server waiting for clients. */
        sys::RunnableThread<TcpServer> server;

        /** The client sockets. */
        Array<Socket> sockets;

        /** Lock for protecting the 'sockets' member. */
        mutable sys::CriticalSection socketsLock;

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SERVERNODEADAPTER_H_INCLUDED */

