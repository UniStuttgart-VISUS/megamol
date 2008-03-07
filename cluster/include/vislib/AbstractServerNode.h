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
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractServerNode& operator =(const AbstractServerNode& rhs);

        /** The TCP server waiting for clients. */
        sys::RunnableThread<TcpServer> server;

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSERVERNODE_H_INCLUDED */

