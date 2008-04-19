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
#include "vislib/TcpServer.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class defines the interface of a server node.
     * 
     * Classes that want to implement a server node should inherit from 
     * ServerNodeAdapter, which implements all the required server 
     * functionality.
     *
     * This class uses virtual inheritance to implement the "delegate to 
     * sister" pattern.
     */
    class AbstractServerNode : public virtual AbstractClusterNode,
            public TcpServer::Listener {

    public:

        /** Dtor. */
        virtual ~AbstractServerNode(void);

        /**
         * Answer the socket address the server is binding to.
         *
         * @return The address the server is binding to.
         */
        virtual const SocketAddress& GetBindAddress(void) const = 0;

        /**
         * Set a new socket address the server should bind to. 
         * 
         * This has only an effect if the server is not yet running.
         *
         * @param bindAddress The address to bind to.
         */
        virtual void SetBindAddress(const SocketAddress& bindAddress) = 0;

        /**
         * Make the server bind to any adapter, but use the specified port.
         *
         * This has only an effect if the server is not yet running.
         *
         * @param port The port to bind to.
         */
        virtual void SetBindAddress(const unsigned short port) = 0;

    protected:

        /** Ctor. */
        AbstractServerNode(void);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
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

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSERVERNODE_H_INCLUDED */
