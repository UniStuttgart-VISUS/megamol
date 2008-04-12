/*
 * AbstractClientNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCLIENTNODE_H_INCLUDED
#define VISLIB_ABSTRACTCLIENTNODE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractClusterNode.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class defines additional functionality that a cluster client node
     * must provide.
     *
     * Classes that want to implement a client node should inherit from 
     * ClientNodeAdapter, which implements all the required functionality.
     *
     * AbstractClientNodes uses virtual inheritance to implement the "delegate
     * to sister" pattern.
     */
    class AbstractClientNode : public virtual AbstractClusterNode {

    public:

        /** Dtor. */
        ~AbstractClientNode(void);

        /**
         * Answer the address of the server to connect to.
         *
         * @return The address of the server to connect to.
         */
        virtual const SocketAddress& GetServerAddress(void) const = 0;

        /**
         * Set the address of the server to connect to. This must be done 
         * before the node connects to the server.
         *
         * @param serverAddress The new server added.
         */
        virtual void SetServerAddress(const SocketAddress& serverAddress) = 0;

    protected:

        /** Ctor. */
        AbstractClientNode(void);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        AbstractClientNode(const AbstractClientNode& rhs);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractClientNode& operator =(const AbstractClientNode& rhs);
    };

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCLIENTNODE_H_INCLUDED */
