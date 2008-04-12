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
     */
    class AbstractClientNode : public AbstractClusterNode {

    public:

        /** Dtor. */
        ~AbstractClientNode(void);

        virtual const SocketAddress& GetServerAddress(void) const = 0;

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

