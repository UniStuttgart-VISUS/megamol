/*
 * AbstractClientEndpoint.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCLIENTENDPOINT_H_INCLUDED
#define VISLIB_ABSTRACTCLIENTENDPOINT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/ReferenceCounted.h"


namespace vislib {
namespace net {


    /**
     * This class defines the interface of a communication channel end point 
     * that acts as a client, i. e. connects to a server node.
     *
     * Rationale: The address is a string that must be parsed by implementing
     * classes. This choice takes into account that strings are the most 
     * flexible way of specifying an address, which allows the interface to be
     * the same for different network families. It is, however, encouraged that
     * subclasses provide additional methods using their native address format.
     */
    class AbstractClientEndpoint : public virtual ReferenceCounted {

    public:

        /**
         * Connects the end point to the peer node at the specified address.
         *
         * Note: The default implementation redirects the method to the UNICODE
         * version of the method.
         *
         * @param address The address to connect to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Connect(const char *address);

        /**
         * Connects the end point to the peer node at the specified address.
         *
         * @param address The address to connect to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Connect(const wchar_t *address) = 0;

    protected:

        /** Ctor. */
        AbstractClientEndpoint(void);

        /** Dtor. */
        virtual ~AbstractClientEndpoint(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCLIENTENDPOINT_H_INCLUDED */

