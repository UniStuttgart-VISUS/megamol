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
     * This is the superclass for all specialised VISlib graphics cluster
     * application nodes.
     */
    class AbstractClusterNode {

    public:

        /** Dtor. */
        ~AbstractClusterNode(void);

        /**
         * Initialise the node.
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
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws
         */
        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine);

        /**
         * Run the node. Initialise must have been called before.
         *
         * @return The exit code that should be returned as exit code of the
         *         application.
         */
        virtual DWORD Run(void) = 0;

    protected:


        /** Ctor. */
        AbstractClusterNode(void);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        AbstractClusterNode(const AbstractClusterNode& rhs);

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

