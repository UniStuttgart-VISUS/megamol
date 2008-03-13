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


#include "AbstractClientNode.h"


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

