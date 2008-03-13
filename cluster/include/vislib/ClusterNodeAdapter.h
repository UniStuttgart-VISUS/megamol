/*
 * ClusterNodeAdapter.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CLUSTERNODEADAPTER_H_INCLUDED
#define VISLIB_CLUSTERNODEADAPTER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "AbstractClusterNode.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class implements the common parts of AbstractClusterNode that are
     * the same by all implementations of cluster nodes. 
     *
     * Note that this class is not literally an adapter, because it does not 
     * implement all pure virtual methods of its parent classes. It therefore
     * cannot be instantiated.
     */
    class ClusterNodeAdapter : public AbstractClusterNode {

    public:

        /** Dtor. */
        virtual ~ClusterNodeAdapter(void);

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
        typedef AbstractClusterNode Super;

        /** Ctor. */
        ClusterNodeAdapter(void);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        ClusterNodeAdapter(const ClusterNodeAdapter& rhs);

        /**
         * Send 'cntData' bytes of data beginning at 'data' to each known peer
         * node.
         *
         * This is a blocking call, which returns after all messages have been
         * successfully delivered or the communication has eventually failed.
         *
         * @param data    Pointer to the data to be sent.
         * @param cntData The number of bytes to be sent.
         *
         * @return The number of messages successfully delivered.
         */
        virtual SIZE_T sendToEachPeer(const BYTE *data, const SIZE_T cntData);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        ClusterNodeAdapter& operator =(const ClusterNodeAdapter& rhs);

    private:

        /**
         * This is the context parameter for SendToPeerFunc().
         */
        typedef struct SendToPeerCtx_t {
            const BYTE *Data;
            SIZE_T CntData;
        } SendToPeerCtx;

        /**
         * This function is a callback that sends the data specified in
         * the 'context' structure, which must be of type SendToPeerCtx to
         * the specified peer node.
         *
         * The function blocks until all of the context->CntData bytes have
         * been sent.
         *
         * @param thisPtr    The pointer to the node object calling the 
         *                   callback function, which allows the callback to 
         *                   access instance members.
         * @param peerSocket The socket representing the peer node.
         * @param context    Pointer to a SendToPeerCtx passed to forEachPeer().
         *
         * @return The function returns true in order to indicate that the 
         *         enumeration should continue, or false in order to stop after 
         *         the current node.
         *
         * @throws SocketException In case that sending the data failed.
         */
        static bool SendToPeerFunc(AbstractClusterNode *thisPtr,
            Socket& peerSocket, void *context);
    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CLUSTERNODEADAPTER_H_INCLUDED */

