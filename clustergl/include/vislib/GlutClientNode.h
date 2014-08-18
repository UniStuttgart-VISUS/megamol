/*
 * GlutClientNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLUTCLIENTNODE_H_INCLUDED
#define VISLIB_GLUTCLIENTNODE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractClientNode.h"
#include "vislib/AbstractControlledNode.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/GlutClusterNode.h"


namespace vislib {
namespace net {
namespace cluster {


#ifdef _WIN32
#pragma warning(disable: 4250)  // I know what I am doing ...
#endif /* _WIN32 */

    /**
     * TODO: comment class
     */
    template<class T> class GlutClientNode
            : public GlutClusterNode<T>, public AbstractClientNode,
            public AbstractControlledNode {

    public:

        /** Dtor. */
        ~GlutClientNode(void);

        virtual void Initialise(sys::CmdLineProviderA& inOutCmdLine);

        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine);

        virtual DWORD Run(void);

    protected:

        /** Ctor. */
        GlutClientNode(void);

        /**
         * Answer the number of known peer nodes.
         *
         * @return The number of known peer nodes.
         *
         * @throws MissingImplementation If not overwritten by subclasses.
         *                               Calling this interface implementation
         *                               is a severe logic error.
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
         *
         * @throws MissingImplementation If not overwritten by subclasses.
         *                               Calling this interface implementation
         *                               is a severe logic error.
         */
        virtual SIZE_T forEachPeer(ForeachPeerFunc func, void *context);

        /**
         * Call 'func' for the peer node that has the specified ID 'peerId'. If
         * such a peer node is not known, nothing should be done.
         *
         * On server nodes, the function should check for a client with the
         * specified ID; on client nodes the implementation should check whether
         * 'peerId' references the server node.
         *
         * @param peerId  The identifier of the node to run 'func' for.
         * @param func    The function to be execured for the specified peer 
         *                node.
         * @param context This is an additional pointer that is passed to 
         *                'func'.
         *
         * @return true if 'func' was executed, false otherwise.
         */
        virtual bool forPeer(const PeerIdentifier& peerId, ForeachPeerFunc func,
            void *context);

        /**
         * This method is called when data have been received and a valid 
         * message has been found in the packet.
         *
         * @param src     The socket the message has been received from.
         * @param msgId   The message ID.
         * @param body    Pointer to the message body.
         * @param cntBody The number of bytes designated by 'body'.
         *
         * @return true in order to signal that the message has been processed, 
         *         false if the implementation did ignore it.
         */
        virtual bool onMessageReceived(const Socket& src, const UINT msgId,
            const BYTE *body, const SIZE_T cntBody);

        /**
         * The message receiver thread calls this method once it exists.
         *
         * @param socket The socket that was used from communication with the
         *               peer node.
         * @param rmc    The receive context that was used by the receiver 
         *               thread. The method takes ownership of the context and
         *               should release if not needed any more.
         */
        virtual void onMessageReceiverExiting(Socket& socket,
            PReceiveMessagesCtx rmc);

        /** 
         * The camera subclasses should use in order to synchronise its 
         * parameters to client nodes.
         */
        graphics::gl::CameraOpenGL camera;

    };
#ifdef _WIN32
#pragma warning(default: 4250)
#endif /* _WIN32 */


    /*
     * vislib::net::cluster::GlutClientNode<T>::~GlutClientNode
     */
    template<class T> GlutClientNode<T>::~GlutClientNode(void) {
    }


    /*
     *  vislib::net::cluster::GlutClientNode<T>::Initialise
     */
    template<class T> 
    void GlutClientNode<T>::Initialise(sys::CmdLineProviderA& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        AbstractClientNode::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutClientNode<T>::Initialise
     */
    template<class T> 
    void GlutClientNode<T>::Initialise(sys::CmdLineProviderW& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        AbstractClientNode::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutClientNode<T>::Run
     */
    template<class T> DWORD GlutClientNode<T>::Run(void) {
        AbstractClientNode::Run();          // Let the client node connect.
        return GlutClusterNode<T>::Run();   // Enter GLUT message loop.
    }


    /*
     * vislib::net::cluster::GlutClientNode<T>::GlutClientNode
     */
    template<class T> GlutClientNode<T>::GlutClientNode(void)
            : GlutClusterNode<T>(), AbstractClientNode(), 
            AbstractControlledNode() {
        // TODO: incomplete
        //this->camera.SetParameters(this->getParameters());
        this->setParameters(this->camera.Parameters());
    }


    /*
     * vislib::net::cluster::GlutClientNode<T>::countPeers
     */
    template<class T> SIZE_T GlutClientNode<T>::countPeers(void) const {
        // GCC cannot resolve this conflict via dominance, even if it is obvious
        // that we do not want to inherit the pure virtual implementation.
        return AbstractClientNode::countPeers();
    }


    /*
     * vislib::net::cluster::GlutClientNode<T>::forEachPeer
     */
    template<class T> 
    SIZE_T GlutClientNode<T>::forEachPeer(ForeachPeerFunc func, void *context) {
        // GCC cannot resolve this conflict via dominance, even if it is obvious
        // that we do not want to inherit the pure virtual implementation.
        return AbstractClientNode::forEachPeer(func, context);
    }


    /*
     * vislib::net::cluster::GlutClientNode<T>::forPeer
     */
    template<class T> 
    bool GlutClientNode<T>::forPeer(const PeerIdentifier& peerId, 
            ForeachPeerFunc func, void *context) {
        return AbstractClientNode::forPeer(peerId, func, context);
    }


    /*
     *  vislib::net::cluster::GlutClientNode<T>::onMessageReceived
     */
    template<class T> bool GlutClientNode<T>::onMessageReceived(
            const Socket& src, const UINT msgId, const BYTE *body, 
            const SIZE_T cntBody) {
        bool retval = AbstractControlledNode::onMessageReceived(src, msgId,
            body, cntBody);
        if (this->isWindowReady) {
            ::glutPostRedisplay();
        }
        return retval;
    }


    /*
     * GlutClientNode<T>::onMessageReceiverExiting
     */
    template<class T> 
    void GlutClientNode<T>::onMessageReceiverExiting(Socket& socket, 
            PReceiveMessagesCtx rmc) {
        AbstractClientNode::onMessageReceiverExiting(socket, rmc);
    }

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLUTCLIENTNODE_H_INCLUDED */
