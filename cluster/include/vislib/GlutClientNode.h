/*
 * GlutClientNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLUTCLIENTNODE_H_INCLUDED
#define VISLIB_GLUTCLIENTNODE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#if defined(VISLIB_CLUSTER_WITH_OPENGL) && (VISLIB_CLUSTER_WITH_OPENGL != 0)

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
         * This method is called when data have been received and a valid 
         * message has been found in the packet.
         *
         * @param src     The socket the message has been received from.
         * @param msgId   The message ID.
         * @param body    Pointer to the message body.
         * @param cntBody The number of bytes designated by 'body'.
         */
        virtual void onMessageReceived(const Socket& src, const UINT msgId,
            const BYTE *body, const SIZE_T cntBody);

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
     *  vislib::net::cluster::GlutClientNode<T>::onMessageReceived
     */
    template<class T> void GlutClientNode<T>::onMessageReceived(
            const Socket& src, const UINT msgId, const BYTE *body, 
            const SIZE_T cntBody) {
        AbstractControlledNode::onMessageReceived(src, msgId, body, cntBody);
        ::glutPostRedisplay();
    }

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#endif /* defined(VISLIB_CLUSTER_WITH_OPENGL) ... */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLUTCLIENTNODE_H_INCLUDED */
