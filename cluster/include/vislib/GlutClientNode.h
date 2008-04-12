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

#include "vislib/AbstractControlledNode.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/ClientNodeAdapter.h"
#include "vislib/GlutClusterNode.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * TODO: comment class
     */
    template<class T> class GlutClientNode
            : public GlutClusterNode<T>, public ClientNodeAdapter,
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
         * The camera subclasses should use in order to synchronise its 
         * parameters to client nodes.
         */
        graphics::gl::CameraOpenGL camera;

    };


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
        ClientNodeAdapter::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutClientNode<T>::Initialise
     */
    template<class T> 
    void GlutClientNode<T>::Initialise(sys::CmdLineProviderW& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        ClientNodeAdapter::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutClientNode<T>::Run
     */
    template<class T> DWORD GlutClientNode<T>::Run(void) {
        ClientNodeAdapter::Run();           // Let the client node connect.
        return GlutClusterNode<T>::Run();   // Enter GLUT message loop.
    }


    /*
     * vislib::net::cluster::GlutClientNode<T>::GlutClientNode
     */
    template<class T> GlutClientNode<T>::GlutClientNode(void) 
            : GlutClusterNode<T>(), ClientNodeAdapter(), AbstractControlledNode() {
        // TODO: incomplete
        //this->camera.SetParameters(this->getParameters());
    }

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#endif /* defined(VISLIB_CLUSTER_WITH_OPENGL) ... */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLUTCLIENTNODE_H_INCLUDED */
