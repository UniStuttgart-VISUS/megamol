/*
 * GlutServerNode.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLUTSERVERNODE_H_INCLUDED
#define VISLIB_GLUTSERVERNODE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#if defined(VISLIB_CLUSTER_WITH_OPENGL) && (VISLIB_CLUSTER_WITH_OPENGL != 0)

#include "vislib/AbstractControllerNode.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/GlutClusterNode.h"
#include "vislib/ServerNodeAdapter.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class provides all necessary parts to create a GLUT server node that
     * reports camera parameters to client nodes.
     *
     * Applications should inherit from this class and use the 'camera' member 
     * as their camera. They must only implement the drawing and interaction 
     * logic, the communication and resolution of ambiguities is done within 
     * this class.
     */
    template<class T> class GlutServerNode 
            : public GlutClusterNode<T>, public ServerNodeAdapter,
            public AbstractControllerNode {

    public:

        /** Dtor. */
        ~GlutServerNode(void);

        virtual void Initialise(sys::CmdLineProviderA& inOutCmdLine);

        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine);

        virtual DWORD Run(void);

    protected:

        /** Ctor. */
        GlutServerNode(void);

        /** 
         * The camera subclasses should use in order to synchronise its 
         * parameters to client nodes.
         */
        graphics::gl::CameraOpenGL camera;

    };


    /*
     * vislib::net::cluster::GlutServerNode<T>::~GlutServerNode
     */
    template<class T> GlutServerNode<T>::~GlutServerNode(void) {
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Initialise
     */
    template<class T> 
    void GlutServerNode<T>::Initialise(sys::CmdLineProviderA& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        ServerNodeAdapter::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Initialise
     */
    template<class T> 
    void GlutServerNode<T>::Initialise(sys::CmdLineProviderW& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        ServerNodeAdapter::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Run
     */
    template<class T> DWORD GlutServerNode<T>::Run(void) {
        ServerNodeAdapter::Run();           // First, start the server.
        return GlutClusterNode<T>::Run();   // Afterwards, enter message loop.
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::GlutServerNode
     */
    template<class T> GlutServerNode<T>::GlutServerNode(void) 
            : GlutClusterNode<T>(), ServerNodeAdapter(), 
            AbstractControllerNode(
            new vislib::graphics::ObservableCameraParams()) {
        this->camera.SetParameters(this->getParameters());
    }

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#endif /* defined(VISLIB_CLUSTER_WITH_OPENGL) ... */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLUTSERVERNODE_H_INCLUDED */
