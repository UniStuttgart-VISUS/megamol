/*
 * GlutServer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_GLUTSERVER_H_INCLUDED
#define VISLIBTEST_GLUTSERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/GlutServerNode.h"
#include "vislib/OpenGLVISLogo.h"


#if defined(VISLIB_CLUSTER_WITH_OPENGL) && (VISLIB_CLUSTER_WITH_OPENGL != 0)

class GlutServer : public vislib::net::cluster::GlutServerNode<GlutServer> {

public:

    virtual ~GlutServer(void);

protected:

    GlutServer(void);

    virtual void initialiseController(
        vislib::graphics::AbstractCameraController *& inOutRotateController,
        vislib::graphics::AbstractCameraController *& inOutZoomController);

    virtual void onFrameRender(void);

    virtual void onInitialise(void);

    vislib::graphics::gl::OpenGLVISLogo logo;

    friend class vislib::net::cluster::GlutClusterNode<GlutServer>;

};

#endif /*defined(VISLIB_CLUSTER_WITH_OPENGL) && ... */
#endif /* VISLIBTEST_GLUTSERVER_H_INCLUDED */
