/*
 * GlutClient.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_GLUTCLIENT_H_INCLUDED
#define VISLIBTEST_GLUTCLIENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/GlutClientNode.h"
#include "vislib/OpenGLVISLogo.h"


#if defined(VISLIB_CLUSTER_WITH_OPENGL) && (VISLIB_CLUSTER_WITH_OPENGL != 0)

class GlutClient : public vislib::net::cluster::GlutClientNode<GlutClient> {

public:

    virtual ~GlutClient(void);

protected:

    GlutClient(void);

    virtual void onFrameRender(void);

    virtual void onInitialise(void);

    vislib::graphics::gl::OpenGLVISLogo logo;

    friend class vislib::net::cluster::GlutClusterNode<GlutClient>;

};

#endif /*defined(VISLIB_CLUSTER_WITH_OPENGL) && ... */
#endif /* VISLIBTEST_GLUTCLIENT_H_INCLUDED */
