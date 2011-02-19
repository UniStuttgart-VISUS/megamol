/*
 * FBOTestApp.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_FBOTESTAPP_H_INCLUDED
#define VISLIBTEST_FBOTESTAPP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "AbstractGlutApp.h"
#include "vislib/FramebufferObject.h"
#include "vislib/OpenGLVISLogo.h"


/*
 * Test for mono tiled display frustrum generation
 */
class FBOTestApp: public AbstractGlutApp {
public:

    FBOTestApp(void);
    virtual ~FBOTestApp(void);
    virtual int GLInit(void);
    virtual void GLDeinit(void);
    virtual void Render(void);

private:

    vislib::graphics::gl::FramebufferObject fbo;
    vislib::graphics::gl::OpenGLVISLogo logo;
};

#endif /* VISLIBTEST_FBOTESTAPP_H_INCLUDED */
