/*
 * FBOTest2App.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_FBOTEST2APP_H_INCLUDED
#define VISLIBTEST_FBOTEST2APP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "AbstractGlutApp.h"
#include "vislib/FramebufferObject.h"
#include "vislib/CameraOpenGL.h"


/*
 * Test for frame buffer objects with common buffers
 */
class FBOTest2App: public AbstractGlutApp {
public:

    FBOTest2App(void);
    virtual ~FBOTest2App(void);
    virtual int GLInit(void);
    virtual void OnResize(unsigned int w, unsigned int h);
    virtual void GLDeinit(void);
    virtual void Render(void);

private:

    void RenderTestBox(float s);

    vislib::graphics::gl::CameraOpenGL camera;
    vislib::graphics::gl::CameraOpenGL camera2;

    vislib::graphics::gl::FramebufferObject fboAll;
    vislib::graphics::gl::FramebufferObject fboRed;
    vislib::graphics::gl::FramebufferObject fboGreen;

};

#endif /* VISLIBTEST_FBOTEST2APP_H_INCLUDED */
