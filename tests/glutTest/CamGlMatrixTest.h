/*
 * CamGlMatrixTest.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_CAMGLMATRIXTEST_H_INCLUDED
#define VISLIBTEST_CAMGLMATRIXTEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "vislib/CameraOpenGL.h"
#include "vislib/MouseInteractionAdapter.h"
#include "vislib/OpenGLVISLogo.h"

#include "AbstractGlutApp.h"


/**
 * Test for manual matrix generation of the OpenGL camera.
 */
class CamGlMatrixTest: public AbstractGlutApp {

public:

    CamGlMatrixTest(void);
    virtual ~CamGlMatrixTest(void);

    virtual int GLInit(void);
    virtual void GLDeinit(void);

    virtual void OnResize(unsigned int w, unsigned int h);
    virtual void Render(void);
    virtual bool OnKeyPress(unsigned char key, int x, int y);
    virtual void OnMouseEvent(int button, int state, int x, int y);
    virtual void OnMouseMove(int x, int y);
    virtual void OnSpecialKey(int key, int x, int y);

private:

    vislib::graphics::gl::CameraOpenGL camera;

    bool isManual;

    vislib::graphics::MouseInteractionAdapter *mia;

    vislib::graphics::gl::OpenGLVISLogo vislogo;

};
#endif /* VISLIBTEST_CAMGLMATRIXTEST_H_INCLUDED */
