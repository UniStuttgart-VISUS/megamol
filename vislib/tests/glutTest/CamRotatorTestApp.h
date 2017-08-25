/*
 * CamRotatorTestApp.h
 *
 * Copyright (C) 2006-2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_CAMROTATORTESTAPP_H_INCLUDED
#define VISLIBTEST_CAMROTATORTESTAPP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "AbstractGlutApp.h"
#include "vislib/graphics/Camera.h"
#include "vislib/graphics/CameraMove2D.h"
#include "vislib/graphics/gl/CameraOpenGL.h"
#include "vislib/graphics/CameraRotate2D.h"
#include "vislib/graphics/CameraRotate2DLookAt.h"
#include "vislib/graphics/Cursor2D.h"
#include "vislib/graphics/InputModifiers.h"
#include "vislib/types.h"
#include "CamParamObserver.h"
#include "vislib/graphics/gl/OpenGLVISLogo.h"



/*
 * Test application of camera rotations
 */
class CamRotatorTestApp: public AbstractGlutApp {
public:
    CamRotatorTestApp(void);
    virtual ~CamRotatorTestApp(void);

    virtual int GLInit(void);
    virtual void GLDeinit(void);

    virtual void OnResize(unsigned int w, unsigned int h);
    virtual void Render(void);
    virtual bool OnKeyPress(unsigned char key, int x, int y);
    virtual void OnMouseEvent(int button, int state, int x, int y);
    virtual void OnMouseMove(int x, int y);
    virtual void OnSpecialKey(int key, int x, int y);

private:
    void RenderLogo(void);

    vislib::graphics::gl::CameraOpenGL camera;
    vislib::graphics::InputModifiers modkeys;
    vislib::graphics::Cursor2D cursor;
    vislib::graphics::CameraRotate2D rotator1;
    vislib::graphics::CameraRotate2DLookAt rotator2;
    vislib::graphics::CameraMove2D mover;
    CamParamObserver testObserver;
    vislib::graphics::gl::OpenGLVISLogo logo;

    void SetupRotator1(void);
    void SetupRotator2(void);
};

#endif /* VISLIBTEST_CAMROTATORTESTAPP_H_INCLUDED */
