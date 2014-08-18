/*
 * CamEulerRotatorTestApp.h
 *
 * Copyright (C) 2006-2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_CAMEULERROTATORTESTAPP_H_INCLUDED
#define VISLIBTEST_CAMEULERROTATORTESTAPP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "AbstractGlutApp.h"
#include "vislib/Camera.h"
#include "vislib/CameraMove2D.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/CameraRotate2DEulerLookAt.h"
#include "vislib/Cursor2D.h"
#include "vislib/InputModifiers.h"
#include "vislib/types.h"
#include "CamParamObserver.h"
#include "vislib/OpenGLVISLogo.h"



/*
 * Test application of camera rotations
 */
class CamEulerRotatorTestApp: public AbstractGlutApp {
public:
    CamEulerRotatorTestApp(void);
    virtual ~CamEulerRotatorTestApp(void);

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
    vislib::graphics::CameraRotate2DEulerLookAt rotator;
    vislib::graphics::CameraMove2D mover;
    vislib::graphics::gl::OpenGLVISLogo logo;

};

#endif /* VISLIBTEST_CAMEULERROTATORTESTAPP_H_INCLUDED */
