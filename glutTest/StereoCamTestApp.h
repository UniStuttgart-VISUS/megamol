/*
 * StereoCamTestApp.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_STEREOCAMTESTAPP_H_INCLUDED
#define VISLIBTEST_STEREOCAMTESTAPP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractGlutApp.h"

#include "vislib/CameraOpenGL.h"
#include "vislib/CameraParameters.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/OpenGLVISLogo.h"


#define TILE_RIGHT_EYE 3


class StereoCamTestApp: public AbstractGlutApp {
public:
    StereoCamTestApp(void);
    virtual ~StereoCamTestApp(void);

    virtual int GLInit(void);
    virtual void GLDeinit(void);

    virtual void OnResize(unsigned int w, unsigned int h);
    virtual void Render(void);
    virtual bool OnKeyPress(unsigned char key, int x, int y);

private:
    void RenderTestBox(void);

    float angle;
    UINT64 lastTime;

    vislib::SmartPtr<vislib::graphics::CameraParameters> parameters;
    vislib::graphics::gl::CameraOpenGL cameraLeft;
#ifdef TILE_RIGHT_EYE
    vislib::graphics::gl::CameraOpenGL cameraRight[TILE_RIGHT_EYE * TILE_RIGHT_EYE];
#else /* TILE_RIGHT_EYE */
    vislib::graphics::gl::CameraOpenGL cameraRight;
#endif /* TILE_RIGHT_EYE */
    vislib::graphics::gl::OpenGLVISLogo logo;
};

#endif /* VISLIBTEST_STEREOCAMTESTAPP_H_INCLUDED */
