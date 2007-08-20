/*
 * GLSLGeomShaderTest.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_GLSLGEOMSHADERTEST_H_INCLUDED
#define VISLIBTEST_GLSLGEOMSHADERTEST_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "AbstractGlutApp.h"
#include "vislib/types.h"
#include "vislib/Beholder.h"
#include "vislib/Camera.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/InputModifiers.h"
#include "vislib/Cursor2D.h"
#include "vislib/BeholderRotator2D.h"
#include "vislib/BeholderLookAtRotator2D.h"
#include "vislib/GLSLGeometryShader.h"


/*
 * Test for mono tiled display frustrum generation
 */
class GLSLGeomShaderTest: public AbstractGlutApp {
public:
    GLSLGeomShaderTest(void);
    virtual ~GLSLGeomShaderTest(void);

    virtual int GLInit(void);
    virtual void GLDeinit(void);

    virtual void OnResize(unsigned int w, unsigned int h);
    virtual void Render(void);
    virtual bool OnKeyPress(unsigned char key, int x, int y);
    virtual void OnMouseEvent(int button, int state, int x, int y);
    virtual void OnMouseMove(int x, int y);
    virtual void OnSpecialKey(int key, int x, int y);

private:

    vislib::graphics::Beholder beholder;
    vislib::graphics::gl::CameraOpenGL camera;
    vislib::graphics::InputModifiers modkeys;
    vislib::graphics::Cursor2D cursor;
    vislib::graphics::BeholderLookAtRotator2D rotator2;

    vislib::graphics::gl::GLSLGeometryShader schade;
};

#endif /* VISLIBTEST_GLSLGEOMSHADERTEST_H_INCLUDED */
