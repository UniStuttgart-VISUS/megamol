/*
 * CamTestApp.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_CAMTESTAPP_H_INCLUDED
#define VISLIBTEST_CAMTESTAPP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "AbstractGlutApp.h"
#include "vislib/types.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/OpenGLVISLogo.h"


/*
 * Test for mono tiled display frustrum generation
 */
class CamTestApp: public AbstractGlutApp {
public:
    CamTestApp(void);
    virtual ~CamTestApp(void);

    virtual int GLInit(void);
    virtual void GLDeinit(void);

    virtual void OnResize(unsigned int w, unsigned int h);
    virtual void Render(void);
    virtual bool OnKeyPress(unsigned char key, int x, int y);
    virtual void OnMouseEvent(int button, int state, int x, int y);
    virtual void OnMouseMove(int x, int y);
    virtual void OnSpecialKey(int key, int x, int y);

private:
    class Lens {
    public:
        Lens(void);
        ~Lens(void);

        void SetCameraParameters(
            const vislib::SmartPtr<vislib::graphics::CameraParameters>& params);

        void Update(float sec);

        void BeginDraw(unsigned int ww, unsigned int wh, bool ortho);
        void EndDraw(void);

    private:
        float x, y, w, h, ax, ay;
        vislib::graphics::gl::CameraOpenGL camera;

    };

    unsigned int lensCount;
    Lens *lenses;
    float angle;
    UINT64 lastTime;
    float walkSpeed, rotSpeed;
    bool ortho;
    bool nativeFull;

    vislib::graphics::gl::CameraOpenGL camera;
    vislib::graphics::gl::OpenGLVISLogo logo;
};

#endif /* VISLIBTEST_CAMTESTAPP_H_INCLUDED */
