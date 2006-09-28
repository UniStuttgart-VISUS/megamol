/*
 * CamTestApp.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_CAMTESTAPP_H_INCLUDED
#define VISLIBTEST_CAMTESTAPP_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "AbstractGlutApp.h"
#include "vislib/types.h"
#include "vislib/Beholder.h"
#include "vislib/Camera.h"
#define VISLIB_ENABLE_OPENGL
#include "vislib/CameraOpenGL.h"


/*
 * Test for mono tiled display frustrum generation
 */
class CamTestApp: public AbstractGlutApp {
public:
    CamTestApp(void);
    virtual ~CamTestApp(void);

    virtual int PreGLInit(void);
    virtual int PostGLInit(void);

    virtual void Resize(unsigned int w, unsigned int h);
    virtual void Render(void);
    virtual bool KeyPress(unsigned char key);

private:
    class Lens {
    public:
        Lens(void);
        ~Lens(void);

        void Update(float sec, vislib::graphics::CameraOpenGL camera);

        void BeginDraw(unsigned int ww, unsigned int wh, bool ortho);
        void EndDraw(void);

    private:
        float x, y, w, h, ax, ay;
        vislib::graphics::CameraOpenGL camera;

    };

    void RenderLogo(void);

    unsigned int lensCount;
    Lens *lenses;
    float angle;
    UINT64 lastTime;
    float walkSpeed, rotSpeed;
    bool ortho;
    bool nativeFull;

    vislib::graphics::Beholder<double> beholder;
    vislib::graphics::CameraOpenGL camera;
};

#endif /* VISLIBTEST_CAMTESTAPP_H_INCLUDED */
