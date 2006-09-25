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

/*
 * Test for mono tiled display frustrum generation
 */
class CamTestApp: public AbstractGlutApp {
public:
    CamTestApp(void);
    virtual ~CamTestApp(void);

    virtual int PreGLInit(void);
    virtual int PostGLInit(void);

    virtual void Render(void);

private:
    class Lens {
    public:
        Lens(void);
        ~Lens(void);

        void Update(float sec);

        void BeginDraw(unsigned int ww, unsigned int wh);
        void EndDraw(void);

    private:
        float x, y, w, h, ax, ay;

    };

    void RenderLogo(void);

    unsigned int lensCount;
    Lens *lenses;
    float angle;
    UINT64 lastTime;
    float walkSpeed, rotSpeed;
};

#endif /* VISLIBTEST_CAMTESTAPP_H_INCLUDED */
