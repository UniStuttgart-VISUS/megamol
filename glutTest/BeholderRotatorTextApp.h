/*
 * CamTestApp.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_BEHOLDERROTATORTESTAPP_H_INCLUDED
#define VISLIBTEST_BEHOLDERROTATORTESTAPP_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "AbstractGlutApp.h"
#include "vislib/types.h"
#include "vislib/Beholder.h"
#include "vislib/Camera.h"
#define VISLIB_ENABLE_OPENGL
#include "vislib/CameraOpenGL.h"
#include "vislib/Cursor2D.h"
#include "vislib/BeholderRotator2D.h"
#include "vislib/BeholderLookAtRotator2D.h"


/*
 * Test for mono tiled display frustrum generation
 */
class BeholderRotatorTextApp: public AbstractGlutApp {
public:
    BeholderRotatorTextApp(void);
    virtual ~BeholderRotatorTextApp(void);

    virtual int GLInit(void);

    virtual void Resize(unsigned int w, unsigned int h);
    virtual void Render(void);
    virtual bool KeyPress(unsigned char key, int x, int y);
    virtual void MouseEvent(int button, int state, int x, int y);
    virtual void MouseMove(int x, int y);
    virtual void SpecialKey(int key, int x, int y);

private:
    void RenderLogo(void);

    vislib::graphics::Beholder beholder;
    vislib::graphics::gl::CameraOpenGL camera;
    vislib::graphics::Cursor2D cursor;
    vislib::graphics::BeholderRotator2D rotator1;
    vislib::graphics::BeholderLookAtRotator2D rotator2;

    void SetupRotator1(void);
    void SetupRotator2(void);
};

#endif /* VISLIBTEST_BEHOLDERROTATORTESTAPP_H_INCLUDED */
