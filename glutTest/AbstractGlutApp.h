/*
 * AbstractGlutApp.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_ABSTRACTGLUTAPP_H_INCLUDED
#define VISLIBTEST_ABSTRACTGLUTAPP_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


/*
 * Abstract Interface class for glut based applications
 */
class AbstractGlutApp {
public:

    /*
     * ctor
     */
    AbstractGlutApp(void);

    /*
     * dtor
     */
    virtual ~AbstractGlutApp(void);

    /*
     * Initializes the application before the opengl state machine is initialized
     *
     * @return 0 if function succeeded.
     */
    virtual int PreGLInit(void) = 0;

    /*
     * Initializes the application after the opengl state machine is initialized
     *
     * @return 0 if function succeeded.
     */
    virtual int PostGLInit(void) = 0;

    /*
     * TODO: comment
     */
    virtual void Resize(unsigned int w, unsigned int h);

    /*
     * TODO: comment
     */
    virtual void Render(void) = 0;

    /*
     * TODO: comment
     */
    inline unsigned int GetWidth(void) { return this->width; }

    /*
     * TODO: comment
     */
    inline unsigned int GetHeight(void) { return this->height; }

    /*
     * TODO: comment
     */
    inline float GetAspectRatio(void) { return this->aspectRatio; }

private:

    /* Widht of the glut window */
    unsigned int width;

    /* Height of the glut window */
    unsigned int height;

    /* aspectRatio of the glut window */
    float aspectRatio;

};

#endif /* VISLIBTEST_ABSTRACTGLUTAPP_H_INCLUDED */
