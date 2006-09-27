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


/**
 * Abstract Interface class for glut based applications
 */
class AbstractGlutApp {
public:

    /**
     * ctor
     */
    AbstractGlutApp(void);

    /**
     * dtor
     */
    virtual ~AbstractGlutApp(void);

    /**
     * Initializes the application before the opengl state machine is initialized
     *
     * @return 0 if function succeeded.
     */
    virtual int PreGLInit(void) = 0;

    /**
     * Initializes the application after the opengl state machine is initialized
     *
     * @return 0 if function succeeded.
     */
    virtual int PostGLInit(void) = 0;

    /**
     * Callback for resizing events of the glut window.
     * Calculates the aspect ratio of the window.
     *
     * @param w New width of the window
     * @param h New Height of the window
     */
    virtual void Resize(unsigned int w, unsigned int h);

    /**
     * Callback for rendering events.
     */
    virtual void Render(void) = 0;

    /**
     * Returns the width of the glut window.
     *
     * @return The width of the glut window.
     */
    inline unsigned int GetWidth(void) { return this->width; }

    /*
     * Returns the height of the glut window.
     *
     * @return The height of the glut window.
     */
    inline unsigned int GetHeight(void) { return this->height; }

    /*
      * Returns the aspect ratio of the glut window.
     *
     * @return The aspect ratio of the glut window.
     */
    inline float GetAspectRatio(void) { return this->aspectRatio; }

    /**
     * Callback for key presses
     *
     * @param key Ascii code of pressed key
     *
     * @return true if the key was accepted in terms, that the pressing of the 
     *         key has an effect, false if the key was ignored.
     */
    virtual bool KeyPress(unsigned char key);

private:

    /* Widht of the glut window */
    unsigned int width;

    /* Height of the glut window */
    unsigned int height;

    /* aspectRatio of the glut window */
    float aspectRatio;

};

#endif /* VISLIBTEST_ABSTRACTGLUTAPP_H_INCLUDED */
