/*
 * ClusterSignRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES 1
#include "ClusterSignRenderer.h"
#include "vislib/mathfunctions.h"
#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include <GL/gl.h>
//#include <GL/glu.h>
#ifndef _USE_MATH_DEFINES
#error _USE_MATH_DEFINES had been undefined
#endif
#include <cmath>
#include "vislib/sysfunctions.h"

using namespace megamol::core;

#define USE_SMOOTH_LINES 1


/*
 * special::ClusterSignRenderer::RenderBroken
 */
void special::ClusterSignRenderer::RenderBroken(int width, int height, bool stereo, bool rightEye) {
    setupMatrices(width, height, stereo, rightEye);
    ::glScaled(0.9, 0.9, 1.0);
    setupScene();
    if (tick()) {
        renderBorder(0x000000C0, 0x00000000, 0x00000000);
    }
    renderCross(0x000000C0);
    cleanupScene();
}


/*
 * special::ClusterSignRenderer::RenderNo
 */
void special::ClusterSignRenderer::RenderNo(int width, int height, bool stereo, bool rightEye) {
    setupMatrices(width, height, stereo, rightEye);
    ::glScaled(0.8, 0.8, 1.0);
    setupScene();
    if (tick()) {
        renderBorder(0x000000FF, 0x00000000, 0x000000FF);
        renderCross(0x00000000);
    } else {
        renderBorder(0x000000FF, 0x00000000, 0x00000000);
        renderCross(0x000000FF);
    }
    cleanupScene();
}


/*
 * special::ClusterSignRenderer::RenderYes
 */
void special::ClusterSignRenderer::RenderYes(int width, int height, bool stereo, bool rightEye) {
    setupMatrices(width, height, stereo, rightEye);
    ::glScaled(0.8, 0.8, 1.0);
    setupScene();
    if (tick()) {
        renderBorder(0x0000FF00, 0x00000000, 0x0000FF00);
        renderCheck(0x00000000);
    } else {
        renderBorder(0x0000FF00, 0x00000000, 0x00000000);
        renderCheck(0x0000FF00);
    }
    cleanupScene();
}


/*
 * special::ClusterSignRenderer::setupMatrices
 */
void special::ClusterSignRenderer::setupMatrices(int width, int height, bool stereo, bool rightEye) {
    double w, h;

    if (width < 1) { width = 1; }
    if (height < 1) { height = 1; }

    if (width > (height * (stereo ? 2 : 1))) {
        w = double(width) / double(height * (stereo ? 2 : 1));
        h = stereo ? 0.5 : 1.0;
    } else {
        w = 1.0;
        h = double(height) / double(width);
    }

    ::glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    ::glScaled(1.0 / w, 1.0 / h, 1.0);

    if (stereo) {
        if (rightEye) {
            ::glTranslated(0.5, 0.0, 0.0);
        } else {
            ::glTranslated(-0.5, 0.0, 0.0);
        }
        ::glScaled(0.5, 0.5, 1.0);
    }
}


/*
 * special::ClusterSignRenderer::setupScene
 */
void special::ClusterSignRenderer::setupScene(void) {
#ifdef USE_SMOOTH_LINES
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.5f);
#endif /* USE_SMOOTH_LINES */
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glDisable(GL_TEXTURE_2D);
    ::glDisable(GL_LIGHTING);
}


/*
 * special::ClusterSignRenderer::cleanupScene
 */
void special::ClusterSignRenderer::cleanupScene(void) {
#ifdef USE_SMOOTH_LINES
    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
#endif /* USE_SMOOTH_LINES */
}


/*
 * special::ClusterSignRenderer::renderBorder
 */
void special::ClusterSignRenderer::renderBorder(unsigned int c1,
        unsigned int c2, unsigned int c3) {

    const double rad1 = 0.3;
    const double rad0 = 1.0 - rad1;
    const unsigned int curveStep = 25;
#ifdef USE_SMOOTH_LINES
    for (int metapass = 0; metapass < 6; metapass++) {
        int pass = metapass / 2;
#else /* USE_SMOOTH_LINES */
    for (int pass = 0; pass < 3; pass++) {
#endif /* USE_SMOOTH_LINES */
        double rad2 = rad1 * (1.0 - double(pass) / 4.0);
        switch (pass) {
            case 0:
                ::glColor3ubv(reinterpret_cast<const GLubyte*>(&c1));
                break;
            case 1:
                ::glColor3ubv(reinterpret_cast<const GLubyte*>(&c2));
                break;
            case 2:
                ::glColor3ubv(reinterpret_cast<const GLubyte*>(&c3));
                break;
        }
#ifdef USE_SMOOTH_LINES
        ::glBegin(((metapass % 2) == 0) ? GL_TRIANGLE_FAN : GL_LINE_LOOP);
#else /* USE_SMOOTH_LINES */
        ::glBegin(GL_TRIANGLE_FAN);
#endif /* USE_SMOOTH_LINES */
        for (int vert = 0; vert < 4; vert++) {
            const double vnx = -1.0 + 2.0 * double(vert / 2);
            const double vny = 1.0 - 2.0 * double(((vert + 1) % 4) / 2);

            for (unsigned int c = 0; c <= curveStep; c++) {
                const double a = (double(vert) + double(c) / double(curveStep) - 1.0) * 0.5 * M_PI;

                ::glVertex2d(vnx * rad0 - cos(a) * rad2, vny * rad0 - sin(a) * rad2);

            }
        }
        ::glEnd();
    }
}


/*
 * special::ClusterSignRenderer::renderCross
 */
void special::ClusterSignRenderer::renderCross(unsigned int col) {
    ::glColor3ubv(reinterpret_cast<const GLubyte*>(&col));
    ::glBegin(GL_TRIANGLE_FAN);
    ::glVertex2d(0.0, 0.0);
#ifdef USE_SMOOTH_LINES
    for (int i = 0; i < 2; i++) {
#endif /* USE_SMOOTH_LINES */

        ::glVertex2d(-0.6, -0.4);
        ::glVertex2d(-0.4, -0.6);
        ::glVertex2d( 0.0, -0.2);
        ::glVertex2d( 0.4, -0.6);
        ::glVertex2d( 0.6, -0.4);
        ::glVertex2d( 0.2,  0.0);
        ::glVertex2d( 0.6,  0.4);
        ::glVertex2d( 0.4,  0.6);
        ::glVertex2d( 0.0,  0.2);
        ::glVertex2d(-0.4,  0.6);
        ::glVertex2d(-0.6,  0.4);
        ::glVertex2d(-0.2,  0.0);

#ifdef USE_SMOOTH_LINES
        if (i == 1) break;
        ::glVertex2d(-0.6, -0.4);
        ::glEnd();
        ::glBegin(GL_LINE_LOOP);
    }
#endif /* USE_SMOOTH_LINES */
    ::glEnd();
}


/*
 * special::ClusterSignRenderer::renderCheck
 */
void special::ClusterSignRenderer::renderCheck(unsigned int col) {
    ::glColor3ubv(reinterpret_cast<const GLubyte*>(&col));
    ::glBegin(GL_TRIANGLE_FAN);
#ifdef USE_SMOOTH_LINES
    for (int i = 0; i < 2; i++) {
#endif /* USE_SMOOTH_LINES */

        ::glVertex2d(-0.2, -0.1);
        ::glVertex2d(-0.4,  0.1);
        ::glVertex2d(-0.6, -0.1);
        ::glVertex2d(-0.2, -0.5);
        ::glVertex2d( 0.6,  0.3);
        ::glVertex2d( 0.4,  0.5);

#ifdef USE_SMOOTH_LINES
        if (i == 1) break;
        ::glEnd();
        ::glBegin(GL_LINE_LOOP);
    }
#endif /* USE_SMOOTH_LINES */
    ::glEnd();
}


/*
 * special::ClusterSignRenderer::tick
 */
bool special::ClusterSignRenderer::tick(void) {
    return (vislib::sys::GetTicksOfDay() % 2000) < 1000;
}


/*
 * special::ClusterSignRenderer::ClusterSignRenderer
 */
special::ClusterSignRenderer::ClusterSignRenderer(void) {
    // intentionally empty
}


/*
 * special::ClusterSignRenderer::~ClusterSignRenderer
 */
special::ClusterSignRenderer::~ClusterSignRenderer(void) {
    // intentionally empty
}
