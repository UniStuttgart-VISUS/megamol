/*
 * VisLogoRenderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "VisLogoRenderer.h"
#include "param/FloatParam.h"
#include <GL/gl.h>

using namespace megamol::core;


/*
 * special::VisLogoRenderer::VisLogoRenderer
 */
special::VisLogoRenderer::VisLogoRenderer(void) : RendererModule(),
        visLogo(), scale("scale", "Scales the vis logo") {
    this->scale << new param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->scale);
}


/*
 * special::VisLogoRenderer::~VisLogoRenderer
 */
special::VisLogoRenderer::~VisLogoRenderer(void) {
    this->Release();
}


/*
 * special::VisLogoRenderer::create
 */
bool special::VisLogoRenderer::create(void) {
    this->visLogo.Create();
    return true;
}


/*
 * special::VisLogoRenderer::release
 */
void special::VisLogoRenderer::release(void) {
    this->visLogo.Release();
}


/*
 * special::VisLogoRenderer::Render
 */
bool special::VisLogoRenderer::Render(Call& call) {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float scale = *this->scale.Param<param::FloatParam>();
    glScalef(scale, scale, scale);
    this->visLogo.Draw();

    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);

    glColor3ub(255, 0, 0); glVertex3i(0, 0, 0); glVertex3i(1, 0, 0);
    glColor3ub(0, 255, 0); glVertex3i(0, 0, 0); glVertex3i(0, 1, 0);
    glColor3ub(0, 0, 255); glVertex3i(0, 0, 0); glVertex3i(0, 0, 1);

    glEnd();

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    return true;
}
