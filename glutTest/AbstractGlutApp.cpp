/*
 * AbstractGlutApp.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "AbstractGlutApp.h"

#include <GL/glut.h>
#include <GL/gl.h>


/*
 * AbstractGlutApp::AbstractGlutApp
 */
AbstractGlutApp::AbstractGlutApp(void) {
}


/*
 * AbstractGlutApp::~AbstractGlutApp
 */
AbstractGlutApp::~AbstractGlutApp(void) {
}


/*
 * AbstractGlutApp::Resize
 */
void AbstractGlutApp::Resize(unsigned int w, unsigned int h) {
    this->aspectRatio = float(this->width = w) / float(this->height = ((h > 0) ? h : 1));
	glViewport(0, 0, this->width, this->height);
}
