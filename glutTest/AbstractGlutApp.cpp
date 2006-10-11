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


/*
 * AbstractGlutApp::KeyPress
 */
bool AbstractGlutApp::KeyPress(unsigned char key, int x, int y) {
    return false;
}


/*
 * AbstractGlutApp::MouseEvent
 */
void AbstractGlutApp::MouseEvent(int button, int state, int x, int y) {
}


/*
 * AbstractGlutApp::MouseMove
 */
void AbstractGlutApp::MouseMove(int x, int y) {
}


/*
 * AbstractGlutApp::SpecialKey
 */
void AbstractGlutApp::SpecialKey(int key, int x, int y) {
}
