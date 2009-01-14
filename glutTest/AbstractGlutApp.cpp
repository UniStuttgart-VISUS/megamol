/*
 * AbstractGlutApp.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "AbstractGlutApp.h"

#include "vislibGlutInclude.h"
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
 * AbstractGlutApp::OnResize
 */
void AbstractGlutApp::OnResize(unsigned int w, unsigned int h) {
    this->aspectRatio = float(this->width = w) / float(this->height = ((h > 0) ? h : 1));
	glViewport(0, 0, this->width, this->height);
}


/*
 * AbstractGlutApp::OnKeyPress
 */
bool AbstractGlutApp::OnKeyPress(unsigned char key, int x, int y) {
    return false;
}


/*
 * AbstractGlutApp::OnMouseEvent
 */
void AbstractGlutApp::OnMouseEvent(int button, int state, int x, int y) {
}


/*
 * AbstractGlutApp::OnMouseMove
 */
void AbstractGlutApp::OnMouseMove(int x, int y) {
}


/*
 * AbstractGlutApp::OnSpecialKey
 */
void AbstractGlutApp::OnSpecialKey(int key, int x, int y) {
}
