/*
 * GlutClient.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "GlutClient.h"

#if defined(VISLIB_CLUSTER_WITH_OPENGL) && (VISLIB_CLUSTER_WITH_OPENGL != 0)
using namespace vislib::net::cluster;


/*
 * GlutClient::~GlutClient
 */
GlutClient::~GlutClient(void) {
}


/*
 * GlutClient::GlutClient
 */
GlutClient::GlutClient(void) : GlutClientNode<GlutClient>() {
    this->SetReconnectAttempts(5);
    this->logo.Create();
}


/*
 * GlutClient::onFrameRender
 */
void GlutClient::onFrameRender(void) {
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    this->camera.glMultProjectionMatrix();

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();
    this->camera.glMultViewMatrix();

    this->logo.Draw();

    ::glFlush();
    ::glutSwapBuffers();
}


/*
 * GlutClient::onInitialise
 */
void GlutClient::onInitialise(void) {
    GlutClientNode<GlutClient>::onInitialise();
    ::glEnable(GL_DEPTH_TEST);
}

#endif /*defined(VISLIB_CLUSTER_WITH_OPENGL) && ... */
