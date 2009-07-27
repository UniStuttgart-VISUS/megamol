/*
 * GlutServer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "GlutServer.h"

#if defined(VISLIB_CLUSTER_WITH_OPENGL) && (VISLIB_CLUSTER_WITH_OPENGL != 0)
using namespace vislib::net::cluster;


/*
 * GlutServer::~GlutServer
 */
GlutServer::~GlutServer(void) {
}

/*
 * GlutServer::GlutServer
 */
GlutServer::GlutServer(void) : GlutServerNode<GlutServer>() {
    this->logo.Create();
}


/*
 * GlutServer::initialiseController
 */
void GlutServer::initialiseController(
        vislib::graphics::AbstractCameraController *& inOutRotateController,
        vislib::graphics::AbstractCameraController *& inOutZoomController) {
    // TODO: warum funktioniert das dynamische binden nicht?
    GlutServerNode<GlutServer>::initialiseController(inOutRotateController,
        inOutZoomController);
    this->camera.Parameters()->SetView(
        vislib::graphics::SceneSpacePoint3D(0.0, -3.0, 0.0),
        vislib::graphics::SceneSpacePoint3D(0.0, 0.0, 0.0),
        vislib::graphics::SceneSpaceVector3D(0.0, 0.0, 1.0));
}


/*
 * GlutServer::onFrameRender
 */
void GlutServer::onFrameRender(void) {
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
 * GlutServer::onInitialise
 */
void GlutServer::onInitialise(void) {
    GlutServerNode<GlutServer>::onInitialise();
    ::glEnable(GL_DEPTH_TEST);
}

#endif /*defined(VISLIB_CLUSTER_WITH_OPENGL) && ... */
