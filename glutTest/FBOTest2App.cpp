/*
 * FBOTest2App.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "FBOTest2App.h"
#include "glh/glh_genext.h"
#include "vislibGlutInclude.h"
#include <iostream>

#include "vislib/glverify.h"
#include "vislib/CameraParamsStore.h"


/*
 * FBOTest2App::FBOTest2App
 */
FBOTest2App::FBOTest2App(void) : AbstractGlutApp(), camera(), camera2(), fboAll(), fboRed(), fboGreen() {

    this->camera.Parameters()->SetView(
        vislib::math::Point<float, 3>(0.0, -4.0, 0.0),
        vislib::math::Point<float, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<float, 3>(0.0, 0.0, 1.0));
    this->camera.Parameters()->SetClip(1.0f, 10.0f);
    this->camera.Parameters()->SetFocalDistance(4.0f);
    this->camera.Parameters()->SetApertureAngle(50.0f);
    this->camera.Parameters()->SetVirtualViewSize(512.0f, 512.0f);
    this->camera.Parameters()->SetProjection(vislib::graphics::CameraParameters::MONO_PERSPECTIVE);

    this->camera2.Parameters()->CopyFrom(this->camera.Parameters());
}


/*
 * FBOTest2App::~FBOTest2App
 */
FBOTest2App::~FBOTest2App(void) {
}


/*
 * FBOTest2App::GLInit
 */
int FBOTest2App::GLInit(void) {
    using vislib::graphics::gl::FramebufferObject;

    if (!FramebufferObject::InitialiseExtensions()) {
        return -1;
    };

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera.glMultViewMatrix();

    float lp[4] = {-2.0f, -2.0f, 2.0f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lp);

    float la[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, la);

    float ld[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    glLightfv(GL_LIGHT0, GL_DIFFUSE, ld);

    glEnable(GL_COLOR_MATERIAL);

    if (!this->fboAll.Create(512, 512, GL_RGB16F_ARB, GL_RGB, GL_FLOAT, FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT32)) {
        printf("fboAll was not created!");
        return -2;
    }

    if (!this->fboRed.Create(512, 512, GL_RGB16F_ARB, GL_RGB, GL_FLOAT, FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT32)) {
        printf("fboRed was not created!");
        return -3;
    }

    FramebufferObject::ColourAttachParams cap;
    cap.internalFormat = GL_RGB16F_ARB;
    cap.format = GL_RGB;
    cap.type = GL_FLOAT;

    FramebufferObject::DepthAttachParams dap;
    dap.format = GL_DEPTH_COMPONENT32;
    dap.state = FramebufferObject::ATTACHMENT_EXTERNAL_TEXTURE;
    dap.externalID = this->fboRed.DepthTextureID();

    FramebufferObject::StencilAttachParams sap;
    sap.state = FramebufferObject::ATTACHMENT_DISABLED;

    if (!this->fboGreen.Create(512, 512, 1, &cap, dap, sap)) {
        printf("fboGreen was not created!");
        return -4;
    }

    return 0;
}


/*
 * FBOTest2App::OnResize
 */
void FBOTest2App::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->camera.Parameters()->SetVirtualViewSize(
        static_cast<vislib::graphics::ImageSpaceType>(w),
        static_cast<vislib::graphics::ImageSpaceType>(h));
}


/*
 * FBOTest2App::GLDeinit
 */
void FBOTest2App::GLDeinit(void) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glDisable(GL_COLOR_MATERIAL);
}


/*
 * FBOTest2App::Render
 */
void FBOTest2App::Render(void) {
    USES_GL_VERIFY;

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);

    // render all 
    ::glDrawBuffer(GL_BACK);
    GL_VERIFY_EXPR(this->fboAll.Enable());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->camera2.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera2.glMultViewMatrix();

    glColor3ub(255, 0, 0);
    glPushMatrix();
    glRotatef(60.0, 1.0, 0.0, 0.0);
    glRotatef(60.0, 0.0, 0.0, 1.0);
    this->RenderTestBox(1.0f);
    glPopMatrix();

    glColor3ub(0, 255, 0);
    glPushMatrix();
    glRotatef(30.0, 1.0, 0.0, 0.0);
    glRotatef(30.0, 0.0, 0.0, 1.0);
    this->RenderTestBox(1.0f);
    glPopMatrix();

    this->fboAll.Disable();

    // render red 
    GL_VERIFY_EXPR(this->fboRed.Enable());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->camera2.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera2.glMultViewMatrix();

    glColor3ub(255, 0, 0);
    glPushMatrix();
    glRotatef(60.0, 1.0, 0.0, 0.0);
    glRotatef(60.0, 0.0, 0.0, 1.0);
    this->RenderTestBox(1.0f);
    glPopMatrix();

    this->fboRed.Disable();

    // render green 
    GL_VERIFY_EXPR(this->fboGreen.Enable());

    glClear(GL_COLOR_BUFFER_BIT); /* do not clear depth */

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->camera2.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera2.glMultViewMatrix();

    glColor3ub(0, 255, 0);
    glPushMatrix();
    glRotatef(30.0, 1.0, 0.0, 0.0);
    glRotatef(30.0, 0.0, 0.0, 1.0);
    this->RenderTestBox(1.0f);
    glPopMatrix();

    this->fboGreen.Disable();

    // render viewing
    glViewport(0, 0, this->GetWidth(), this->GetHeight());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->camera.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera.glMultViewMatrix();

    glColor3ub(255, 255, 255);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    glPushMatrix();
    glTranslated(0.0, 3.0, 0.0);
    this->fboRed.BindColourTexture();
    glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0); ::glVertex3d(-1.0, 0.0, -1.0);  // bottom left
        glTexCoord2d(1.0, 0.0); ::glVertex3d(1.0, 0.0, -1.0);  // bottom right
        glTexCoord2d(1.0, 1.0); ::glVertex3d(1.0, 0.0, 1.0);   // top right
        glTexCoord2d(0.0, 1.0); ::glVertex3d(-1.0, 0.0, 1.0);   // top left
    glEnd();  
    glPopMatrix();

    glPushMatrix();
    glRotated(38.0, 0.0, 0.0, 1.0);
    glTranslated(0.0, 3.0, 0.0);
    this->fboAll.BindColourTexture();
    glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0); ::glVertex3d(-1.0, 0.0, -1.0);  // bottom left
        glTexCoord2d(1.0, 0.0); ::glVertex3d(1.0, 0.0, -1.0);  // bottom right
        glTexCoord2d(1.0, 1.0); ::glVertex3d(1.0, 0.0, 1.0);   // top right
        glTexCoord2d(0.0, 1.0); ::glVertex3d(-1.0, 0.0, 1.0);   // top left
    glEnd();  
    glPopMatrix();

    glPushMatrix();
    glRotated(-38.0, 0.0, 0.0, 1.0);
    glTranslated(0.0, 3.0, 0.0);
    this->fboGreen.BindColourTexture();
    glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0); ::glVertex3d(-1.0, 0.0, -1.0);  // bottom left
        glTexCoord2d(1.0, 0.0); ::glVertex3d(1.0, 0.0, -1.0);  // bottom right
        glTexCoord2d(1.0, 1.0); ::glVertex3d(1.0, 0.0, 1.0);   // top right
        glTexCoord2d(0.0, 1.0); ::glVertex3d(-1.0, 0.0, 1.0);   // top left
    glEnd();  
    glPopMatrix();

    glFlush();
    glutSwapBuffers();
}


/*
 * FBOTest2App::RenderTestBox
 */
void FBOTest2App::RenderTestBox(float s) {
    glBegin(GL_QUADS);
        glNormal3f( 1,  0,  0); glVertex3f( s,  s,  s); glVertex3f( s, -s,  s); glVertex3f( s, -s, -s); glVertex3f( s,  s, -s);
        glNormal3f(-1,  0,  0); glVertex3f(-s,  s,  s); glVertex3f(-s,  s, -s); glVertex3f(-s, -s, -s); glVertex3f(-s, -s,  s);
        glNormal3f( 0,  1,  0); glVertex3f( s,  s,  s); glVertex3f( s,  s, -s); glVertex3f(-s,  s, -s); glVertex3f(-s,  s,  s);
        glNormal3f( 0, -1,  0); glVertex3f( s, -s,  s); glVertex3f(-s, -s,  s); glVertex3f(-s, -s, -s); glVertex3f( s, -s, -s);
        glNormal3f( 0,  0,  1); glVertex3f( s,  s,  s); glVertex3f(-s,  s,  s); glVertex3f(-s, -s,  s); glVertex3f( s, -s,  s);
        glNormal3f( 0,  0, -1); glVertex3f( s,  s, -s); glVertex3f( s, -s, -s); glVertex3f(-s, -s, -s); glVertex3f(-s,  s, -s);
    glEnd();
}
