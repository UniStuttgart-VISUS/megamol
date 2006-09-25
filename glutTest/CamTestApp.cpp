/*
 * CamTestApp.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#include "CamTestApp.h"

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "vislib/PerformanceCounter.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include "vislogo.h"
#include <stdlib.h>


/*
 * CamTestApp::Lens::Lens
 */
CamTestApp::Lens::Lens(void) {
    this->w = 0.25f;
    this->h = 0.25f;
    this->x = float(::rand() % 1000) / 1000.0f * (1.0f - this->w);
    this->y = float(::rand() % 1000) / 1000.0f * (1.0f - this->h);
    this->ax = float(::rand() % 2000) / 2000.0f * M_PI;
    this->ay = sin(this->ax);
    this->ax = cos(this->ax);
}


/*
 * CamTestApp::Lens::~Lens
 */
CamTestApp::Lens::~Lens(void) {
}


/*
 * CamTestApp::Lens::Update
 */
void CamTestApp::Lens::Update(float sec) {
    bool pong = false;

    this->x += ax * sec;
    this->y += ay * sec;

    if (((this->x < 0.0f) && (this->ax < 0.0f)) || ((this->x > 1.0f - this->w) && (this->ax > 0.0f))) {
        this->ax = -this->ax;
        pong = true;
    }
    if (((this->y < 0.0f) && (this->ay < 0.0f)) || ((this->y > 1.0f - this->h) && (this->ay > 0.0f))) {
        this->ay = -this->ay;
        pong = true;
    }

    if (pong) {
        double d = atan2(this->ay, this->ax);
        d += (double(::rand() % 2001 - 1000) / 1000.0f) * 0.5f;
        this->ax = cos(d);
        this->ay = sin(d);
    }

}


/*
 * CamTestApp::Lens::BeginDraw
 */
void CamTestApp::Lens::BeginDraw(unsigned int ww, unsigned int wh) {
    glViewport(0, 0, ww, wh);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glTranslatef(-1.0f, -1.0f, 0.0f);
    glScalef(2.0f, 2.0f, 1.0f);
    glTranslatef(this->x, this->y, 0.0f);

    glDisable(GL_DEPTH_TEST);
    glColor3ub(255, 255, 0);
    glBegin(GL_LINE_LOOP);
        glVertex2f(0.0f, 0.0f);
        glVertex2f(this->w, 0.0f);
        glVertex2f(this->w, this->h);
        glVertex2f(0.0f, this->h);
    glEnd();
    glEnable(GL_DEPTH_TEST);

    glViewport(int(this->x * ww), int(this->y * wh), int(this->w * ww), int(this->h * wh));

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, float(this->w * ww) / float(this->h * wh), 1.0, 10.0); // TODO: Parameters!

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, -2.0, 0.0,  // TODO: Parameters!
        0.0, 0.0, 0.0,  // TODO: Parameters!
        0.0, 0.0, 1.0); // TODO: Parameters!
}


/*
 * CamTestApp::Lens::EndDraw
 */
void CamTestApp::Lens::EndDraw(void) {
	glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
	glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}


/*
 * CamTestApp::CamTestApp
 */
CamTestApp::CamTestApp(void) : AbstractGlutApp() {
    this->lensCount = 8;
    this->lenses = new Lens[this->lensCount];
    this->walkSpeed = 0.25f;
    this->rotSpeed = 0.5f;

    this->walkSpeed = 0.01f;
    this->rotSpeed = 0.05f;
}


/*
 * CamTestApp::~CamTestApp
 */
CamTestApp::~CamTestApp(void) {
    delete[] this->lenses;
}


/*
 * CamTestApp::PreGLInit
 */
int CamTestApp::PreGLInit(void) {
    VisLogoDoStuff();
    VisLogoTwistLogo();
    return 0;
}


/*
 * CamTestApp::PostGLInit
 */
int CamTestApp::PostGLInit(void) {
    glEnable(GL_DEPTH_TEST);
    this->lastTime = vislib::sys::PerformanceCounter::Query();
    return 0;
}


/*
 * CamTestApp::Render
 */
void CamTestApp::Render(void) {
    UINT64 time = vislib::sys::PerformanceCounter::Query();
    float sec = this->walkSpeed * float(time - this->lastTime) / 1000.0f;

    this->lastTime = time;
    this->angle = static_cast<float>(static_cast<int>(static_cast<float>(time) * this->rotSpeed) % 3600) * 0.1f;

    glViewport(0, 0, this->GetWidth(), this->GetHeight());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(45.0, this->GetAspectRatio(), 1.0, 10.0); // TODO: Parameters!

    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

    gluLookAt(0.0, -2.0, 0.0,  // TODO: Parameters!
        0.0, 0.0, 0.0,  // TODO: Parameters!
        0.0, 0.0, 1.0); // TODO: Parameters!

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    this->RenderLogo();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glClear(GL_DEPTH_BUFFER_BIT);

    for (unsigned int i = 0; i < this->lensCount; i++) {
        this->lenses[i].Update(sec);
        this->lenses[i].BeginDraw(this->GetWidth(), this->GetHeight());
        this->RenderLogo();
        this->lenses[i].EndDraw();
    }

	glFlush();

	glutSwapBuffers();
    glutPostRedisplay();
}


/*
 * CamTestApp::RenderLogo
 */
void CamTestApp::RenderLogo(void) {
    glPushMatrix();
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    glRotatef(this->angle, 0.0f, -1.0f, 0.0f);
    unsigned int vCount = VisLogoCountVertices();
	unsigned int p;
	glBegin(GL_QUAD_STRIP);
	for (unsigned int i = 0; i < 20; i++) {
		for (unsigned int j = 0; j < vCount / 20; j++) {
			p = (i + j * 20) % vCount;
			glColor3dv(VisLogoVertexColor(p)->f);
			glNormal3dv(VisLogoVertexNormal(p)->f);
			glVertex3dv(VisLogoVertex(p)->f);
			p = ((i + 1) % 20 + j * 20) % vCount;
			glColor3dv(VisLogoVertexColor(p)->f);
			glNormal3dv(VisLogoVertexNormal(p)->f);
			glVertex3dv(VisLogoVertex(p)->f);
		}
	}
	p = 0; // closing strip
	glColor3dv(VisLogoVertexColor(p)->f);
	glNormal3dv(VisLogoVertexNormal(p)->f);
	glVertex3dv(VisLogoVertex(p)->f);
	p = 1;
	glColor3dv(VisLogoVertexColor(p)->f);
	glNormal3dv(VisLogoVertexNormal(p)->f);
	glVertex3dv(VisLogoVertex(p)->f);
	glEnd();    
    glPopMatrix();
}
