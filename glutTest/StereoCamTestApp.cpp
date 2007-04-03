/*
 * StereoCamTestApp.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "StereoCamTestApp.h"

#include "vislib/PerformanceCounter.h"

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>

#include "vislogo.h"

StereoCamTestApp::StereoCamTestApp(void) : AbstractGlutApp() {
    this->beholder.SetView(
        vislib::math::Point<float, 3>(0.0, -10.0, 0.0),
        vislib::math::Point<float, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<float, 3>(0.0, 0.0, 1.0));

    this->cameraLeft.SetBeholder(&this->beholder);
    this->cameraLeft.SetNearClipDistance(1.0f);
    this->cameraLeft.SetFarClipDistance(15.0f);
    this->cameraLeft.SetFocalDistance(10.0f);
    this->cameraLeft.SetApertureAngle(30.0f);
    this->cameraLeft.SetVirtualWidth(10.0f);
    this->cameraLeft.SetVirtualHeight(10.0f);
    this->cameraLeft.SetStereoDisparity(0.125f);
    this->cameraLeft.SetProjectionType(vislib::graphics::Camera::MONO_PERSPECTIVE);

    this->cameraRight = this->cameraLeft;

    printf("Stereo Projection set to STEREO_PARALLEL\n");
    this->cameraLeft.SetStereoEye(vislib::graphics::Camera::LEFT_EYE);
    this->cameraRight.SetStereoEye(vislib::graphics::Camera::RIGHT_EYE);

    this->UpdateCamTiles();
}

StereoCamTestApp::~StereoCamTestApp(void) {
}

int StereoCamTestApp::GLInit(void) {
    VisLogoDoStuff();
    VisLogoTwistLogo();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    float lp[4] = {0.0f, -2.0f, 2.0f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lp);

    float la[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, la);

    float ld[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    glLightfv(GL_LIGHT0, GL_DIFFUSE, ld);

    glEnable(GL_COLOR_MATERIAL);

    this->lastTime = vislib::sys::PerformanceCounter::Query();
    return 0;
}

void StereoCamTestApp::GLDeinit(void) {
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void StereoCamTestApp::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->cameraRight.SetVirtualWidth(float(w));
    this->cameraRight.SetVirtualHeight(float(h));
    this->cameraLeft.SetVirtualWidth(float(w));
    this->cameraLeft.SetVirtualHeight(float(h));
    this->UpdateCamTiles();
}

void StereoCamTestApp::Render(void) {
    UINT64 time = vislib::sys::PerformanceCounter::Query();
    this->lastTime = time;
    this->angle = static_cast<float>(static_cast<int>(static_cast<float>(time) * 0.5f) % 3600) * 0.1f;

    // render left eye image
    glViewport(0, 0, this->GetWidth(), this->GetHeight());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3ub(255, 0, 0);

    /*
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    this->cameraLeft.glMultProjectionMatrix();
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
    this->cameraLeft.glMultViewMatrix();

    this->RenderTestBox();
  */  
    vislib::math::Rectangle<vislib::graphics::ImageSpaceType> rect;
    for (int y = 0; y < SCTA_CY_TILES; y++)
        for (int x = 0; x < SCTA_CX_TILES; x++) {
            rect = this->camTilesLeft[x][y].GetTileRectangle();
            glViewport(int(rect.Left() + 0.5f), 
                int(rect.Bottom() + 0.5f), int(rect.Width() + 0.5f), 
                int(rect.Height() + 0.5f));

            glMatrixMode(GL_PROJECTION);
	        glLoadIdentity();
            this->camTilesLeft[x][y].glMultProjectionMatrix();
            glMatrixMode(GL_MODELVIEW);
	        glLoadIdentity();
            this->camTilesLeft[x][y].glMultViewMatrix();

            this->RenderTestBox();
        }


    // render right eye image
    glViewport(0, 0, this->GetWidth(), this->GetHeight());
	glClear(GL_DEPTH_BUFFER_BIT);
    glColor3ub(0, 255, 255);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    this->cameraRight.glMultProjectionMatrix();
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
    this->cameraRight.glMultViewMatrix();

    this->RenderTestBox();

    // done rendering
    glFlush();
	glutSwapBuffers();
    glutPostRedisplay(); // because we do animation stuff
}

bool StereoCamTestApp::OnKeyPress(unsigned char key, int x, int y) {
    switch(key) {
        case '1':
            printf("Stereo Projection set to STEREO_PARALLEL\n");
            this->cameraLeft.SetProjectionType(vislib::graphics::Camera::STEREO_PARALLEL);
            this->cameraRight.SetProjectionType(vislib::graphics::Camera::STEREO_PARALLEL);
            break;
        case '2':
            printf("Stereo Projection set to STEREO_OFF_AXIS\n");
            this->cameraLeft.SetProjectionType(vislib::graphics::Camera::STEREO_OFF_AXIS);
            this->cameraRight.SetProjectionType(vislib::graphics::Camera::STEREO_OFF_AXIS);
            break;
        case '3':
            printf("Stereo Projection set to STEREO_TOE_IN\n");
            this->cameraLeft.SetProjectionType(vislib::graphics::Camera::STEREO_TOE_IN);
            this->cameraRight.SetProjectionType(vislib::graphics::Camera::STEREO_TOE_IN);
            break;
        case 'a': {
            float sd = this->cameraLeft.GetStereoDisparity();
            sd /= 1.2f;
            this->cameraLeft.SetStereoDisparity(sd);
            this->cameraRight.SetStereoDisparity(sd);
            printf("Stereo Disparity set to %f\n", sd);
        } break;
        case 'y': {
            float sd = this->cameraLeft.GetStereoDisparity();
            sd *= 1.2f;
            this->cameraLeft.SetStereoDisparity(sd);
            this->cameraRight.SetStereoDisparity(sd);
            printf("Stereo Disparity set to %f\n", sd);
        } break;
        case 's': {
            float sd = this->cameraLeft.GetFocalDistance();
            sd -= 0.1f;
            this->cameraLeft.SetFocalDistance(sd);
            this->cameraRight.SetFocalDistance(sd);
            printf("Focal distance set to %f\n", sd);
        } break;
        case 'x': {
            float sd = this->cameraLeft.GetFocalDistance();
            sd += 0.1f;
            this->cameraLeft.SetFocalDistance(sd);
            this->cameraRight.SetFocalDistance(sd);
            printf("Focal distance set to %f\n", sd);
        } break;

        default:
            return AbstractGlutApp::OnKeyPress(key, x, y);
    }
    this->UpdateCamTiles();
    glutPostRedisplay();
    return true;
}

void StereoCamTestApp::RenderTestBox(void) {
    glDisable(GL_LIGHTING);
    glBegin(GL_LINE_LOOP);
        glVertex3i(-1, -1, -1);
        glVertex3i( 1, -1, -1);
        glVertex3i( 1,  1, -1);
        glVertex3i(-1,  1, -1);
    glEnd();
    glBegin(GL_LINE_LOOP);
        glVertex3i(-1, -1,  1);
        glVertex3i( 1, -1,  1);
        glVertex3i( 1,  1,  1);
        glVertex3i(-1,  1,  1);
    glEnd();
    glBegin(GL_LINES);
        glVertex3i(-1, -1, -1);
        glVertex3i(-1, -1,  1);
        glVertex3i( 1, -1, -1);
        glVertex3i( 1, -1,  1);
        glVertex3i(-1,  1, -1);
        glVertex3i(-1,  1,  1);
        glVertex3i( 1,  1, -1);
        glVertex3i( 1,  1,  1);
    glEnd();

    glBegin(GL_POINTS);
    for (float y = -1.0f; y < 1.0f; y += 0.2f)
        glVertex3f(0.0f, y, -1.0f);
    glEnd();

    glEnable(GL_LIGHTING);

    glPushMatrix();
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    glRotatef(this->angle, 0.0f, -1.0f, 0.0f);
    unsigned int vCount = VisLogoCountVertices();
	unsigned int p;
	glBegin(GL_QUAD_STRIP);
	for (unsigned int i = 0; i < 20; i++) {
		for (unsigned int j = 0; j < vCount / 20; j++) {
			p = (i + j * 20) % vCount;
			glNormal3dv(VisLogoVertexNormal(p)->f);
			glVertex3dv(VisLogoVertex(p)->f);
			p = ((i + 1) % 20 + j * 20) % vCount;
			glNormal3dv(VisLogoVertexNormal(p)->f);
			glVertex3dv(VisLogoVertex(p)->f);
		}
	}
	p = 0; // closing strip
	glNormal3dv(VisLogoVertexNormal(p)->f);
	glVertex3dv(VisLogoVertex(p)->f);
	p = 1;
	glNormal3dv(VisLogoVertexNormal(p)->f);
	glVertex3dv(VisLogoVertex(p)->f);
	glEnd();    
    glPopMatrix();
}

void StereoCamTestApp::UpdateCamTiles(void) {
    vislib::math::Rectangle<vislib::graphics::ImageSpaceType> rect;
    vislib::graphics::ImageSpaceType w, h;
    for (int y = 0; y < SCTA_CY_TILES; y++)
        for (int x = 0; x < SCTA_CX_TILES; x++) {
            this->camTilesLeft[x][y] = this->cameraLeft;
            rect = this->cameraLeft.GetTileRectangle();
            w = rect.Width();
            h = rect.Height();
            rect.Set(
                float(x) * float(w) / float(SCTA_CX_TILES),
                float(y) * float(h) / float(SCTA_CY_TILES),
                float(x + 1) * float(w) / float(SCTA_CX_TILES),
                float(y + 1) * float(h) / float(SCTA_CY_TILES));
            this->camTilesLeft[x][y].SetTileRectangle(rect);            
        }
}
