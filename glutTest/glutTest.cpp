/*
 * glutTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "AbstractGlutApp.h"
#include "GlutAppManager.h"

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#endif /* _WIN32 */

#include <stdio.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <cassert>
#include <stdlib.h>

#ifdef _WIN32
#else /* _WIN32 */
#include <GL/glx.h>
#endif /* _WIN32 */

#if (_MSC_VER > 1000)
#pragma warning(disable: 4996)
#endif /* (_MSC_VER > 1000) */
#define GLH_EXT_SINGLE_FILE
#include "glh/glh_extensions.h"
#if (_MSC_VER > 1000)
#pragma warning(default: 4996)
#endif /* (_MSC_VER > 1000) */

#include "CamTestApp.h"
#include "StereoCamTestApp.h"
#include "BeholderRotatorTextApp.h"
#include "FBOTestApp.h"


/*
 * std glut callbacks
 */
void reshape(int w, int h) {

    // at least one pixel size!
    if (w < 1) w = 1;
    if (h < 1) h = 1;

    GlutAppManager::GetInstance()->SetSize(w, h);
    if (GlutAppManager::CurrentApp()) {
        GlutAppManager::CurrentApp()->OnResize(w, h);
    }
}

void display(void) {
    if (GlutAppManager::CurrentApp()) {
        GlutAppManager::CurrentApp()->Render();
    } else {
        GlutAppManager::GetInstance()->glRenderEmptyScreen();
    }
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: // esc
            GlutAppManager::ExitApplication(0);
            break;
        default:
            if (!(GlutAppManager::CurrentApp() && GlutAppManager::CurrentApp()->OnKeyPress(key, x, y))) {
                fprintf(stderr, "Warning: Key %u is not used\n", key); 
            }
            break;
    };
}

void mouse(int button, int state, int x, int y) {
    if (GlutAppManager::CurrentApp()) {
        GlutAppManager::CurrentApp()->OnMouseEvent(button, state, x, y);
    }
}

void motion(int x, int y) {
    if (GlutAppManager::CurrentApp()) {
        GlutAppManager::CurrentApp()->OnMouseMove(x, y);
    }
}

void special(int key, int x, int y) {
    if (GlutAppManager::CurrentApp()) {
        GlutAppManager::CurrentApp()->OnSpecialKey(key, x, y);
    }
}

/*
 * main
 */
int main(int argc, char* argv[]) {
    printf("VISlib glut test program\n");

    // register factories:
    GlutAppManager::InstallFactory<CamTestApp>("Camera Test");
    GlutAppManager::InstallFactory<StereoCamTestApp>("Stereo Camera Test");
    GlutAppManager::InstallFactory<BeholderRotatorTextApp>("Beholder Rotator Test");
    GlutAppManager::InstallFactory<FBOTestApp>("Framebuffer Object Test");

    // run test application
    glutInit(&argc, argv);

    // TODO: Startup settings object with compatibility check
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);  // TODO: should be moved to AbstractGlutApp
    glutInitWindowSize(512, 512);                               // TODO: should be configured by AbstractGlutApp
    glutInitWindowPosition(128, 128);                           // TODO: should be configured by AbstractGlutApp
    glutCreateWindow("VISlib Glut Test");                       // TODO: should be configured by AbstractGlutApp

    glutDisplayFunc(display);
	glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(motion);
    glutSpecialFunc(special);

    GlutAppManager::GetInstance()->InitGlutWindow();

    // glutFullScreen();
    // app->GLInit();

    glutMainLoop();

	return 0;
}

