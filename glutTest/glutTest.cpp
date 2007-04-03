/*
 * glutTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "AbstractGlutApp.h"

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


// static global functions
AbstractGlutApp *app = NULL;


/*
 * std glut callbacks
 */
void reshape(int w, int h) {
    assert(app != NULL);
    app->OnResize(w, h);
}

void display(void) {
    assert(app != NULL);
    app->Render();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: // esc
            exit(0); 
            break;
        default:
            if (!(app && app->OnKeyPress(key, x, y))) {
                fprintf(stderr, "Warning: Key %u is not used\n", key); 
            }
            break;
    };
}

void mouse(int button, int state, int x, int y) {
    assert(app != NULL);
    app->OnMouseEvent(button, state, x, y);
}

void motion(int x, int y) {
    assert(app != NULL);
    app->OnMouseMove(x, y);
}

void special(int key, int x, int y) {
    assert(app != NULL);
    app->OnSpecialKey(key, x, y);
}

/*
 * main
 */
int main(int argc, char* argv[]) {
    printf("VISlib glut test program\n");

    // select test application:
    //CamTestApp cta; app = &cta;
    //StereoCamTestApp scta; app = &scta;
    BeholderRotatorTextApp brta; app = &brta;
    //FBOTestApp fbota; app = &fbota;

    // run test application
    if (app) {
        glutInit(&argc, argv);

        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA); // TODO: should be moved to AbstractGlutApp
        glutInitWindowSize(512, 512); // TODO: should be configured by AbstractGlutApp
        glutInitWindowPosition(128, 128); // TODO: should be configured by AbstractGlutApp
    	glutCreateWindow("VISlib Glut Test"); // TODO: should be configured by AbstractGlutApp

    	glutDisplayFunc(display);
	    glutReshapeFunc(reshape);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutPassiveMotionFunc(motion);
        glutSpecialFunc(special);

        // glutFullScreen();

        app->GLInit();

    	glutMainLoop();
    }

	return 0;
}

