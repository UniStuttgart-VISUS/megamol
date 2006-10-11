/*
 * glutTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

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
#endif /* _WIN32 */


#include "AbstractGlutApp.h"

#include "CamTestApp.h"
#include "StereoCamTestApp.h"


// static global functions
AbstractGlutApp *app = NULL;


/*
 * std glut callbacks
 */
void reshape(int w, int h) {
    assert(app != NULL);
    app->Resize(w, h);
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
            if (!(app && app->KeyPress(key, x, y))) {
                fprintf(stderr, "Warning: Key %u is not used\n", key); 
            }
            break;
    };
}

void mouse(int button, int state, int x, int y) {
    assert(app != NULL);
    app->MouseEvent(button, state, x, y);
}

void motion(int x, int y) {
    assert(app != NULL);
    app->MouseMove(x, y);
}

void special(int key, int x, int y) {
    assert(app != NULL);
    app->SpecialKey(key, x, y);
}

/*
 * main
 */
int main(int argc, char* argv[]) {
    printf("VISlib glut test program\n");

    // select test application:
    CamTestApp cta; app = &cta;
    //StereoCamTestApp scta; app = &scta;

    // run test application
    if (app) {
        glutInit(&argc, argv);

        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA); // TODO: should be moved to AbstractGlutApp
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

