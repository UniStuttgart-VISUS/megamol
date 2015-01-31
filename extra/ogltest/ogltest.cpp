// ogltest.cpp : Defines the entry point for the console application.
//
#include "vislib/types.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glut.h>


namespace my {

    namespace {

        // Normals for the 6 faces of a cube.
        const float n[6][3] = {
            { -1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 },
            { 0.0, -1.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0, -1.0 }
        };

        // Vertex indices for the 6 faces of a cube.
        const int faces[6][4] = {  
            { 0, 1, 2, 3 }, { 3, 2, 6, 7 }, { 7, 6, 5, 4 },
            { 4, 5, 1, 0 }, { 5, 6, 2, 1 }, { 7, 4, 0, 3 }
        };

        // Will be filled in with X,Y,Z vertexes.
        const float v[8][3] = {
            { 1.0, -1.0, 1.0 },
            { 1.0, -1.0, -1.0 },
            { 1.0, 1.0, -1.0 },
            { 1.0, 1.0, 1.0 },
            { -1.0, -1.0, 1.0 },
            { -1.0, -1.0, -1.0 },
            { -1.0, 1.0, -1.0 },
            { -1.0, 1.0, 1.0 }
        };

    }

    void display(void) {
        ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (int i = 0; i < 6; ++i) {
            ::glBegin(GL_QUADS);
            ::glNormal3fv(&n[i][0]);
            ::glVertex3fv(&v[faces[i][0]][0]);
            ::glVertex3fv(&v[faces[i][1]][0]);
            ::glVertex3fv(&v[faces[i][2]][0]);
            ::glVertex3fv(&v[faces[i][3]][0]);
            ::glEnd();
        }

        ::glutSwapBuffers();
    }

    void init(void) {
        ::ogl_LoadFunctions();

        // Red diffuse light
        const float light_diffuse[] = { 1.0, 0.0, 0.0, 1.0 };
        // Directional light
        const float light_position[] = { 1.0, 1.0, 1.0, 0.0 };

        // Enable a single OpenGL light.
        ::glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
        ::glLightfv(GL_LIGHT0, GL_POSITION, light_position);
        ::glEnable(GL_LIGHT0);
        ::glEnable(GL_LIGHTING);

        // Use depth buffering for hidden surface elimination.
        ::glEnable(GL_DEPTH_TEST);

        // Setup the view of the cube.
        ::glMatrixMode(GL_PROJECTION);
        ::gluPerspective(40.0, 1.0, 1.0, 10.0);
        ::glMatrixMode(GL_MODELVIEW);
        ::gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        // Adjust cube position to be asthetic angle.
        ::glTranslated(0.0, 0.0, -1.0);
        ::glRotated(60.0, 1.0, 0.0, 0.0);
        ::glRotated(-20.0, 0.0, 0.0, 1.0);
    }

}


int main(int argc, char* argv[]) {
    ::glutInit(&argc, argv);
    ::glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    ::glutInitWindowSize(400, 400);
    ::glutCreateWindow("ogltest");
    ::glutDisplayFunc(&my::display);
    my::init();
    ::glutMainLoop();
    return 0;
}

