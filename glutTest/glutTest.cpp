/*
 * glutTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma warning(disable: 4996)
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#define GLH_EXT_SINGLE_FILE
#include "glh/glh_extensions.h"
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma warning(default: 4996)
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "AbstractGlutApp.h"
#include "GlutAppManager.h"

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#endif /* _WIN32 */

#include <stdio.h>
#include "vislibGlutInclude.h"
#include <cassert>
#include <stdlib.h>

#ifdef _WIN32
#else /* _WIN32 */
#include <GL/glx.h>
#endif /* _WIN32 */

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma warning(disable: 4996)
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#define GLH_EXT_SINGLE_FILE
#include "glh/glh_extensions.h"
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma warning(default: 4996)
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CamTestApp.h"
#include "StereoCamTestApp.h"
#include "CamRotatorTestApp.h"
#include "FBOTestApp.h"
#include "FBOTest2App.h"
#include "GLSLShaderTest.h"
#include "GLSLGeomShaderTest.h"
#include "CamGlMatrixTest.h"
#include "SimpleFontTest.h"
#include "CamEulerRotatorTestApp.h"

#include "vislib/FpsCounter.h"
#include "vislib/Thread.h"
#include "vislib/glfunctions.h"

/** the startup test selection */
static int startupTest;

/** the global frames per second counter */
vislib::graphics::FpsCounter fpsCounter;

/*
 * fps output function
 */
unsigned int OutputFps(void *) {
    while (true) {
        vislib::sys::Thread::Sleep(1000);
        printf("FPS: (%f average; %f minimum; %f maximum)\n", 
            fpsCounter.FPS(), fpsCounter.MinFPS(), fpsCounter.MaxFPS());
    }
    return 0;
} 

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
    fpsCounter.FrameBegin();
    if (GlutAppManager::CurrentApp()) {
        GlutAppManager::CurrentApp()->Render();
    } else {
        GlutAppManager::GetInstance()->glRenderEmptyScreen();
        if (startupTest >= 0) {
            GlutAppManager::OnMenuItemClicked(startupTest);
            startupTest = -1;
        }
    }
    fpsCounter.FrameEnd();
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
    vislib::sys::Thread fpsOutputThread((vislib::sys::Runnable::Function)OutputFps);

    printf("VISlib glut test program\n");
    startupTest = -1;

    // register factories:
    GlutAppManager::InstallFactory<CamTestApp>("Camera Test");
    GlutAppManager::InstallFactory<StereoCamTestApp>("Stereo Camera Test");
    GlutAppManager::InstallFactory<CamRotatorTestApp>("Camera Rotation Test");
    GlutAppManager::InstallFactory<FBOTestApp>("Framebuffer Object Test");
    GlutAppManager::InstallFactory<FBOTest2App>("Framebuffer Object Test #2");
    GlutAppManager::InstallFactory<GLSLShaderTest>("GLSLShader Test");
    GlutAppManager::InstallFactory<GLSLGeomShaderTest>("GLSLGeometryShader Test");
    GlutAppManager::InstallFactory<CamGlMatrixTest>("Manual Matrix Generation Test");
    GlutAppManager::InstallFactory<SimpleFontTest>("SimpleFont Test");
    GlutAppManager::InstallFactory<CamEulerRotatorTestApp>("Camera Euler Rotation Test");

    // run test application
    glutInit(&argc, argv);

    // TODO: Startup settings object with compatibility check
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);  // TODO: should be moved to AbstractGlutApp
    glutInitWindowSize(512, 512);                               // TODO: should be configured by AbstractGlutApp
    glutInitWindowPosition(128, 128);                           // TODO: should be configured by AbstractGlutApp
    glutCreateWindow("VISlib Glut Test");                       // TODO: should be configured by AbstractGlutApp

#ifdef _WIN32
#if defined(VISGLUT_EXTENSIONS)
    ::glutSetWindowIconI(101);
#endif /* VISGLUT_EXTENSIONS */
#endif

    printf("VSync is %s\n", vislib::graphics::gl::IsVSyncEnabled() ? "Enabled" : "Disabled");
    //vislib::graphics::gl::DisableVSync();
    //printf("VSync is %s\n", vislib::graphics::gl::IsVSyncEnabled() ? "Enabled" : "Disabled");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(motion);
    glutSpecialFunc(special);

    GlutAppManager::GetInstance()->InitGlutWindow();

    fpsOutputThread.Start();

    // start a glut application
    if (argc > 1) {
        int id;
#if (_MSC_VER >= 1400)
#pragma warning(disable: 4996)
#endif
        if (sscanf(argv[1], "%d", &id) == 1) {
#if (_MSC_VER >= 1400)
#pragma warning(default: 4996)
#endif
            printf("Trying to start Test %d\n", id);
            startupTest = id;
        }
    }

    // glutFullScreen();
    // app->GLInit();

    glutMainLoop();

    return 0;
}

