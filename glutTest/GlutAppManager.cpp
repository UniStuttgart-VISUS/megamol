/*
 * FBOTestApp.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#endif /* _WIN32 */

#include <stdio.h>
#include "vislibGlutInclude.h"
#include <GL/gl.h>
#include "GlutAppManager.h"
#include "vislib/FpsCounter.h"
#include "vislib/VersionNumber.h"
#include "vislib/glfunctions.h"


/** not nice! */
extern vislib::graphics::FpsCounter fpsCounter;


/*
 * GlutAppManager::AbstractFactory::AbstractFactory
 */
GlutAppManager::AbstractFactory::AbstractFactory(const char *name) 
        : name(name) {
}


/*
 * GlutAppManager::GlutAppManager
 */
GlutAppManager::GlutAppManager(void) 
        : app(NULL), factories(), windowMenu(0), appMenu(0) {
}


/*
 * GlutAppManager::~GlutAppManager
 */
GlutAppManager::~GlutAppManager(void) {
    if (app) {
        app->GLDeinit();
        delete app;
    }

    for (int i = int(this->factories.Count()) - 1; i >= 0; i--) {
        AbstractFactory *f = this->factories[i];
        this->factories[i] = NULL;
        delete f;
    }
    this->factories.Clear();

    if (this->windowMenu != 0) {
        glutDetachMenu(this->windowMenu);
        glutDestroyMenu(this->windowMenu);
        this->windowMenu = 0;
    }
    if (this->appMenu != 0) {
        glutDestroyMenu(this->appMenu);
        this->appMenu = 0;
    }
}


/*
 * GlutAppManager::GetInstance
 */
GlutAppManager * GlutAppManager::GetInstance(void) {
    static GlutAppManager instance;
    return &instance;
}


/*
 * GlutAppManager::InstallFactory
 */
void GlutAppManager::InstallFactory(AbstractFactory *factory) {
    if (factory == NULL) return;

    GlutAppManager::GetInstance()->factories.Append(factory);
}


/*
 * GlutAppManager::InitGlutWindow
 */
void GlutAppManager::InitGlutWindow(void) {
    if (this->windowMenu != 0) {
        return;
    }

    this->appMenu = glutCreateMenu(GlutAppManager::OnMenuItemClicked);
    vislib::StringA name;
    for (int i = 0; i < int(this->factories.Count()); i++) {
        if (this->factories[i] != NULL) {
            name.Format("%d: %s", (i + 1), this->factories[i]->GetName());
            glutAddMenuEntry(name.PeekBuffer(), i + 1);
        }
    }

    this->windowMenu = glutCreateMenu(GlutAppManager::OnMenuItemClicked);

    glutAddSubMenu("Select Test", this->appMenu);
#if defined(VISGLUT_EXTENSIONS)
    ::glutAddMenuSeparator();
#endif /* VISGLUT_EXTENSIONS */
    glutAddMenuEntry("Restart Test", -2);
    glutAddMenuEntry("Exit", -1);

    glutAttachMenu(GLUT_RIGHT_BUTTON);
}


/*
 * GlutAppManager::OnMenuItemClicked
 */
void GlutAppManager::OnMenuItemClicked(int menuID) {
    if (menuID == -1) {
        GlutAppManager::ExitApplication(0);
    } else if (menuID == -2) {
        GlutAppManager *This = GlutAppManager::GetInstance();
        if (This->app) {
            This->app->GLDeinit();
            if (This->app->GLInit() == 0) {
                // TODO: initializes the glut stuff
                This->app->OnResize(This->width, This->height);

                printf("Test restarted.\n");

                fpsCounter.Reset();

            } else {
                delete This->app;
                This->app = NULL;
                printf("Test could not be restarted.\n");
            }
        }
    } else if ((menuID > 0) && (menuID <= int(GlutAppManager::GetInstance()->factories.Count()))) {
        GlutAppManager *This = GlutAppManager::GetInstance();
        // select an test application factory
        printf("Selecting Test: %s\n", This->factories[menuID - 1]->GetName());
        if (This->app) {
            if (This->factories[menuID - 1]->HasCreated(This->app)) {
                printf("  Test already selected.\n");
            } else {
                This->app->GLDeinit();
                delete This->app;
                This->app = NULL;
            }
        }
        if (!This->app) {
            This->app = This->factories[menuID - 1]->CreateApplication();
            if (This->app) {
                if (This->app->GLInit() == 0) {
                    // TODO: initializes the glut stuff
                    This->app->OnResize(This->width, This->height);

                    printf("  Test selected.\n");

                    fpsCounter.Reset();

                } else {
                    delete This->app;
                    This->app = NULL;
                    printf("  Test could not be initialized.\n");
                }

            } else {
                printf("  Test could not be created.\n");
            }
        }

        glutPostRedisplay();
    }
}


/*
 * GlutAppManager::ExitApplication
 */
void GlutAppManager::ExitApplication(int exitcode) {
    exit(exitcode);
}


/*
 * GlutAppManager::SetSize
 */
void GlutAppManager::SetSize(int w, int h) {
    this->width = w;
    this->height = h;
    if (this->app == NULL) {
        glViewport(0, 0, this->width, this->height);
    }
}


/*
 * glprintf
 */
static void glprintf(float x, float y, const void *font, const char *string) {
    glRasterPos2f(x, y);
    while (*string) {
        glutBitmapCharacter((void *)font, *string++);
    }
}

/*
 * GlutAppManager::glRenderEmptyScreen
 */
void GlutAppManager::glRenderEmptyScreen(void) {
    GlutAppManager *This = GlutAppManager::GetInstance();
    glViewport(0, 0, This->width, This->height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);

    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-0.5, This->width, -0.5, This->height, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(0.7f, 0.8f, 1.0f);

    glprintf(10.0f, float(This->height - 28), GLUT_BITMAP_HELVETICA_18,
        "VISlib glutTest Application");
    glprintf(10.0f, float(This->height - 44), GLUT_BITMAP_HELVETICA_12,
        "Copyright  2007, Universität Stuttgart (VIS). Alle Rechte vorbehalten.");
    vislib::StringA txt;
    txt.Format("OpenGL Version: %s", 
        vislib::graphics::gl::GLVersion().ToStringA(3).PeekBuffer());
    glprintf(10.0f, float(This->height - 60), GLUT_BITMAP_HELVETICA_12, 
        txt.PeekBuffer());
    glprintf(10.0f, float(This->height - 76), GLUT_BITMAP_HELVETICA_12,
        "Use the right click context menu to select a test.");

    glFlush();

    glutSwapBuffers();
}
