/*
 * Viewer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Window.h"
#include "visglut.h"
#include "vislib/glfunctions.h"
#include "vislib/KeyCode.h"
#include "vislib/Log.h"
#include <GL/gl.h>


/*
 * megamol::viewer::Window::Window
 */
megamol::viewer::Window::Window(megamol::viewer::Viewer& owner)
        : ApiHandle(), glutID(0), owner(owner), left(0), top(0), width(1),
        height(1), isFullscreen(false), renderCallback(), resizeCallback(),
        keyCallback(), mouseButtonCallback(), mouseMoveCallback(),
        closeCallback(), commandCallback(), updateFreeze(), glutMainMenu(0),
        glutSizeMenu(0), glutCommandMenu(0), modifiers(0),
        presentationMode(0) {

    Window *fw = this->owner.FirstWindow();
    if (fw != NULL) {
        ::glutSetWindow(fw->glutID);
        ::glutShareContextWithNextWindow();
    }

    this->glutID = ::glutCreateWindow("MegaMol");
    ::glutSetWindowData(static_cast<void*>(this));
    ::glutDisplayFunc(megamol::viewer::Window::glutDisplayCallback);
    ::glutReshapeFunc(megamol::viewer::Window::glutReshapeCallback);
    ::glutKeyboardFunc(megamol::viewer::Window::glutKeyboardCallback);
    ::glutSpecialFunc(megamol::viewer::Window::glutSpecialCallback);
    ::glutMouseFunc(megamol::viewer::Window::glutMouseCallback);
    ::glutMotionFunc(megamol::viewer::Window::glutMotionCallback);
    ::glutPassiveMotionFunc(megamol::viewer::Window::glutMotionCallback);
    ::glutCloseFunc(megamol::viewer::Window::glutCloseCallback);

#ifdef FREEGLUT_WITH_VIS_FUNCTIONS
#ifdef _WIN32
    ::glutSetWindowIconI(200);
#else /* _WIN32 */
    // ::glutSetWindowIcon(NULL); // TODO: Implement
#endif /* _WIN32 */
#endif /* FREEGLUT_WITH_VIS_FUNCTIONS */

    ::glutShowWindow();
}


/*
 * megamol::viewer::Window::~Window
 */
megamol::viewer::Window::~Window(void) {
    this->renderCallback.Clear();
    this->resizeCallback.Clear();
    this->keyCallback.Clear();
    this->mouseButtonCallback.Clear();
    this->mouseMoveCallback.Clear();
    this->closeCallback.Clear();
    this->commandCallback.Clear();
    this->updateFreeze.Clear();
    this->owner.UnownWindow(this);
    if (this->glutID != 0) {
        int oldWin = ::glutGetWindow();
        int id = this->glutID;
        this->glutID = 0;
        ::glutSetWindow(id);
        ::glutSetWindowData(NULL);
        ::glutHideWindow();
        ::glutDestroyWindow(id);
        if (oldWin != 0) ::glutSetWindow(oldWin);
    }
}


/*
 * megamol::viewer::Window::Close
 */
void megamol::viewer::Window::Close(void) {
    if (this->glutID != 0) {
        ::glutDestroyWindow(this->glutID);
    }
}


/*
 * megamol::viewer::Window::InstallContextMenu
 */
void megamol::viewer::Window::InstallContextMenu(void) {
    if (this->glutMainMenu != 0) return;
    if (this->glutID == 0) return;

    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);

    // The window size sub menu
    this->glutSizeMenu = ::glutCreateMenu(glutSizeMenuCallback);
    ::glutAddMenuEntry("", 1);
    this->updateFullscreenMenuItem();
    ::glutAddMenuSeparator();
    ::glutAddMenuEntry("256 x 256", 2);
    ::glutAddMenuEntry("512 x 512", 3);
    ::glutAddMenuEntry("640 x 480", 4);
    ::glutAddMenuEntry("768 x 768", 5);
    ::glutAddMenuEntry("800 x 600", 6);
    ::glutAddMenuEntry("1024 x 768", 7);
    ::glutAddMenuEntry("1024 x 1024", 8);

    // The main context menu
    this->glutMainMenu = ::glutCreateMenu(glutMainMenuCallback);
    ::glutAddSubMenu("Window Size", this->glutSizeMenu);
    ::glutAddMenuSeparator();
    ::glutAddMenuEntry("Close", 1);
    ::glutAddMenuEntry("Exit", 2);

    ::glutAttachMenu(GLUT_RIGHT_BUTTON);
    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::ResizeWindow
 */
void megamol::viewer::Window::ResizeWindow(
        unsigned int width, unsigned int height) {
    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);

    if (this->isFullscreen) {
        this->toogleFullscreen();
    }
    ::glutReshapeWindow(width, height);

    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::MoveWindowTo
 */
void megamol::viewer::Window::MoveWindowTo(int x, int y) {
    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);

    if (this->isFullscreen) {
        this->toogleFullscreen();
    }
    ::glutPositionWindow(x, y);

    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::AddCommand
 */
void megamol::viewer::Window::AddCommand(const char *caption, int value) {
    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);

    if (this->glutMainMenu == 0) return;
    int om = ::glutGetMenu();
    if (this->glutCommandMenu == 0) {
        this->glutCommandMenu = ::glutCreateMenu(glutCommandMenuCallback);
        ::glutSetMenu(this->glutMainMenu);

        ::glutRemoveMenuItem(::glutGet(GLUT_MENU_NUM_ITEMS));
        ::glutRemoveMenuItem(::glutGet(GLUT_MENU_NUM_ITEMS));

        ::glutAddSubMenu("Commands", this->glutCommandMenu);
        ::glutAddMenuSeparator();
        ::glutAddMenuEntry("Close", 1);
        ::glutAddMenuEntry("Exit", 2);

    }
    ::glutSetMenu(this->glutCommandMenu);
    ::glutAddMenuEntry(caption, value);
    if (om != 0) {
        ::glutSetMenu(om);
    }

    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::SetTitle
 */
void megamol::viewer::Window::SetTitle(const char *title) {
    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);

    vislib::StringA t("MegaMol");
    {
        vislib::StringW tm(0x2122, 1);
        vislib::StringA tmA(tm);
        if (vislib::StringW(tmA).Equals(tm)) {
            t.Append(tmA);
        } else {
            t.Append("(TM)");
        }
    }
    if ((title != NULL) && (title[0] != '\0')) {
        t.Append(" - ");
        t.Append(title);
    }

    ::glutSetWindowTitle(t.PeekBuffer());
    ::glutSetIconTitle(t.PeekBuffer());

    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::SetTitle
 */
void megamol::viewer::Window::SetTitle(const wchar_t *title) {
    this->SetTitle(vislib::StringA(title));
}


/*
 * megamol::viewer::Window::SetFullscreen
 */
void megamol::viewer::Window::SetFullscreen(void) {
    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);

    if (!this->isFullscreen) {
        this->toogleFullscreen();
    }

    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::ShowDecorations
 */
void megamol::viewer::Window::ShowDecorations(bool dec) {
    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);
    ::glutShowWindowDecorations(dec ? 1 : 0);
    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::SetCursorVisibility
 */
void megamol::viewer::Window::SetCursorVisibility(bool visible) {
    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);
    ::glutSetCursor(visible ? GLUT_CURSOR_INHERIT : GLUT_CURSOR_NONE);
    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::StayOnTop
 */
void megamol::viewer::Window::StayOnTop(bool stay) {
    int oldWin = ::glutGetWindow();
    ::glutSetWindow(this->glutID);
    ::glutWindowStayOnTop(stay ? 1 : 0);
    if (oldWin != 0) ::glutSetWindow(oldWin);
}


/*
 * megamol::viewer::Window::thisWindow
 */
megamol::viewer::Window* megamol::viewer::Window::thisWindow(void) {
    return static_cast<Window*>(::glutGetWindowData());
}


/*
 * megamol::viewer::Window::glutDisplayCallback
 */
void megamol::viewer::Window::glutDisplayCallback(void) {
    try {

        Window *t = thisWindow();
        if (t == NULL) return;
        bool contRedraw = true;

        t->renderCallback.Call(*thisWindow(), &contRedraw);

        if (t->presentationMode > 0) {
            t->presentationMode = 2;
            t->owner.PresentationModeSwap();

        } else {
            ::glutSwapBuffers();
            if (contRedraw) {
                ::glutPostRedisplay();
            }
        }

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutReshapeCallback
 */
void megamol::viewer::Window::glutReshapeCallback(int w, int h) {
    try {

        if (thisWindow() == NULL) return;
        if (w <= 0) w = 1;
        if (h <= 0) h = 1;
        if (!thisWindow()->isFullscreen) {
            thisWindow()->width = w;
            thisWindow()->height = h;
        }
        ::glViewport(0, 0, w, h);
        unsigned int size[2] = { w, h };
        thisWindow()->resizeCallback.Call(*thisWindow(), size);

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutKeyboardCallback
 */
void megamol::viewer::Window::glutKeyboardCallback(unsigned char key,
        int x, int y) {
    try {

        if (thisWindow() == NULL) return;
        mmvKeyParamsStruct params;

        vislib::sys::KeyCode keycode;
        switch (key) {
            case 8: keycode = vislib::sys::KeyCode::KEY_BACKSPACE; break;
            case 9: keycode = vislib::sys::KeyCode::KEY_TAB; break;
            case 13: keycode = vislib::sys::KeyCode::KEY_ENTER; break;
            case 27: keycode = vislib::sys::KeyCode::KEY_ESC; break;
            case 127: keycode = vislib::sys::KeyCode::KEY_DELETE; break;
            default: keycode = key; break;
        }

        params.mouseX = x;
        params.mouseY = y;
        
        thisWindow()->modifiers = ::glutGetModifiers();
        params.modShift = (thisWindow()->modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT;
        params.modCtrl = (thisWindow()->modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL;
        params.modAlt = (thisWindow()->modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT;

        if (params.modShift) keycode = vislib::sys::KeyCode::KEY_MOD_SHIFT | keycode;
        if (params.modCtrl) keycode = vislib::sys::KeyCode::KEY_MOD_CTRL | keycode;
        if (params.modAlt) keycode = vislib::sys::KeyCode::KEY_MOD_ALT | keycode;

        if (params.modCtrl && (key < ' ')) {
            keycode = vislib::sys::KeyCode::KEY_MOD_CTRL | ('a' - 1 + key);
            if (params.modShift) keycode = vislib::sys::KeyCode::KEY_MOD_SHIFT | keycode;
            if (params.modAlt) keycode = vislib::sys::KeyCode::KEY_MOD_ALT | keycode;
        }

        params.keycode = keycode;

        thisWindow()->keyCallback.Call(*thisWindow(), &params);

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutSpecialCallback
 */
void megamol::viewer::Window::glutSpecialCallback(int key, int x, int y) {
    try {

        if (thisWindow() == NULL) return;
        mmvKeyParams params;

        vislib::sys::KeyCode keycode;

        params.mouseX = x;
        params.mouseY = y;

        thisWindow()->modifiers = ::glutGetModifiers();
        params.modShift = (thisWindow()->modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT;
        params.modCtrl = (thisWindow()->modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL;
        params.modAlt = (thisWindow()->modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT;

        switch (key) {
            case GLUT_KEY_F1:
                keycode = vislib::sys::KeyCode::KEY_F1;
                break;
            case GLUT_KEY_F2:
                keycode = vislib::sys::KeyCode::KEY_F2;
                break;
            case GLUT_KEY_F3:
                keycode = vislib::sys::KeyCode::KEY_F3;
                break;
            case GLUT_KEY_F4:
                keycode = vislib::sys::KeyCode::KEY_F4;
                break;
            case GLUT_KEY_F5:
                keycode = vislib::sys::KeyCode::KEY_F5;
                break;
            case GLUT_KEY_F6:
                keycode = vislib::sys::KeyCode::KEY_F6;
                break;
            case GLUT_KEY_F7:
                keycode = vislib::sys::KeyCode::KEY_F7;
                break;
            case GLUT_KEY_F8:
                keycode = vislib::sys::KeyCode::KEY_F8;
                break;
            case GLUT_KEY_F9:
                keycode = vislib::sys::KeyCode::KEY_F9;
                break;
            case GLUT_KEY_F10:
                keycode = vislib::sys::KeyCode::KEY_F10;
                break;
            case GLUT_KEY_F11:
                keycode = vislib::sys::KeyCode::KEY_F11;
                break;
            case GLUT_KEY_F12:
                keycode = vislib::sys::KeyCode::KEY_F12;
                break;
            case GLUT_KEY_LEFT:
                keycode = vislib::sys::KeyCode::KEY_LEFT;
                break;
            case GLUT_KEY_UP:
                keycode = vislib::sys::KeyCode::KEY_UP;
                break;
            case GLUT_KEY_RIGHT:
                keycode = vislib::sys::KeyCode::KEY_RIGHT;
                break;
            case GLUT_KEY_DOWN:
                keycode = vislib::sys::KeyCode::KEY_DOWN;
                break;
            case GLUT_KEY_PAGE_UP:
                keycode = vislib::sys::KeyCode::KEY_PAGE_UP;
                break;
            case GLUT_KEY_PAGE_DOWN:
                keycode = vislib::sys::KeyCode::KEY_PAGE_DOWN;
                break;
            case GLUT_KEY_HOME:
                keycode = vislib::sys::KeyCode::KEY_HOME;
                break;
            case GLUT_KEY_END:
                keycode = vislib::sys::KeyCode::KEY_END;
                break;
            case GLUT_KEY_INSERT:
                keycode = vislib::sys::KeyCode::KEY_INSERT;
                break;
            default:
                fprintf(stderr, "Glut special key %d not handled.\n", key);
                break;
        }

        if (params.modShift) keycode = vislib::sys::KeyCode::KEY_MOD_SHIFT | keycode;
        if (params.modCtrl) keycode = vislib::sys::KeyCode::KEY_MOD_CTRL | keycode;
        if (params.modAlt) keycode = vislib::sys::KeyCode::KEY_MOD_ALT | keycode;

        if (params.modCtrl && (key < ' ')) {
            keycode = vislib::sys::KeyCode::KEY_MOD_CTRL | ('a' - 1 + key);
            if (params.modShift) keycode = vislib::sys::KeyCode::KEY_MOD_SHIFT | keycode;
            if (params.modAlt) keycode = vislib::sys::KeyCode::KEY_MOD_ALT | keycode;
        }

        params.keycode = keycode;

        thisWindow()->keyCallback.Call(*thisWindow(), &params);

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutMouseCallback
 */
void megamol::viewer::Window::glutMouseCallback(int button, int state,
        int x, int y) {
    try {

        if (thisWindow() == NULL) return;
        mmvMouseButtonParams params;
        switch (button) {
            case GLUT_LEFT_BUTTON: params.button = 0; break;
            case GLUT_RIGHT_BUTTON: params.button = 1; break;
            case GLUT_MIDDLE_BUTTON: params.button = 2; break;
            // case 3: TODO: mouse wheel up
            // case 4: TODO: mouse wheel down
            default: return;
        }
        params.buttonDown = state == GLUT_DOWN;
        params.mouseX = x;
        params.mouseY = y;
        thisWindow()->modifiers = ::glutGetModifiers();
        params.modShift = (thisWindow()->modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT;
        params.modCtrl = (thisWindow()->modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL;
        params.modAlt = (thisWindow()->modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT;
        thisWindow()->mouseButtonCallback.Call(*thisWindow(), &params);

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutMotionCallback
 */
void megamol::viewer::Window::glutMotionCallback(int x, int y) {
    try {

        if (thisWindow() == NULL) return;
        mmvMouseMoveParams params;
        params.mouseX = x;
        params.mouseY = y;
        params.modShift = (thisWindow()->modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT;
        params.modCtrl = (thisWindow()->modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL;
        params.modAlt = (thisWindow()->modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT;
        thisWindow()->mouseMoveCallback.Call(*thisWindow(), &params);

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutCloseCallback
 */
void megamol::viewer::Window::glutCloseCallback(void) {
    try {

        if (thisWindow() == NULL) return;
        // remove window from the list of active windows,
        // but keep the handles valid!
        if (thisWindow()->glutID != 0) {
            thisWindow()->closeCallback.Call(*thisWindow(), NULL);
            thisWindow()->owner.UnownWindow(thisWindow());
            thisWindow()->glutID = 0;
        }

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutMainMenuCallback
 */
void megamol::viewer::Window::glutMainMenuCallback(int item) {
    try {

        switch (item) {
            case 0: break; // none or separator
            case 1: // close
                ::glutDestroyWindow(thisWindow()->glutID);
                break;
            case 2: { // exit
                Viewer &viewer = thisWindow()->owner;
                viewer.CloseAllWindows();
                viewer.RequestAppTermination();
            } break;
            default: break; // unknown
        }

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutSizeMenuCallback
 */
void megamol::viewer::Window::glutSizeMenuCallback(int item) {
    try {

        switch (item) {
            case 0: break; // none or separator
            case 1: thisWindow()->toogleFullscreen(); break;
            case 2: thisWindow()->ResizeWindow(256, 256); break;
            case 3: thisWindow()->ResizeWindow(512, 512); break;
            case 4: thisWindow()->ResizeWindow(640, 480); break;
            case 5: thisWindow()->ResizeWindow(768, 768); break;
            case 6: thisWindow()->ResizeWindow(800, 600); break;
            case 7: thisWindow()->ResizeWindow(1024, 768); break;
            case 8: thisWindow()->ResizeWindow(1024, 1024); break;
            default: break; // unknown
        }

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::glutCommandMenuCallback
 */
void megamol::viewer::Window::glutCommandMenuCallback(int item) {
    try {

        thisWindow()->commandCallback.Call(*thisWindow(), &item);

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: %s [%s, %d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unhandled exception: native exception");
    }
}


/*
 * megamol::viewer::Window::updateFullscreenMenuItem
 */
void megamol::viewer::Window::updateFullscreenMenuItem(void) const {
    if (this->glutSizeMenu != 0) {
        int oldMenu = ::glutGetMenu();
        ::glutSetMenu(this->glutSizeMenu);
        ::glutChangeToMenuEntry(1, this->isFullscreen 
            ? "Window Mode" : "Fullscreen Mode", 1);
        ::glutSetMenu(oldMenu);
    }
}


/*
 * megamol::viewer::Window::toogleFullscreen
 */
void megamol::viewer::Window::toogleFullscreen(void) {
    if (this->isFullscreen) {
#ifdef FREEGLUT_WITH_VIS_FUNCTIONS
        ::glutClientPositionWindow(this->left, this->top); 
#else /* FREEGLUT_WITH_VIS_FUNCTIONS */
        ::glutPositionWindow(this->left, this->top); 
#endif /* FREEGLUT_WITH_VIS_FUNCTIONS */
        ::glutReshapeWindow(this->width, this->height);
        this->isFullscreen = false;
    } else {
        this->left = ::glutGet(GLUT_WINDOW_X);
        this->top = ::glutGet(GLUT_WINDOW_Y);
        this->width = ::glutGet(GLUT_WINDOW_WIDTH);
        this->height = ::glutGet(GLUT_WINDOW_HEIGHT);
        this->isFullscreen = true;
        ::glutFullScreen();
    }
    this->updateFullscreenMenuItem();
}


/*
 * megamol::viewer::Window::SetPresentationMode
 */
void megamol::viewer::Window::SetPresentationMode(bool presentation) {
    if (presentation) {
        int freeze = 1;
        this->updateFreeze.Call(*this, &freeze);
        this->presentationMode = 1;
    } else {
        int unfreeze = 0;
        this->presentationMode = 0;
        this->updateFreeze.Call(*this, &unfreeze);
    }
}


/*
 * megamol::viewer::Window::PresentationModeSwap
 */
void megamol::viewer::Window::PresentationModeSwap(void) {
    ::glutSetWindow(this->glutID);
    ::glutSwapBuffers();
}


/*
 * megamol::viewer::Window::PresentationModeUpdate
 */
void megamol::viewer::Window::PresentationModeUpdate(void) {
    int update = 1;
    this->updateFreeze.Call(*this, &update);
}


/*
 * megamol::viewer::Window::PresentationModeRefresh
 */
void megamol::viewer::Window::PresentationModeRefresh(void) {
    ::glutSetWindow(this->glutID);
    ::glutPostRedisplay();
    this->presentationMode = 1;
}


/*
 * megamol::viewer::Window::SetVSync
 */
void megamol::viewer::Window::SetVSync(bool active) {
    vislib::graphics::gl::EnableVSync(active);
}


/*
 * megamol::viewer::Window::ShowParameterGUI
 */
void megamol::viewer::Window::ShowParameterGUI(bool show) {

    // TODO: Implement

}
