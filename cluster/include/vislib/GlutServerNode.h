/*
 * GlutServerNode.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLUTSERVERNODE_H_INCLUDED
#define VISLIB_GLUTSERVERNODE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#if defined(VISLIB_CLUSTER_WITH_OPENGL) && (VISLIB_CLUSTER_WITH_OPENGL != 0)

#include "vislib/AbstractControllerNode.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/CameraRotate2DLookAt.h"
#include "vislib/Cursor2D.h"
#include "vislib/GlutClusterNode.h"
#include "vislib/InputModifiers.h"
#include "vislib/ServerNodeAdapter.h"


namespace vislib {
namespace net {
namespace cluster {


#ifdef _WIN32
#pragma warning(disable: 4250)  // I know what I am doing ...
#endif /* _WIN32 */

    /**
     * This class provides all necessary parts to create a GLUT server node that
     * reports camera parameters to client nodes.
     *
     * Applications should inherit from this class and use the 'camera' member 
     * as their camera. They must only implement the drawing and interaction 
     * logic, the communication and resolution of ambiguities is done within 
     * this class. However, this only works for a limited set of 
     * rapid-prototyping as you may not have sufficient control over the system
     * functionality.
     */
    template<class T> class GlutServerNode 
            : public GlutClusterNode<T>, public ServerNodeAdapter,
            public AbstractControllerNode {

    public:

        /** Dtor. */
        ~GlutServerNode(void);

        virtual void Initialise(sys::CmdLineProviderA& inOutCmdLine);

        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine);

        virtual DWORD Run(void);

    protected:

        /** Ctor. */
        GlutServerNode(void);

        virtual void initialiseController(
            graphics::AbstractCameraController *& inOutController);

        virtual void initialiseCursor(graphics::Cursor2D& inOutCursor);

        virtual void initialiseInputModifiers(
            graphics::InputModifiers& inOutInputModifiers);

        /**
         * The method is called when a mouse button is pressed or released.
         *
         * The implementation in GlutServerNode updates the VISlib cursor that
         * controls the camera.
         *
         * @param button The button that has been pressed or released.
         * @param state  The new button state.
         * @param x      The mouse x-coordinates where the event occurred.
         * @param y      The mouse y-coordinates where the event occurred.
         */
        virtual void onMouseButton(const int button, const int state,
            const int x, const int y);

        /**
         * This method is called when the mouse is moved.
         *
         * The implementation in GlutServerNode updates the VISlib cursor that
         * controls the camera.
         *
         * @param x The new x-coordinates of the mouse cursor.
         * @param y The new y-coordiantes of the mouse cursor.
         */
        virtual void onMouseMove(const int x, const int y);

        /**
         * This method is called when a special key was pressed.
         *
         * The implementation in GlutServerNode updates the input modifiers for
         * the camera control.
         *
         * @param key The virtual key code of the special key that was pressed.
         * @param x   The cursor x-coordinates at the time when the key was 
         *            pressed.
         * @param y   The cursor y-coordinates at the time when the key was
         *            pressed.
         */
        virtual void onSpecialKeyDown(const int key, const int x, const int y);

        /**
         * Update the cursor position.
         *
         * @param x The new x-value.
         * @param y The new y-value.
         */
        inline void updateCursorPosition(const int x, const int y) {
            this->cursor.SetPosition(
                static_cast<vislib::graphics::ImageSpaceType>(x),
                static_cast<vislib::graphics::ImageSpaceType>(y), true);
        }

        /**
         * Updates the 'inputModifiers' state from the current GLUT state.
         */
        void updateInputModifiers(void);

        /** 
         * The camera subclasses should use in order to synchronise its 
         * parameters to client nodes.
         */
        graphics::gl::CameraOpenGL camera;

    private:

        /** The controller that manipulates the camera. */
        graphics::AbstractCameraController *controller;

        /** The 2D cursor that manipulates the camera. */
        graphics::Cursor2D cursor;

        /** The input modifiers for 'cursor'. */
        graphics::InputModifiers inputModifiers;

    };
#ifdef _WIN32
#pragma warning(default: 4250)
#endif /* _WIN32 */


    /*
     * vislib::net::cluster::GlutServerNode<T>::~GlutServerNode
     */
    template<class T> GlutServerNode<T>::~GlutServerNode(void) {
        SAFE_DELETE(this->controller);
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Initialise
     */
    template<class T> 
    void GlutServerNode<T>::Initialise(sys::CmdLineProviderA& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        ServerNodeAdapter::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Initialise
     */
    template<class T> 
    void GlutServerNode<T>::Initialise(sys::CmdLineProviderW& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        ServerNodeAdapter::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Run
     */
    template<class T> DWORD GlutServerNode<T>::Run(void) {
        ServerNodeAdapter::Run();           // First, start the server.
        return GlutClusterNode<T>::Run();   // Afterwards, enter message loop.
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::GlutServerNode
     */
    template<class T> GlutServerNode<T>::GlutServerNode(void) 
            : GlutClusterNode<T>(), ServerNodeAdapter(), 
            AbstractControllerNode(
            new vislib::graphics::ObservableCameraParams()),
            controller(NULL) {
        this->camera.SetParameters(this->getParameters());
        this->initialiseCursor(this->cursor);
        this->initialiseInputModifiers(this->inputModifiers);
        this->initialiseController(this->controller);
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::initialiseController
     */
    template<class T> void GlutServerNode<T>::initialiseController(
            graphics::AbstractCameraController *& inOutController) {
        graphics::CameraRotate2DLookAt *ctrl = NULL;

        /* Clean up possible old controller. */
        if (inOutController != NULL) {
            BECAUSE_I_KNOW(dynamic_cast<graphics::AbstractCursorEvent *>(
                inOutController) != NULL);
            this->cursor.UnregisterCursorEvent(dynamic_cast<
                graphics::AbstractCursorEvent *>(inOutController));
        }
        SAFE_DELETE(inOutController);

        /* Create new controller. */
        inOutController = ctrl = new graphics::CameraRotate2DLookAt(
            this->camera.Parameters());
        ctrl->SetTestButton(0);          // left
        ctrl->SetModifierTestCount(0);
        ctrl->SetAltModifier(graphics::InputModifiers::MODIFIER_SHIFT);

        /* Reset the view. */
        this->camera.Parameters()->SetView(
            math::Point<double, 3>(0.0, -1.0, 0.0),
            math::Point<double, 3>(0.0, 0.0, 0.0),
            math::Vector<double, 3>(0.0, 0.0, 1.0));

        /* Connect controller with cursor device. */
        this->cursor.RegisterCursorEvent(ctrl);
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::initialiseCursor
     */
    template<class T> 
    void GlutServerNode<T>::initialiseCursor(graphics::Cursor2D& inOutCursor) {
        this->cursor.SetButtonCount(3);
        this->cursor.SetInputModifiers(&this->inputModifiers);
        this->cursor.SetCameraParams(this->camera.Parameters());
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::initialiseInputModifiers
     */
    template<class T> void GlutServerNode<T>::initialiseInputModifiers(
            graphics::InputModifiers& inOutInputModifiers) {
        this->inputModifiers.SetModifierCount(3);
        this->inputModifiers.RegisterObserver(&this->cursor);
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::onMouseButton
     */
    template<class T> void GlutServerNode<T>::onMouseButton(const int button,
            const int state, const int x, const int y) {
        unsigned int btn = 0;

        switch (button) {
            case GLUT_LEFT_BUTTON: btn = 0; break;
            case GLUT_RIGHT_BUTTON: btn = 1; break;
            case GLUT_MIDDLE_BUTTON: btn = 2; break;
        }
        this->cursor.SetButtonState(btn, (state == GLUT_DOWN));

        this->updateInputModifiers();
        this->updateCursorPosition(x, y);
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::onMouseMove
     */
    template<class T> 
    void GlutServerNode<T>::onMouseMove(const int x, const int y) {
        this->updateCursorPosition(x, y);
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::onSpecialKeyDown
     */
    template<class T> void GlutServerNode<T>::onSpecialKeyDown(const int key,
            const int x, const int y) {
        this->updateInputModifiers();
        this->updateCursorPosition(x, y);
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::updateInputModifiers
     */
    template<class T> void GlutServerNode<T>::updateInputModifiers(void) {
        int modifiers = ::glutGetModifiers();

        this->inputModifiers.SetModifierState(
            graphics::InputModifiers::MODIFIER_SHIFT,
            (modifiers & GLUT_ACTIVE_SHIFT) != 0);
        this->inputModifiers.SetModifierState(
            graphics::InputModifiers::MODIFIER_CTRL,
            (modifiers & GLUT_ACTIVE_CTRL) != 0);
        this->inputModifiers.SetModifierState(
            graphics::InputModifiers::MODIFIER_ALT,
            (modifiers & GLUT_ACTIVE_ALT) != 0);
    }

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#endif /* defined(VISLIB_CLUSTER_WITH_OPENGL) ... */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLUTSERVERNODE_H_INCLUDED */
