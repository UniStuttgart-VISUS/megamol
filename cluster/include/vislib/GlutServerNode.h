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
#include "vislib/AbstractServerNode.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/CameraRotate2DLookAt.h"
#include "vislib/CameraZoom2DMove.h"
#include "vislib/Cursor2D.h"
#include "vislib/GlutClusterNode.h"
#include "vislib/InputModifiers.h"


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
            : public GlutClusterNode<T>, public AbstractServerNode,
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

        /**
         * This method creates and initialises the controllers. The controllers
         * are allocated using new and stored to the in-out-parameters.
         *
         * The method deletes any exisiting controller if the in-out-parameters
         * are not NULL.
         *
         * Subclasses have the following two possibilites:
         * 1. Call this implementation and possibly modify the controllers 
         *    afterwards. The 'inOutRotateController' created by this method is
         *    a CameraRotate2DLookAt, the 'inOutZoomController' a 
         *    CameraZoom2DMove.
         * 2. Completely overwrite the method. In this case, the implementation
         *    must perform the following steps: (i) Check, whether the pointers
         *    already designate an object and optionally unregister there from
         *    this->camera and then delete it using delete, (ii) create new 
         *    controller instances using new and store their pointers to the 
         *    in-out-parameters, and (iii) finally register the events to the
         *    camera this->camera.
         *
         * @param inOutRotateController
         * @param inOutZoomController
         */
        virtual void initialiseController(
            graphics::AbstractCameraController *& inOutRotateController,
            graphics::AbstractCameraController *& inOutZoomController);

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
         * This method is called when the window was resized.
         *
         * This implementation in GlutServerNode calls the parent method and 
         * updates the virtual view dimensions of the camera.
         *
         * @param width  The new width of the window.
         * @param height The new height of the window.
         */
        virtual void onResize(const int width, const int height);

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

        /** The 2D cursor that manipulates the camera. */
        graphics::Cursor2D cursor;

        /** The input modifiers for 'cursor'. */
        graphics::InputModifiers inputModifiers;

        /** The controller that manipulates the rotation of the camera. */
        graphics::AbstractCameraController *rotateController;

        /** The controller that manipulates the aperture of the camera. */
        graphics::AbstractCameraController *zoomController;

    };
#ifdef _WIN32
#pragma warning(default: 4250)
#endif /* _WIN32 */


    /*
     * vislib::net::cluster::GlutServerNode<T>::~GlutServerNode
     */
    template<class T> GlutServerNode<T>::~GlutServerNode(void) {
        SAFE_DELETE(this->rotateController);
        SAFE_DELETE(this->zoomController);
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Initialise
     */
    template<class T> 
    void GlutServerNode<T>::Initialise(sys::CmdLineProviderA& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        AbstractServerNode::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Initialise
     */
    template<class T> 
    void GlutServerNode<T>::Initialise(sys::CmdLineProviderW& inOutCmdLine) {
        GlutClusterNode<T>::Initialise(inOutCmdLine);
        AbstractServerNode::Initialise(inOutCmdLine);
    }


    /*
     *  vislib::net::cluster::GlutServerNode<T>::Run
     */
    template<class T> DWORD GlutServerNode<T>::Run(void) {
        AbstractServerNode::Run();          // First, start the server.
        return GlutClusterNode<T>::Run();   // Afterwards, enter message loop.
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::GlutServerNode
     */
    template<class T> GlutServerNode<T>::GlutServerNode(void) 
            : GlutClusterNode<T>(), AbstractServerNode(),
            AbstractControllerNode(new graphics::ObservableCameraParams()),
            rotateController(NULL), zoomController(NULL) {
        this->camera.SetParameters(this->getParameters());
        this->initialiseCursor(this->cursor);
        this->initialiseInputModifiers(this->inputModifiers);
        this->initialiseController(this->rotateController,
            this->zoomController);
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::initialiseController
     */
    template<class T> void GlutServerNode<T>::initialiseController(
            graphics::AbstractCameraController *& inOutRotateController,
            graphics::AbstractCameraController *& inOutZoomController) {
        graphics::CameraRotate2DLookAt *rotCtrl = NULL;
        graphics::CameraZoom2DMove *zoomCtrl = NULL;

        /* Clean up possible old controllers. */
        if (inOutRotateController != NULL) {
            BECAUSE_I_KNOW(dynamic_cast<graphics::AbstractCursorEvent *>(
                inOutRotateController) != NULL);
            this->cursor.UnregisterCursorEvent(dynamic_cast<
                graphics::AbstractCursorEvent *>(inOutRotateController));
        }
        SAFE_DELETE(inOutRotateController);

        if (inOutZoomController != NULL) {
            BECAUSE_I_KNOW(dynamic_cast<graphics::AbstractCursorEvent *>(
                inOutZoomController) != NULL);
            this->cursor.UnregisterCursorEvent(dynamic_cast<
                graphics::AbstractCursorEvent *>(inOutZoomController));
        }

        /* Create new controllers. */
        inOutRotateController = rotCtrl = new graphics::CameraRotate2DLookAt(
            this->camera.Parameters());
        rotCtrl->SetTestButton(GLUT_LEFT_BUTTON);
        rotCtrl->SetModifierTestCount(0);
        rotCtrl->SetAltModifier(graphics::InputModifiers::MODIFIER_SHIFT);

        inOutZoomController = zoomCtrl = new graphics::CameraZoom2DMove(
            this->camera.Parameters());
        zoomCtrl->SetTestButton(GLUT_MIDDLE_BUTTON);
        zoomCtrl->SetModifierTestCount(0);
        zoomCtrl->SetSpeed(10.0f);
        zoomCtrl->SetZoomBehaviour(graphics::CameraZoom2DMove::MOVE_IF_CLOSE);

        /* Reset the view. */
        this->camera.Parameters()->SetView(
            graphics::SceneSpacePoint3D(0.0, -5.0, 0.0),
            graphics::SceneSpacePoint3D(0.0, 0.0, 0.0),
            graphics::SceneSpaceVector3D(0.0, 0.0, 1.0));

        /* Connect controllers with cursor device. */
        this->cursor.RegisterCursorEvent(zoomCtrl);
        this->cursor.RegisterCursorEvent(rotCtrl);
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
        BECAUSE_I_KNOW(GLUT_LEFT_BUTTON < 3);
        BECAUSE_I_KNOW(GLUT_MIDDLE_BUTTON < 3);
        this->cursor.SetButtonState(button, (state == GLUT_DOWN));

        this->updateInputModifiers();
        this->updateCursorPosition(x, y);
        ::glutPostRedisplay();
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::onMouseMove
     */
    template<class T> 
    void GlutServerNode<T>::onMouseMove(const int x, const int y) {
        this->updateCursorPosition(x, y);
        ::glutPostRedisplay();
    }


    /*
     * vislib::net::cluster::GlutServerNode<T>::onResize
     */
    template<class T> 
    void GlutServerNode<T>::onResize(const int width, const int height) {
        GlutClusterNode<T>::onResize(width, height);
        this->camera.Parameters()->SetVirtualViewSize(
            static_cast<graphics::ImageSpaceType>(width),
            static_cast<graphics::ImageSpaceType>(height));
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
