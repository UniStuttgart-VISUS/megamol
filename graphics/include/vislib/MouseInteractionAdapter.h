/*
 * MouseInteractionAdapter.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MOUSEINTERACTIONADAPTER_H_INCLUDED
#define VISLIB_MOUSEINTERACTIONADAPTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraRotate2D.h"
#include "vislib/CameraRotate2DLookAt.h"
#include "vislib/CameraZoom2DAngle.h"
#include "vislib/CameraZoom2DMove.h"
#include "vislib/Cursor2D.h"
#include "vislib/InputModifiers.h"


namespace vislib {
namespace graphics {

    /**
     * This is a convenience class for configuring a VISlib camera with 
     * 2D mouse interaction.
     */
    class MouseInteractionAdapter {

    public:

        /** Symolic constants for the mouse buttons. */
        typedef enum Button_t {
            BUTTON_LEFT = 0,
            BUTTON_RIGHT = 1,
            BUTTON_MIDDLE = 2
        } Button;

        /** The supported rotation types. */
        typedef enum RotationType_t {
            ROTATION_FREE = 1,
            ROTATION_LOOKAT
        } RotationType;

        /** The supported zoom types. */
        typedef enum ZoomType_t {
            ZOOM_ANGLE = 1,
            ZOOM_MOVE
        } ZoomType;

        /** 
         * Create a new instance with default behaviour. This is equivalent to
         * calling ConfigureRotation() and ConfigureZoom() with default 
         * parameters.
         *
         * @param params     The camera parameters that are manipulated by this
         *                   adapter.
         * @param cntButtons The number of buttons the mouse has. The order is
         *                   left, right, middle, others.
         */
        MouseInteractionAdapter(const SmartPtr<CameraParameters>& params, 
            const unsigned int cntButtons = 3);

        /** Dtor. */
        ~MouseInteractionAdapter(void);

        /**
         * Configure the behaviour of the rotation controller.
         *
         * @param type        The type of rotation controller, which can be one 
         *                    of RotationType's members.
         * @param button      The button that should trigger a rotation.
         * @param altModifier The modifiert for the alternative rotation 
         *                    behaviour. See CameraRotate2DLookAt and 
         *                    CameraRotate2D for more information.
         */
        void ConfigureRotation(const RotationType type = ROTATION_LOOKAT, 
            const Button button = BUTTON_LEFT,
            const InputModifiers::Modifier altModifier 
            = InputModifiers::MODIFIER_SHIFT);

        /**
         * Configure the behaviour of the zoom controller.
         *
         * @param type      Specify how the zoom should be implemented.
         * @param button    The button that should trigger the zoom.
         * @param speed     The motion speed of the zoom. This is only 
         *                  meaningful if 'type' is ZOOM_MOVE.
         * @param behaviour The proximity move behaviour of the zoom. This
         *                  is only meaningful if 'type' is ZOOM_MOVE.
         */
        void ConfigureZoom(const ZoomType type = ZOOM_MOVE, 
            const Button button = BUTTON_MIDDLE,
            const SceneSpaceType speed = 10.0f,
            const CameraZoom2DMove::ZoomBehaviourType behaviour
            = CameraZoom2DMove::FIX_LOOK_AT);

        /**
         * Answer the camera parameters that are modified.
         *
         * @return A smart pointer to the camera parameters being modified by
         *         this interaction adapter.
         */
        SmartPtr<CameraParameters> GetCamera(void);

        /**
         * Change the camera parameters to be modfied.
         *
         * @param camera The camera to modify the parameters of.
         */
        inline void SetCamera(const Camera& camera) {
            this->SetCamera(camera.Parameters());
        }

        /**
         * Change the camera parameters to be modfied.
         *
         * @param params The parameters to be modified.
         */
        void SetCamera(const SmartPtr<CameraParameters>& params);

        /**
         * Change the state of the modifier buttons.
         *
         * @param modifier
         * @param isDown
         */
        inline void SetModifierState(const InputModifiers::Modifier modifier, 
                const bool isDown) {
            this->modifiers.SetModifierState(modifier, isDown);
        }

        /**
         * Change the state of the mouse buttons.
         *
         * @param button
         * @param isDown
         */
        inline void SetMouseButtonState(const Button button, bool isDown) {
            this->cursor.SetButtonState(static_cast<unsigned int>(button), isDown);
        }

        /**
         * Change the cursor position.
         *
         * @param x
         * @param y
         * @param invertY
         */
        inline void SetMousePosition(const int x, const int y, 
                const bool invertY) {
            this->cursor.SetPosition(
                static_cast<vislib::graphics::ImageSpaceType>(x),
                static_cast<vislib::graphics::ImageSpaceType>(y),
                invertY);
        }

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        MouseInteractionAdapter(const MouseInteractionAdapter& rhs);

        /**
         * Answer the 'rotator' as AbstractCameraController.
         *
         * @return A pointer in case of success, NULL otherwise.
         */
        inline AbstractCameraController *getRotateCtrl(void) {
            return dynamic_cast<AbstractCameraController *>(this->rotator);
        }

        /**
         * Answer the 'zoomer' as AbstractCameraController.
         *
         * @return A pointer in case of success, NULL otherwise.
         */
        inline AbstractCameraController *getZoomCtrl(void) {
            return dynamic_cast<AbstractCameraController *>(this->zoomer);
        }

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        MouseInteractionAdapter& operator =(const MouseInteractionAdapter& rhs);

        /** The cursor controlled by the mouse. */
        Cursor2D cursor;
    
        /** Input modifier states for alternate interaction methods. */
        InputModifiers modifiers;

        /** The rotation controller. */
        AbstractCursorEvent *rotator;
    
        /** The zoom controller. */
        AbstractCursorEvent *zoomer;
    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MOUSEINTERACTIONADAPTER_H_INCLUDED */

