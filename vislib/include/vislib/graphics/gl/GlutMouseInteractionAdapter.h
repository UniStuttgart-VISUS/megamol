/*
 * GlutMouseInteractionAdapter.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLUTMOUSEINTERACTIONADAPTER_H_INCLUDED
#define VISLIB_GLUTMOUSEINTERACTIONADAPTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/graphics/MouseInteractionAdapter.h"


namespace vislib {
namespace graphics {
namespace gl {

#ifndef GLUT_API_VERSION
#error You must include GLUT before you include GlutMouseInteractionAdapter.h
#endif

    /**
     * Traits class connecting to standard GLUT
     */
    typedef struct _GlutTraits_t {
        /* int Constant values */
        enum constants {
            LEFT_BUTTON = GLUT_LEFT_BUTTON,
            RIGHT_BUTTON = GLUT_RIGHT_BUTTON,
            MIDDLE_BUTTON = GLUT_MIDDLE_BUTTON,
            BUTTON_DOWN = GLUT_DOWN,
            ACTIVE_SHIFT = GLUT_ACTIVE_SHIFT,
            ACTIVE_CTRL = GLUT_ACTIVE_CTRL,
            ACTIVE_ALT = GLUT_ACTIVE_ALT
        };
        /* GetModifiers */
        static inline int GetModifiers(void) {
            return ::glutGetModifiers();
        }
    } GlutTraits;


    /**
     * This is a convenience class for using a VISlib camera with 2D mouse 
     * interaction in a GLUT application.
     *
     * @param T GlutTraits class
     */
    template<class T>
    class GlutMouseInteractionAdapter {
    public:

        /** Ctor. */
        GlutMouseInteractionAdapter(const SmartPtr<CameraParameters>& params,
                const unsigned int cntButtons = 3) : mia(params, cntButtons) {
            // intentionally empty 
        }

        /** Dtor. */
        ~GlutMouseInteractionAdapter(void) {
            // intentionally empty
        }

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
        inline void ConfigureRotation(const 
                MouseInteractionAdapter::RotationType type 
                = MouseInteractionAdapter::ROTATION_LOOKAT,
                const MouseInteractionAdapter::Button button 
                = MouseInteractionAdapter::BUTTON_LEFT,
                const InputModifiers::Modifier altModifier 
                = InputModifiers::MODIFIER_SHIFT) {
            this->mia.ConfigureRotation(type, button, altModifier);
        }

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
        inline void ConfigureZoom(const MouseInteractionAdapter::ZoomType type 
                = MouseInteractionAdapter::ZOOM_MOVE, 
                const MouseInteractionAdapter::Button button 
                = MouseInteractionAdapter::BUTTON_MIDDLE,
                const SceneSpaceType speed = 10.0f,
                const CameraZoom2DMove::ZoomBehaviourType behaviour
                = CameraZoom2DMove::FIX_LOOK_AT) {
            this->mia.ConfigureZoom(type, button, speed, behaviour);
        }

        /**
         * Answer the camera parameters that are modified.
         *
         * @return A smart pointer to the camera parameters being modified by
         *         this interaction adapter.
         */
        inline SmartPtr<CameraParameters> GetCamera(void) {
            return this->mia.GetCamera();
        }

        /**
         * This method should be called once a key was pressed.
         *
         * @param key The character of the key that was pressed.
         * @param x   The new x-coordinate of the mouse cursor.
         * @param y   The new y-coordinate of the mouse cursor.
         */
        inline void OnKeyDown(const unsigned char key, const int x, const int y) {
            this->mia.SetMousePosition(x, y, true);
        }

        /**
         * This method should be called once a mouse button was pressed or 
         * released
         * 
         * @param button The GLUT constant for the respective mouse button.
         * @param state  The GLUT constant for the new mouse button state.
         * @param x      The new x-coordinate of the mouse cursor.
         * @param y      The new y-coordinate of the mouse cursor.
         */
        inline void OnMouseButton(const int button, const int state,
                const int x, const int y) {
            MouseInteractionAdapter::Button btn
                = MouseInteractionAdapter::BUTTON_LEFT;

            switch (button) {
            case T::LEFT_BUTTON:
                btn = MouseInteractionAdapter::BUTTON_LEFT;
                break;

            case T::RIGHT_BUTTON:
                btn = MouseInteractionAdapter::BUTTON_RIGHT;
                break;

            case T::MIDDLE_BUTTON:
                btn = MouseInteractionAdapter::BUTTON_MIDDLE;
                break;
            }

            this->mia.SetMousePosition(x, y, true);
            this->mia.SetMouseButtonState(btn, (state == T::BUTTON_DOWN));
            this->setModifierState(T::GetModifiers());
        }

        /**
         * This method should be called once the mouse was moved.
         *
         * @param x The new x-coordinate of the mouse cursor.
         * @param y The new y-coordinate of the mouse cursor.
         */
        inline void OnMouseMove(const int x, const int y) {
            this->mia.SetMousePosition(x, y, true);
        }

        /**
         * This method should be called once the mouse was moved outside the
         * window.
         *
         * @param x The new x-coordinate of the mouse cursor.
         * @param y The new y-coordinate of the mouse cursor.
         */
        inline void OnMousePassiveMove(const int x, const int y) {
            this->mia.SetMousePosition(x, y, true);
        }

        /**
         * This method should be called once the window was resized.
         *
         * @param width  The new width of the window.
         * @param height The new height of the window.
         */
        inline void OnResize(const int width, const int height) {
            SmartPtr<CameraParameters> params = this->mia.GetCamera();
            if (!params.IsNull()) {
                params->SetVirtualViewSize(
                    static_cast<ImageSpaceType>(width),
                    static_cast<ImageSpaceType>(height));
            }
        }

        /**
         * This method should be called once a special key was pressed.
         *
         * @param key The GLUT symbolic constant of the special key.
         * @param x   The new x-coordinate of the mouse cursor.
         * @param y   The new y-coordinate of the mouse cursor.
         */
        inline void OnSpecialKeyDown(const int key, const int x,
                const int y) {
            this->mia.SetMousePosition(x, y, true);
            this->setModifierState(T::GetModifiers());
        }

        /**
         * Change the camera parameters to be modfied.
         *
         * @param camera The camera to modify the parameters of.
         */
        inline void SetCamera(const Camera& camera) {
            this->mia.SetCamera(camera);
        }

        /**
         * Change the camera parameters to be modfied.
         *
         * @param params The parameters to be modified.
         */
        inline void SetCamera(const SmartPtr<CameraParameters>& params) {
            this->mia.SetCamera(params);
        }

    private:

        /**
         * Update the modifier state of 'mia' by using the given GLUT modifier
         * mask.
         *
         * @param glutModifiers The GLUT modifier mask.
         */
        inline void setModifierState(const int glutModifiers) {
            this->mia.SetModifierState(InputModifiers::MODIFIER_SHIFT,
                (glutModifiers & T::ACTIVE_SHIFT) == T::ACTIVE_SHIFT);
            this->mia.SetModifierState(InputModifiers::MODIFIER_CTRL,
                (glutModifiers & T::ACTIVE_CTRL) == T::ACTIVE_CTRL);
            this->mia.SetModifierState(InputModifiers::MODIFIER_ALT,
                (glutModifiers & T::ACTIVE_ALT) == T::ACTIVE_ALT);
        }

        /** The interaction adapter that performs the actual work. */
        MouseInteractionAdapter mia;

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLUTMOUSEINTERACTIONADAPTER_H_INCLUDED */
