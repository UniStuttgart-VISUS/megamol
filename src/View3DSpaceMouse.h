/*
 * View3DSpaceMouse.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEW3DSPACEMOUSE_H_INCLUDED
#define MEGAMOLCORE_VIEW3DSPACEMOUSE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/View3D.h"
#include "vislib/graphics/RelativeCursor3D.h"
#include "vislib/graphics/CameraAdjust3D.h"
#include "Raw3DRelativeMouseInput.h"

namespace megamol {
namespace protein_cuda {


    class View3DSpaceMouse : public megamol::core::view::View3D {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "View3DSpaceMouse";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "3D View Module with 3D mouse support";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        View3DSpaceMouse(void);

        /** Dtor. */
        virtual ~View3DSpaceMouse(void);

        /**
         * Resets the view. This normally sets the camera parameters to
         * default values.
         * This also resets the object center for the relative cursor
         */
        virtual void ResetView(void);

        /**
         * Callback function for 3d mouse motion input.
         *
         * @param tx The translation vector x component.
         * @param ty The translation vector y component.
         * @param tz The translation vector z component.
         * @param rx The rotation vector x component.
         * @param ry The rotation vector y component.
         * @param rz The rotation vector z component.
         */
        virtual void On3DMouseMotion(float tx, float ty, float tz, float rx, float ry, float rz);

        /**
         * Callback function for 3d mouse button input.
         *
         * @param keyState 32 bits representing button 0 through 31. A 1 means the
         * button is pressed, a 0 means it is not pressed.
         */
        virtual void On3DMouseButton(unsigned long keyState);

    private:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * Updates translation and rotation speed when the parameter
         * slots are changed.
         *
         * @param p The parameter slot that triggered the callback.
         *
         * @return Always true.
         */
        virtual bool update3DSpeed(megamol::core::param::ParamSlot& p);

        /**
         * Updates the various camera modes with the parameter slots 
         * are changed.
         *
         * @param p The parameter slot that triggered the callback.
         *
         * @return Always true.
         */
        virtual bool updateModes(megamol::core::param::ParamSlot& p);

        /**
         * Changes the camera control mode to the one specified by the
         * param slot.
         *
         * @param p The parameter slot that triggered the callback.
         *
         * @return Always true.
         */
        virtual bool updateCameraModes(megamol::core::param::ParamSlot& p);

        // private variables //

        /** the relative 3d cursor of this view */
        vislib::graphics::RelativeCursor3D relativeCursor3d;

#ifdef _WIN32
        /** the raw input class associated with this 3d cursor */
        Raw3DRelativeMouseInput rawInput;
#endif /* _WIN32 */

        /** 3d camera adjustor */
        vislib::graphics::CameraAdjust3D adjustor;

        /** Slot used to control 3d mouse translation speed */
        megamol::core::param::ParamSlot translateSpeed3DSlot;

        /** Slot used to control 3d mouse rotation speed */
        megamol::core::param::ParamSlot rotateSpeed3DSlot;

        /** Slot used to toggle 3d mouse translation */
        megamol::core::param::ParamSlot translateToggleSlot;

        /** Slot used to toggle 3d mouse rotation */
        megamol::core::param::ParamSlot rotateToggleSlot;

        /** Slot used to toggle on/off single (dominant) axis mode */
        megamol::core::param::ParamSlot singleAxisToggleSlot;

        /** Slot used to change between the different camera control modes */
        megamol::core::param::ParamSlot cameraControlModeSlot;

    };

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEW3DSPACEMOUSE_H_INCLUDED */
