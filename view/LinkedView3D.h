//
// LinkedView3D.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MEGAMOLCORE_LINKEDVIEW3D_H_INCLUDED
#define MEGAMOLCORE_LINKEDVIEW3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/View3D.h"
#include "Call.h"
#include "CallerSlot.h"

#include "vislib/CameraParameterObserver.h"
#include "vislib/ObservableCameraParams.h"

namespace megamol {
namespace core {
namespace view {

class LinkedView3D: public core::view::View3D {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "LinkedView3D";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "3D View Module enabling linked views using shared camera " \
                   "parameters.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
#if (VISLIB_VERSION_MAJOR > 1) || (VISLIB_VERSION_MINOR > 0)
        	return false;
#else
            return true;
#endif
        }

        /** Ctor. */
        LinkedView3D(void);

        /** Dtor. */
        virtual ~LinkedView3D(void);

    protected:

        class LinkedCameraParameterObserver:
                public vislib::graphics::CameraParameterObserver {
            public:

                /** Ctor. */
                LinkedCameraParameterObserver(void)
                        : camChanged(false) {
                }

                /** Dtor. */
                virtual ~LinkedCameraParameterObserver(void) {
                    this->camChanged = true;
                }

                /**
                 * Answers the current state of the camChanged flag.
                 *
                 * @return 'True' if the camera has changed, 'false' otherwise.
                 */
                bool HasCamChanged() {
                    return this->camChanged;
                }

                /**
                 * This method is called if the aperture angle changed.
                 *
                 * @param newValue The new aperture angle.
                 */
                virtual void OnApertureAngleChanged(
                        const vislib::math::AngleDeg newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the stereo eye changed.
                 *
                 * @param newValue The new stereo eye.
                 */
                virtual void OnEyeChanged(
                        const vislib::graphics::CameraParameters::StereoEye newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the far clipping plane changed.
                 *
                 * @param newValue The new far clipping plane.
                 */
                virtual void OnFarClipChanged(
                        const vislib::graphics::SceneSpaceType newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the focal distance changed.
                 *
                 * @param newValue The new forcal distance.
                 */
                virtual void OnFocalDistanceChanged(
                        const vislib::graphics::SceneSpaceType newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the look at point changed.
                 *
                 * @param newValue The new look at point.
                 */
                virtual void OnLookAtChanged(
                        const vislib::graphics::SceneSpacePoint3D& newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the near clipping plane changed.
                 *
                 * @param newValue The new near clipping plane.
                 */
                virtual void OnNearClipChanged(
                        const vislib::graphics::SceneSpaceType newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the camera position changed.
                 *
                 * @param newValue The new camera position.
                 */
                virtual void OnPositionChanged(
                        const vislib::graphics::SceneSpacePoint3D& newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the projection type changed.
                 *
                 * @param newValue The new projection type.
                 */
                virtual void OnProjectionChanged(
                        const vislib::graphics::CameraParameters::ProjectionType newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the stereo disparity changed.
                 *
                 * @param newValue The new stereo disparity.
                 */
                virtual void OnStereoDisparityChanged(
                        const vislib::graphics::SceneSpaceType newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the screen tile changed.
                 *
                 * @param newValue The new screen tile.
                 */
                virtual void OnTileRectChanged(
                        const vislib::graphics::ImageSpaceRectangle& newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the camera up vector changed.
                 *
                 * @param newValue The new camera up vector.
                 */
                virtual void OnUpChanged(
                        const vislib::graphics::SceneSpaceVector3D& newValue) {
                    this->camChanged = true;
                }

                /**
                 * This method is called if the virtual screen size changed.
                 *
                 * @param newValue The new virtual screen size.
                 */
                virtual void OnVirtualViewSizeChanged(
                        const vislib::graphics::ImageSpaceDimension& newValue) {
                    this->camChanged = true;
                }

                /**
                 * Resets the camChanged flag to 'false'.
                 */
                void ResetCamChanged() {
                    this->camChanged = false;
                }

            protected:
            private:
                bool camChanged;
        };

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         *
         * @param context
         */
        virtual void Render(const mmcRenderViewContext& context);

    private:

        /// Caller slot to access shared camera parameters
        core::CallerSlot sharedCamParamsSlot;

        /// Observable camera parameters
        vislib::SmartPtr<vislib::graphics::ObservableCameraParams> observableCamParams;

        /// Camera parameters observer
        LinkedCameraParameterObserver observer;

};

} // namespace view
} // namespace core
} // namespace megamol

#endif // MEGAMOLCORE_LINKEDVIEW3D_H_INCLUDED
