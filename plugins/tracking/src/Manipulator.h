/*
 * Manipulator.h
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <vector>

#include "mmcore/param/ParamSlot.h"

#include "mmcore/view/CallCamParamSync.h"

#include "vislib/graphics/graphicstypes.h"

#include "vislib/math/Point.h"
#include "vislib/math/Matrix4.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"


namespace megamol {
namespace tracking {

    /**
     * Implements the camera manipulation for a 3DOF device.
     */
    class Manipulator {

    public:

        typedef core::view::CallCamParamSync::CamParams CamParamsType;
        typedef vislib::math::Point<vislib::graphics::SceneSpaceType, 3> PointType;
        typedef vislib::math::Quaternion<vislib::graphics::SceneSpaceType> QuaternionType;
        typedef vislib::math::Matrix4<vislib::graphics::SceneSpaceType, vislib::math::COLUMN_MAJOR> MatrixType;
        typedef vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> VectorType;

        Manipulator(void);
        ~Manipulator(void);

        void ApplyTransformations(void);

        inline CamParamsType GetCamParams(void) {
            return this->camParams;
        }

        inline const QuaternionType& GetOrientation(void) const {
            return this->curOrientation;
        }

        inline std::vector<core::param::ParamSlot *>& GetParams(void) {
            return this->params;
        }

        inline const PointType& GetPosition(void) const {
            return this->curPosition;
        }

        void OnButtonChanged(const int button, const bool isPressed);

        inline void SetOrientation(const QuaternionType& o) {
            this->curOrientation = o;
        }

        inline void SetPosition(const PointType& p) {
            this->curPosition = p;
        }

        inline void SetCamParams(const CamParamsType camParams) {
            this->camParams = camParams;
        }

    private:

        /**
         * Answer the (normalised) rotation for matching 'u' on 'v'.
         */
        static QuaternionType xform(const VectorType& u, const VectorType& v);

        /**
         * Answer if any of the interactions is currently performed (ie a button
         * was pressed).
         */
        inline bool isInteracting(void) const {
            return (this->isRotating || this->isTranslating || this->isZooming);
        }

        CamParamsType camParams;
        QuaternionType curOrientation;
        PointType curPosition;
        bool isRotating;
        bool isTranslating;
        bool isZooming;
        core::param::ParamSlot paramInvertRotate;
        core::param::ParamSlot paramInvertZoom;
        core::param::ParamSlot paramRotateButton;
        std::vector<core::param::ParamSlot *> params;
        core::param::ParamSlot paramSingleInteraction;
        core::param::ParamSlot paramTranslateButton;
        core::param::ParamSlot paramTranslateSpeed;
        core::param::ParamSlot paramZoomButton;
        core::param::ParamSlot paramZoomSpeed;
        PointType startCamLookAt;
        PointType startCamPosition;
        VectorType startCamUp;
        QuaternionType startOrientation;
        QuaternionType startRelativeOrientation;
        PointType startPosition;
    };

} /* end namespace tracking */
} /* end namespace megamol */