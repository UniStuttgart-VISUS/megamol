/*
* KeyframeManipulator.h
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATICCAMERA_KEYFRAME_MANIP_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_KEYFRAME_MANIP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"
#include "Keyframe.h"

#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"
#include "vislib/math/Matrix.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/math/Cuboid.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
    namespace cinematiccamera {

        class KeyframeManipulator {

        public:

            /** CTOR */
            KeyframeManipulator(void);

            /** DTOR */
            ~KeyframeManipulator(void);

            // enumeration of manipulator types
            enum manipType {
                SELECTED_KF_POS_X      = 0,
                SELECTED_KF_POS_Y      = 1,
                SELECTED_KF_POS_Z      = 2,
                SELECTED_KF_POS_LOOKAT = 3,
                SELECTED_KF_LOOKAT_X   = 4,
                SELECTED_KF_LOOKAT_Y   = 5,
                SELECTED_KF_LOOKAT_Z   = 6,
                SELECTED_KF_UP         = 7,
                CTRL_POINT_POS_X       = 8,
                CTRL_POINT_POS_Y       = 9,
                CTRL_POINT_POS_Z       = 10,
                NUM_OF_SELECTED_MANIP  = 11,
                KEYFRAME_POS           = 12,
                NONE                   = 13
            }; // DON'T CHANGE ORDER OR NUMBERING
               // Add new manipulator type before NUM_OF_SELECTED_MANIP ...


            /** Update rednering data of manipulators.
            *
            * @param am    Array of manipulator types
            * @param kfa   Pointer to the array of keyframes
            * @param skf   The currently selected keyframe
            * @param vps   The current viewport size
            * @param mvpm  The current Model-View-Projection-Matrix
            * @param wclad The lookat direction of the world camera
            * @param wcmd  The  direction between the world camera position and the model center
            * @param mob   If true manipulators always lie outside of model bbox
            * @param fcp   First control point for interpolation curve 
            * @param lcp   Last control point for interpolation curve 

            * @return True if data was updated successfully.
            *
            */
            bool Update(vislib::Array<KeyframeManipulator::manipType> am, vislib::Array<Keyframe>* kfa, Keyframe skf, 
                        float vph, float vpw, vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> mvpm,
                        vislib::math::Vector<float, 3> wclad, vislib::math::Vector<float, 3> wcmd, bool mob, 
                        vislib::math::Vector<float, 3> fcp, vislib::math::Vector<float, 3> lcp);

            /** Update extents.
            *   Grows bounding box to manipulators.
            *   If manipulator lies inside of bounding box:
            *   Get bounding box of model to determine minimum length of manipulator axes.
            *
            */
            void SetExtents(vislib::math::Cuboid<float> *bb);

            /** */
            bool Draw(void);

            /** */
            int CheckKeyframePositionHit(float x, float y);

            /** */
            bool CheckManipulatorHit(float x, float y);

            /** */
            bool ProcessManipulatorHit(float x, float y);

            /** */
            Keyframe GetManipulatedKeyframe(void);

            /** */
            vislib::math::Vector<float, 3> GetFirstControlPointPosition();
            /** */
            vislib::math::Vector<float, 3> GetLastControlPointPosition();

        private:

            /**********************************************************************
            * variables
            **********************************************************************/

            // Class for manipulator values
            class manipPosData {
            public:
                // Comparison needed for use in vislib::Array
                bool operator==(manipPosData const& rhs) {
                    return ((this->wsPos == rhs.wsPos) &&
                        (this->ssPos == rhs.ssPos) &&
                        (this->offset == rhs.offset));
                }
                // Variables
                vislib::math::Vector<float, 3> wsPos;   // position in world space
                vislib::math::Vector<float, 2> ssPos;   // position in screeen space
                float                          offset;  // screen space offset for ssPos to count as hit
                bool                           available;
            };

            // Some fixed values
            const float                      circleRadiusFac;  // Factor for world cam lookat direction which is used as adaptive circle radius
            const float                      axisLengthFac;    // Factor for world cam lookat direction which is used as adaptive axis length
            const unsigned int               circleSubDiv;     // Amount of subdivisions of an circle primitive
            const float                      lineWidth;
            const float                      sensitivity;      // Relationship between mouse movement and length changes of coordinates

            // Positions of keyframes
            vislib::Array<manipPosData>      kfArray;     // Array of keyframe positions

            // Selected keyframe
            Keyframe                         selectedKf;   // Copy of currently selected Keyframe
            vislib::Array<manipPosData>      manipArray;   // Array of manipulators for selected keyframe
            vislib::math::Vector<float, 2>   sKfSsPos;     // Screen space position of selected keyframe
            vislib::math::Vector<float, 2>   sKfSsLookAt;  // Screen space lookat of selected keyframe
            bool                             sKfInArray;   // Inidcates if selected keyframe exists in keyframe array
            manipType                        activeType;   // Indicates the type of the active selected manipulator
            vislib::math::Vector<float, 2>   lastMousePos;


            vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix;
            vislib::math::Dimension<int, 2>  viewportSize;
            vislib::math::Vector<float, 3>   worldCamLaDir;
            vislib::math::Vector<float, 3>   worldCamModDir;
            bool                             isDataSet;
            bool                             isDataDirty;
            vislib::math::Cuboid<float>      modelBbox;
            bool                             manipOusideBbox;

            vislib::math::Vector<float, 3>   startCtrllPos;
            vislib::math::Vector<float, 3>   endCtrllPos;
            bool                             selectedIsFirst;
            bool                             selectedIsLast;

            vislib::Array<vislib::math::Vector<float, 3> > circleVertices;

            /**********************************************************************
            * functions
            **********************************************************************/

            /** */
            void calculateCircleVertices(void);

            /** */
            void drawCircle(vislib::math::Vector<float, 3> pos, float factor);

            /* */
            void drawManipulator(vislib::math::Vector<float, 3> kp, vislib::math::Vector<float, 3> mp);

            /** */
            vislib::math::Vector<float, 2> getScreenSpace(vislib::math::Vector<float, 3> wp);

            /** */
            bool updateManipulatorPositions(void);

            /** Convert Vector to Point*/
            inline vislib::math::Point<float, 3> V2P(vislib::math::Vector<float, 3> v) {
                return vislib::math::Point<float, 3>(v.X(), v.Y(), v.Z());
            }

        };

    } /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_KEYFRAME_MANIP_H_INCLUDED */