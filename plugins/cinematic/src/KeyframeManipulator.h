/*
* KeyframeManipulator.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_KEYFRAMEMANIPULATOR_H_INCLUDED
#define MEGAMOL_CINEMATIC_KEYFRAMEMANIPULATOR_H_INCLUDED

#include "Cinematic/Cinematic.h"
#include "Keyframe.h"

#include "vislib/math/Cuboid.h"
#include "vislib/sys/Log.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


namespace megamol {
namespace cinematic {

    /*
     * Keyframe Manipulators.
     */
    class KeyframeManipulator {

    public:

        /** CTOR */
        KeyframeManipulator(void);

        /** DTOR */
        ~KeyframeManipulator(void);

        // Enumeration of manipulator types
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

        /** 
        * Update rednering data of manipulators.
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
        *
        * @return True if data was updated successfully.
        *
        */
        bool Update(std::vector<KeyframeManipulator::manipType> am, std::shared_ptr<std::vector<Keyframe>> kfa, Keyframe skf,
                    float vph, float vpw, glm::mat4 mvpm, glm::vec3 wclad, glm::vec3 wcmd, bool mob, glm::vec3 fcp, glm::vec3 lcp);

        /** 
        * Update extents.
        * Grows bounding box to manipulators.
        * If manipulator lies inside of bounding box:
        * Get bounding box of model to determine minimum length of manipulator axes.
        */
        void SetExtents(vislib::math::Cuboid<float>& bb);

        /** 
        *
        */
        bool Draw(void);

        /**
        *
        */
        int CheckKeyframePositionHit(float x, float y);

        /**
        *
        */
        bool CheckManipulatorHit(float x, float y);

        /**
        *
        */
        bool ProcessManipulatorHit(float x, float y);

        /**
        *
        */
        Keyframe GetManipulatedKeyframe(void);

        /**
        *
        */
        glm::vec3 GetFirstControlPointPosition(void);

        /**
        *
        */
        glm::vec3 GetLastControlPointPosition(void);

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

            glm::vec3 wsPos;	 // position in world space
            glm::vec2 ssPos;	 // position in screeen space
            float     offset;	 // screen space offset for ssPos to count as hit
            bool      available;
        };

        // Some fixed values
        const float						 circleRadiusFac;	// Factor for world cam lookat direction which is used as adaptive circle radius
        const float						 axisLengthFac;		// Factor for world cam lookat direction which is used as adaptive axis length
        const unsigned int				 circleSubDiv;		// Amount of subdivisions of an circle primitive
        const float						 lineWidth;
        const float						 sensitivity;		// Relationship between mouse movement and length changes of coordinates
        std::vector<manipPosData>        kfArray;			// Array of keyframe positions

        Keyframe                         selectedKf;		// Copy of currently selected Keyframe
        std::vector<manipPosData>        manipArray;		// Array of manipulators for selected keyframe
        glm::vec2                        sKfSsPos;			// Screen space position of selected keyframe
        glm::vec2                        sKfSsLookAt;		// Screen space lookat of selected keyframe
        int                              sKfInArray;		// Inidcates if selected keyframe exists in keyframe array if >= 0
        manipType                        activeType;		// Indicates the type of the active selected manipulator
        glm::vec2                        lastMousePos;

        glm::mat4                        modelViewProjMatrix;
        glm::vec2                        viewportSize;
        glm::vec3						 worldCamLaDir;
        glm::vec3						 worldCamModDir;
        bool                             isDataSet;
        bool                             isDataDirty;
        vislib::math::Cuboid<float>      modelBbox;
        bool                             manipOusideBbox;

        glm::vec3						 startCtrllPos;
        glm::vec3						 endCtrllPos;
        bool                             selectedIsFirst;
        bool                             selectedIsLast;

        std::vector<glm::vec3 >		    circleVertices;

        /**********************************************************************
        * functions
        **********************************************************************/

        /**
        *
        */
        void calculateCircleVertices(void);

        /**
        *
        */
        void drawCircle(glm::vec3 pos, float factor);

        /**
        *
        */
        void drawManipulator(glm::vec3 kp, glm::vec3 mp);

        /**
        *
        */
        glm::vec2 getScreenSpace(glm::vec3 wp);

        /**
        *
        */
        bool updateManipulatorPositions(void);
    };

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_KEYFRAMEMANIPULATOR_H_INCLUDED