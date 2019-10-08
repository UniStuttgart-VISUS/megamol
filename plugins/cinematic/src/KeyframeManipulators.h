/*
* KeyframeManipulators.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_KEYFRAMEMANIPULATORS_H_INCLUDED
#define MEGAMOL_CINEMATIC_KEYFRAMEMANIPULATORS_H_INCLUDED

#include "Cinematic/Cinematic.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/ButtonParam.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/Log.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Keyframe.h"
#include "CinematicUtils.h"


namespace megamol {
namespace cinematic {


    /*
     * Manipulator
     */
    class Manipulator {
    public:
        enum Rigging {
            X_DIRECTION,
            Y_DIRECTION,
            Z_DIRECTION,
            ROTATION
        };

        enum Group {
            SELECTION_KEYFRAME_POSITIONS,
            MANIPULATOR_CTRLPOINTS_POSITION,
            MANIPULATOR_SELECTED_KEYFRAME_POSITION,
            MANIPULATOR_SELECTED_KEYFRAME_POSITION_LOOKAT,
            MANIPULATOR_SELECTED_KEYFRAME_LOOKAT_POSITION,
            MANIPULATOR_SELECTED_KEYFRAMEF_UP_ROTATION,
        };
        
        bool show;
        Group group;
        Rigging rig; 
        glm::vec3 position;
        int keyframe_index;

        const float point_radius = 5.0f;
    };


    /*
     * Keyframe Manipulators.
     */
    class KeyframeManipulators {

    public:

        /** CTOR */
        KeyframeManipulators(void);

        /** DTOR */
        ~KeyframeManipulators(void);

        enum VisibleGroup : int {
            KEYFRAME_AND_CTRPOINT = 0,
            LOOKAT_AND_UP         = 1,
            GROUP_COUNT           = 2
        };

        /** 
        * Update rendering data.
        *
        * @param utils              ...
        * @param keyframes          ...
        * @param selected_keyframe  ...
        * @param dim_vp             ...
        * @param mvp                ...
        * @param snapshot           ...
        * @param first_ctrl_pos     ...
        * @param last_ctrl_pos      ...
        *
        * @return True if data was updated successfully.
        *
        */
        bool UpdateRendering(std::shared_ptr<CinematicUtils> utils, std::vector<Keyframe> const& keyframes, Keyframe selected_keyframe, glm::vec2 viewport_dim, glm::mat4 mvp,
            camera_state_type snapshot, glm::vec3 first_ctrl_pos, glm::vec3 last_ctrl_pos);

        /** 
        * Update extents.
        *
        * Grows bounding box to manipulators.
        * If manipulator lies inside of bounding box:
        * Get bounding box of model to determine minimum length of manipulator axes.
        * 
        * @param inout_bbox   ...
        */
        void UpdateExtents(vislib::math::Cuboid<float>& inout_bbox);

        bool Draw(void);

        int GetSelectedKeyframePositionIndex(float mouse_x, float mouse_y);

        bool CheckForHitManipulator(float mouse_x, float mouse_y);

        bool ProcessHitManipulator(float mouse_x, float mouse_y);

        inline Keyframe GetManipulatedSelectedKeyframe(void) const  {
            return this->current_selected_keyframe;
        }

        inline glm::vec3 GetFirstControlPointPosition(void) const {
            return this->current_first_ctrl_point;
        }

        inline glm::vec3 GetLastControlPointPosition(void) const {
            return this->current_last_ctrl_point;
        }

        /**
        * GetParams
        *
        * @return List with pointers to the parameter slots.
        */
        inline std::vector<std::shared_ptr<megamol::core::param::ParamSlot>>& GetParams(void) {
            return this->paramSlots;
        }

    private:

        /**********************************************************************
        * variables
        **********************************************************************/

        std::vector<std::shared_ptr<megamol::core::param::ParamSlot>> paramSlots;
        core::param::ParamSlot visibleGroupParam;
        core::param::ParamSlot togglevisibleGroupParam;
        core::param::ParamSlot toggleOusideBboxParam;
        VisibleGroup visibleGroup;
        bool toggleOusideBbox;

        std::vector<Manipulator> manipulators;

        // Current state variables
        std::shared_ptr<CinematicUtils> current_utils;
        Keyframe current_selected_keyframe;
        glm::vec2 current_viewport;
        glm::mat4 current_mvp;
        glm::vec3 current_first_ctrl_point;
        glm::vec3 current_last_ctrl_point;
        camera_state_type current_cam_snapshot;

        vislib::math::Cuboid<float> current_bbox;

        glm::vec2 current_mouse;
        std::shared_ptr<std::vector<Manipulator>> current_hit;

        /**********************************************************************
        * functions
        **********************************************************************/

        bool updateManipulators(void);

        glm::vec2 world2ScreenSpace(glm::vec3 vec);

    };

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_KEYFRAMEMANIPULATOR_H_INCLUDED
