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
     * Keyframe Manipulators.
     */
    class KeyframeManipulators {

    public:

        /** CTOR */
        KeyframeManipulators(void);

        /** DTOR */
        ~KeyframeManipulators(void);

        void UpdateExtents(vislib::math::Cuboid<float>& inout_bbox);

        bool UpdateRendering(const std::shared_ptr<std::vector<Keyframe>> keyframes, Keyframe selected_keyframe, glm::vec3 first_ctrl_pos, glm::vec3 last_ctrl_pos,
            const camera_state_type& minimal_snapshot, glm::vec2 viewport_dim, glm::mat4 mvp);

        bool PushRendering(CinematicUtils &utils);

        int GetSelectedKeyframePositionIndex(float mouse_x, float mouse_y);

        bool CheckForHitManipulator(float mouse_x, float mouse_y);

        bool ProcessHitManipulator(float mouse_x, float mouse_y);

        inline Keyframe GetManipulatedSelectedKeyframe(void) const  {
            return this->state.selected_keyframe;
        }

        inline glm::vec3 GetFirstControlPointPosition(void) const {
            return this->state.first_ctrl_point;
        }

        inline glm::vec3 GetLastControlPointPosition(void) const {
            return this->state.last_ctrl_point;
        }

        inline std::vector<megamol::core::param::ParamSlot*>& GetParams(void) {
            return this->paramSlots;
        }

        inline void ResetHitManipulator(void) {
            this->state.hit = nullptr;
        }

    private:

        /**********************************************************************
        * types and classes
        **********************************************************************/

        enum VisibleGroup : int {
            SELECTED_KEYFRAME_AND_CTRLPOINT_POSITION = 0,
            SELECTED_KEYFRAME_LOOKAT_AND_UP_VECTOR   = 1,
            VISIBLEGROUP_COUNT                       = 2
        };

        class Manipulator {
        public:
            enum Rigging {
                NONE,
                X_DIRECTION,
                Y_DIRECTION,
                Z_DIRECTION,
                VECTOR_DIRECTION,
                ROTATION
            };

            enum Variety {
                SELECTOR_KEYFRAME_POSITION,
                MANIPULATOR_FIRST_CTRLPOINT_POSITION,
                MANIPULATOR_LAST_CTRLPOINT_POSITION,
                MANIPULATOR_SELECTED_KEYFRAME_POSITION,
                MANIPULATOR_SELECTED_KEYFRAME_POSITION_USING_LOOKAT,
                MANIPULATOR_SELECTED_KEYFRAME_LOOKAT_VECTOR,
                MANIPULATOR_SELECTED_KEYFRAME_UP_VECTOR,
            };

            bool show;
            Variety variety;
            Rigging rigging;
            // Storing position for selectable keyframes and
            // direction for manipulators:
            glm::vec3 vector;
        };

        struct CurrentState {
            Keyframe selected_keyframe;
            glm::vec2 viewport;
            glm::mat4 mvp;
            glm::vec3 first_ctrl_point;
            glm::vec3 last_ctrl_point;
            camera_state_type cam_min_snapshot;
            vislib::math::Cuboid<float> bbox;
            std::shared_ptr<Manipulator> hit;
            glm::vec2 last_mouse;
            int selected_index;
            float point_radius;
            float line_width;
            float line_length;
            float lookat_length;
        };

        /**********************************************************************
        * variables
        **********************************************************************/

        core::param::ParamSlot visibleGroupParam;
        core::param::ParamSlot toggleVisibleGroupParam;
        core::param::ParamSlot toggleOusideBboxParam;

        std::vector<megamol::core::param::ParamSlot*> paramSlots;
        VisibleGroup visibleGroup;
        bool toggleOusideBbox;

        std::vector<Manipulator> manipulators;
        std::vector<Manipulator> selectors;
        CurrentState state;

        /**********************************************************************
        * functions
        **********************************************************************/
        
        glm::vec3 getManipulatorOrigin(Manipulator &manipulator);

        glm::vec3 getActualManipulatorPosition(Manipulator &manipulator);

        glm::vec4 getManipulatorColor(Manipulator &manipulator, CinematicUtils &utils);

        bool checkMousePointIntersection(Manipulator &manipulator, glm::vec2 mouse, glm::vec3 cam_up);

        glm::vec2 world2ScreenSpace(glm::vec3 vec);

    };

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_KEYFRAMEMANIPULATOR_H_INCLUDED
