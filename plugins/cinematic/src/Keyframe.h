/*
 *Keyframe.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED
#define MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED

#include "Cinematic/Cinematic.h"

#include "mmcore/view/Camera_2.h"
#include "vislib/sys/Log.h"

#include "json.hpp"
#include <glm/glm.hpp>


namespace megamol {
namespace cinematic {


    /**
    * Keyframe Keeper.
    */
	class Keyframe{
	public:

        // Camera State Type
        typedef megamol::core::thecam::camera<cam_maths_type>::minimal_state_type cam_state_type;

		/** CTOR */
        Keyframe();

        Keyframe(float at, float st, cam_state_type cam);

		/** DTOR */
		~Keyframe();

        inline bool operator==(Keyframe const& rhs){
			return ((this->camStatesEqual(this->camera_state, rhs.camera_state)) && (this->anim_time == rhs.anim_time) && (this->sim_time == rhs.sim_time));
		}

        inline bool operator!=(Keyframe const& rhs) {
            return (!((*this) == rhs));
        }

        // GET ----------------------------------------------------------------
 
        inline float GetAnimTime() {
            return this->anim_time;
        }

        inline float GetSimTime() {
            return (this->sim_time == 1.0f)?(1.0f-0.0000001f):(this->sim_time);
        }

        inline cam_state_type GetCameraState(){
            return this->camera_state;
		}

        // SET ----------------------------------------------------------------

        inline void SetAnimTime(float t) {
            this->anim_time = (t < 0.0f)?(0.0f):(t);
        }

        inline void SetSimTime(float t) {
            this->sim_time = glm::clamp(t, 0.0f, 1.0f);
        }

        inline void SetCameraState(const cam_state_type& cam){
            this->camera_state = cam;
		}
    
        // SERIALISATION ------------------------------------------------------

        bool Serialise(std::string& json_string);

        bool Deserialise(const std::string& json_string);

	private:

        /**********************************************************************
        * variables
        **********************************************************************/

        cam_state_type camera_state;
        float sim_time; // Simulation time value is relative (always in [0,1])
		float anim_time;

        /**********************************************************************
        * functions
        **********************************************************************/

        /** Returns true if both states are euqal, false otherwise. */
        bool camStatesEqual(cam_state_type ls, cam_state_type rs);
    
	};

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED