/*
 *Keyframe.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED
#define MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED

#include "Cinematic/Cinematic.h"

#include "json.hpp"
#include <glm/glm.hpp>

#include "CinematicUtils.h"


namespace megamol {
namespace cinematic {


    /**
    * Keyframe Keeper.
    */
	class Keyframe{
	public:

		/** CTOR */
        Keyframe();

        Keyframe(float at, float st, camera_state_type cam);

		/** DTOR */
		~Keyframe();

        inline bool operator ==(Keyframe const& rhs){
			return ((this->camera_state ==  rhs.camera_state) && (this->anim_time == rhs.anim_time) && (this->sim_time == rhs.sim_time));
		}

        inline bool operator !=(Keyframe const& rhs) {
            return (!((*this) == rhs));
        }

        // GET ----------------------------------------------------------------
 
        inline float GetAnimTime() const {
            return this->anim_time;
        }

        inline float GetSimTime() const {
            return this->sim_time; // (this->sim_time == 1.0f) ? (1.0f - 0.0000001f) : (this->sim_time);
        }

        inline camera_state_type GetCameraState() const {
            return this->camera_state;
		}

        // SET ----------------------------------------------------------------

        inline void SetAnimTime(float t) {
            this->anim_time = (t < 0.0f)?(0.0f):(t);
        }

        inline void SetSimTime(float t) {
            this->sim_time = glm::clamp(t, 0.0f, 1.0f);
        }

        inline void SetCameraState(const camera_state_type& cam){
            this->camera_state = cam;
		}
    
        // SERIALISATION ------------------------------------------------------

        bool Serialise(nlohmann::json& inout_json, size_t index);

        bool Deserialise(const nlohmann::json& in_json);

	private:

        /**********************************************************************
        * variables
        **********************************************************************/

        camera_state_type camera_state;
        float sim_time; // Simulation time value is relative (always in [0,1])
		float anim_time;    
	};

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED