/*
 *Keyframe.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED
#define MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED

#include "Cinematic/Cinematic.h"

#include "vislib/graphics/Camera.h"
#include "vislib/Serialisable.h"

#include <glm/glm.hpp>


namespace megamol {
namespace cinematic {


    /**
    * Keyframe Keeper.
    */
	class Keyframe{
	public:

		/** CTOR */
        Keyframe();

        Keyframe(float at, float st, glm::vec3 pos, glm::vec3 up, glm::vec3 lookat, float aperture);

		/** DTOR */
		~Keyframe();

        inline bool operator==(Keyframe const& rhs){
			return ((this->camera == rhs.camera) && (this->animTime == rhs.animTime) && (this->simTime == rhs.simTime));
		}

        inline bool operator!=(Keyframe const& rhs) {
            return (!(this->camera == rhs.camera) || (this->animTime != rhs.animTime) || (this->simTime != rhs.simTime));
        }

        ///// GET /////
 
        inline float GetAnimTime() {
            return this->animTime;
        }

        inline float GetSimTime() {
            return (this->simTime == 1.0f)?(1.0f-0.0000001f):(this->simTime);
        }

        inline glm::vec3 GetCamPosition(){
            return this->camera.position;
		}

        inline glm::vec3 GetCamLookAt(){
            return this->camera.lookat;
		}

        inline glm::vec3 GetCamUp(){
            return this->camera.up;
		}

        inline float GetCamApertureAngle(){
            return this->camera.apertureangle;
		}

        ///// SET /////

        inline void SetAnimTime(float t) {
            this->animTime = (t < 0.0f)?(0.0f):(t);
        }

        inline void SetSimTime(float t) {
            this->simTime = glm::clamp(t, 0.0f, 1.0f);
        }

        inline void SetCameraPosition(glm::vec3 pos){
            this->camera.position = pos;
		}

        inline void SetCameraLookAt(glm::vec3 look){
            this->camera.lookat = look;
		}

        inline void SetCameraUp(glm::vec3 up){
            this->camera.up = up;
		}

        inline void SetCameraApertureAngele(float apertureangle){
            this->camera.apertureangle = glm::clamp(apertureangle, 0.0f, 180.0f);
		}

        ///// SERIALISATION /////

        void Serialise(vislib::Serialiser& serialiser);

        void Deserialise(vislib::Serialiser& serialiser);

	private:

        /**********************************************************************
        * classes
        **********************************************************************/

        class Camera {
        public:
            bool operator==(Keyframe::Camera const& rhs) {
                return ((this->lookat == rhs.lookat) && (this->position == rhs.position) && 
                        (this->apertureangle == rhs.apertureangle) && (this->up == rhs.up));
            }
            glm::vec3 up;
            glm::vec3 position;
            glm::vec3 lookat;
            float apertureangle;
        };

        /**********************************************************************
        * variables
        **********************************************************************/

        /// Simulation time is always in [0,1] and is relative to absolute total simulation time.
        float simTime;
		float animTime;
        Keyframe::Camera camera;

	};

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED