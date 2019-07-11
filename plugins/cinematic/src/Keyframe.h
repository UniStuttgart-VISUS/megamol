/*
 *Keyframe.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "Cinematic/Cinematic.h"

#include "vislib/graphics/Camera.h"
#include "vislib/math/Point.h"
#include "vislib/Serialisable.h"
#include "vislib/math/Vector.h"

namespace megamol {
namespace cinematic {

    /**
    * Keyframe Keeper.
    */
	class Keyframe{
	public:

		/** CTOR */
        Keyframe();

        Keyframe(float at, float st, vislib::math::Point<float, 3> pos, vislib::math::Vector<float, 3> up,
                    vislib::math::Point<float, 3> lookat, float aperture);

		/** DTOR */
		~Keyframe();

        /**
        *
        */
        inline bool operator==(Keyframe const& rhs){
			return ((this->camera == rhs.camera) && (this->animTime == rhs.animTime) && (this->simTime == rhs.simTime));
		}

        /**
        *
        */
        inline bool operator!=(Keyframe const& rhs) {
            return (!(this->camera == rhs.camera) || (this->animTime != rhs.animTime) || (this->simTime != rhs.simTime));
        }

        ///// GET /////

        /**
        *
        */     
        inline float GetAnimTime() {
            return this->animTime;
        }

        /**
        *
        */
        inline float GetSimTime() {
            return (this->simTime == 1.0f)?(1.0f-0.0000001f):(this->simTime);
        }

        /**
        *
        */
        inline vislib::math::Point<float, 3> GetCamPosition(){
            return this->camera.position;
		}

        /**
        *
        */
        inline vislib::math::Point<float, 3> GetCamLookAt(){
            return this->camera.lookat;
		}

        /**
        *
        */
        inline vislib::math::Vector<float, 3> GetCamUp(){
            return this->camera.up;
		}

        /**
        *
        */
        inline float GetCamApertureAngle(){
            return this->camera.apertureangle;
		}

        ///// SET /////

        /**
        *
        */
        inline void SetAnimTime(float t) {
            this->animTime = (t < 0.0f)?(0.0f):(t);
        }

        /**
        *
        */
        inline void SetSimTime(float t) {
            this->simTime = vislib::math::Clamp(t, 0.0f, 1.0f);
        }

        /**
        *
        */
        inline void SetCameraPosition(vislib::math::Point <float, 3> pos){
            this->camera.position = pos;
		}

        /**
        *
        */
        inline void SetCameraLookAt(vislib::math::Point <float, 3> look){
            this->camera.lookat = look;
		}

        /**
        *
        */
        inline void SetCameraUp(vislib::math::Vector<float, 3> up){
            this->camera.up = up;
		}

        /**
        *
        */
        inline void SetCameraApertureAngele(float apertureangle){
            this->camera.apertureangle = vislib::math::Clamp(apertureangle, 0.0f, 180.0f);
		}

        ///// SERIALISATION /////

        /**
        *
        */
        void Serialise(vislib::Serialiser& serialiser);

        /**
        *
        */
        void Deserialise(vislib::Serialiser& serialiser);

	private:

        /**********************************************************************
        * classes
        **********************************************************************/

        // Hard copy of camera parameters
        class Camera {
        public:
            bool operator==(Keyframe::Camera const& rhs) {
                return ((this->lookat == rhs.lookat) && (this->position == rhs.position) && 
                        (this->apertureangle == rhs.apertureangle) && (this->up == rhs.up));
            }
            vislib::math::Vector<float, 3> up;
            vislib::math::Point<float, 3>  position;
            vislib::math::Point<float, 3>  lookat;
            float                          apertureangle;
        };

        /**********************************************************************
        * variables
        **********************************************************************/

        // Simulation time is always in [0,1] and is relative to absolute total simulation time.
        float                    simTime;
        // Animation time [in seconds]
		float                    animTime;
        Keyframe::Camera         camera;

	};

} /* end namespace cinematic */
} /* end namespace megamol */
