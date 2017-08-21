/*
 *Keyframe.h
 */


#ifndef MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED
#pragma once

#include "CinematicCamera/CinematicCamera.h"

#include "vislib/graphics/Camera.h"
#include "vislib/math/Point.h"
#include "vislib/Serialisable.h"
#include "vislib/math/Vector.h"


namespace megamol {
	namespace cinematiccamera {

		class Keyframe{

		public:
			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "Keyframe";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Holds a camera as well as a time";
			}

			/** ctor */
            Keyframe();
            Keyframe(vislib::graphics::Camera c, float t);

			/** dtor */
			~Keyframe();

			vislib::graphics::Camera getCamera(){
				return this->camera;
			}

			void setCamera(vislib::graphics::Camera c) {
				this->camera = c;
			}

			float getTime(){
				return time;
			}

			void setTime(float t) {
				this->time = t;
			}

			bool operator==(Keyframe const& rhs){
				return ((this->camera == rhs.camera) && (this->time == rhs.time));
			}

			vislib::math::Point<FLOAT, 3> getCamPosition(){
				return this->camera.Parameters()->Position();
			}

			vislib::math::Point<FLOAT, 3> getCamLookAt(){
				return this->camera.Parameters()->LookAt();
			}

			vislib::math::Vector<FLOAT, 3> getCamUp(){
				return this->camera.Parameters()->Up();
			}

			float getCamApertureAngle(){
				return this->camera.Parameters()->ApertureAngle();
			}
			
			vislib::SmartPtr<vislib::graphics::CameraParameters> getCamParameters(){
				return this->camera.Parameters();
			}

			void setCameraPosition(vislib::math::Point <float, 3> pos){
                this->camera.Parameters()->SetPosition(pos);
			}

			void setCameraLookAt(vislib::math::Point <float, 3> look){
                this->camera.Parameters()->SetLookAt(look);
			}

			void setCameraUp(vislib::math::Vector<float, 3> up){
                this->camera.Parameters()->SetUp(up);
			}

			void setCameraApertureAngele(float appertureAngle){
                this->camera.Parameters()->SetApertureAngle(appertureAngle);
			}

			void setCameraParameters(vislib::SmartPtr<vislib::graphics::CameraParameters> params){
                this->camera.SetParameters(params);
			}			

		private:

			vislib::graphics::Camera camera;
			float time;

		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */
#endif /* MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED */