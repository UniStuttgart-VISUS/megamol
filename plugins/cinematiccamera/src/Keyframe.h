/*
 *Keyframe.h
 */


#ifndef MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED
#pragma once

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
			Keyframe(int ID);
			Keyframe(vislib::graphics::Camera camera, float time, int ID);
			/** dtor */
			~Keyframe();


			vislib::graphics::Camera getCamera(){
				return this->camera;
			}

			void setCamera(vislib::graphics::Camera camera) {
				this->camera = camera;
			}

			float getTime(){
				return time;
			}

			void setTime(float time) {
				this->time = time;
			}

			bool operator==(Keyframe const& rhs){
				return (camera == rhs.camera && time == rhs.time);
			}

			vislib::math::Point<FLOAT, 3> getCamPosition(){
				return camera.Parameters()->Position();
			}

			vislib::math::Point<FLOAT, 3> getCamLookAt(){
				return camera.Parameters()->LookAt();
			}

			vislib::math::Vector<FLOAT, 3> getCamUp(){
				return camera.Parameters()->Up();
			}

			float getCamApertureAngle(){
				return camera.Parameters()->ApertureAngle();
			}
			
			vislib::SmartPtr<vislib::graphics::CameraParameters> getCamParameters(){
				return camera.Parameters();
			}

			void putCamParameters(vislib::SmartPtr<vislib::graphics::CameraParameters> params){
				auto p = camera.Parameters();
				params->SetView(p->Position(), p->LookAt(), p->Up());
				params->SetApertureAngle(p->ApertureAngle());
			}

			void setCameraPosition(vislib::math::Point <float, 3> pos){
				camera.Parameters()->SetPosition(pos);
			}

			void setCameraLookAt(vislib::math::Point <float, 3> look){
				camera.Parameters()->SetLookAt(look);
			}

			void setCameraUp(vislib::math::Vector<float, 3> up){
				camera.Parameters()->SetUp(up);
			}

			void setCameraApertureAngele(float appertureAngle){
				camera.Parameters()->SetApertureAngle(appertureAngle);
			}

			void setCameraParameters(vislib::SmartPtr<vislib::graphics::CameraParameters> params){
				camera.SetParameters(params);
			}			

			int getID(){
				return ID;
			}
			/**
			* ONLY USE THIS IF AN INTERPOLATED KEYFRAME
			* IS TO BE ADDED TO THE KEYFRAME ARRAY!!!
			*/
			void setID(int id){
				ID = id;
			}

		private:
			vislib::graphics::Camera camera;
			// normalized
			float time;

			// interpolated keyframes have ID -1
			// dummy keyframe has ID -2
			int ID;
		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */
#endif /* MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED */