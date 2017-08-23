/*
* CallCinematicCamera.h
*
*/

#ifndef MEGAMOL_CINEMATICCAMERA_CALLCINCAM_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_CALLCINCAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/Array.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Point.h"

#include "Keyframe.h"

#include <iostream>


using namespace vislib;


namespace megamol {
	namespace cinematiccamera {

		/**
		* Call transporting keyframe data
		*
		*/
		class CallCinematicCamera : public core::AbstractGetDataCall {
		public:

			/** function name for getting all Keyframes */
			static const unsigned int CallForUpdateKeyframeKeeperData     = 0;
			static const unsigned int CallForSetTotalTime                 = 1;
			static const unsigned int CallForRequestInterpolatedKeyframe  = 2;
			static const unsigned int CallForSetSelectedKeyframe          = 3;
			static const unsigned int CallForSetCameraForKeyframe         = 4;

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "CallCinematicCamera";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Call keyframes and keyframe data";
			}

			/**
			* Answer the number of functions used for this call.
			*
			* @return The number of functions used for this call.
			*/
			static unsigned int FunctionCount(void) {
				return 5;
			}

			/**
			* Answer the name of the function used for this call.
			*
			* @param idx The index of the function to return it's name.
			*
			* @return The name of the requested function.
			*/
			static const char * FunctionName(unsigned int idx) {
				switch (idx) {
					case CallForUpdateKeyframeKeeperData:     return "CallForUpdateKeyframeKeeperData";
					case CallForSetTotalTime:                 return "CallForSetTotalTime";
					case CallForRequestInterpolatedKeyframe:  return "CallForRequestInterpolatedKeyframe";
					case CallForSetSelectedKeyframe:          return "CallForSetSelectedKeyframe";
					case CallForSetCameraForKeyframe:         return "CallForSetCameraForKeyframe";
					default: return "";
				}
				
			}

			/** Ctor */
			CallCinematicCamera(void);

			/** Dtor */
			virtual ~CallCinematicCamera(void);


            /**********************************************************************
            * functions
            **********************************************************************/

            // KEYFRAME ARRAY
			inline vislib::Array<Keyframe>* getKeyframes(){
				return this->keyframes;
			}
            inline void setKeyframes(vislib::Array<Keyframe>* kfs) {
                this->keyframes = kfs;
            }


            // BOUNDINGBOX
            inline void setBoundingBox(vislib::math::Cuboid<float>* bbx) {
                this->boundingbox = bbx;
            }
            inline vislib::math::Cuboid<float> *getBoundingBox() {
                return this->boundingbox;
            }


            // SELECTED KEYFRAME
            inline void setSelectedKeyframeTime(float t) { 
                this->selectedTime = t;
            }
            inline float getSelectedKeyframeTime() {
                return this->selectedTime;
            }


            inline Keyframe getSelectedKeyframe() {
                return this->selectedKeyframe;
            }
            inline void setSelectedKeyframe(Keyframe k) {
                this->selectedKeyframe = k;
            }


            // INTERPOLATED KEYFRAME
			inline void setInterpolatedKeyframeTime(float t){
                this->interpolatedTime = t;
			}
            inline float getInterpolatedKeyframeTime(){
                return this->interpolatedTime;
            }


			inline Keyframe getInterpolatedKeyframe(){
				return this->interpolatedKeyframe;
			}
            inline void setInterpolatedKeyframe(Keyframe k){
                this->interpolatedKeyframe = k;
            }


            // TOTAL TIME
            inline void setTotalTime(float f) {
                this->totalTime = f;
            }
			inline float getTotalTime(){
				return this->totalTime;
			}


            // CAMERA PARAMETER
            inline void setCameraParameter(SmartPtr<graphics::CameraParameters> c) {
                this->cameraParam = c;
            }
            inline SmartPtr<graphics::CameraParameters> getCameraParameter() {
                return this->cameraParam;
            }


		private:

            /**********************************************************************
            * variables
            **********************************************************************/

			// Pointer to array of keyframes
			vislib::Array<Keyframe>				*keyframes;
			vislib::math::Cuboid<float>		    *boundingbox;
            Keyframe						     selectedKeyframe;
            float								 selectedTime;
			Keyframe						     interpolatedKeyframe;
			float							     interpolatedTime;
			float								 totalTime;
            SmartPtr<graphics::CameraParameters> cameraParam;

		};

		/** Description class typedef */
		typedef core::factories::CallAutoDescription<CallCinematicCamera> CallCinematicCameraDescription;

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_CALLCINCAM_H_INCLUDED */
