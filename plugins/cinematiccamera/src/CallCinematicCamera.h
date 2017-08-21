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
			static const unsigned int CallForUpdateKeyframeKeeper = 0;

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
				return 1;
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
					case CallForUpdateKeyframeKeeper: return "CallForUpdateKeyframeKeeper";
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

            inline void setChangedKeyframes(bool b) {
                this->keyframesChanged = b;
            }
            inline bool changedKeyframes() {
                return this->keyframesChanged;
            }


            // BOUNDINGBOX
            inline void setBoundingBox(vislib::math::Cuboid<float>* bbx) {
                this->boundingbox = bbx;
            }
            inline vislib::math::Cuboid<float> *getBoundingBox() {
                return this->boundingbox;
            }


            // SELECTED KEYFRAME
            inline void setSelectedKeyframeTime(float t) { // set value for one specific ccc (called from keyframe keeper)
                this->selectedTime = t;
            }
            inline float getSelectedKeyframeTime() {
                return this->selectedTime;
            }

            inline void setChangedSelectedKeyframeTime(bool b) {
                this->selTimeChanged = b;
            }
            inline bool changedSelectedKeyframeTime() {
                return this->selTimeChanged;
            }

            inline Keyframe* getSelectedKeyframe() {
                return this->selectedKeyframe;
            }
            inline void setSelectedKeyframe(Keyframe* k) {
                this->selectedKeyframe = k;
            }


            // INTERPOLATED KEYFRAME
			inline void setInterpolatedKeyframeTime(float t){
                this->interpolatedTime = t;
			}
            inline float getInterpolatedKeyframeTime(){
                return this->interpolatedTime;
            }

            inline void setChangedInterpolatedKeyframeTime(bool b) {
                this->intTimeChanged = b;
            }
            inline bool changedInterpolatedKeyframeTime(){
                return this->intTimeChanged;
            }

			inline Keyframe* getInterpolatedKeyframe(){
				return this->interpolatedKeyframe;
			}
            inline void setInterpolatedKeyframe(Keyframe* k){
                this->interpolatedKeyframe = k;
            }


            // TOTAL TIME
            inline void setTotalTime(float f) {
                this->totalTime = f;
            }
			inline float getTotalTime(){
				return this->totalTime;
			}

            inline void setChangedTotalTime(bool b) {
                this->totTimeChanged = b;
            }
            inline bool changedTotalTime(){
                return this->totTimeChanged;
            }

            // CAMERA PARAMETER
            inline void setCameraParameter(SmartPtr<graphics::CameraParameters> c) {
                this->cameraParam = c;
            }
            inline SmartPtr<graphics::CameraParameters> getCameraParameter() {
                return this->cameraParam;
            }

            inline void setChangedCameraParameter(bool b) {
                this->camParamChanged = b;
            }
            inline bool changedCameraParameter() {
                return this->camParamChanged;
            }


		private:

            /**********************************************************************
            * variables
            **********************************************************************/

			// Pointer to array of keyframes
			vislib::Array<Keyframe>     *keyframes;
            bool                        keyframesChanged;

			vislib::math::Cuboid<float> *boundingbox;

            Keyframe                    *selectedKeyframe;
            float                       selectedTime;
            bool                        selTimeChanged;

			Keyframe                    *interpolatedKeyframe;
			float                       interpolatedTime;
            bool                        intTimeChanged;

			float                       totalTime;
            bool                        totTimeChanged;

            SmartPtr<graphics::CameraParameters> cameraParam;
            bool                       camParamChanged;
		};

		/** Description class typedef */
		typedef core::factories::CallAutoDescription<CallCinematicCamera> CallCinematicCameraDescription;

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_CALLCINCAM_H_INCLUDED */
