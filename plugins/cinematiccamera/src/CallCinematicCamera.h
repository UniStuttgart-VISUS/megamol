/*
* CallCinematicCamera.h
*
*/

#ifndef MEGAMOL_CINEMATICCAMERA_CALLCINCAM_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_CALLCINCAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"
#include "Keyframe.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Point.h"

namespace megamol {
	namespace cinematiccamera {

		/**
		* Call transporting keyframe data
		*
		*/
		class CallCinematicCamera : public core::AbstractGetDataCall {
		public:

			/** function name for getting all Keyframes */
			static const unsigned int CallForGetKeyframes = 0;
			/** function name for getting selected Keyframes */
			static const unsigned int CallForGetSelectedKeyframe = 1;
			/** function name for setting the selected Keyframe */
			static const unsigned int CallForSelectKeyframe = 2;
			/**function name for getting interpolated Keyframe */
			static const unsigned int CallForInterpolatedKeyframe = 3;
			/**function name for getting total time */
			static const unsigned int CallForGetTotalTime = 4;
			/**function name for getting a keyframe at a certain time*/
			static const unsigned int CallForGetKeyframeAtTime = 5;
			/**function name for adding a keyframe at a certain position*/
			static const unsigned int CallForNewKeyframeAtPosition = 6;
			/**function name for load of a keyframe*/
			static const unsigned int CallForLoadKeyframe = 7;
			/**function name for setting of total time*/
			static const unsigned int CallForSetTotalTime = 8;
            /** Requests the selected key frame being replaced by 'cameraForNewKeyframe' */
            static const unsigned int CallForKeyFrameUpdate = 9;

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
				return 10;
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
					case CallForGetKeyframes: return "CallForGetKeyframes";
					case CallForGetSelectedKeyframe: return "CallForGetSelectedKeyframe";
					case CallForSelectKeyframe: return "CallForSelectKeyframe";
					case CallForInterpolatedKeyframe: return "CallForInterpolatedKeyframe";
					case CallForGetTotalTime: return "CallForGetTotalTime";
					case CallForGetKeyframeAtTime : return "CallForGetKeyframeAtTime";
					case CallForNewKeyframeAtPosition: return "CallForNewKeyframeAtPosition";
					case CallForLoadKeyframe: return "CallForLoadKeyframe";
					case CallForSetTotalTime: return "CallForSetTotalTime";
                    case CallForKeyFrameUpdate: return "CallForKeyFrameUpdate";
					default: return "";
				}
				
			}

			/** Ctor */
			CallCinematicCamera(void);

			/** Dtor */
			virtual ~CallCinematicCamera(void);

			inline void setTimeofKeyframeToGet(float time){
				timeToGet = time;
			}
			
			inline float getTimeofKeyframeToGet(){
				return timeToGet;
			}
			/**
			* TODO
			*/
			inline Keyframe getKeyframeAtTime(float t){
				
			}

			inline vislib::Array <Keyframe>* getKeyframes(){
				return keyframes;
			}

			/**
			* Returning the selected Keyframe if an exact Keyframe is selected.
			* Returning a generic keyframe otherwise
			*/
			inline cinematiccamera::Keyframe getSelectedKeyframe() {

				if (ceilf(selectedKeyframe) == selectedKeyframe) {
					return (*keyframes)[(int)ceil(selectedKeyframe)];
				}
				else {
					return interpolatedKeyframe;
				}
			}

			inline bool setSelectedKeyframeIndex(float select){
				if (select >= 0 && select <= keyframes->Count())
					selectedKeyframe = select;
				else
					return false;
				return true;
			}

			inline void addKeyframe (Keyframe keyframe) {
				keyframes->Add(keyframe);
				boundingbox->GrowToPoint(keyframe.getCamPosition());
			}

			inline void deleteKeyframe(int number) {
				keyframes->RemoveAt(number);
			}


			inline void setKeyframes(vislib::Array <Keyframe> *kfs){
				keyframes = kfs;
			}

			inline float getSelectedKeyframeIndex(){
				return selectedKeyframe;
			}

			inline void setBoundingBox(vislib::math::Cuboid<float> *bbx){
				this->boundingbox = bbx;
			}

			inline vislib::math::Cuboid<float>* getBoundingBox(){
				return this->boundingbox;
			}

			inline void setIndexToInterpolate(float f){
				interpolatedIndex = f;
			}

			inline float getIndexToInterPolate(){
				return interpolatedIndex;
			}

			inline void setInterpolatedKeyframe(Keyframe k){
				interpolatedKeyframe = k;
			}
			inline Keyframe getInterpolatedKeyframe(){
				return interpolatedKeyframe;
			}

			inline void setTotalTime(float f){
				totalTime = f;
			}

			inline float getTotalTime(){
				return totalTime;
			}

			inline bool setTimeToSet(float f){
				if (f >= 0 && f <= 1){
					timeToSet = f;
				}
				else {
					vislib::sys::Log::DefaultLog.WriteError("Tried to set the time of a Keyframe out of the interval of [0,1]");
					return false;
				}
			}

			inline float getTimeToSet(){
				return timeToSet;
			}

			inline void setPosToSet(vislib::math::Point<float, 3> pt){
				posToSet = pt;
			}
			inline void setLookToSet(vislib::math::Point<float, 3> pt){
				lookToSet = pt;
			}

			inline vislib::math::Point<float, 3> getPosToSet(){
				return posToSet;
			}
			inline vislib::math::Point<float, 3> getLookToSet(){
				return lookToSet;
			}

			inline void setCameraForNewKeyframe(vislib::graphics::Camera cam){
				cameraForNewKeyframe = cam;
			}

			vislib::graphics::Camera getCameraForNewKeyframe(){
				return cameraForNewKeyframe;
			}

		protected:

			/** Array of keyframes */
			vislib::Array <Keyframe> *keyframes;
			float selectedKeyframe;

			vislib::math::Cuboid<float> *boundingbox;

			Keyframe interpolatedKeyframe;
			float interpolatedIndex;
			float totalTime;
			float timeToSet;	
			vislib::math::Point<float, 3> posToSet;
			vislib::math::Point<float, 3> lookToSet;
			float timeToGet;
			vislib::graphics::Camera cameraForNewKeyframe;
		};

		/** Description class typedef */
		typedef core::factories::CallAutoDescription<CallCinematicCamera> CallCinematicCameraDescription;

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_CALLCINCAM_H_INCLUDED */
