/*
* KeyframeKeeper.h
*
*/

#ifndef MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/Module.h"
#include "vislib/Array.h"
#include "CinematicCamera/CinematicCamera.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "Keyframe.h"
#include "vislib/math/Cuboid.h"

namespace megamol {
	namespace cinematiccamera {

		/**
		* Call transporting keyframe data
		*
		*/
		class KeyframeKeeper : public megamol::core::Module {
		public:

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "KeyframeKeeper";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Manages Keyframes";
			}

			/** Ctor */
			KeyframeKeeper(void);

			/** Dtor */
			virtual ~KeyframeKeeper(void);

			static bool IsAvailable(void) {
				return true;
			}
			
		protected:
			virtual bool create(void);
			virtual void release(void);

			/** Array of keyframes */
			vislib::Array <Keyframe> keyframes;
			float selectedKeyframeIndex;

			int IDCounter;

		private:
			/** knowing the total Time before the Variable is changed is mandatory
				in order to scale the time of the keyframes correctly*/
			float prevTotalTime;

			/** Callback for getting all Keyframes*/
			bool cbAllKeyframes(core::Call& c);

			/** Callback for getting selected Keyframe*/
			bool cbSelectedKeyframe(core::Call& c);

			/** Callback for selecting a new Keyframe */
			bool cbSelectKeyframe(core::Call& c);

			/** Callback for creating new Keyframe */
			bool cbNewKeyframe(core::Call& c);

			/** Callback for deleting selected Keyframe */
			bool cbDelKeyframe(core::param::ParamSlot& slot);

			/**Callback for interpolating Keyframe */
			bool cbInterpolateKeyframe(core::Call& c);

			/**Callback for getting total Time */
			bool cbGetTotalTime(core::Call& c);

			/**Callback for setting total Time*/
			bool cbSetTotalTime(core::Call& c);

			/**Callback for getting Keyframe at a certain time */
			bool cbGetKeyframeAtTime(core::Call& c);

			/**Callback for saving Keyframes */
			bool cbSave(core::param::ParamSlot& slot);

			/**Callback for loading Keyframes */
			bool cbLoad(core::param::ParamSlot& slot);
			bool cbLoad(core::Call& c);

			bool cbAddKeyframeAtSelectedPosition(core::param::ParamSlot& slot);

            bool cbUpdate(core::Call& call);


			void updateParameters();

			Keyframe interpolateKeyframe(float idx);

		

			megamol::core::CalleeSlot cinematicRendererSlot;

			core::param::ParamSlot saveKeyframesParam;
			core::param::ParamSlot loadKeyframesParam;
			core::param::ParamSlot fileNameParam;
			core::param::ParamSlot totalTime;
			core::param::ParamSlot selectedKeyframe;
			core::param::ParamSlot addKeyframeAtSelectedPosition;
			core::param::ParamSlot deleteKeyframe;
			core::param::ParamSlot keyframeDistance;

			/**param for currentkeyframe Time */
			megamol::core::param::ParamSlot currentKeyframeTime;

			/**param for currentkeyframe Position */
			core::param::ParamSlot currentPos;

			/**param for currentkeyframe LookAt */
			core::param::ParamSlot currentLookAt;

			/**param for currentkeyframe Up */
			core::param::ParamSlot currentUp;

			vislib::math::Cuboid<float> boundingBox;

	//		megamol::core::AbstractSlot::Listener lisCreateKeyframe;

			vislib::Array<Keyframe> getKeyframes(){
				return keyframes;
			}

			float getSelectedKeyframeIndex(){
				return selectedKeyframeIndex;
			}

			/**sorts Keyframes by time*/
			void sortKeyframes();

		};

		/** Description class typedef */
		typedef core::factories::CallAutoDescription<KeyframeKeeper> KeyframeKeeperDescription;

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED */
