/*
* KeyframeKeeper.h
*
*/

#ifndef MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"

#include "Keyframe.h"


using namespace vislib;


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

            // ...
			static bool IsAvailable(void) {
				return true;
			}
			
		protected:

            // ...
			virtual bool create(void);

            // ...
			virtual void release(void);

		private:

            /**********************************************************************
            * functions
            ***********************************************************************/

            // Get an interpolated keyframe at time
            Keyframe interpolateKeyframe(float time);

            // Add new keyframe to keyframe array
            bool addKeyframe(Keyframe kf);

            // Replace keyframe in keyframe array
            bool replaceKeyframe(Keyframe kf);

            // Delete keyframe from keyframe array
            bool deleteKeyframe(Keyframe kf);

            // Load keyframes from file
            void loadKeyframes();

            // Save keyframes to file
            void saveKeyframes();

            /**********************************************************************
            * variables
            **********************************************************************/

            // variables shared with call
            vislib::Array<Keyframe>              keyframes;
            vislib::math::Cuboid<float>          boundingBox;
            Keyframe                             selectedKeyframe;
            float                                totalTime;
            SmartPtr<graphics::CameraParameters> cameraParam; 

            // variables only used in keyframe keeper
            vislib::StringA                     filename;

            /**********************************************************************
            * callback stuff
            **********************************************************************/

            megamol::core::CalleeSlot cinematicCallSlot;

			/** Callback for updating the keyframe keeper */
			bool CallForUpdateKeyframeKeeperData(core::Call& c);
			/** */
			bool CallForSetTotalTime(core::Call& c);
			/** */
			bool CallForRequestInterpolatedKeyframe(core::Call& c);
			/** */
			bool CallForSetSelectedKeyframe(core::Call& c);
			/** */
			bool CallForSetCameraForKeyframe(core::Call& c);

            /**********************************************************************
            * parameters
            **********************************************************************/

            /** */
            core::param::ParamSlot addKeyframeParam;
            /** */
            core::param::ParamSlot replaceKeyframeParam;
            /** */
            core::param::ParamSlot deleteSelectedKeyframeParam;
			/**param for currentkeyframe Time */
			core::param::ParamSlot editCurrentTimeParam;
			/**param for currentkeyframe Position */
			core::param::ParamSlot editCurrentPosParam;
			/**param for currentkeyframe LookAt */
			core::param::ParamSlot editCurrentLookAtParam;
			/**param for currentkeyframe Up */
			core::param::ParamSlot editCurrentUpParam;
            /** */
            core::param::ParamSlot setTotalTimeParam;
            /** */
            core::param::ParamSlot saveKeyframesParam;
            /** */
            core::param::ParamSlot loadKeyframesParam;
            /** */
            core::param::ParamSlot fileNameParam;
            /** */
            core::param::ParamSlot  editCurrentApertureParam;

		};

		/** Description class typedef */
		typedef core::factories::CallAutoDescription<KeyframeKeeper> KeyframeKeeperDescription;

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED */
