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

            /** create */
			virtual bool create(void);

            /** release */
			virtual void release(void);

		private:

            /**********************************************************************
            * functions
            ***********************************************************************/

            /** Get an interpolated keyframe at specific time. */
            Keyframe interpolateKeyframe(float time);

            /** Add new keyframe to keyframe array. */
            bool addKeyframe(Keyframe kf);

            /** Replace existing keyframe in keyframe array.*/
            bool replaceKeyframe(Keyframe kf);

            /** Delete keyframe from keyframe array.*/
            bool deleteKeyframe(Keyframe kf);

            /** Load keyframes from file.*/
            void loadKeyframes();

            /** Save keyframes to file.*/
            void saveKeyframes();

            /** Refresh interpolated camera positions (called when keyframe array changes). */
            void refreshInterpolCamPos(unsigned int s);

            /** Updating edit parameters without setting them dirty.*/
            void updateEditParameters(Keyframe k);

            /** Set speed between al keyframes to same speed 
             *  Uses interpolSteps for approximation of keyframe positions 
             *  
             */
            void setSameSpeed();

            /**********************************************************************
            * variables
            **********************************************************************/

            // Variables shared/updated with call
            Array<Keyframe>                      keyframes;
            Array<math::Point<float, 3> >        interpolCamPos;
            math::Cuboid<float>                  boundingBox;
            Keyframe                             selectedKeyframe;
            Keyframe                             dragDropKeyframe;
            float                                totalTime;
            float                                maxAnimTime;
            unsigned int                         interpolSteps;
            SmartPtr<graphics::CameraParameters> cameraParam; 
            vislib::math::Point<float, 3>        bboxCenter;

            // Variables only used in keyframe keeper
            vislib::StringA                      filename;
            bool                                 sameSpeed;
            float                                totVelocity;

            /**********************************************************************
            * callback stuff
            **********************************************************************/

            megamol::core::CalleeSlot cinematicCallSlot;

			/** Callback for updating parameters of the keyframe keeper */
			bool CallForGetUpdatedKeyframeData(core::Call& c);
			/** Callback for applying total time of animation */
			bool CallForSetAnimationData(core::Call& c);
			/** Callback for calculating new interpolated camera positions */
			bool CallForInterpolatedCamPos(core::Call& c);
			/** Callback for updating selected keyframe at new given time */
			bool CallForSetSelectedKeyframe(core::Call& c);
			/** Callback for updating current camera parameters for new keyframe */
			bool CallForSetCameraForKeyframe(core::Call& c);
            /** Callback for dragging selected keyframe */
            bool CallForDragKeyframe(core::Call& c); 
            /** Callback for dropping selected keyframe */
            bool CallForDropKeyframe(core::Call& c);

            /**********************************************************************
            * parameters
            **********************************************************************/

            /** */
            core::param::ParamSlot addKeyframeParam;
            /** */
            core::param::ParamSlot replaceKeyframeParam;
            /** */
            core::param::ParamSlot deleteSelectedKeyframeParam;
            /** */
            core::param::ParamSlot  setKeyframesToSameSpeed;
			/**param for current keyframe Time */
			core::param::ParamSlot editCurrentTimeParam;
			/**param for current keyframe Position */
			core::param::ParamSlot editCurrentPosParam;
			/**param for current keyframe LookAt */
			core::param::ParamSlot editCurrentLookAtParam;
			/**param for current keyframe Up */
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
            /** */
            core::param::ParamSlot  resetLookAtParam;


		};

		/** Description class typedef */
		typedef core::factories::CallAutoDescription<KeyframeKeeper> KeyframeKeeperDescription;

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED */
