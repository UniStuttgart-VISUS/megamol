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
            * variables
            **********************************************************************/

            // Variables shared/updated with call
            vislib::Array<vislib::math::Point<float, 3> > interpolCamPos;
            vislib::Array<Keyframe>              keyframes;
            vislib::math::Cuboid<float>          boundingBox;
            Keyframe                             selectedKeyframe;
            Keyframe                             dragDropKeyframe;
            float                                totalAnimTime;
            float                                totalSimTime;
            unsigned int                         interpolSteps;
            vislib::math::Point<float, 3>        bboxCenter;
            unsigned int                         fps;

            vislib::math::Vector<float, 3>       camViewUp;
            vislib::math::Point<float, 3>        camViewPosition;
            vislib::math::Point<float, 3>        camViewLookat;
            float                                camViewApertureangle;
            
            // Variables only used in keyframe keeper
            vislib::StringA                      filename;

            /**********************************************************************
            * functions
            ***********************************************************************/

            /** Get an interpolated keyframe at specific time. */
            Keyframe interpolateKeyframe(float time);

            /** Add new keyframe to keyframe array. */
            bool addKeyframe(Keyframe kf);

            /** change existing keyframe in keyframe array.*/
            bool changeKeyframe(Keyframe kf);

            /** Delete keyframe from keyframe array.*/
            bool deleteKeyframe(Keyframe kf);

            /** Load keyframes from file.*/
            void loadKeyframes(void);

            /** Save keyframes to file.*/
            void saveKeyframes(void);

            /** Refresh interpolated camera positions (called when keyframe array changes). */
            void refreshInterpolCamPos(unsigned int s);

            /** Updating edit parameters without setting them dirty.*/
            void updateEditParameters(Keyframe kf, bool setDirty);

            /** Set speed between all keyframes to same speed 
             *  Uses interpolSteps for approximation of path between keyframe positions 
             */
            void setSameSpeed(void);

            /** */
            void snapKeyframe2AnimFrame(Keyframe *kf);

            /** */
            void snapKeyframe2SimFrame(Keyframe *kf);

            /**********************************************************************
            * callback stuff
            **********************************************************************/

            megamol::core::CalleeSlot cinematicCallSlot;

			/** Callback for updating parameters of the keyframe keeper */
			bool CallForGetUpdatedKeyframeData(core::Call& c);
			/** Callback for  */
			bool CallForSetSimulationData(core::Call& c);
			/** Callback for  */
			bool CallForGetInterpolCamPositions(core::Call& c);
			/** Callback for  */
			bool CallForSetSelectedKeyframe(core::Call& c);
            /** */
            bool CallForGetSelectedKeyframeAtTime(core::Call& c);
			/** Callback for  */
			bool CallForSetCameraForKeyframe(core::Call& c);
            /** Callback for dragging selected keyframe */
            bool CallForSetDragKeyframe(core::Call& c); 
            /** Callback for dropping selected keyframe */
            bool CallForSetDropKeyframe(core::Call& c);


            /**********************************************************************
            * parameters
            **********************************************************************/

            /** */
            core::param::ParamSlot addKeyframeParam;
            /** */
            core::param::ParamSlot changeKeyframeParam;
            /** */
            core::param::ParamSlot deleteSelectedKeyframeParam;
            /** */
            core::param::ParamSlot setKeyframesToSameSpeed;
			/**param for current keyframe aniamtion time */
			core::param::ParamSlot editCurrentAnimTimeParam;
            /**param for current keyframe simulation time */
            core::param::ParamSlot editCurrentSimTimeParam;
			/**param for current keyframe Position */
			core::param::ParamSlot editCurrentPosParam;
			/**param for current keyframe LookAt */
			core::param::ParamSlot editCurrentLookAtParam;
			/**param for current keyframe Up */
			core::param::ParamSlot editCurrentUpParam;
            /** */
            core::param::ParamSlot setTotalAnimTimeParam;
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
            /** */
            core::param::ParamSlot  snapAnimFramesParam;
            /** */
            core::param::ParamSlot  snapSimFramesParam;
		};

		/** Description class typedef */
		typedef core::factories::CallAutoDescription<KeyframeKeeper> KeyframeKeeperDescription;

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED */
