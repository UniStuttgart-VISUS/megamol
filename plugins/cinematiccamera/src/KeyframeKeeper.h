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
            vislib::math::Vector<float, 3>       firstCtrllPos;
            vislib::math::Vector<float, 3>       lastCtrllPos;
            float                                totalAnimTime;
            float                                totalSimTime;
            unsigned int                         interpolSteps;
            vislib::math::Point<float, 3>        modelBboxCenter;
            unsigned int                         fps;

            vislib::math::Vector<float, 3>       camViewUp;
            vislib::math::Point<float, 3>        camViewPosition;
            vislib::math::Point<float, 3>        camViewLookat;
            float                                camViewApertureangle;
            
            // Variables only used in keyframe keeper
            vislib::StringA                      filename;
            bool                                 simTangentStatus;
            float                                tl; // Global interpolation spline tangent length of keyframes

            // undo queue stuff -----------------------------------------------

            enum UndoActionEnum {
                UNDO_NONE   = 0,
                UNDO_ADD    = 1,
                UNDO_DELETE = 2,
                UNDO_MODIFY = 3
            };

            class UndoAction {  
                public:
                    /** functions **/
                    UndoAction() {
                        this->action       = KeyframeKeeper::UndoActionEnum::UNDO_NONE;
                        this->keyframe     = Keyframe();
                        this->prevKeyframe = Keyframe();
                    }
                    UndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prevkf) {
                        this->action       = act;
                        this->keyframe     = kf;
                        this->prevKeyframe = prevkf;
                    }
                    ~UndoAction() { }
                    inline bool operator==(UndoAction const& rhs) {
                        return ((this->action == rhs.action) && (this->keyframe == rhs.keyframe) && (this->prevKeyframe == rhs.prevKeyframe));
                    }
                    inline bool operator!=(UndoAction const& rhs) {
                        return (!(this->action == rhs.action) || (this->keyframe != rhs.keyframe) || (this->prevKeyframe != rhs.prevKeyframe));
                    }
                    /** variables **/
                   UndoActionEnum action;
                   Keyframe       keyframe;
                   Keyframe       prevKeyframe;
            };

            vislib::Array<UndoAction> undoQueue;

            int undoQueueIndex;

            /**********************************************************************
            * functions
            ***********************************************************************/

            /** Get an interpolated keyframe at specific time. */
            Keyframe interpolateKeyframe(float time);

            /** Add new keyframe to keyframe array. */
            bool addKeyframe(Keyframe kf, bool undo);

            /** replace existing keyframe in keyframe array.*/
            bool replaceKeyframe(Keyframe oldkf, Keyframe newkf, bool undo);

            /** Delete keyframe from keyframe array.*/
            bool deleteKeyframe(Keyframe kf, bool undo);

            /** Load keyframes from file.*/
            void loadKeyframes(void);

            /** Save keyframes to file.*/
            void saveKeyframes(void);

            /** Refresh interpolated camera positions (called when keyframe array changes). */
            void refreshInterpolCamPos(unsigned int s);

            /** Updating edit parameters without setting them dirty.*/
            void updateEditParameters(Keyframe kf);

            /** Set speed between all keyframes to same speed 
             *  Uses interpolSteps for approximation of path between keyframe positions 
             */
            void setSameSpeed(void);

            /** */
            void linearizeSimTangent(Keyframe stkf);

            /** */
            void snapKeyframe2AnimFrame(Keyframe *kf);

            /** */
            void snapKeyframe2SimFrame(Keyframe *kf);

            /**   */
            bool undo();

            /**   */
            bool redo();

            /**   */
            bool addNewUndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prevkf);

            /** */
            vislib::math::Vector<float, 3> interpolation(float u, vislib::math::Vector<float, 3> v0, vislib::math::Vector<float, 3> v1, vislib::math::Vector<float, 3> v2, vislib::math::Vector<float, 3> v3);
            float interpolation(float u, float f0, float f1, float f2, float f3);

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
            /** Callback for getting new control point positions */
            bool CallForSetCtrlPoints(core::Call& c);

            /**********************************************************************
            * parameters
            **********************************************************************/

            /** */
            core::param::ParamSlot applyKeyframeParam;
            /** */
            core::param::ParamSlot undoChangesParam;
            /** */
            core::param::ParamSlot redoChangesParam;
            /** */
            core::param::ParamSlot deleteSelectedKeyframeParam;
            /** */
            //UNUSED core::param::ParamSlot setKeyframesToSameSpeed;
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
            /** */
            core::param::ParamSlot  simTangentParam;
            /** */
            core::param::ParamSlot  interpolTangentParam;
		};

		/** Description class typedef */
		typedef core::factories::CallAutoDescription<KeyframeKeeper> KeyframeKeeperDescription;

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_KEYKEEP_H_INCLUDED */
