/*
* KeyframeKeeper.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "Cinematic/Cinematic.h"

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"

#include "Keyframe.h"

namespace megamol {
namespace cinematic {

	/**
	* Keyframe Keeper.
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
			return "Manages keyframes";
		}

		/** Ctor */
		KeyframeKeeper(void);

		/** Dtor */
		virtual ~KeyframeKeeper(void);

        /**
        *
        */
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
        * typedefs
        **********************************************************************/

        typedef vislib::math::Vector<float, 3> v3f;
        typedef vislib::math::Point<float, 3>  p3f;

        /**********************************************************************
        * variables
        **********************************************************************/

        // Variables shared/updated with call
        vislib::Array<p3f >          interpolCamPos;
        vislib::Array<Keyframe>      keyframes;
        vislib::math::Cuboid<float>  boundingBox;
        Keyframe                     selectedKeyframe;
        Keyframe                     dragDropKeyframe;
        v3f                          startCtrllPos;
        v3f                          endCtrllPos;
        float                        totalAnimTime;
        float                        totalSimTime;
        unsigned int                 interpolSteps;
        p3f                          modelBboxCenter;
        unsigned int                 fps;

        v3f                          camViewUp;
        p3f                          camViewPosition;
        p3f                          camViewLookat;
        float                        camViewApertureangle;
            
        // Variables only used in keyframe keeper
        vislib::StringA             filename;
        bool                        simTangentStatus;
        float                       tl; // Global interpolation spline tangent length of keyframes

        // undo queue stuff -----------------------------------------------

        enum UndoActionEnum {
            UNDO_NONE      = 0,
            UNDO_KF_ADD    = 1,
            UNDO_KF_DELETE = 2,
            UNDO_KF_MODIFY = 3,
            UNDO_CP_MODIFY = 4
        };

        class UndoAction {  
            public:
                /** functions **/
                UndoAction() {
                    this->action        = KeyframeKeeper::UndoActionEnum::UNDO_NONE;
                    this->keyframe      = Keyframe();
                    this->prev_keyframe = Keyframe();
                    this->startcp       = v3f();
                    this->endcp         = v3f();
                    this->prev_startcp  = v3f();
                    this->prev_endcp    = v3f();
                }

                UndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prev_kf, v3f scp, v3f ecp, v3f prev_scp, v3f prev_ecp) {
                    this->action        = act;
                    this->keyframe      = kf;
                    this->prev_keyframe = prev_kf;
                    this->startcp       = scp;
                    this->endcp         = ecp;
                    this->prev_startcp  = prev_scp;
                    this->prev_endcp    = prev_ecp;
                }

                ~UndoAction() { }

                inline bool operator==(UndoAction const& rhs) {
                    return ((this->action == rhs.action) && (this->keyframe == rhs.keyframe) && (this->prev_keyframe == rhs.prev_keyframe) && 
                            (this->startcp == rhs.startcp) && (this->endcp == rhs.endcp) && (this->prev_startcp == rhs.prev_startcp) && (this->prev_endcp == rhs.prev_endcp));
                }

                inline bool operator!=(UndoAction const& rhs) {
                    return (!(this->action == rhs.action) || (this->keyframe != rhs.keyframe) || (this->prev_keyframe != rhs.prev_keyframe) || 
                                (this->startcp != rhs.startcp) || (this->endcp != rhs.endcp) || (this->prev_startcp != rhs.prev_startcp) || (this->prev_endcp != rhs.prev_endcp));
                }

                /** variables **/
                UndoActionEnum action;
                Keyframe       keyframe;
                Keyframe       prev_keyframe;
                v3f            startcp;
                v3f            endcp;
                v3f            prev_startcp;
                v3f            prev_endcp;
        };

        vislib::Array<UndoAction> undoQueue;

        int undoQueueIndex;

        /**********************************************************************
        * functions
        ***********************************************************************/

        /** 
        * Get an interpolated keyframe at specific time. 
        */
        Keyframe interpolateKeyframe(float time);

        /**  
        * Add new keyframe to keyframe array. 
        */
        bool addKeyframe(Keyframe kf, bool undo);

        /**  
        * Replace existing keyframe in keyframe array.
        */
        bool replaceKeyframe(Keyframe oldkf, Keyframe newkf, bool undo);

        /**  
        * Delete keyframe from keyframe array.
        */
        bool deleteKeyframe(Keyframe kf, bool undo);

        /**  
        * Load keyframes from file.
        */
        void loadKeyframes(void);

        /**  
        * Save keyframes to file.
        */
        void saveKeyframes(void);

        /**  
        * Refresh interpolated camera positions (called when keyframe array changes). 
        */
        void refreshInterpolCamPos(unsigned int s);

        /**  
        * Updating edit parameters without setting them dirty.
        */
        void updateEditParameters(Keyframe kf);

        /** 
        * Set speed between all keyframes to same speed 
        * Uses interpolSteps for approximation of path between keyframe positions 
        */
        void setSameSpeed(void);

        /**
        *
        */
        void linearizeSimTangent(Keyframe stkf);

        /**
        *
        */
        void snapKeyframe2AnimFrame(Keyframe *kf);

        /**
        *
        */
        void snapKeyframe2SimFrame(Keyframe *kf);

        /**
        *
        */
        bool undoAction();

        /**
        *
        */
        bool redoAction();

        /**
        *
        */
        bool addUndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prev_kf, v3f startcp, v3f endcp, v3f prev_startcp, v3f prev_endcp);

        /**
        *
        */
        bool addKfUndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe pre_vkf);

        /**
        *
        */
        bool addCpUndoAction(KeyframeKeeper::UndoActionEnum act, v3f startcp, v3f endcp, v3f prev_startcp, v3f prev_endcp);

        /**
        *
        */
        float interpolate_f(float u, float f0, float f1, float f2, float f3);

        /**
        *
        */
        vislib::math::Vector<float, 3> interpolate_v3f(float u, v3f v0, v3f v1, v3f v2, v3f v3);

        /**********************************************************************
        * callback stuff
        **********************************************************************/

        megamol::core::CalleeSlot cinematicCallSlot;

		bool CallForGetUpdatedKeyframeData(core::Call& c);
		bool CallForSetSimulationData(core::Call& c);
		bool CallForGetInterpolCamPositions(core::Call& c);
		bool CallForSetSelectedKeyframe(core::Call& c);
        bool CallForGetSelectedKeyframeAtTime(core::Call& c);
		bool CallForSetCameraForKeyframe(core::Call& c);
        bool CallForSetDragKeyframe(core::Call& c); 
        bool CallForSetDropKeyframe(core::Call& c);
        bool CallForSetCtrlPoints(core::Call& c);

        /**********************************************************************
        * parameters
        **********************************************************************/

        core::param::ParamSlot applyKeyframeParam;
        core::param::ParamSlot undoChangesParam;
        core::param::ParamSlot redoChangesParam;
        core::param::ParamSlot deleteSelectedKeyframeParam;
        core::param::ParamSlot setTotalAnimTimeParam;
        core::param::ParamSlot  snapAnimFramesParam;
        core::param::ParamSlot  snapSimFramesParam;
        core::param::ParamSlot  simTangentParam;
        core::param::ParamSlot  interpolTangentParam;
        core::param::ParamSlot setKeyframesToSameSpeed;
		/**param for current keyframe aniamtion time */
		core::param::ParamSlot editCurrentAnimTimeParam;
        /**param for current keyframe simulation time */
        core::param::ParamSlot editCurrentSimTimeParam;
		/**param for current keyframe Position */
		core::param::ParamSlot editCurrentPosParam;
		/**param for current keyframe LookAt */
		core::param::ParamSlot editCurrentLookAtParam;
        core::param::ParamSlot  resetLookAtParam;
		/**param for current keyframe Up */
		core::param::ParamSlot editCurrentUpParam;
        core::param::ParamSlot  editCurrentApertureParam;
        core::param::ParamSlot fileNameParam;
        core::param::ParamSlot saveKeyframesParam;
        core::param::ParamSlot loadKeyframesParam;
	};


	/** Description class typedef */
	typedef core::factories::CallAutoDescription<KeyframeKeeper> KeyframeKeeperDescription;


} /* end namespace cinematic */
} /* end namespace megamol */

