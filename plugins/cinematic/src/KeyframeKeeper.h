/*
* KeyframeKeeper.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_KEYFRAMEKEEPER_H_INCLUDED
#define MEGAMOL_CINEMATIC_KEYFRAMEKEEPER_H_INCLUDED

#include "Cinematic/Cinematic.h"
#include "Keyframe.h"
#include "CallKeyframeKeeper.h"

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/utility/xml/XmlParser.h"
#include "mmcore/utility/xml/XmlReader.h"

#include "vislib/math/Cuboid.h"
#include "vislib/StringSerialiser.h"
#include "vislib/assert.h"

#include <iostream>
#include <fstream>
#include <ctime>


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
        * variables
        **********************************************************************/

        // Variables shared/updated with call
		std::shared_ptr<std::vector<glm::vec3 >>      interpolCamPos;
		std::shared_ptr<std::vector<Keyframe>>        keyframes;
		std::shared_ptr<vislib::math::Cuboid<float>>  boundingBox;
        Keyframe                     selectedKeyframe;
        Keyframe                     dragDropKeyframe;
        glm::vec3                    startCtrllPos;
        glm::vec3                    endCtrllPos;
        float                        totalAnimTime;
        float                        totalSimTime;
        unsigned int                 interpolSteps;
        glm::vec3                    modelBboxCenter;
        unsigned int                 fps;

        glm::vec3                    camViewUp;
        glm::vec3                    camViewPosition;
        glm::vec3                    camViewLookat;
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
                    this->startcp       = glm::vec3();
                    this->endcp         = glm::vec3();
                    this->prev_startcp  = glm::vec3();
                    this->prev_endcp    = glm::vec3();
                }

                UndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prev_kf, glm::vec3 scp, glm::vec3 ecp, glm::vec3 prev_scp, glm::vec3 prev_ecp) {
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
                glm::vec3            startcp;
                glm::vec3            endcp;
                glm::vec3            prev_startcp;
                glm::vec3            prev_endcp;
        };

        std::vector<UndoAction> undoQueue;

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
        bool addUndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prev_kf, glm::vec3 startcp, glm::vec3 endcp, glm::vec3 prev_startcp, glm::vec3 prev_endcp);

        /**
        *
        */
        bool addKfUndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe pre_vkf);

        /**
        *
        */
        bool addCpUndoAction(KeyframeKeeper::UndoActionEnum act, glm::vec3 startcp, glm::vec3 endcp, glm::vec3 prev_startcp, glm::vec3 prev_endcp);

        /**
        *
        */
        float interpolate_f(float u, float f0, float f1, float f2, float f3);

        /**
        *
        */
        glm::vec3 interpolate_vec3(float u, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);

        /**
        *
        */
        int getKeyframeIndex(std::shared_ptr<std::vector<Keyframe>> keyframes, Keyframe keyframe);

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
        core::param::ParamSlot snapAnimFramesParam;
        core::param::ParamSlot snapSimFramesParam;
        core::param::ParamSlot simTangentParam;
        core::param::ParamSlot interpolTangentParam;
        core::param::ParamSlot setKeyframesToSameSpeed;
		/**param for current keyframe aniamtion time */
		core::param::ParamSlot editCurrentAnimTimeParam;
        /**param for current keyframe simulation time */
        core::param::ParamSlot editCurrentSimTimeParam;
		/**param for current keyframe Position */
		core::param::ParamSlot editCurrentPosParam;
		/**param for current keyframe LookAt */
		core::param::ParamSlot editCurrentLookAtParam;
        core::param::ParamSlot resetLookAtParam;
		/**param for current keyframe Up */
		core::param::ParamSlot editCurrentUpParam;
        core::param::ParamSlot editCurrentApertureParam;
        core::param::ParamSlot fileNameParam;
        core::param::ParamSlot saveKeyframesParam;
        core::param::ParamSlot loadKeyframesParam;
	};


	/** Description class typedef */
	typedef core::factories::CallAutoDescription<KeyframeKeeper> KeyframeKeeperDescription;


} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_KEYFRAMEKEEPER_H_INCLUDED