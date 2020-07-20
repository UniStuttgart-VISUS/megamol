/*
* KeyframeKeeper.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_KEYFRAMEKEEPER_H_INCLUDED
#define MEGAMOL_CINEMATIC_KEYFRAMEKEEPER_H_INCLUDED

#include "Cinematic/Cinematic.h"

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
#include "vislib/assert.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>

#include "Keyframe.h"
#include "CallKeyframeKeeper.h"
#include "CinematicUtils.h"


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

        // Variables shared/updated with call ---------------------------------
        camera_state_type            cameraState;
		std::vector<glm::vec3 >      interpolCamPos;
		std::vector<Keyframe>        keyframes;
        Keyframe                     selectedKeyframe;
        Keyframe                     dragDropKeyframe;
        glm::vec3                    startCtrllPos;
        glm::vec3                    endCtrllPos;
        float                        totalAnimTime;
        float                        totalSimTime;
        unsigned int                 interpolSteps;
        glm::vec3                    modelBboxCenter;
        unsigned int                 fps;
            
        // Undo queue  --------------------------------------------------------
        class Undo {  
            public:

            enum Action {
                UNDO_NONE = 0,
                UNDO_KEYFRAME_ADD = 1,
                UNDO_KEYFRAME_DELETE = 2,
                UNDO_KEYFRAME_MODIFY = 3,
                UNDO_CONTROLPOINT_MODIFY = 4
            };

            Undo() {
                this->action = Action::UNDO_NONE;
                this->keyframe = Keyframe();
                this->previous_keyframe = Keyframe();
                this->first_controlpoint = glm::vec3();
                this->last_controlpoint = glm::vec3();
                this->previous_first_controlpoint = glm::vec3();
                this->previous_last_controlpoint = glm::vec3();
            }

            Undo(Action act, Keyframe kf, Keyframe prev_kf, glm::vec3 fcp, glm::vec3 lcp, glm::vec3 prev_fcp, glm::vec3 prev_lcp) {
                this->action = act;
                this->keyframe = kf;
                this->previous_keyframe = prev_kf;
                this->first_controlpoint = fcp;
                this->last_controlpoint = lcp;
                this->previous_first_controlpoint = prev_fcp;
                this->previous_last_controlpoint = prev_lcp;
            }

            ~Undo() { }

            inline bool operator==(Undo const& rhs) {
                return ((this->action == rhs.action) && (this->keyframe == rhs.keyframe) && (this->previous_keyframe == rhs.previous_keyframe) && 
                        (this->first_controlpoint == rhs.first_controlpoint) && (this->last_controlpoint == rhs.last_controlpoint) && 
                        (this->previous_first_controlpoint == rhs.previous_first_controlpoint) && (this->previous_last_controlpoint == rhs.previous_last_controlpoint));
            }

            inline bool operator!=(Undo const& rhs) {
                return (!(this->action == rhs.action) || (this->keyframe != rhs.keyframe) || (this->previous_keyframe != rhs.previous_keyframe) || 
                         (this->first_controlpoint != rhs.first_controlpoint) || (this->last_controlpoint != rhs.last_controlpoint) || 
                         (this->previous_first_controlpoint != rhs.previous_first_controlpoint) || (this->previous_last_controlpoint != rhs.previous_last_controlpoint));
            }

            Action action;
            Keyframe       keyframe;
            Keyframe       previous_keyframe;
            glm::vec3      first_controlpoint;
            glm::vec3      previous_first_controlpoint;
            glm::vec3      last_controlpoint;
            glm::vec3      previous_last_controlpoint;
        };

        // Variables only used in keyframe keeper -----------------------------
        std::string       filename;
        bool              simTangentStatus;
        float             splineTangentLength;
        int               undoQueueIndex;
        std::vector<Undo> undoQueue;
          
        /**********************************************************************
        * functions
        ***********************************************************************/

        Keyframe interpolateKeyframe(float time);

        bool addKeyframe(Keyframe kf, bool undo);

        bool replaceKeyframe(Keyframe oldkf, Keyframe newkf, bool undo);

        bool deleteKeyframe(Keyframe kf, bool undo);

        bool loadKeyframes(void);

        bool saveKeyframes(void);

        void refreshInterpolCamPos(unsigned int s);

        void updateEditParameters(Keyframe kf);

        void setSameSpeed(void);

        void linearizeSimTangent(Keyframe kf);

        void snapKeyframe2AnimFrame(Keyframe& inout_kf);

        void snapKeyframe2SimFrame(Keyframe& inout_kf);

        bool undoAction(void);

        bool redoAction(void);

        bool addUndoAction(KeyframeKeeper::Undo::Action act, Keyframe kf, Keyframe prev_kf, glm::vec3 first_controlpoint, glm::vec3 last_controlpoint, glm::vec3 previous_first_controlpoint, glm::vec3 previous_last_controlpoint);

        bool addKeyframeUndoAction(KeyframeKeeper::Undo::Action act, Keyframe kf, Keyframe pre_kf);

        bool addControlPointUndoAction(KeyframeKeeper::Undo::Action act, glm::vec3 first_controlpoint, glm::vec3 last_controlpoint, glm::vec3 previous_first_controlpoint, glm::vec3 previous_last_controlpoint);

        float float_interpolation(float u, float f0, float f1, float f2, float f3);

        glm::vec3 vec3_interpolation(float u, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);

        glm::quat quaternion_interpolation(float u, glm::quat q0, glm::quat q1);

        int getKeyframeIndex(std::vector<Keyframe>& keyframes, Keyframe keyframe);

        /**********************************************************************
        * callbacks
        **********************************************************************/

        megamol::core::CalleeSlot keyframeCallSlot;

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
		core::param::ParamSlot editCurrentAnimTimeParam;
        core::param::ParamSlot editCurrentSimTimeParam;
		core::param::ParamSlot editCurrentPosParam;
        core::param::ParamSlot resetViewParam;
        core::param::ParamSlot editCurrentViewParam;
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