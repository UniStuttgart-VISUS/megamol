/*
* CallKeyframeKeeper.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_CALLKEYFRAMEKEEPER_H_INCLUDED
#define MEGAMOL_CINEMATIC_CALLKEYFRAMEKEEPER_H_INCLUDED

#include "Cinematic/Cinematic.h"

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/Camera_2.h"

#include "Keyframe.h"


namespace megamol {
namespace cinematic {

	/**
	* Call transporting keyframe data
	*/
	class CallKeyframeKeeper : public core::AbstractGetDataCall {
	public:

		/** function name for Getting all Keyframes */
		static const unsigned int CallForGetUpdatedKeyframeData     = 0;
        static const unsigned int CallForGetSelectedKeyframeAtTime  = 1;
		static const unsigned int CallForGetInterpolCamPositions    = 2;
        static const unsigned int CallForSetSelectedKeyframe        = 3;
        static const unsigned int CallForSetSimulationData          = 4;
		static const unsigned int CallForSetCameraForKeyframe       = 5;
        static const unsigned int CallForSetDragKeyframe            = 6;
        static const unsigned int CallForSetDropKeyframe            = 7;
        static const unsigned int CallForSetCtrlPoints              = 8;

		/**
		* Answer the name of the objects of this description.
		*
		* @return The name of the objects of this description.
		*/
		static const char *ClassName(void) {
			return "CallKeyframeKeeper";
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
			return 9;
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
				case CallForGetUpdatedKeyframeData:     return "CallForGetUpdatedKeyframeData";
				case CallForGetInterpolCamPositions:    return "CallForGetInterpolCamPositions";
                case CallForGetSelectedKeyframeAtTime:  return "CallForGetSelectedKeyframeAtTime";
                case CallForSetSelectedKeyframe:        return "CallForSetSelectedKeyframe";
                case CallForSetSimulationData:          return "CallForSetSimulationData";
				case CallForSetCameraForKeyframe:       return "CallForSetCameraForKeyframe";
                case CallForSetDragKeyframe:            return "CallForSetDragKeyframe";
                case CallForSetDropKeyframe:            return "CallForSetDropKeyframe";
                case CallForSetCtrlPoints:              return "CallForSetCtrlPoints";
				default: return "";
			}
		}

		/** Ctor */
		CallKeyframeKeeper(void);

		/** Dtor */
		virtual ~CallKeyframeKeeper(void);


        /**********************************************************************
        * functions
        **********************************************************************/

        // KEYFRAME ARRAY
		inline std::shared_ptr<std::vector<Keyframe>> GetKeyframes(){
			return this->keyframes;
		}
        inline void SetKeyframes(std::shared_ptr<std::vector<Keyframe>> kfs) {
            this->keyframes = kfs;
        }

        // SELECTED KEYFRAME 
        inline void SetSelectedKeyframeTime(float t) { 
            this->selectedKeyframe.SetAnimTime(t);
        }
        inline void SetSelectedKeyframe(Keyframe k) {
            this->selectedKeyframe = k;
        }

        inline Keyframe GetSelectedKeyframe() {
            return this->selectedKeyframe;
        }

        // BOUNDINGBOX
        inline void SetBoundingBox(std::shared_ptr<vislib::math::Cuboid<float>> bbx) {
            this->boundingbox = bbx;
        }
        inline std::shared_ptr<vislib::math::Cuboid<float>> GetBoundingBox() {
            return this->boundingbox;
        }

        // INTERPOLATED KEYFRAME
        inline void SetInterpolationSteps(unsigned int s) {
            this->interpolSteps = s;
        }
        inline unsigned int GetInterpolationSteps() {
            return this->interpolSteps;
        }

		inline std::shared_ptr<std::vector<glm::vec3 >> GetInterpolCamPositions(){
			return this->interpolCamPos;
		}
        inline void SetInterpolCamPositions(std::shared_ptr<std::vector<glm::vec3 >> k){
            this->interpolCamPos = k;
        }

        // TOTAL ANIMATION TIME
        inline void SetTotalAnimTime(float f) {
            this->totalAnimTime = f;
        }
		inline float GetTotalAnimTime(){
			return this->totalAnimTime;
		}

        // TOTAL SIMULATION TIME
        inline void SetTotalSimTime(float f) {
            this->totalSimTime = f;
        }
        inline float GetTotalSimTime() {
            return this->totalSimTime;
        }

        // CAMERA STATE
        inline void SetCameraState(std::shared_ptr<Keyframe::cam_state_type> cs) {
            this->cameraState = cs;
        }
        inline std::shared_ptr<Keyframe::cam_state_type> GetCameraState() {
            return this->cameraState;
        }

        // DROP OF DRAGGED KEYFRAME
        inline void SetDropTimes(float at, float st) {
            this->dropAnimTime = at;
            this->dropSimTime  = st;
        }
        inline float GetDropAnimTime() {
            return this->dropAnimTime;
        }
        inline float GetDropSimTime() {
            return this->dropSimTime;
        }

        // BOUNDING-BOX CENTER
        inline void SetBboxCenter(glm::vec3  c) {
            this->bboxCenter = c;
        }
        inline glm::vec3 GetBboxCenter() {
            return this->bboxCenter;
        }

        // FRAMES PER SECOND
        inline void SetFps(unsigned int f) {
            this->fps = f;
        }
        inline unsigned int GetFps() {
            return this->fps;
        }

        // CONTROL POINT POSITIONS
        inline void SetControlPointPosition(glm::vec3 firstcp, glm::vec3 lastcp) {
            this->startCtrllPos = firstcp;
            this->endCtrllPos  = lastcp;
        }
        inline glm::vec3 GetStartControlPointPosition() {
            return this->startCtrllPos;
        }
        inline glm::vec3 GetEndControlPointPosition() {
            return this->endCtrllPos;
        }

	private:

        /**********************************************************************
        * variables
        **********************************************************************/

        // Out Data (set by called KeyframeKeeper ) ---------------------------
		std::shared_ptr<Keyframe::cam_state_type>      cameraState;
		std::shared_ptr<std::vector<glm::vec3 >>       interpolCamPos;
		std::shared_ptr<std::vector<Keyframe>>	       keyframes;
        std::shared_ptr<vislib::math::Cuboid<float>>   boundingbox;
        Keyframe						               selectedKeyframe;
        glm::vec3                                      startCtrllPos;
        glm::vec3                                      endCtrllPos;
        float								           totalAnimTime;
        float                                          totalSimTime;
        unsigned int                                   fps;

        // In Data (set by calling modules) -----------------------------------
        unsigned int                                   interpolSteps;
        float                                          dropAnimTime;
        float                                          dropSimTime;
        glm::vec3                                      bboxCenter;
	};


	/** Description class typedef */
	typedef core::factories::CallAutoDescription<CallKeyframeKeeper> CallKeyframeKeeperDescription;


} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_CALLKEYFRAMEKEEPER_H_INCLUDED