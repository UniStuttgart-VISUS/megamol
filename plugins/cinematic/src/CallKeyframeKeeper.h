/*
* CallKeyframeKeeper.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "Cinematic/Cinematic.h"

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/Array.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Point.h"

#include "Keyframe.h"


// GLOBAL CC DEFINES
#ifndef CC_MENU_HEIGHT
    #define CC_MENU_HEIGHT (25.0f)
#endif


namespace megamol {
namespace cinematic {

	/**
	* Call transporting keyframe data
	*/
	class CallKeyframeKeeper : public core::AbstractGetDataCall {
	public:

		/** function name for getting all Keyframes */
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
		inline vislib::Array<Keyframe>* getKeyframes(){
			return this->keyframes;
		}
        inline void setKeyframes(vislib::Array<Keyframe>* kfs) {
            this->keyframes = kfs;
        }

        // SELECTED KEYFRAME 
        inline void setSelectedKeyframeTime(float t) { 
            this->selectedKeyframe.SetAnimTime(t);
        }
        inline void setSelectedKeyframe(Keyframe k) {
            this->selectedKeyframe = k;
        }

        inline Keyframe getSelectedKeyframe() {
            return this->selectedKeyframe;
        }

        // BOUNDINGBOX
        inline void setBoundingBox(vislib::math::Cuboid<float>* bbx) {
            this->boundingbox = bbx;
        }
        inline vislib::math::Cuboid<float> *getBoundingBox() {
            return this->boundingbox;
        }

        // INTERPOLATED KEYFRAME
        inline void setInterpolationSteps(unsigned int s) {
            this->interpolSteps = s;
        }
        inline unsigned int getInterpolationSteps() {
            return this->interpolSteps;
        }

		inline vislib::Array<vislib::math::Point<float, 3> >* getInterpolCamPositions(){
			return this->interpolCamPos;
		}
        inline void setInterpolCamPositions(vislib::Array<vislib::math::Point<float, 3> >* k){
            this->interpolCamPos = k;
        }

        // TOTAL ANIMATION TIME
        inline void setTotalAnimTime(float f) {
            this->totalAnimTime = f;
        }
		inline float getTotalAnimTime(){
			return this->totalAnimTime;
		}

        // TOTAL SIMULATION TIME
        inline void setTotalSimTime(float f) {
            this->totalSimTime = f;
        }
        inline float getTotalSimTime() {
            return this->totalSimTime;
        }

        // CAMERA PARAMETERS
        inline void setCameraParameters(vislib::SmartPtr<vislib::graphics::CameraParameters> c) {
            this->cameraParam = c;
        }
        inline vislib::SmartPtr<vislib::graphics::CameraParameters> getCameraParameters() {
            return this->cameraParam;
        }

        // DROP OF DRAGGED KEYFRAME
        inline void setDropTimes(float at, float st) {
            this->dropAnimTime = at;
            this->dropSimTime  = st;
        }
        inline float getDropAnimTime() {
            return this->dropAnimTime;
        }
        inline float getDropSimTime() {
            return this->dropSimTime;
        }

        // BOUNDING-BOX CENTER
        inline void setBboxCenter(vislib::math::Point<float, 3>  c) {
            this->bboxCenter = c;
        }
        inline vislib::math::Point<float, 3> getBboxCenter() {
            return this->bboxCenter;
        }

        // FRAMES PER SECOND
        inline void setFps(unsigned int f) {
            this->fps = f;
        }
        inline unsigned int getFps() {
            return this->fps;
        }

        // CONTROL POINT POSITIONS
        inline void setControlPointPosition(vislib::math::Vector<float, 3> firstcp, vislib::math::Vector<float, 3> lastcp) {
            this->startCtrllPos = firstcp;
            this->endCtrllPos  = lastcp;
        }
        inline vislib::math::Vector<float, 3> getStartControlPointPosition() {
            return this->startCtrllPos;
        }
        inline vislib::math::Vector<float, 3> getEndControlPointPosition() {
            return this->endCtrllPos;
        }

	private:

        /**********************************************************************
        * variables
        **********************************************************************/

		// Pointer to array of keyframes
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraParam;
        vislib::Array<vislib::math::Point<float, 3> > *interpolCamPos;
		vislib::Array<Keyframe>				          *keyframes;
        vislib::math::Cuboid<float>		              *boundingbox;
        unsigned int                                   interpolSteps;
        Keyframe						               selectedKeyframe;
        float                                          dropAnimTime;
        float                                          dropSimTime;
		float								           totalAnimTime;
        float                                          totalSimTime;
        vislib::math::Point<float, 3>                  bboxCenter;
        unsigned int                                   fps;
        vislib::math::Vector<float, 3>                 startCtrllPos;
        vislib::math::Vector<float, 3>                 endCtrllPos;

	};


	/** Description class typedef */
	typedef core::factories::CallAutoDescription<CallKeyframeKeeper> CallKeyframeKeeperDescription;


} /* end namespace cinematic */
} /* end namespace megamol */

