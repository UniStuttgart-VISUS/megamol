/*
 *Keyframe.h
 *
 */

#ifndef MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"

#include "vislib/graphics/Camera.h"
#include "vislib/math/Point.h"
#include "vislib/Serialisable.h"
#include "vislib/math/Vector.h"

namespace megamol {
	namespace cinematiccamera {

		class Keyframe{

		public:

			/** CTOR */
            Keyframe();
            Keyframe(float at, float st, vislib::math::Point<float, 3> pos, vislib::math::Vector<float, 3> up,
                     vislib::math::Point<float, 3> lookat, float aperture);

			/** DTOR */
			~Keyframe();

            /** */
            inline bool operator==(Keyframe const& rhs){
				return ((this->camera == rhs.camera) && (this->animTime == rhs.animTime) && (this->simTime == rhs.simTime));
			}

            /** */
            inline bool operator!=(Keyframe const& rhs) {
                return (!(this->camera == rhs.camera) || (this->animTime != rhs.animTime) || (this->simTime != rhs.simTime));
            }

            ///// GET /////

            /** */
            inline float getAnimTime() {
                return this->animTime;
            }
            /** */
            inline float getSimTime() {
                return (this->simTime == 1.0f)?(1.0f-0.0000001f):(this->simTime);
            }
            /** */
            inline vislib::math::Point<float, 3> getCamPosition(){
                return this->camera.position;
			}
            /** */
            inline vislib::math::Point<float, 3> getCamLookAt(){
                return this->camera.lookat;
			}
            /** */
            inline vislib::math::Vector<float, 3> getCamUp(){
                return this->camera.up;
			}
            /** */
            inline float getCamApertureAngle(){
                return this->camera.apertureangle;
			}

            ///// SET /////

            /** */
            inline void setAnimTime(float t) {
                this->animTime = (t < 0.0f)?(0.0f):(t);
            }
            /** */
            inline void setSimTime(float t) {
                this->simTime = vislib::math::Clamp(t, 0.0f, 1.0f);
            }
            /** */
            inline void setCameraPosition(vislib::math::Point <float, 3> pos){
                this->camera.position = pos;
			}
            /** */
            inline void setCameraLookAt(vislib::math::Point <float, 3> look){
                this->camera.lookat = look;
			}
            /** */
            inline void setCameraUp(vislib::math::Vector<float, 3> up){
                this->camera.up = up;
			}
            /** */
            inline void setCameraApertureAngele(float apertureangle){
                this->camera.apertureangle = vislib::math::Clamp(apertureangle, 0.0f, 180.0f);
			}

            ///// SERIALISATION /////

            /** Serialise */
            void serialise(vislib::Serialiser& serialiser);

            /** Deserialise */
            void deserialise(vislib::Serialiser& serialiser);

		private:

            /**********************************************************************
            * classes
            **********************************************************************/

            // Own camera class without nasty smart pointer foobar :-P
            class Camera {
            public:
                bool operator==(Keyframe::Camera const& rhs) {
                    return ((this->lookat == rhs.lookat) && (this->position == rhs.position) && 
                            (this->apertureangle == rhs.apertureangle) && (this->up == rhs.up));
                }
                vislib::math::Vector<float, 3> up;
                vislib::math::Point<float, 3>  position;
                vislib::math::Point<float, 3>  lookat;
                float                          apertureangle;
            };

            /**********************************************************************
            * variables
            **********************************************************************/

            // Simulation time is always in [0,1] and is relative to absolute total simulation time 
            float                    simTime;
            // Unit of animation time are seconds
			float                    animTime;
            Keyframe::Camera         camera;


		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */
#endif /* MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED */