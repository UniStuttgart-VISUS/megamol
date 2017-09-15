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
            Keyframe(float t);
            Keyframe(float t, vislib::math::Point<float, 3> pos, vislib::math::Vector<float, 3> up,
                     vislib::math::Point<float, 3> lookat, float aperture);

			/** DTOR */
			~Keyframe();

            /** */
			bool operator==(Keyframe const& rhs){
				return ((this->camera == rhs.camera) && (this->animTime == rhs.animTime));
			}

            /** */
            bool operator!=(Keyframe const& rhs) {
                return (!(this->camera == rhs.camera) || (this->animTime != rhs.animTime));
            }

            ///// GET /////

            /** */
            float getAnimTime() {
                return animTime;
            }
            /** */
			vislib::math::Point<float, 3> getCamPosition(){
                return this->camera.position;
			}
            /** */
			vislib::math::Point<float, 3> getCamLookAt(){
                return this->camera.lookat;
			}
            /** */
			vislib::math::Vector<float, 3> getCamUp(){
                return this->camera.up;
			}
            /** */
			float getCamApertureAngle(){
                return this->camera.apertureangle;
			}

            ///// SET /////

            /** */
            void setAnimTime(float t) {
                this->animTime = t;
            }
            /** */
			void setCameraPosition(vislib::math::Point <float, 3> pos){
                this->camera.position = pos;
			}
            /** */
			void setCameraLookAt(vislib::math::Point <float, 3> look){
                this->camera.lookat = look;
			}
            /** */
			void setCameraUp(vislib::math::Vector<float, 3> up){
                this->camera.up = up;
			}
            /** */
			void setCameraApertureAngele(float apertureangle){
                this->camera.apertureangle = apertureangle;
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

			float                    animTime;
            Keyframe::Camera         camera;


		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */
#endif /* MEGAMOL_CINEMATICCAMERA_KEYFRAME_H_INCLUDED */