/*
*CinematicView.h
*
*/

///////////////////////////////////////////////////////////////////////////////
///// DISCLAIMER: Code for png export is adapted from "ScreenShooter.cpp" /////
///////////////////////////////////////////////////////////////////////////////

#ifndef MEGAMOL_CINEMATICCAMERA_CINEMATICVIEW_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_CINEMATICVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"

#include "mmcore/view/View3D.h"

#include "vislib/graphics/Camera.h"
#include "vislib/math/Point.h"
#include "vislib/Serialisable.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/sys/FastFile.h"

#include "Keyframe.h"
#include "png.h"

namespace megamol {
	namespace cinematiccamera {

		class CinematicView : public core::view::View3D {

		public:

			enum SkyboxSides {
				SKYBOX_NONE  = 0,
				SKYBOX_FRONT = 1,
				SKYBOX_BACK  = 2,
				SKYBOX_LEFT  = 4,
				SKYBOX_RIGHT = 8, 
				SKYBOX_UP    = 16,
				SKYBOX_DOWN  = 32
			};

			typedef core::view::View3D Base;

			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "CinematicView";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Screenshot View Module";
			}

			/**
			* Answers whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
                if (!vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable())
                    return false;
				return true;
			}

			/**
			* Disallow usage in quickstarts
			*
			* @return false
			*/
			static bool SupportQuickstart(void) {
				return false;
			}

			/** Ctor. */
			CinematicView(void);

			/** Dtor. */
			virtual ~CinematicView(void);

			/**
			* Renders this AbstractView3D in the currently active OpenGL context.
			*/
			virtual void Render(const mmcRenderViewContext& context);

		private:

            /**********************************************************************
            * variables
            **********************************************************************/

            clock_t                                 deltaAnimTime;
            Keyframe                                shownKeyframe;
            bool                                    playAnim;

            int                                     cineWidth;
            int                                     cineHeight;
            int                                     vpH;
            int                                     vpW;

            CinematicView::SkyboxSides              sbSide;

            vislib::graphics::gl::FramebufferObject fbo;
            bool                                    resetFbo;
            bool                                    rendering;
            unsigned int                            fps;
            unsigned int                            expFrameCnt;

            struct pngData {
                BYTE                  *buffer;
                vislib::sys::FastFile  file;
                unsigned int           width;
                unsigned int           height;
                unsigned int           bpp; 
                vislib::TString        path;
                vislib::TString        filename;
                unsigned int           cnt;
                png_structp            ptr;
                png_infop              infoptr;
                float                  animTime;
                bool                   lock;
            } pngdata;

            /**********************************************************************
            * functions
            **********************************************************************/

            /** */
            bool setSimTime(float st);

            /** Render to file functions */
            bool rtf_setup();

            /** */
            bool rtf_set_time_and_camera();

            /** */
            bool rtf_create_frame();

            /** */
            bool rtf_write_frame();

            /** */
            bool rtf_finish();

            /**
            * My error handling function for png export
            *
            * @param pngPtr The png structure pointer
            * @param msg The error message
            */
            static void PNGAPI pngError(png_structp pngPtr, png_const_charp msg) {
                throw vislib::Exception(msg, __FILE__, __LINE__);
            }

            /**
            * My error handling function for png export
            *
            * @param pngPtr The png structure pointer
            * @param msg The error message
            */
            static void PNGAPI pngWarn(png_structp pngPtr, png_const_charp msg) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "Png-Warning: %s\n", msg);
            }

            /**
            * My write function for png export
            *
            * @param pngPtr The png structure pointer
            * @param buf The pointer to the buffer to be written
            * @param size The number of bytes to be written
            */
            static void PNGAPI pngWrite(png_structp pngPtr, png_bytep buf, png_size_t size) {
                vislib::sys::File *f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
                f->Write(buf, size);
            }

            /**
            * My flush function for png export
            *
            * @param pngPtr The png structure pointer
            */
            static void PNGAPI pngFlush(png_structp pngPtr) {
                vislib::sys::File *f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
                f->Flush();
            }

            /**********************************************************************
            * callback
            **********************************************************************/
			/** The keyframe keeper caller slot */
			core::CallerSlot keyframeKeeperSlot;

            /**********************************************************************
            * parameters
            **********************************************************************/
            /** */
			core::param::ParamSlot selectedSkyboxSideParam;
            /** */
            core::param::ParamSlot resHeightParam;
            /** */
            core::param::ParamSlot resWidthParam;
            /** */
            core::param::ParamSlot fpsParam;
            /** */
            core::param::ParamSlot renderParam;
            /** */
            core::param::ParamSlot toggleAnimPlayParam;
		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_CINEMATICVIEW_H_INCLUDED */