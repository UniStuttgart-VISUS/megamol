/*
 * SimpleMoleculeRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_SIMPLEMOLECULERENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_SIMPLEMOLECULERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MolecularDataCall.h"
#include "Color.h"
#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "view/Renderer3DModuleDS.h"
#include "view/AbstractCallRender3D.h"
#include "vislib/GLSLShader.h"



namespace megamol {
namespace protein {

    /*
     * Simple Molecular Renderer class
     */

    class SimpleMoleculeRenderer : public megamol::core::view::Renderer3DModuleDS
    {
    public:

        /** The names of the render modes */
        enum RenderMode {
            LINES            = 0,
            STICK            = 1,
            BALL_AND_STICK   = 2,
            SPACEFILLING     = 3,
            SAS              = 4,
            LINES_FILTER     = 5,
            STICK_FILTER     = 6,
            SPACEFILL_FILTER = 7
        };

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "SimpleMoleculeRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void)
        {
            return "Offers molecule renderings.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void)
        {
            return true;
        }

        /** Ctor. */
        SimpleMoleculeRenderer(void);

        /** Dtor. */
        virtual ~SimpleMoleculeRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'release'.
         */
        virtual void release(void);

    private:

       /**********************************************************************
        * 'render'-functions
        **********************************************************************/

        /**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities( megamol::core::Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents( megamol::core::Call& call);

        /**
         * The Open GL Render callback.
         *
         * @param call The calling call.
         * @return The return value of the function.
         */
        virtual bool Render( megamol::core::Call& call);

        /**
         * Render the molecular data using lines and points.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderLines( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the molecular data in stick mode.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderStick( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the molecular data in ball-and-stick mode.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderBallAndStick( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the molecular data in spacefilling mode.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderSpacefilling( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the molecular data in solvent accessible surface mode.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderSAS( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Test the filter module.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderPointsFilter(const MolecularDataCall *mol, const float *atomPos);

		/**
         * Test the filter module.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderLinesFilter(const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the molecular data in stick mode.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderStickFilter( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the molecular data in spacefilling mode.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderSpacefillingFilter( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Update all parameter slots.
         *
         * @param mol   Pointer to the data call.
         */
        void UpdateParameters( const MolecularDataCall *mol);


        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        megamol::core::CallerSlot molDataCallerSlot;

        /** camera information */
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        /** parameter slot for color table filename */
        megamol::core::param::ParamSlot colorTableFileParam;
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam0;
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam1;
        /** parameter slot for coloring mode weighting*/
        megamol::core::param::ParamSlot cmWeightParam;
        /** parameter slot for rendering mode */
        megamol::core::param::ParamSlot renderModeParam;
        /** parameter slot for stick radius */
        megamol::core::param::ParamSlot stickRadiusParam;
        /** parameter slot for SAS probe radius */
        megamol::core::param::ParamSlot probeRadiusParam;
        /** parameter slot for min color of gradient color mode */
        megamol::core::param::ParamSlot minGradColorParam;
        /** parameter slot for mid color of gradient color mode */
        megamol::core::param::ParamSlot midGradColorParam;
        /** parameter slot for max color of gradient color mode */
        megamol::core::param::ParamSlot maxGradColorParam;
        /** list of molecule indices */
        megamol::core::param::ParamSlot molIdxListParam;
        /** parameter slot for special color */
        megamol::core::param::ParamSlot specialColorParam;
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;
		/** Toggle offscreen rendering */
        megamol::core::param::ParamSlot offscreenRenderingParam;

        /** shader for the spheres (raycasting view) */
        vislib::graphics::gl::GLSLShader sphereShader;
		vislib::graphics::gl::GLSLShader sphereShaderOR;
        /** shader for the cylinders (raycasting view) */
        vislib::graphics::gl::GLSLShader cylinderShader;
		vislib::graphics::gl::GLSLShader cylinderShaderOR;
		/** Shader for the spheres that uses filter information */
        vislib::graphics::gl::GLSLShader filterSphereShader;
        vislib::graphics::gl::GLSLShader filterCylinderShader;

        // attribute locations for GLSL-Shader
        GLint attribLocInParams;
        GLint attribLocQuatC;
        GLint attribLocColor1;
        GLint attribLocColor2;
        GLint attribLocAtomFilter;
        GLint attribLocConFilter;

        /** The current coloring mode */
        Color::ColoringMode currentColoringMode0;
        Color::ColoringMode currentColoringMode1;

        /** The color lookup table (for chains, amino acids,...) */
        vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;
        /** The color lookup table which stores the rainbow colors */
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;

        /** The atom color table for rendering */
        vislib::Array<float> atomColorTable;

        /** The current rendering mode */
        RenderMode currentRenderMode;

        /** vertex array for spheres */
        vislib::Array<float> vertSpheres;
        /** vertex array for cylinders */
        vislib::Array<float> vertCylinders;
        /** attribute array for quaterinons of the cylinders */
        vislib::Array<float> quatCylinders;
        /** attribute array for inParam of the cylinders (radius and length) */
        vislib::Array<float> inParaCylinders;
        /** first color array for cylinder */
        vislib::Array<float> color1Cylinders;
        /** second color array for cylinder */
        vislib::Array<float> color2Cylinders;
        /** Connections filter */
        vislib::Array<float> conFilter;

        // the list of molecular indices
        vislib::Array<vislib::StringA> molIdxList;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_SIMPLEMOLECULERENDERER_H_INCLUDED
