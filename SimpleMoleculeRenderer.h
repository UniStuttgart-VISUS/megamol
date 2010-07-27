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
#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"
#include "vislib/GLSLShader.h"

namespace megamol {
namespace protein {

    /*
     * Simple Molecular Renderer class
     */

    class SimpleMoleculeRenderer : public megamol::core::view::Renderer3DModule
    {
    public:

        /** The names of the render modes */
        enum RenderMode {
            LINES            = 0,
            STICK            = 1,
            BALL_AND_STICK   = 2,
            SPACEFILLING     = 3,
            SAS              = 4
        };

        /** The names of the coloring modes */
        enum ColoringMode {
            ELEMENT     = 0,
            RESIDUE     = 1,
            STRUCTURE   = 2,
            BFACTOR     = 3,
            CHARGE      = 4,
            OCCUPANCY   = 5,
            CHAIN       = 6,
            MOLECULE    = 7,
            RAINBOW     = 8,
            CHAINBOW    = 9     // TODO
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
         * Read color table from file.
         *
         * @param filename The filename of the color table file.
         */
        void ReadColorTableFromFile( vislib::StringA filename);

        /**
         * Update all parameter slots.
         *
         * @param mol   Pointer to the data call.
         */
        void UpdateParameters( const MolecularDataCall *mol);
        
        /**
         * Make color table for all atoms acoording to the current coloring mode.
         * The color table is only computed if it is empty or if the recomputation 
         * is forced by parameter.
         *
         * @param mol               The data interface.
         * @param forceRecompute    Force recomputation of the color table.
         */
        void MakeColorTable( const MolecularDataCall *mol, bool forceRecompute = false);
 
         /**
         * Creates a rainbow color table with 'num' entries.
         *
         * @param num The number of color entries.
         */
        void MakeRainbowColorTable( unsigned int num);
        
        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        megamol::core::CallerSlot molDataCallerSlot;
		// caller slot
		megamol::core::CallerSlot molRendererCallerSlot;

        /** camera information */
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        /** parameter slot for color table filename */
        megamol::core::param::ParamSlot colorTableFileParam;
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam;
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

        /** shader for the spheres (raycasting view) */
        vislib::graphics::gl::GLSLShader sphereShader;
        /** shader for the cylinders (raycasting view) */
        vislib::graphics::gl::GLSLShader cylinderShader;

        // attribute locations for GLSL-Shader
        GLint attribLocInParams;
        GLint attribLocQuatC;
        GLint attribLocColor1;
        GLint attribLocColor2;

        /** The current coloring mode */
        ColoringMode currentColoringMode;

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

        // the list of molecular indices
        vislib::Array<vislib::StringA> molIdxList;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_SIMPLEMOLECULERENDERER_H_INCLUDED
