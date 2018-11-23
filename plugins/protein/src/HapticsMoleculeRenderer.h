/*
 * HapticsMoleculeRenderer.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifdef WITH_OPENHAPTICS

#ifndef MMPROTEINPLUGIN_HAPTICSMOLECULERENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_HAPTICSMOLECULERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/MolecularDataCall.h"
#include "Color.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "PhantomDeviceWrapper.h"
#include "vislib/graphics/gl/FramebufferObject.h"

namespace megamol {
namespace protein {

    /*
     * Simple Molecular Renderer class
     */

    class HapticsMoleculeRenderer : public megamol::core::view::Renderer3DModule
    {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "HapticsMoleculeRenderer";
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
        HapticsMoleculeRenderer(void);

        /** Dtor. */
        virtual ~HapticsMoleculeRenderer(void);

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
         * Render the IDs of all atoms to a FBO
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void WriteAtomId( const MolecularDataCall *mol, const float *atomPos);
        
        /**
         * Renders all atoms using GPU ray casting and write atom ID to red color channel.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderAtomIdGPU( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the molecular data in spacefilling mode.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderSpacefilling( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the molecular data in spacefilling mode using OpenGL
         * immediate mode rendering.
         *
         * @param mol        Pointer to the data call.
         * @param atomPos    Pointer to the interpolated atom positions.
         */
        void RenderSpacefillingImmediateMode( const MolecularDataCall *mol, const float *atomPos);

        /**
         * Render the 3d mouse cursor as an arrow.
         */
        void Render3DCursor(void);

        /**
         * Render the force arrows applied by the haptic pen.
         */
        void RenderForceArrows(void);

        /**
         * Update all parameter slots.
         *
         * @param mol   Pointer to the data call.
         */
        void UpdateParameters( const MolecularDataCall *mol);

        /**
         * Responds to button events from phantom device.
         * If param is true, gets atom id that cursor is touching and returns it.
         * Also sets flag that renderer needs to update this atom position for
         * phantom device.
         * If false, unsets update flag and returns -1.
         *
         * @param click True if button was clicked, false if released.
         * @param touch True if device thinks it is touching an object, false otherwise.
         *
         * @return Atom id or -1 for no atom.
         */
        unsigned int getAtomID(bool click, bool touch);

        /**
         * Callback function for setting the force data to be used by MD Driver.
         *
         * @param force The ForceDataCall that triggered this function.
         *
         * @return True on success.
         */
        bool getForceData( core::Call& call);


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
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;
        /** parameter slot for multi-force capability */
        megamol::core::param::ParamSlot multiforceParam;

        /** The data callee slot */
        core::CalleeSlot forceDataOutSlot;

        /** shader for the spheres (raycasting view) */
        vislib::graphics::gl::GLSLShader sphereShader;
        /** shader for the cylinders (raycasting view) */
        vislib::graphics::gl::GLSLShader cylinderShader;
        // shader for the arrow cursor
        vislib::graphics::gl::GLSLShader arrowShader;
        // shader for the offscreen rendering / atom id writing
        vislib::graphics::gl::GLSLShader writeSphereIdShader;

        // radius for movement arrows
        float radiusArrow;

        // attribute locations for GLSL-Shader
        GLint attribLocInParams;
        GLint attribLocQuatC;
        GLint attribLocColor1;
        GLint attribLocColor2;
        GLint attribLocAtomFilter;
        GLint attribLocConFilter;

        /** The current coloring mode */
        Color::ColoringMode currentColoringMode;

        /** The color lookup table (for chains, amino acids,...) */
        vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;
        /** The color lookup table which stores the rainbow colors */
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;

        /** The atom color table for rendering */
        vislib::Array<float> atomColorTable;

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

        /** Absolute cursor to associate with phantom device */
        vislib::graphics::AbsoluteCursor3D absoluteCursor3d; 

        /** Phantom device wrapper class */
        PhantomDeviceWrapper phantom;

        /* Shape id for logo shape and shader object we will render haptically. */
        HLuint shaderID;

        /* Atom id for the atom being dragged (-1 for no atom) */
        unsigned int currentDragAtom;

        /* Closest atom id */
        unsigned int closestAtomID;

        /* Closest atom proxy-to-surface distance */
        float closestAtomDistance;
        
        /* Array of temporary closest atom ids */
        vislib::Array<unsigned int> tmpClosestAtomID;

        /* Array of temporary closest atom proxy-to-surface distances */
        vislib::Array<float> tmpClosestAtomDistance;
        
        /** current width and height of the view */
        unsigned int width, height;
        /** FBO for sphere id writing */
        vislib::graphics::gl::FramebufferObject sphereIdFbo;
        /** array for atom ids and radii */
        vislib::Array<float> atomParams;

        /** the atom positions */
        float *atomPositions;
        /** the size of the atom position array */
        unsigned int atomPositionCount;

        /* Force data */
        unsigned int forceCount; // number of forces being transferred
        vislib::Array<unsigned int> forceAtomIDs; // array of atom ids that correspond to forces
        vislib::Array<float> forces; // array of forces in x,y,z,x,y,z order
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_HAPTICSMOLECULERENDERER_H_INCLUDED

#endif // WITH_OPENHAPTICS
