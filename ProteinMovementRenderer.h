/*
 * ProteinMovementRenderer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROTEINMOVEMENTRENDERER_H_INCLUDED
#define MEGAMOLCORE_PROTEINMOVEMENTRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallProteinMovementData.h"
#include "CallFrame.h"
#include "Color.h"
#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"
#include "vislib/GLSLShader.h"
#include "vislib/SimpleFont.h"
#include <vector>

namespace megamol {
namespace protein {

    /**
     * Protein Movement Renderer class
     */
    class ProteinMovementRenderer : public megamol::core::view::Renderer3DModule
    {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) 
        {
            return "ProteinMovementRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) 
        {
            return "Offers protein movement renderings.";
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
        ProteinMovementRenderer(void);

        /** Dtor. */
        virtual ~ProteinMovementRenderer(void);
        
        enum RenderMode
        {
            LINES             = 0,
            STICK_RAYCASTING  = 1,
            BALL_AND_STICK    = 2
        };

        enum ArrowColoringMode
        {
            PROTEIN_COLOR   = 0,
            UNIFORM_COLOR   = 1,
            DISTANCE        = 2
        };

       /**********************************************************************
         * 'get'-functions
        **********************************************************************/

        /** Get radius for stick rendering mode */
        inline float GetRadiusStick(void) const { return this->radiusStick; };

        /** Get the color of a certain atom of the protein. */
        const float * GetProteinAtomColor( unsigned int idx) { return &this->protAtomColorTable[idx*3]; };

       /**********************************************************************
         * 'set'-functions
        **********************************************************************/

        /** Set current render mode */
        inline void SetRenderMode( RenderMode rm) { currentRenderMode = rm; RecomputeAll(); };

        /** Set current coloring mode */
        inline void SetColoringMode( Color::ColoringMode cm) { currentColoringMode = cm; RecomputeAll(); };

        /** Set radius for stick rendering mode */
        inline void SetRadiusStick( const float rad ) { radiusStick = rad; RecomputeAll(); };

        /** Set radius for movement arrows */
        inline void SetRadiusArrow( const float rad ) { radiusArrow = rad; };

        /** Set scale factor for movement arrows */
        inline void SetScaleArrow( const float scale ) { scaleArrow = scale; };

        /** Set minimum length for movement arrows */
        inline void SetMinLenghtArrow( const float len ) { minLenArrow = len; };

        /** Set if atoms are drawn as dots in LINES mode */
        inline void DrawAtomsAsDotsWithLine( bool drawDot ) { drawDotsWithLine = drawDot; RecomputeAll(); };

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
        * Render protein data in LINES mode.
        *
        * @param prot The data interface.
        */
        void RenderLines( const CallProteinMovementData *prot);
        
        /**
        * Render protein data in STICK_RAYCASTING mode.
        *
        * @param prot The data interface.
        */
        void RenderStickRaycasting( const CallProteinMovementData *prot);
        
        /**
        * Render protein data in BALL_AND_STICK mode using GPU raycasting.
        *
        * @param prot The data interface.
        */
        void RenderBallAndStick( const CallProteinMovementData *prot);
        
        /** 
        * Recompute all values.
        * This function has to be called after every change rendering attributes,
        * e.g. coloring or render mode.
        */
        void RecomputeAll(void);

        /**********************************************************************
         * variables
         **********************************************************************/
        
        // caller slot
        megamol::core::CallerSlot protDataCallerSlot;
        
        // camera information
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        
		megamol::core::param::ParamSlot renderingModeParam;
        megamol::core::param::ParamSlot coloringModeParam;
        megamol::core::param::ParamSlot stickRadiusParam;
        megamol::core::param::ParamSlot arrowRadiusParam;
        megamol::core::param::ParamSlot arrowScaleParam;
        megamol::core::param::ParamSlot arrowMinLengthParam;
        megamol::core::param::ParamSlot arrowRadiusScaleParam;
        megamol::core::param::ParamSlot arrowBaseLengthParam;
        megamol::core::param::ParamSlot arrowScaleLogParam;
        megamol::core::param::ParamSlot arrowColoringModeParam;
        
        // shader for the spheres (raycasting view)
        vislib::graphics::gl::GLSLShader sphereShader;
        // shader for the cylinders (raycasting view)
        vislib::graphics::gl::GLSLShader cylinderShader;
        // shader for the arrows
        vislib::graphics::gl::GLSLShader arrowShader;
        
        // current render mode
        RenderMode currentRenderMode;
        // current coloring mode
        Color::ColoringMode currentColoringMode;
        
        // attribute locations for GLSL-Shader
        GLint attribLocInParams;
        GLint attribLocQuatC;
        GLint attribLocColor1;
        GLint attribLocColor2;
        
        // display list [LINES]
        GLuint proteinDisplayListLines;
        // has the STICK_RAYCASTING render mode to be prepared?
        bool prepareStickRaycasting;
        // has the BALL_AND_STICK render mode to be prepared?
        bool prepareBallAndStick;

        // vertex array for spheres [STICK_RAYCASTING]
        vislib::Array<float> vertSphereStickRay;
        // vertex array for cylinders [STICK_RAYCASTING]
        vislib::Array<float> vertCylinderStickRay;
        // attribute array for quaterinons of the cylinders [STICK_RAYCASTING]
        vislib::Array<float> quatCylinderStickRay;
        // attribute array for inParameters of the cylinders (radius and length) [STICK_RAYCASTING]
        vislib::Array<float> inParaCylStickRaycasting;
        // color array for spheres [STICK_RAYCASTING]
        vislib::Array<float> colorSphereStickRay;
        // first color array for cylinder [STICK_RAYCASTING]
        vislib::Array<float> color1CylinderStickRay;
        // second color array for cylinder [STICK_RAYCASTING]
        vislib::Array<float> color2CylinderStickRay;
        
        // draw dots for atoms in LINE mode
        bool drawDotsWithLine;
        
        // radius for spheres and sticks with STICK_ render modes
        float radiusStick;
        
        // radius for movement arrows
        float radiusArrow;
        // scale for radius of movement arrows
        float scaleRadiusArrow;
        // scale factor for movement arrows
        float scaleArrow;
        // minimum length for movement arrows
        float minLenArrow;
        // additional scaling factor for logarithmic component
        float scaleLogArrow;
        // base length of vector
        float baseLengthArrow;
        // coloring mode for arrows
        ArrowColoringMode arrowColorMode;
        
        // color table for amino acids
        vislib::Array<vislib::math::Vector<float, 3> > aminoAcidColorTable;
        // color palette vector: stores the color for chains
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;
        // color table for protein atoms
        vislib::Array<float> protAtomColorTable;
        
		vislib::math::Vector<int, 3> colMax;
		vislib::math::Vector<int, 3> colMid;
		vislib::math::Vector<int, 3> colMin;
		vislib::math::Vector<int, 3> col;

        unsigned int atomCount;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_PROTEINMOVEMENTRENDERER_H_INCLUDED
