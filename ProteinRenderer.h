/*
 * ProteinRenderer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROTEINRENDERER_H_INCLUDED
#define MEGAMOLCORE_PROTEINRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallProteinData.h"
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

        /*
     * Protein Renderer class
         *
         * TODO:
         * - add coloring mode:
         *    o value
         *    o rainbow / "chain"-bow(?)
     */

        class ProteinRenderer : public megamol::core::view::Renderer3DModule
        {
        public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
                {
            return "ProteinRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void)
                {
            return "Offers protein renderings.";
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
        ProteinRenderer(void);

        /** Dtor. */
        virtual ~ProteinRenderer(void);

                enum RenderMode
                {
                        LINES            = 0,
                        STICK_RAYCASTING = 1,
                        STICK_POLYGON    = 2,
                        BALL_AND_STICK   = 3,
                        SPACEFILLING     = 4,
                        SAS              = 5
                };

                /*enum ColoringMode
                {
                        ELEMENT   = 0,
                        AMINOACID = 1,
                        STRUCTURE = 2,
                        VALUE     = 3,
                        CHAIN_ID  = 4,
                        RAINBOW   = 5,
                        CHARGE    = 6
                };*/

           /**********************************************************************
                * 'get'-functions
            **********************************************************************/

                /** Get radius for stick rendering mode */
                inline float GetRadiusStick(void) const { return this->m_radiusStick; };

                /** Get the color of a certain atom of the protein. */
                const unsigned char * GetProteinAtomColor( unsigned int idx) { return &this->m_protAtomColorTable[idx*3]; };

                /** Get probe radius for Solvent Accessible Surface mode. */
                inline float GetProbeRadius() const { return this->m_probeRadius; };

           /**********************************************************************
                * 'set'-functions
            **********************************************************************/

                /** Set current render mode */
                inline void SetRenderMode( RenderMode rm) { m_currentRenderMode = rm; RecomputeAll(); };

                /** Set current coloring mode */
                inline void SetColoringMode( Color::ColoringMode cm) { m_currentColoringMode = cm; RecomputeAll(); };

                /** Set radius for stick rendering mode */
                inline void SetRadiusStick( const float rad ) { m_radiusStick = rad; RecomputeAll(); };

            /** Set probe radius for Solvent Accessible Surface mode. */
            inline void SetRadiusProbe( const float rad) { m_probeRadius = rad; RecomputeAll(); };

                /** Set if atoms are drawn as dots in LINES mode */
                inline void DrawAtomsAsDotsWithLine( bool drawDot ) { m_drawDotsWithLine = drawDot; RecomputeAll(); };

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
         * Draw label for current loaded RMS frame.
         *
         * @param call Ths calling CallFrame.
         */
        void DrawLabel(unsigned int frameID);

        /**
         * The CallFrame callback.
         *
         * @param call The calling call.
         * @return The return value of the function.
         */
        bool ProcessFrameRequest( megamol::core::Call& call);

                /**
                 * Render protein data in LINES mode.
                 *
                 * @param prot The data interface.
                 */
                void RenderLines( const CallProteinData *prot);

                /**
                 * Render protein data in STICK_RAYCASTING mode.
                 *
                 * @param prot The data interface.
                 */
                void RenderStickRaycasting( const CallProteinData *prot);

                /**
                 * Render protein data in BALL_AND_STICK mode using GPU raycasting.
                 *
                 * @param prot The data interface.
                 */
                void RenderBallAndStick( const CallProteinData *prot);

                /**
                 * Render protein data in SPACEFILLING mode using GPU raycasting.
                 *
                 * @param prot The data interface.
                 */
                void RenderSpacefilling( const CallProteinData *prot);

                 /* Render protein data in SAS mode (Solvent Accessible Surface) using GPU raycasting.
                 *
                 * @param prot The data interface.
                 */
                void RenderSolventAccessibleSurface( const CallProteinData *prot);

                /**
                 * Render disulfide bonds using GL_LINES.
                 *
                 * @param prot The data interface.
                 */
                void RenderDisulfideBondsLine( const CallProteinData *prot);

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
                megamol::core::CallerSlot m_protDataCallerSlot;
        // callee slot
        megamol::core::CalleeSlot m_callFrameCalleeSlot;
                // caller slot
                megamol::core::CallerSlot solventRendererCallerSlot;

        // 'true' if there is rms data to be rendered
        bool m_renderRMSData;

        // label with id of current loaded frame
                vislib::graphics::AbstractFont *m_frameLabel;

                // camera information
                vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        megamol::core::param::ParamSlot m_renderingModeParam;
        megamol::core::param::ParamSlot m_coloringModeParam;
        megamol::core::param::ParamSlot m_drawBackboneParam;
        megamol::core::param::ParamSlot m_drawDisulfideBondsParam;
                megamol::core::param::ParamSlot m_stickRadiusParam;
                megamol::core::param::ParamSlot m_probeRadiusParam;

                // shader for the spheres (raycasting view)
                vislib::graphics::gl::GLSLShader m_sphereShader;
                // shader for the cylinders (raycasting view)
                vislib::graphics::gl::GLSLShader m_cylinderShader;

                // current render mode
                RenderMode m_currentRenderMode;
                // current coloring mode
                Color::ColoringMode m_currentColoringMode;

                // attribute locations for GLSL-Shader
                GLint m_attribLocInParams;
                GLint m_attribLocQuatC;
                GLint m_attribLocColor1;
                GLint m_attribLocColor2;

                // draw only the backbone atoms of the protein?
                bool m_drawBackbone;
                // draw the disulfide bonds?
                bool m_drawDisulfideBonds;

                // display list [LINES]
                GLuint m_proteinDisplayListLines;
                // display list [disulfide bonds]
                GLuint m_disulfideBondsDisplayList;
                // has the STICK_RAYCASTING render mode to be prepared?
                bool m_prepareStickRaycasting;
                // has the BALL_AND_STICK render mode to be prepared?
                bool m_prepareBallAndStick;
                // has the SPACEFILLING render mode to be prepared?
                bool m_prepareSpacefilling;
                // has the SAS render mode to be prepared?
                bool m_prepareSAS;

                // vertex array for spheres [STICK_RAYCASTING]
                vislib::Array<float> m_vertSphereStickRay;
                // vertex array for cylinders [STICK_RAYCASTING]
                vislib::Array<float> m_vertCylinderStickRay;
                // attribute array for quaterinons of the cylinders [STICK_RAYCASTING]
                vislib::Array<float> m_quatCylinderStickRay;
                // attribute array for inParameters of the cylinders (radius and length) [STICK_RAYCASTING]
                vislib::Array<float> m_inParaCylStickRaycasting;
                // color array for spheres [STICK_RAYCASTING]
                vislib::Array<unsigned char> m_colorSphereStickRay;
                // first color array for cylinder [STICK_RAYCASTING]
                vislib::Array<float> m_color1CylinderStickRay;
                // second color array for cylinder [STICK_RAYCASTING]
                vislib::Array<float> m_color2CylinderStickRay;

                // draw dots for atoms in LINE mode
                bool m_drawDotsWithLine;

                // radius for spheres and sticks with STICK_ render modes
                float m_radiusStick;

                // probe radius for SAS rendering
                float m_probeRadius;

                // color table for amino acids
                vislib::Array<vislib::math::Vector<unsigned char, 3> > m_aminoAcidColorTable;
                // color palette vector: stores the color for chains
                std::vector<vislib::math::Vector<float,3> > rainbowColors;
                // color table for protein atoms
                vislib::Array<unsigned char> m_protAtomColorTable;

                // the Id of the current frame (for dynamic data)
                unsigned int m_currentFrameId;

        // the number of protein atoms
        unsigned int atomCount;
        };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_PROTEINRENDERER_H_INCLUDED
