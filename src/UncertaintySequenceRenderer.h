/*
 * UncertaintySequenceRenderer.h
 *
 * Author: Matthias Braun
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "SequenceRenderer" in megamol protein plugin (svn revision 1500).
 *
 */


#ifndef MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYSEQUENCERENDERER_H_INCLUDED
#define MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYSEQUENCERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//#define USE_SIMPLE_FONT


#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer2DModule.h"

#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/graphics/gl/GLSLShader.h"

#ifdef USE_SIMPLE_FONT
#include "vislib/graphics/gl/SimpleFont.h"
#else //  USE_SIMPLE_FONT
#include "vislib/graphics/gl/OutlineFont.h"
#include "vislib/graphics/gl/Verdana.inc"
#endif //  USE_SIMPLE_FONT

#include "UncertaintyDataCall.h"
#include "protein_calls/ResidueSelectionCall.h"
#include "protein_calls/BindingSiteCall.h"


namespace megamol {
	namespace protein_uncertainty {

    class UncertaintySequenceRenderer : public megamol::core::view::Renderer2DModule {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "UncertaintySequenceRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers sequence renderings of protein secondary structure uncertainty.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** ctor */
        UncertaintySequenceRenderer(void);

        /** dtor */
        ~UncertaintySequenceRenderer(void);

    protected:
        
        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);
        
        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * Callback for mouse events (move, press, and release)
         *
         * @param x The x coordinate of the mouse in world space
         * @param y The y coordinate of the mouse in world space
         * @param flags The mouse flags
         */
        virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);

        /**
         * Prepares the data for rendering.
         *
         * @param mol The molecular data call.
         * @return true if preparation was successful, false otherwise
         */
        bool PrepareData(UncertaintyDataCall *udc, protein_calls::BindingSiteCall *bs);
        
        /**
         * TODO
         */
        bool LoadTexture(vislib::StringA filename);

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
        virtual bool GetExtents(megamol::core::view::CallRender2D& call);

        /**
        * The Open GL Render callback.
        *
        * @param call The calling call.
        * @return The return value of the function.
        */
        virtual bool Render(megamol::core::view::CallRender2D& call);

        /**********************************************************************
         * other functions
         **********************************************************************/
         
        /**
         * Returns the single letter code for an amino acid given the three letter code.
         *
         * @param resName The name of the residue as three letter code.
         * @return The single letter code for the amino acid.
         */
        char GetAminoAcidOneLetterCode(vislib::StringA resName);
        
        /**
         * Draws the texture tiles for the secondary structure types.
         *
         * @param secStructPre The secondary structure type on position i-1
         * @param secStructi   The secondary structure type on position i
         * @param secStructSuc The secondary structure type on position i+1
         * @param m            The flag indicating if it is a missing amino-acid
         * @param x            The x position
         * @param y            The y position
         * @param defColor     The the default color
         */
        void drawSecStructTextureTiles(UncertaintyDataCall::secStructure secStructPre, 
                                       UncertaintyDataCall::secStructure secStructi, 
                                       UncertaintyDataCall::secStructure secStructSuc, bool m, float x, float y, float defColor[4]);

        /**
        * Renders a two lined tooltip text.
        *
        * @param start   The height the tooltip starts.
        * @param end     The height the tooltip ends.
        * @param str1    The string for the first line.
        * @param str2    The string for the second line.
        * @param fgColor The foreground color.
        * @param bgColor The background color.
        */
        void renderToolTip(float start, float end, vislib::StringA str1, vislib::StringA str2, float fgColor[4], float bgColor[4]);

        /**
         * enumeration of available uncertainty visualizations
         */
        enum visualization {
            STACK_SIMPLE   = 0,  
            STACK_EXT_VERT = 1,       
            STACK_EXT_HORI = 2,
        };
        
        /**
         * Renders the simple STACK uncertainty visualization.
         *
         * @param yPos The y position the rendering should start.
         */        
        void renderUncertaintyStackSimple(float yPos);
        

        /**
         * Renders the extended STACK uncertainty visualization with horizontal ordered tiles.
         *
         * @param yPos The y position the rendering should start.
         */        
        void renderUncertaintyStackHorizontal(float yPos);
        
        /**
         * Renders the extended STACK uncertainty visualization with vertical ordered tiles.
         *
         * @param yPos The y position the rendering should start.
         */        
        void renderUncertaintyMorphingVertical(float yPos);
        
        
        /**********************************************************************
         * variables
         **********************************************************************/
        
	    /** The call for uncertainty data */
        core::CallerSlot uncertaintyDataSlot;		
        
        /** binding site caller slot */
        core::CallerSlot bindingSiteCallerSlot;
        /** residue selection caller slot */
        core::CallerSlot resSelectionCallerSlot;

        // the number of residues in one row
        megamol::core::param::ParamSlot resCountPerRowParam;
        // the file name for the color table
        megamol::core::param::ParamSlot colorTableFileParam;
        // parameter to show/hide row legend/key
        megamol::core::param::ParamSlot toggleLegendParam;
        // clear the current residue selection
        megamol::core::param::ParamSlot clearResSelectionParam;

        // parameter to show/hide tooltip 
        megamol::core::param::ParamSlot toggleTooltipParam;

        // parameter to show/hide pdb secondary structure 
        megamol::core::param::ParamSlot togglePdbParam;
        // parameter to show/hide stride secondary structure
        megamol::core::param::ParamSlot toggleStrideParam;
        // parameter to show/hide dssp secondary structure
        megamol::core::param::ParamSlot toggleDsspParam;
        // parameter to show/hide disagreements in secondary structure assignment
        megamol::core::param::ParamSlot toggleDiffParam;
        // parameter to show/hide secondary structure uncertainty visualization
        megamol::core::param::ParamSlot toggleUncertaintyParam;
                
        // parameter to turn animation of morphing visualization on/off
        megamol::core::param::ParamSlot animateMorphing;

        // parameter to choose secondary structure uncertainty visualization
        megamol::core::param::ParamSlot uncertaintyVisualizationParam;
        
        // the current uncertainty visualization
        visualization currentVisualization;
        
        // data preparation flag
        bool dataPrepared;
        // the total number of amino-acids 
        unsigned int aminoAcidCount;
        // the total number of binding sites
        unsigned int bindingSiteCount;

        // the number of secondary structure rows which can be shown/hidden
        unsigned int secStructRows;
        // the number of residue columns
        unsigned int resCols;
        // the number of residue rows
        unsigned int resRows;
        // the height of a row
        float rowHeight;
        
        // font rendering
#ifdef USE_SIMPLE_FONT
        vislib::graphics::gl::SimpleFont theFont;
#else
        vislib::graphics::gl::OutlineFont theFont;
#endif
        // the array of amino acid 1-letter codes
        vislib::Array<char> aminoAcidName;
        // the array of pdb amino-acid indices
        vislib::Array<int> aminoAcidIndex;
        // the array of the chain IDs
        vislib::Array<char> chainID;
        // the array for the missing amin-acid flag
        vislib::Array<bool> missingAminoAcids;
 
        // the array of binding site names
        vislib::Array<vislib::StringA> bindingSiteNames;
        // the array of descriptons for the binding sites
        vislib::Array<vislib::StringA> bindingSiteDescription;
        
        // the vertex buffer array for the tiles
        vislib::Array<float> vertices;

        // the vertex buffer array secondary structure
        vislib::Array<vislib::Array<vislib::math::Vector<float, 2> > >secStructVertices;

        // the vertex buffer array for the chain tiles
        vislib::Array<float> chainVertices;
        // the color buffer array for the chain tiles
        vislib::Array<float> chainColors;
        // the vertex buffer array for the chain separator lines
        vislib::Array<float> chainSeparatorVertices;

        // the vertex buffer array for the binding site tiles
        vislib::Array<float> bsVertices;
        // the index array for the binding site tiles
        vislib::Array<unsigned int> bsIndices;
        // the color array for the binding site tiles
        vislib::Array<vislib::math::Vector<float, 3> > bsColors;

        // The secondary structure assignment methods and their secondary structure type assignments
        vislib::Array<vislib::Array<UncertaintyDataCall::secStructure> > secStructAssignment;

        // The values of the secondary structure uncertainty for each amino-acid 
        vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > secUncertainty;
        // The sorted structure types of the uncertainty values
        vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > sortedUncertainty;
                
        // color table
        vislib::Array<vislib::math::Vector<float, 3> > colorTable;
        
        // textures
        vislib::Array<vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D> > markerTextures;

        // mouse hover
        vislib::math::Vector<float, 2> mousePos;
        vislib::math::Vector<float, 2> mousePosDetail;
        int mousePosResIdx;
        bool rightMouseDown;
        bool initialClickSelection;

        // selection 
        vislib::Array<bool> selection;
		protein_calls::ResidueSelectionCall *resSelectionCall;
        
        // PDB ID 
        vislib::StringA pdbID;

        // timer for animation of morphing visualization
        clock_t animTimer;
    };

	} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif // MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYSEQUENCERENDERER_H_INCLUDED
