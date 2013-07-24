/*
 * SequenceRenderer.h
 *
 * Author: Michael Krone
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOLPROTEIN_SEQUENCERENDERER_H_INCLUDED
#define MEGAMOLPROTEIN_SEQUENCERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "view/Renderer2DModule.h"
#include "MolecularDataCall.h"
#include "BindingSiteCall.h"
#include "vislib/GLSLShader.h"
#include "vislib/SimpleFont.h"
#include <vislib/OpenGLTexture2D.h>

namespace megamol {
namespace protein {

    class SequenceRenderer : public megamol::core::view::Renderer2DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SequenceRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers sequence renderings.";
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
        SequenceRenderer(void);

        /** dtor */
        ~SequenceRenderer(void);

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
        bool PrepareData( MolecularDataCall *mol);

    private:

        /**
         * Returns the single letter code for an amino acid given the three letter code.
         *
         * @param resName The name of the residue as three letter code.
         * @return The single letter code for the amino acid.
         */
        char GetAminoAcidOneLetterCode( vislib::StringA resName );

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
         * variables
         **********************************************************************/
        
        /** caller slot */
        core::CallerSlot dataCallerSlot;
        /** caller slot */
        core::CallerSlot bindingSiteCallerSlot;
        
        // the number of residues in one row
        megamol::core::param::ParamSlot resCountPerRowParam;

        // data preparation flag
        bool dataPrepared;

        // the number of residues
        unsigned int resCount;
        // the number of residue columns
        unsigned int resCols;
        // the number of residue rows
        unsigned int resRows;
        // the height of a row
        float rowHeight;
        
        // font rendering
        vislib::graphics::gl::SimpleFont theFont;
        // the array of amino acid 1-letter codes
        vislib::Array<vislib::StringA> aminoAcidStrings;

        // the vertex buffer array for the tiles
        vislib::Array<float> vertices;
        // the index of the residue
        vislib::Array<unsigned int> resIndex;
        // the secondary structure element type of the residue
        vislib::Array<MolecularDataCall::SecStructure::ElementType> resSecStructType;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLPROTEIN_SEQUENCERENDERER_H_INCLUDED
