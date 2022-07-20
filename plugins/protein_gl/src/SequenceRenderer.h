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

//#define USE_SIMPLE_FONT


#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Input.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"
#ifdef USE_SIMPLE_FONT
#include "vislib_gl/graphics/gl/SimpleFont.h"
#else //  USE_SIMPLE_FONT
#include "vislib_gl/graphics/gl/OutlineFont.h"
#include "vislib_gl/graphics/gl/Verdana.inc"
#endif //  USE_SIMPLE_FONT
#include "protein_calls/ResidueSelectionCall.h"
#include "vislib_gl/graphics/gl/OpenGLTexture2D.h"

namespace megamol {
namespace protein_gl {

class SequenceRenderer : public megamol::mmstd_gl::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SequenceRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
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
    bool PrepareData(megamol::protein_calls::MolecularDataCall* mol, protein_calls::BindingSiteCall* bs);

    /**
     * TODO
     */
    bool LoadTexture(vislib::StringA filename);

private:
    /**
     * Returns the single letter code for an amino acid given the three letter code.
     *
     * @param resName The name of the residue as three letter code.
     * @return The single letter code for the amino acid.
     */
    char GetAminoAcidOneLetterCode(vislib::StringA resName);

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
    virtual bool GetExtents(mmstd_gl::CallRender2DGL& call);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(mmstd_gl::CallRender2DGL& call);

    /**********************************************************************
     * variables
     **********************************************************************/

    /** pdb caller slot */
    core::CallerSlot dataCallerSlot;
    /** binding site caller slot */
    core::CallerSlot bindingSiteCallerSlot;
    /** residue selection caller slot */
    core::CallerSlot resSelectionCallerSlot;

    // the number of residues in one row
    megamol::core::param::ParamSlot resCountPerRowParam;
    // the file name for the color table
    megamol::core::param::ParamSlot colorTableFileParam;
    // parameter to turn the binding site legend/key on/off
    megamol::core::param::ParamSlot toggleKeyParam;
    // clear the current residue selection
    megamol::core::param::ParamSlot clearResSelectionParam;

    // data preparation flag
    bool dataPrepared;
    // the total number of atoms
    unsigned int atomCount;
    // the total number of atoms
    unsigned int bindingSiteCount;

    // the number of residues
    unsigned int resCount;
    // the number of residue columns
    unsigned int resCols;
    // the number of residue rows
    unsigned int resRows;
    // the height of a row
    float rowHeight;

    // font rendering
#ifdef USE_SIMPLE_FONT
    vislib_gl::graphics::gl::SimpleFont theFont;
#else
    vislib_gl::graphics::gl::OutlineFont theFont;
#endif
    // the array of amino acid 1-letter codes
    vislib::Array<vislib::StringA> aminoAcidStrings;
    // the array of amino acid chain name and index
    vislib::Array<vislib::Array<vislib::Pair<char, int>>> aminoAcidIndexStrings;
    // the array of binding site names
    vislib::Array<vislib::StringA> bindingSiteNames;
    // the array of descriptons for the binding sites
    vislib::Array<vislib::StringA> bindingSiteDescription;

    // the vertex buffer array for the tiles
    vislib::Array<float> vertices;
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
    vislib::Array<vislib::math::Vector<float, 3>> bsColors;
    // the index of the residue
    vislib::Array<unsigned int> resIndex;
    // the secondary structure element type of the residue
    vislib::Array<megamol::protein_calls::MolecularDataCall::SecStructure::ElementType> resSecStructType;
    // color table
    std::vector<glm::vec3> colorTable;

    vislib::Array<vislib::SmartPtr<vislib_gl::graphics::gl::OpenGLTexture2D>> markerTextures;

    // mouse hover
    vislib::math::Vector<float, 2> mousePos;
    int mousePosResIdx;
    bool leftMouseDown;
    bool initialClickSelection;
    // selection
    vislib::Array<bool> selection;
    protein_calls::ResidueSelectionCall* resSelectionCall;
};

} // namespace protein_gl
} /* end namespace megamol */

#endif // MEGAMOLPROTEIN_SEQUENCERENDERER_H_INCLUDED
