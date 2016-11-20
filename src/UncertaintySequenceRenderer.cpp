/*
* UncertaintySequenceRenderer.cpp
*
* Author: Matthias Braun
* Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*
* This module is based on the source code of "SequenceRenderer" in megamol protein plugin (svn revision 1500).
*
*/


#include "stdafx.h"
#include "UncertaintySequenceRenderer.h"

#include <GL/glu.h>
#include <math.h>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/utility/ResourceWrapper.h"

#include "vislib/math/Rectangle.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include "UncertaintyColor.h" 

#include <iostream> // DEBUG


using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_uncertainty;

using namespace vislib::graphics::gl;

using vislib::sys::Log;


/*
 * UncertaintySequenceRenderer::UncertaintySequenceRenderer (CTOR)
 */
UncertaintySequenceRenderer::UncertaintySequenceRenderer( void ) : Renderer2DModule (),
            uncertaintyDataSlot( "uncertaintyDataSlot", "Connects the sequence diagram rendering with uncertainty data storage." ),
            bindingSiteCallerSlot( "getBindingSites", "Connects the sequence diagram rendering with binding site storage." ),
            resSelectionCallerSlot( "getResSelection", "Connects the sequence diagram rendering with residue selection storage." ),
            resCountPerRowParam( "ResiduesPerRow", "The number of residues per row" ),
            colorTableFileParam( "ColorTableFilename", "The filename of the color table."),
            toggleLegendParam( "LegendDrawing", "Show/hide row legend/key on the left."),
            clearResSelectionParam( "clearResidueSelection", "Clears the current selection (everything will be deselected)."),
            togglePdbParam("Author", "Show/hide PDB secondary structure row." ),
            toggleStrideParam("Stride", "Show/hide STRIDE secondary structure row"),
            toggleDsspParam("Dssp", "Show/hide DSSP secondary structure row"),
            toggleDiffParam("Difference", "Show/hide row with disagreements in secondary structure assignment."),
            toggleUncertaintyParam("Uncertainty", "Show/hide row with uncertainty of secondary structure assignment."), 
            uncertaintyVisualizationParam("Visualization", "Choose uncertainty Visualization"),   
            dataPrepared(false), aminoAcidCount(0), bindingSiteCount(0), resCols(0), resRows(0), rowHeight(2.0f), 
            markerTextures(0), resSelectionCall(nullptr), rightMouseDown(false), secStructRows(0), pdbID("")
#ifndef USE_SIMPLE_FONT
            ,theFont(FontInfo_Verdana)
#endif // USE_SIMPLE_FONT
             {

    // uncertainty data caller slot
    this->uncertaintyDataSlot.SetCompatibleCall<UncertaintyDataCallDescription>();
    this->MakeSlotAvailable(&this->uncertaintyDataSlot);

    // binding site data caller slot
    this->bindingSiteCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bindingSiteCallerSlot);
   
    // residue selection caller slot
    this->resSelectionCallerSlot.SetCompatibleCall<ResidueSelectionCallDescription>();
    this->MakeSlotAvailable(&this->resSelectionCallerSlot);
    

    // param slot for number of residues per row
    this->resCountPerRowParam.SetParameter( new param::IntParam(50, 1));
    this->MakeSlotAvailable( &this->resCountPerRowParam);
    
    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    this->colorTableFileParam.SetParameter(new param::FilePathParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);
    UncertaintyColor::ReadColorTableFromFile( T2A(this->colorTableFileParam.Param<param::FilePathParam>()->Value()), this->colorTable);
    
    // param slot clearing all selections
    this->clearResSelectionParam.SetParameter( new param::ButtonParam('c'));
    this->MakeSlotAvailable( &this->clearResSelectionParam);
        
    // param slot for key/legend toggling
    this->toggleLegendParam.SetParameter( new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleLegendParam);
    
    
    // param slot for PDB toggling
    this->togglePdbParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->togglePdbParam);

    // param slot for STRIDE toggling
    this->toggleStrideParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleStrideParam);

    // param slot for DSSP toggling
    this->toggleDsspParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleDsspParam);
    
    // param slot for uncertainty difference toggling
    this->toggleDiffParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleDiffParam);
    
    // param slot for uncertainty toggling
    this->toggleUncertaintyParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleUncertaintyParam);        
    
    
    // param for uncertainty Visualization
    this->currentVisualization = STACK;
    param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(this->currentVisualization));
    
    tmpEnum->SetTypePair(STACK,     "Stack.");
    // tmpEnum->SetTypePair(..., "...");
    
    this->uncertaintyVisualizationParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertaintyVisualizationParam);       


    // setting initial row height, as if all secondary structure rows would be shown
    this->secStructRows = 5;
    this->rowHeight = 2.0f + static_cast<float>(this->secStructRows);   

}

/*
 * UncertaintySequenceRenderer::~UncertaintySequenceRenderer (DTOR)
 */
UncertaintySequenceRenderer::~UncertaintySequenceRenderer( void ) {
    this->Release(); // DON'T change !
}

/*
 * UncertaintySequenceRenderer::create
 */
bool UncertaintySequenceRenderer::create() {
    
    this->LoadTexture("secStruct-white.png");       // markerTextures[0]
    this->LoadTexture("secStruct-coil.png");        // markerTextures[1]
    this->LoadTexture("secStruct-sheet.png");       // markerTextures[2]
    this->LoadTexture("secStruct-arrow.png");       // markerTextures[3]
    this->LoadTexture("secStruct-helix-left.png");  // markerTextures[4]
    this->LoadTexture("secStruct-helix2.png");      // markerTextures[5]
    this->LoadTexture("secStruct-helix-right.png"); // markerTextures[6]

////////////////////////////////////////////////
// TODO: additional textures for: TURN (?), some new, higher resolution textures ? ....
////////////////////////////////////////////////  
    
    return true;
}

/*
 * UncertaintySequenceRenderer::release
 */
void UncertaintySequenceRenderer::release() {
    /** intentionally left empty ... */
}


/*
* UncertaintySequenceRenderer::GetExtents
*/
bool UncertaintySequenceRenderer::GetExtents(view::CallRender2D& call) {

    // get pointer to UncertaintyDataCall
    UncertaintyDataCall *udc = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if (udc == NULL) return false;
    // execute the call
    if (!(*udc)(UncertaintyDataCall::CallForGetData)) return false;

    // get pointer to BindingSiteCall
    BindingSiteCall *bs = this->bindingSiteCallerSlot.CallAs<BindingSiteCall>();
    if( bs != NULL ) {
        // execute the call
        if( !(*bs)(BindingSiteCall::CallForGetData) ) {
            bs = NULL;
        }
    }
    
    // check whether the number of residues per row or the number of atoms was changed
    if((this->resCountPerRowParam.IsDirty()) || (this->aminoAcidCount != udc->GetAminoAcidCount())) {
        this->dataPrepared = false;
    }

    // check whether the number of binding sites has changed
    if( bs && (this->bindingSiteCount != bs->GetBindingSiteCount())) {
        this->dataPrepared = false;
    }
    
    // check whether the number of shown secondary structure rows has changed
    if(this->togglePdbParam.IsDirty()) {
        this->dataPrepared = false;
    }
    if(this->toggleStrideParam.IsDirty()) {
        this->dataPrepared = false;
    }    
    if(this->toggleDsspParam.IsDirty()) {
        this->dataPrepared = false;
    }
    if(this->toggleDiffParam.IsDirty()) {
        this->dataPrepared = false;
    }
    if(this->toggleUncertaintyParam.IsDirty()) {
        this->dataPrepared = false;
    }
    
    // check whether uncertainty data has changed
    if(udc->GetRecalcFlag()) {
        udc->SetRecalcFlag(false);        
        this->dataPrepared = false;
    }
       
    // prepare the data
    if( !this->dataPrepared ) {
      
        // reset number of shown secondary structure rows
        this->SecStructRows = 0;
        
        // calculate new row height
        if (this->togglePdbParam.Param<param::BoolParam>()->Value())
            this->SecStructRows++;
        if (this->toggleStrideParam.Param<param::BoolParam>()->Value())
            this->SecStructRows++;
        if (this->toggleDsspParam.Param<param::BoolParam>()->Value())
            this->SecStructRows++;
        if (this->toggleDiffParam.Param<param::BoolParam>()->Value())
            this->SecStructRows++;
        if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value())
            this->SecStructRows++;
                
        this->rowHeight = 2.0 + static_cast<float>(this->SecStructRows);
        
        unsigned int oldRowHeight = (unsigned int)this->rowHeight; 
        
        this->dataPrepared = this->PrepareData(udc, bs);
        
        // when binding sites changed row height - prepare data again ...
        if(oldRowHeight != this->rowHeight) {
            this->dataPrepared = this->PrepareData(udc, bs);
        }

        if( this->dataPrepared ) {
            // the data has been prepared for the current number of residues per row
            this->resCountPerRowParam.ResetDirty();
            // store the number of atoms for which the data was prepared (... already done in prepareData)
            this->aminoAcidCount = udc->GetAminoAcidCount();
            //store the number of binding sites
            bs ? this->bindingSiteCount = bs->GetBindingSiteCount() : this->bindingSiteCount = 0;
            
            // the data has been prepared for the selected number of rows
            this->togglePdbParam.ResetDirty();
            this->toggleStrideParam.ResetDirty();
            this->toggleDsspParam.ResetDirty();
            this->toggleDiffParam.ResetDirty();
            this->toggleUncertaintyParam.ResetDirty();
        }
    }
        
    // set the bounding box (minX, minY, maxX, maxY) -> (0,0) is in the upper left corner
    call.SetBoundingBox( 0.0f, -static_cast<float>(this->resRows) * this->rowHeight, static_cast<float>(this->resCols), 0.0f);

    return true;
}


/*
 * UncertaintySequenceRenderer::Render
 */
bool UncertaintySequenceRenderer::Render(view::CallRender2D &call) {

	// get pointer to UncertaintyDataCall
	UncertaintyDataCall *ud = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if( ud == NULL ) return false;
    // execute the call
    if( !(*ud)(UncertaintyDataCall::CallForGetData)) return false;
  
    // check if new Visualization was chosen
	if (this->uncertaintyVisualizationParam.IsDirty()) {
        this->uncertaintyVisualizationParam.ResetDirty();  
        this->currentVisualization = this->uncertaintyVisualizationParam.Param<core::param::EnumParam>()->Value();
    }
      
    // updating selection
    this->resSelectionCall = this->resSelectionCallerSlot.CallAs<ResidueSelectionCall>();
    if (this->resSelectionCall != NULL) {
        (*this->resSelectionCall)(ResidueSelectionCall::CallForGetSelection);

        // clear selection
        if( this->clearResSelectionParam.IsDirty() ) {
            this->resSelectionCall->GetSelectionPointer()->Clear();
            this->clearResSelectionParam.ResetDirty();
        }

        vislib::Array<ResidueSelectionCall::Residue> *resSelPtr = this->resSelectionCall->GetSelectionPointer();
        for(unsigned int i = 0; i < this->aminoAcidCount; i++) {
            this->selection[i] = false;
            if(resSelPtr != NULL) {
                // loop over selections and try to match wit the current amino acid
                for( unsigned int j = 0; j < resSelPtr->Count(); j++) {
                    if((*resSelPtr)[j].chainID == this->chainID[i] && (*resSelPtr)[j].resNum == i)
                    {
                        this->selection[i] = true;
                    }
                }
            }
        }
    }
    else { 
        // when no selection call is present clear selection anyway ...
        if (this->clearResSelectionParam.IsDirty()) {
            this->clearResSelectionParam.ResetDirty();
            for (int i = 0; i < this->selection.Count(); i++) {
                this->selection[i] = false;
            }
        }
    }

    // read and update the color table, if necessary
    if( this->colorTableFileParam.IsDirty() ) {
		UncertaintyColor::ReadColorTableFromFile(T2A(this->colorTableFileParam.Param<param::FilePathParam>()->Value()), this->colorTable);
        this->colorTableFileParam.ResetDirty();
    }
    
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // get the text color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for( unsigned int i = 0; i < 4; i++ ) {
        fgColor[i] -= bgColor[i];
    }

    if( this->dataPrepared ) {
        
        // temporary variables and constants
        vislib::StringA tmpStr;
        float yPos;
        float xPos;
        float wordlength;
        float maxWordLength;
        float rowPos;            
        float fontSize;
        vislib::math::Vector<float, 2> pos;
        float resRow;
        float diff;
        UncertaintyDataCall::secStructure s;
        
///// TEXTURING /////
    
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);      

        yPos = 0.0f;
        //draw texture tiles for secondary structure
        //STRIDE
        if(this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
                this->drawSecStructTextureTiles(((i > 0)?(strideSecStructure[i-1]):(strideSecStructure[i])), strideSecStructure[i], 
                                                ((i < this->aminoAcidCount)?(strideSecStructure[i+1]):(strideSecStructure[i])), 
                                                this->missingAminoAcids[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);
            } 
            yPos += 1.0f;
        }
        // DSSP
        if(this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
                this->drawSecStructTextureTiles(((i > 0)?(dsspSecStructure[i-1]):(dsspSecStructure[i])), dsspSecStructure[i], 
                                                ((i < this->aminoAcidCount)?(dsspSecStructure[i+1]):(dsspSecStructure[i])), 
                                                this->missingAminoAcids[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);                
            }
            yPos += 1.0f;
        }  
        // PDB - AUTHOR
        if(this->togglePdbParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
                this->drawSecStructTextureTiles(((i > 0)?(pdbSecStructure[i-1]):(pdbSecStructure[i])), pdbSecStructure[i], 
                                                ((i < this->aminoAcidCount)?(pdbSecStructure[i+1]):(pdbSecStructure[i])), 
                                                this->missingAminoAcids[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);                
            }
            yPos += 1.0f;
        }
                
        // draw type legend textures
        xPos = 2.0f;
        for (unsigned int i = 0; i < static_cast<int>(UncertaintyDataCall::secStructure::NoE); i++) {
            s = static_cast<UncertaintyDataCall::secStructure>(i);
            this->drawSecStructTextureTiles(s, s, s, false, (static_cast<float>(this->resCols) + 1.0f + xPos), (static_cast<float>(i) + 2.0f), bgColor);  
        }
        
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D); 
        
        // draw uncertainty difference
        if(this->toggleDiffParam.Param<param::BoolParam>()->Value()) {
            glColor3fv(fgColor);
            
            glBegin(GL_QUADS);
            for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
                
                // Max - Min:
                // three assignment methods -> max three different sec structures: max=[0] - middle=[1] - min[2]
                // assuming values are in range 0.0-1.0
                diff = 1.0f - (this->secUncertainty[i][this->sortedUncertainty[0]] - this->secUncertainty[i][this->sortedUncertainty[2]]); 
                   
                if (diff < 1.0f) {
                    glVertex2f( this->vertices[2*i],        -(this->vertices[2*i+1]+yPos) - diff);
                    glVertex2f( this->vertices[2*i],        -(this->vertices[2*i+1]+yPos) - 1.0f);
                    glVertex2f( this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1]+yPos) - 1.0f);
                    glVertex2f( this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1]+yPos) - diff);
                }
            }
            glEnd();
            
            glBegin(GL_LINES);
            for (unsigned int i = 0; i < this->resRows; i++) {
                glVertex2f(0.0f, -((static_cast<float>(i)*this->rowHeight)+yPos));
                glVertex2f(static_cast<float>(this->resCols), -((static_cast<float>(i)*this->rowHeight)+yPos));
            }
            glEnd();
            
            yPos += 1.0f;
        }
        
        ////////////////////////////////////////////////
        // TODO: draw uncertainty 
        ////////////////////////////////////////////////    
        // ...
        if(this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
            switch (this->currentVisualization) {
                case (STACK): this->renderUncertaintyStack(yPos); break;
                default: break;
            }
            yPos += 1.0f;
        }
        
        
///// GEOMETRY /////
        
        // draw tiles for binding sites
        glBegin(GL_QUADS);
        for( unsigned int i = 0; i < this->bsIndices.Count(); i++ ) {
            glColor3fv( this->bsColors[this->bsIndices[i]].PeekComponents());
            glVertex2f( this->bsVertices[2*i] + 0.1f, -this->bsVertices[2*i+1]);
            glVertex2f( this->bsVertices[2*i] + 0.1f, -this->bsVertices[2*i+1] - 0.4f);
            glVertex2f( this->bsVertices[2*i] + 0.9f, -this->bsVertices[2*i+1] - 0.4f);
            glVertex2f( this->bsVertices[2*i] + 0.9f, -this->bsVertices[2*i+1]);
        }
        glEnd();

        // draw tiles for chains
        glBegin( GL_QUADS);
        for( unsigned int i = 0; i < this->chainVertices.Count() / 2; i++ ) {
            glColor3fv( (&this->chainColors[i*3]));
            glVertex2f( this->chainVertices[2*i]       , -this->chainVertices[2*i+1] - 0.2f);
            glVertex2f( this->chainVertices[2*i]       , -this->chainVertices[2*i+1] - 0.3f);
            glVertex2f( this->chainVertices[2*i] + 1.0f, -this->chainVertices[2*i+1] - 0.3f);
            glVertex2f( this->chainVertices[2*i] + 1.0f, -this->chainVertices[2*i+1] - 0.2f);
        }
        glEnd();
        
        // draw chain separators
        glColor3fv(fgColor);
        glEnable( GL_VERTEX_ARRAY);
        glVertexPointer( 2, GL_FLOAT, 0, this->chainSeparatorVertices.PeekElements());
        glDrawArrays( GL_LINES, 0, (GLsizei)this->chainSeparatorVertices.Count() / 2 );
        glDisable( GL_VERTEX_ARRAY);
        
        
///// TEXT /////

        // INFO: DrawString(float x, float y, float w, float h, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) 
        glColor3fv(fgColor);
        resRow = 0.0f;
        fontSize = 1.5f;
        if( theFont.Initialise() ) {
            
            // pdb id as info on top
            tmpStr = this->pdbID.Prepend("PDB-ID: ");
            wordlength = theFont.LineWidth(fontSize, tmpStr);
            theFont.DrawString(0.0f, 0.5f, 1.0f, 1.0f, fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
            
            // row labeling
            for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
                // new row after resCols amino-acids
                if (((i%this->resCols) == 0) && (i != 0))
                    resRow += 1.0f;
                    
                // draw the one-letter amino acid code
                tmpStr.Format("%c", this->aminoAcidName[i]);
                theFont.DrawString(static_cast<float>(i%this->resCols), -(resRow*this->rowHeight + (static_cast<float>(this->secStructRows)+1.0f)), 1.0f, 1.0f,
                                   1.0f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);

                // draw the chain name and amino acid index                  
                tmpStr.Format("%c %i", this->chainID[i], this->aminoAcidIndex[i]);
                theFont.DrawString(static_cast<float>(i%this->resCols), -(resRow*this->rowHeight + (static_cast<float>(this->secStructRows)+1.5f)), 1.0f, 0.5f,
                                   ((this->aminoAcidIndex[i] < 1000)?(0.35f):(0.25f)), true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
                ////////////////////////////////////////////////
                // TODO: check size if aminoAcidIndex > 1000
                ////////////////////////////////////////////////  
            }
        }
        
        // draw legend/key
        if(this->toggleLegendParam.Param<param::BoolParam>()->Value()) {
            glColor3fv(fgColor);
            fontSize = 1.0f;
            if( theFont.Initialise() ) {
                               
                // RIGHT SIDE OF DIAGRAM //

                // draw secondary strucure type legend
                tmpStr = "Secondary Structure Types: ";
                wordlength = theFont.LineWidth( fontSize, tmpStr);
                theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -1.0f, wordlength, 1.0f,
                                   fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                                   
                for( unsigned int i = 0; i < static_cast<int>(UncertaintyDataCall::secStructure::NoE); i++) {
                    
                    s = static_cast<UncertaintyDataCall::secStructure>(i);
                    
                    switch (s) {
                        case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) : tmpStr = "[H] - Alpha Helix"; break;
                        case (UncertaintyDataCall::secStructure::G_310_HELIX) :   tmpStr = "[G] - 3-10 Helix"; break;
                        case (UncertaintyDataCall::secStructure::I_PI_HELIX) :    tmpStr = "[I] - Pi Helix"; break;
                        case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  tmpStr = "[E] - Strand"; break;
                        case (UncertaintyDataCall::secStructure::T_H_TURN) :      tmpStr = "[T] - Turn"; break;
                        case (UncertaintyDataCall::secStructure::B_BRIDGE) :      tmpStr = "[B] - Bridge"; break;
                        case (UncertaintyDataCall::secStructure::S_BEND) :        tmpStr = "[S] - Bend"; break;
                        case (UncertaintyDataCall::secStructure::C_COIL) :        tmpStr = "[C] - Random Coil"; break;
                        default: tmpStr = "No description"; break;
                    }
        
                    wordlength = theFont.LineWidth( fontSize, tmpStr.Add(":"));
                    theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -(static_cast<float>(i) + 2.0f), wordlength, 1.0f,
                                       fontSize * 0.5f, true, this->bindingSiteDescription[i], vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                }
                
                // draw binding site legend
                if( !this->bindingSiteNames.IsEmpty() ) {
                    
                    yPos = static_cast<float>(UncertaintyDataCall::secStructure::NoE) + 1.0f;
                    
                    tmpStr = "Binding Sites: ";
                    wordlength = theFont.LineWidth( fontSize, tmpStr);
                    theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -yPos, wordlength, 1.0f,
                                       fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                    for( unsigned int i = 0; i < this->bindingSiteNames.Count(); i++ ) {
                        // draw the binding site names
                        glColor3fv( this->bsColors[i].PeekComponents());
                        theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -(yPos + static_cast<float>(i) * 2.0f + 2.0f), wordlength, 1.0f,
                                           fontSize, true, this->bindingSiteNames[i], vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                        wordlength = theFont.LineWidth( fontSize, this->bindingSiteDescription[i]);
                        theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -( static_cast<float>(i) * 2.0f + 3.0f), wordlength, 1.0f,
                                           fontSize * 0.5f, true, this->bindingSiteDescription[i], vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                    }
                }
                
                // LEFT SIDE OF DIAGRAM //
                
                // draw diagram row legend for every row
                glColor3fv( fgColor);
                fontSize = 0.5f;
                yPos = 1.0f
                rowPos = 0.0f;
                for (unsigned int i = 0; i < this->resRows; i++) {
                    if (this->togglePdbParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "STRIDE";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString( -wordlength, -(rowPos+yPos), wordlength, 1.0f, fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "DSSP";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString( -wordlength, -(rowPos+yPos), wordlength, 1.0f, fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "AUTHOR (PDB)";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString( -wordlength, -(rowPos+yPos), wordlength, 1.0f, fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->toggleDiffParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Differences";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString( -wordlength, -(rowPos+yPos), wordlength, 1.0f, fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Uncertainty";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString( -wordlength, -(rowPos+yPos), wordlength, 1.0f, fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    tmpStr = "Amino Acid";
                    wordlength = theFont.LineWidth( fontSize, tmpStr) + fontSize;
                    theFont.DrawString( -wordlength,-(rowPos+yPos), wordlength, 1.0f,
                        fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                    tmpStr = "Chain-ID + PDB-Index";
                    wordlength = theFont.LineWidth( fontSize, tmpStr) + fontSize;
                    theFont.DrawString( -wordlength,-(rowPos+yPos)-0.5f, wordlength, 0.5f,
                        fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                    tmpStr = "Chain";
                    wordlength = theFont.LineWidth( fontSize, tmpStr) + fontSize;
                    theFont.DrawString( -wordlength,-(rowPos+yPos)-1.0f, wordlength, 0.5f,
                        fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                    if( !this->bindingSiteNames.IsEmpty() ) {
                        tmpStr = "Binding Sites";
                        wordlength = theFont.LineWidth( fontSize, tmpStr) + fontSize;
                        theFont.DrawString( -wordlength,-(this->rowHeight+rowPos), wordlength, (this->rowHeight+rowPos) - (2.0f + static_cast<float>(this->secStructRows)),
                            fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                    }
                    
                    yPos = 1.0f;
                    rowPos += this->rowHeight;
                }
            }
            
        }
        
            
///// GEOMETRY /////

        glDisable(GL_DEPTH_TEST);

        // draw overlays for selected amino acids
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        fgColor[3] = 0.3f;
        glColor4fv(fgColor);
        for( unsigned int i = 0; i < this->selection.Count(); i++ ) {
            if( this->selection[i] ) {
                pos.Set( static_cast<float>( i % this->resCols),
                    floorf( static_cast<float>(i) / this->resCols) + 1.0f);
                glBegin( GL_QUADS);
                    glVertex2f( pos.X()       , -this->rowHeight * (pos.Y() - 1.0f));
                    glVertex2f( pos.X()       , -this->rowHeight * (pos.Y()) + 0.5f);
                    glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y()) + 0.5f);
                    glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y() - 1.0f));
                glEnd();
            }
        }
        glDisable(GL_BLEND);

        // draw frames for selected amino acids
        fgColor[3] = 1.0f;
        glColor3fv( fgColor);
        glBegin(GL_LINES);
        for( unsigned int i = 0; i < this->selection.Count(); i++ ) {
            if( this->selection[i] ) {
                pos.Set( static_cast<float>( i % this->resCols),
                    floorf( static_cast<float>(i) / this->resCols) + 1.0f);
                // left (only draw if left neighbor ist not selected)
                if( i == 0 || !this->selection[i-1] ) {
                    glVertex2f( pos.X()       , -this->rowHeight * (pos.Y() - 1.0f));
                    glVertex2f( pos.X()       , -this->rowHeight * (pos.Y()) + 0.5f);
                }
                // bottom
                glVertex2f( pos.X()       , -this->rowHeight * (pos.Y()) + 0.5f);
                glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y()) + 0.5f);
                // right (only draw if right neighbor ist not selected)
                if( i == (this->selection.Count() - 1) || !this->selection[i+1] ) {
                    glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y()) + 0.5f);
                    glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y() - 1.0f));
                }
                // top
                glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y() - 1.0f));
                glVertex2f( pos.X()       , -this->rowHeight * (pos.Y() - 1.0f));
            }
        }
        glEnd();
           
        // render mouse hover
        if( this->mousePos.X() > -1.0f && this->mousePos.X() < static_cast<float>(this->resCols) &&
            this->mousePos.Y() >  0.0f && this->mousePos.Y() < static_cast<float>(this->resRows+1) &&
            this->mousePosResIdx > -1 && this->mousePosResIdx < (int)this->aminoAcidCount)
        {
            glColor3f( 1.0f, 0.75f, 0.0f);
            glBegin( GL_LINE_STRIP);
                glVertex2f( this->mousePos.X()       , -this->rowHeight * (this->mousePos.Y() - 1.0f));
                glVertex2f( this->mousePos.X()       , -this->rowHeight * (this->mousePos.Y()) + 0.5f);
                glVertex2f( this->mousePos.X() + 1.0f, -this->rowHeight * (this->mousePos.Y()) + 0.5f);
                glVertex2f( this->mousePos.X() + 1.0f, -this->rowHeight * (this->mousePos.Y() - 1.0f));
                glVertex2f( this->mousePos.X()       , -this->rowHeight * (this->mousePos.Y() - 1.0f));
            glEnd();
        }

        ////////////////////////////////////////////////
        // TODO: draw tooltip ...
        ////////////////////////////////////////////////       
        // draw tooltip with some information 
        /*if( this->mousePos.X() > -1.0f && this->mousePos.X() < static_cast<float>(this->resCols) &&
            this->mousePos.Y() >  0.0f && this->mousePos.Y() < static_cast<float>(this->resRows+1) &&
            this->mousePosResIdx > -1 && this->mousePosResIdx < (int)this->aminoAcidCount)
        {     
            // ...
        }*/
        
        glEnable(GL_DEPTH_TEST);
        
    } // dataPrepared
    
    return true;
}


/*
 * UncertaintySequenceRenderer::renderUncertaintyStack
 */
void UncertaintySequenceRenderer::renderUncertaintyStack(float yPos, float defColor[4]) {

    float uMax, uMid, uMin;
   
    // assuming sum of uncertainty values equals always 1.0
    
    glBegin(GL_QUADS);
    for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
        
        uMax = this->secUncertainty[i][this->sortedUncertainty[0]] 
        
        glColor3f(this->secStructureColor(this->sortedUncertainty[0]));
        glVertex2f( this->vertices[2*i],        -(this->vertices[2*i+1]+yPos) - 1.0f + uMax);
        glVertex2f( this->vertices[2*i],        -(this->vertices[2*i+1]+yPos) - 1.0f);
        glVertex2f( this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1]+yPos) - 1.0f);
        glVertex2f( this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1]+yPos) - 1.0f + uMax);
    
        uMid = uMax + this->secUncertainty[i][this->sortedUncertainty[1]] 
        
        glColor3f(this->secStructureColor(this->sortedUncertainty[1]));
        glVertex2f( this->vertices[2*i],        -(this->vertices[2*i+1]+yPos) - 1.0f + uMid);
        glVertex2f( this->vertices[2*i],        -(this->vertices[2*i+1]+yPos) - 1.0f + uMax);
        glVertex2f( this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1]+yPos) - 1.0f + uMax);
        glVertex2f( this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1]+yPos) - 1.0f + uMid);
        
        uMin = uMid + this->secUncertainty[i][this->sortedUncertainty[2]] 
        
        glColor3f(this->secStructureColor(this->sortedUncertainty[2]));
        glVertex2f( this->vertices[2*i],        -(this->vertices[2*i+1]+yPos) - 1.0f + uMin);
        glVertex2f( this->vertices[2*i],        -(this->vertices[2*i+1]+yPos) - 1.0f + uMid);
        glVertex2f( this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1]+yPos) - 1.0f + uMid);
        glVertex2f( this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1]+yPos) - 1.0f + uMin);
    
    }
    glEnd();
}


/*
 * UncertaintySequenceRenderer::drawSecStructTextureTiles
 */
void UncertaintySequenceRenderer::drawSecStructTextureTiles(UncertaintyDataCall::secStructure> secStructPre, 
                                                            UncertaintyDataCall::secStructure> secStructi, 
                                                            UncertaintyDataCall::secStructure> secStructSuc, bool m, float x, float y, float defColor[4]) {    
    const float eps = 0.0f;
        
    if(m) {
        glColor4fv(defColor);
        this->markerTextures[0]->Bind(); 
    }
    else {    
        glColor3f(this->secStructureColor(secStructi)); 
        switch (secStructi) {
            case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) :
                if ((i > 0) && (secStructPre != secStructi)) {
                    this->markerTextures[4]->Bind();
                } 
                else if ((i + 1) < this->aminoAcidCount) && (secStructSuc != secStructi)) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::G_310_HELIX) :        
                if ((i > 0) && (secStructPre != secStructi)) {
                    this->markerTextures[4]->Bind();
                } 
                else if ((i + 1) < this->aminoAcidCount) && (secStructSuc != secStructi)) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::I_PI_HELIX) :        
                if ((i > 0) && (secStructPre != secStructi)) {
                    this->markerTextures[4]->Bind();
                } 
                else if ((i + 1) < this->aminoAcidCount) && (secStructSuc != secStructi)) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  
                if (((i + 1) < this->aminoAcidCount) && (secStructSuc != secStructi)) {
                    this->markerTextures[3]->Bind();
                } 
                else {
                    this->markerTextures[2]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::T_H_TURN) : 
                this->markerTextures[1]->Bind();
                break;
            case (UncertaintyDataCall::secStructure::B_BRIDGE) :   
                this->markerTextures[1]->Bind(); 
                break;
            case (UncertaintyDataCall::secStructure::S_BEND) :   
                this->markerTextures[1]->Bind(); 
                break;
            case (UncertaintyDataCall::secStructure::C_COIL) :  
                this->markerTextures[1]->Bind();        
                break;
            default: this->markerTextures[0]->Bind(); break;
        }
    }
        
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glBegin( GL_QUADS);
        glTexCoord2f(eps      , eps);
        glVertex2f(  x        , -y);
        glTexCoord2f(eps      , 1.0f -eps);
        glVertex2f(  x        , -y -1.0f);
        glTexCoord2f(1.0f -eps, 1.0f -eps);
        glVertex2f(  x + 1.0f , -y -1.0f)
        glTexCoord2f(1.0f -eps, eps);
        glVertex2f(  x + 1.0f , -y );
    glEnd();
}


/*
* UncertaintySequenceRenderer::secStructureColor
*/
vislib::math::Vector<float, 4> UncertaintySequenceRenderer::secStructureColor(UncertaintyDataCall::secStructure s) {

    vislib::math::Vector<float, 4> color;
    color.Set(1.0f, 1.0f, 1.0f, 1.0f);
    
    switch (s) {
        case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) : color.Set(1.0f, 0.0f, 0.0f); break;
        case (UncertaintyDataCall::secStructure::G_310_HELIX) :   color.Set(1.0f, 0.33f, 0.0f); break;
        case (UncertaintyDataCall::secStructure::I_PI_HELIX) :    color.Set(1.0f, 0.66f, 0.0f); break;
        case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  color.Set(0.0f, 0.0f, 1.0f); break;
        case (UncertaintyDataCall::secStructure::T_H_TURN) :      color.Set(1.0f, 1.0f, 0.0f); break;
        case (UncertaintyDataCall::secStructure::B_BRIDGE) :      color.Set(0.66f, 1.0f, 0.0f); break;
        case (UncertaintyDataCall::secStructure::S_BEND) :        color.Set(0.33f, 1.0f, 0.0f); break;
        case (UncertaintyDataCall::secStructure::C_COIL) :        color.Set(0.5f, 0.5f, 0.5f); break;
        default: break; 
    }
    return color;
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool UncertaintySequenceRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;
    this->mousePos.Set( floorf(x), fabsf(floorf(y / this->rowHeight)));
    this->mousePosResIdx = static_cast<int>(this->mousePos.X() + (this->resCols * (this->mousePos.Y()-1)));
    // do nothing else if mouse is outside bounding box
    if( this->mousePos.X() < 0.0f || this->mousePos.X() > this->resCols ||
        this->mousePos.Y() < 0.0f || this->mousePos.Y() > this->resRows )
    {
        return consumeEvent;
    }
    
    // right click
    if (flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN) {
        if (flags & view::MOUSEFLAG_MODKEY_ALT_DOWN) {
            if( !this->rightMouseDown ) {
                this->initialClickSelection = !this->selection[this->mousePosResIdx];
            }
            this->selection[this->mousePosResIdx] = this->initialClickSelection;
            consumeEvent = true;
        } else {
            if( this->mousePosResIdx > -1 && this->mousePosResIdx < (int)this->aminoAcidCount) {
                this->selection[this->mousePosResIdx] = !this->selection[this->mousePosResIdx];
            }
        }
        this->rightMouseDown = true;
    } else {
        this->rightMouseDown = false;
    }
    
    // propagate selection to selection module
    if (this->resSelectionCall != NULL) {
        vislib::Array<ResidueSelectionCall::Residue> selectedRes;
        for (unsigned int i = 0; i < this->selection.Count(); i++) {
            if (this->selection[i]) {
                selectedRes.Add( ResidueSelectionCall::Residue());
                selectedRes.Last().chainID = this->chainID[i];
                selectedRes.Last().resNum = i;
                selectedRes.Last().id = -1;
            }
        }
        this->resSelectionCall->SetSelectionPointer(&selectedRes);
        (*this->resSelectionCall)(ResidueSelectionCall::CallForSetSelection);
    }

    return consumeEvent;
}



/*
 * UncertaintySequenceRenderer::PrepareData
 */
bool UncertaintySequenceRenderer::PrepareData(UncertaintyDataCall *udc, BindingSiteCall *bs) {

    if( !udc ) return false;

    // initialization
    this->aminoAcidCount = udc->GetAminoAcidCount();

    this->secUncertainty;
    this->secUncertainty.AssertCapacity(this->aminoAcidCount);
    
    this->sortedUncertainty;
    this->sortedUncertainty.AssertCapacity(this->aminoAcidCount);

    this->dsspSecStructure.Clear(),
    this->dsspSecStructure.AssertCapacity(this->aminoAcidCount);

    this->strideSecStructure.Clear();
    this->strideSecStructure.AssertCapacity(this->aminoAcidCount);

    this->pdbSecStructure.Clear();
    this->pdbSecStructure.AssertCapacity(this->aminoAcidCount);

    this->missingAminoAcids.Clear();
    this->missingAminoAcids.AssertCapacity(this->aminoAcidCount);
    
    this->selection.Clear();
    this->selection.AssertCapacity(this->aminoAcidCount);

    this->vertices.Clear();
    this->vertices.AssertCapacity(this->aminoAcidCount * 2);

    this->chainVertices.Clear();
    this->chainVertices.AssertCapacity(this->aminoAcidCount * 2);

    this->chainSeparatorVertices.Clear();
    this->chainSeparatorVertices.AssertCapacity(this->aminoAcidCount * 4);

    this->chainColors.Clear();
    this->chainColors.AssertCapacity(this->aminoAcidCount * 3);

    this->bsVertices.Clear();
    this->bsVertices.AssertCapacity(this->aminoAcidCount * 2);

    this->bsIndices.Clear();
    this->bsIndices.AssertCapacity(this->aminoAcidCount);

    this->bsColors.Clear();

    this->aminoAcidName.Clear();
    this->aminoAcidName.AssertCapacity(this->aminoAcidCount);

    this->chainID.Clear();
    this->chainID.AssertCapacity(this->aminoAcidCount);

    this->aminoAcidIndex.Clear();
    this->aminoAcidIndex.AssertCapacity(this->aminoAcidCount);

    this->bindingSiteDescription.Clear();

    this->bindingSiteNames.Clear();
    if( bs ) {
        this->bindingSiteDescription.SetCount( bs->GetBindingSiteCount());
        this->bindingSiteNames.AssertCapacity( bs->GetBindingSiteCount());
        this->bsColors.SetCount( bs->GetBindingSiteCount());
        // copy binding site names
        for( unsigned int i = 0; i < bs->GetBindingSiteCount(); i++ ) {
            this->bindingSiteDescription[i].Clear();
            this->bindingSiteNames.Add( bs->GetBindingSiteName( i));
            this->bsColors[i] = bs->GetBindingSiteColor(i);
        }
    }

    // temporary variables
    vislib::StringA tmpStr;
    unsigned int maxNumBindingSitesPerRes = 0;
    unsigned int cCnt = 0;
    char currentChainID = udc->GetChainID(0);

    // collect data from call
    for (unsigned int aa = 0; aa < this->aminoAcidCount; aa++) {

        // store the pdb index of the current amino-acid
        this->aminoAcidIndex.Add(udc->GetPDBAminoAcidIndex(aa));
        // store the secondary structure element type of the current amino-acid
        this->strideSecStructure.Add(udc->GetStrideSecStructure(aa));
        // store the amino-acid three letter code and convert it to the one letter code
        this->aminoAcidName.Add(this->GetAminoAcidOneLetterCode(udc->GetAminoAcid(aa)));
        // store the chainID
        this->chainID.Add(udc->GetChainID(aa));
        // init selection
        this->selection.Add(false);
        // store missing amino-acid flag
        this->missingAminoAcids.Add(udc->GetMissingFlag(aa));
        // store uncerteiny values
        this->secUncertainty.Add(udc->GetSecStructUncertainty(aa));
        // store sorted uncertainty structure types
        this->sortedUncertainty.Add(udc->GetSortedSecStructureIndices(aa));
        
        // count different chains and set seperator vertices
        if (udc->GetChainID(aa) != currentChainID) {
            currentChainID = udc->GetChainID(aa);
            this->chainSeparatorVertices.Add(this->vertices[this->vertices.Count() - 2] + 1.0f);
            this->chainSeparatorVertices.Add(-(this->chainVertices[this->chainVertices.Count() - 1] - 1.5f));
            this->chainSeparatorVertices.Add(this->vertices[this->vertices.Count() - 2] + 1.0f);
            this->chainSeparatorVertices.Add(-(this->chainVertices[this->chainVertices.Count() - 1] + 0.5f));
            cCnt++;
        }

        // compute the position of the amino-acid icon
        if (aa == 0) {
            // structure vertices
            this->vertices.Add(0.0f);
            this->vertices.Add(0.0f);
            // chain tile vertices and colors
            this->chainVertices.Add(0.0f);
            this->chainVertices.Add(static_cast<float>(this->secStructRows) + 1.5f);
            this->chainColors.Add(this->colorTable[cCnt].X());
            this->chainColors.Add(this->colorTable[cCnt].Y());
            this->chainColors.Add(this->colorTable[cCnt].Z());
        }
        else {
            if (aa % static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value()) != 0) {
                // structure vertices
                this->vertices.Add(this->vertices[aa * 2 - 2] + 1.0f);
                this->vertices.Add(this->vertices[aa * 2 - 1]);
                // chain tile vertices and colors
                this->chainVertices.Add(this->chainVertices[aa * 2 - 2] + 1.0f);
                this->chainVertices.Add(this->chainVertices[aa * 2 - 1]);
                this->chainColors.Add(this->colorTable[cCnt].X());
                this->chainColors.Add(this->colorTable[cCnt].Y());
                this->chainColors.Add(this->colorTable[cCnt].Z());
            }
            else {
                // structure vertices
                this->vertices.Add(0.0f);
                this->vertices.Add(this->vertices[aa * 2 - 1] + this->rowHeight);
                // chain tile vertices and colors
                this->chainVertices.Add(0.0f);
                this->chainVertices.Add(this->chainVertices[aa * 2 - 1] + this->rowHeight);
                this->chainColors.Add(this->colorTable[cCnt].X());
                this->chainColors.Add(this->colorTable[cCnt].Y());
                this->chainColors.Add(this->colorTable[cCnt].Z());
            }
        }

        // try to match binding sites
        if (bs) {
            vislib::Pair<char, unsigned int> bsRes;
            unsigned int numBS = 0;
            // loop over all binding sites
            for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
                for (unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count(); bsResCnt++) {
                    bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
                    if (udc->GetChainID(aa) == bsRes.First() && udc->GetPDBAminoAcidIndex(aa) == bsRes.Second() ) {
                        this->bsVertices.Add(this->vertices[this->vertices.Count() - 2]);
                        this->bsVertices.Add(this->vertices[this->vertices.Count() - 1] + this->rowHeight    + numBS * 0.5f);
                        this->bsIndices.Add(bsCnt);
                        if (!this->bindingSiteDescription[bsCnt].IsEmpty()) {
                            this->bindingSiteDescription[bsCnt].Append(", ");
                        }
                        this->bindingSiteDescription[bsCnt].Append(tmpStr);
                        numBS++;
                        maxNumBindingSitesPerRes = vislib::math::Max(maxNumBindingSitesPerRes, numBS);
                    }
                }
            }
        }
    }

    this->rowHeight = 2.0f + static_cast<float>(this->secStructRows) + maxNumBindingSitesPerRes * 0.5f + 0.5f;
    
    // loop over all binding sites and add binding site descriptions
    if( bs ) {
        for( unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++ ) {
            this->bindingSiteDescription[bsCnt].Prepend( "\n");
            this->bindingSiteDescription[bsCnt].Prepend( bs->GetBindingSiteDescription( bsCnt));
        }
    }

    // set the number of columns
    this->resCols = vislib::math::Min(this->aminoAcidCount,
        static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value()));
    this->resCountPerRowParam.Param<param::IntParam>()->SetValue( this->resCols);
    // compute the number of rows
    this->resRows = static_cast<unsigned int>( ceilf(static_cast<float>(this->aminoAcidCount) / static_cast<float>(this->resCols)));
    
    return true;
}


/*
 * Check if the residue is an amino acid.
 */
char UncertaintySequenceRenderer::GetAminoAcidOneLetterCode( vislib::StringA resName ) {
    if( resName.Equals( "ALA" ) ) return 'A';
    else if( resName.Equals( "ARG" ) ) return 'R';
    else if( resName.Equals( "ASN" ) ) return 'N';
    else if( resName.Equals( "ASP" ) ) return 'D';
    else if( resName.Equals( "CYS" ) ) return 'C';
    else if( resName.Equals( "GLN" ) ) return 'Q';
    else if( resName.Equals( "GLU" ) ) return 'E';
    else if( resName.Equals( "GLY" ) ) return 'G';
    else if( resName.Equals( "HIS" ) ) return 'H';
    else if( resName.Equals( "ILE" ) ) return 'I';
    else if( resName.Equals( "LEU" ) ) return 'L';
    else if( resName.Equals( "LYS" ) ) return 'K';
    else if( resName.Equals( "MET" ) ) return 'M';
    else if( resName.Equals( "PHE" ) ) return 'F';
    else if( resName.Equals( "PRO" ) ) return 'P';
    else if( resName.Equals( "SER" ) ) return 'S';
    else if( resName.Equals( "THR" ) ) return 'T';
    else if( resName.Equals( "TRP" ) ) return 'W';
    else if( resName.Equals( "TYR" ) ) return 'Y';
    else if( resName.Equals( "VAL" ) ) return 'V';
    else if( resName.Equals( "ASH" ) ) return 'D';
    else if( resName.Equals( "CYX" ) ) return 'C';
    else if( resName.Equals( "CYM" ) ) return 'C';
    else if( resName.Equals( "GLH" ) ) return 'E';
    else if( resName.Equals( "HID" ) ) return 'H';
    else if( resName.Equals( "HIE" ) ) return 'H';
    else if( resName.Equals( "HIP" ) ) return 'H';
    else if( resName.Equals( "MSE" ) ) return 'M';
    else if( resName.Equals( "LYN" ) ) return 'K';
    else if( resName.Equals( "TYM" ) ) return 'Y';
    else return '?';
}


/*
* UncertaintySequenceRenderer::LoadTexture
*/
bool UncertaintySequenceRenderer::LoadTexture(vislib::StringA filename) {
    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    void *buf = NULL;
    SIZE_T size = 0;

    if ((size = megamol::core::utility::ResourceWrapper::LoadResource(
            this->GetCoreInstance()->Configuration(), filename, &buf)) > 0) {
        if (pbc.Load(buf, size)) {
            img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);
            for (unsigned int i = 0; i < img.Width() * img.Height(); i++ ) {
                BYTE r = img.PeekDataAs<BYTE>()[i * 4 + 0];
                BYTE g = img.PeekDataAs<BYTE>()[i * 4 + 1];
                BYTE b = img.PeekDataAs<BYTE>()[i * 4 + 2];
                if (r + g + b > 0) {
                    img.PeekDataAs<BYTE>()[i * 4 + 3] = 255;
                } else {
                    img.PeekDataAs<BYTE>()[i * 4 + 3] = 0;
                }
            }
            markerTextures.Add( vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D>());
            markerTextures.Last() = new vislib::graphics::gl::OpenGLTexture2D();
            if (markerTextures.Last()->Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
                Log::DefaultLog.WriteError("Could not load \"%s\" texture.", filename.PeekBuffer());
                ARY_SAFE_DELETE(buf);
                return false;
            }
            markerTextures.Last()->SetFilter(GL_LINEAR, GL_LINEAR);
            ARY_SAFE_DELETE(buf);
            return true;
        } else {
            Log::DefaultLog.WriteError("Could not read \"%s\" texture.", filename.PeekBuffer());
        }
    } else {
        Log::DefaultLog.WriteError("Could not find \"%s\" texture.", filename.PeekBuffer());
    }
    return false;
}



        
