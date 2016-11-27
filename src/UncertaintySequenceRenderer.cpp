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

//////////////////////////////////////////////////////////////////////////////////////////////
//
// TODO:
//
// - eigene Texturen: Helix, ... ODER komplett Goemetrie?
// - Turn, Bend: Textur, Geometrie (2-5 Aminosäuren)
// - ANDERE Farben für Struktur-Typen ...
// - Anordnung horizontal in Zeilen: "max" unten "min" oben 
// - nur Strang-Ausschnitte zeigen, die sich unterscheiden
// - Animation: gleiches Intervall zwischen Strukturtypen, aber prozentuale Skalierung der Geometrie
// - 
// 
//////////////////////////////////////////////////////////////////////////////////////////////


#include "stdafx.h"
#include "UncertaintySequenceRenderer.h"

#include <GL/glu.h>
#include <math.h>
#include <ctime>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/utility/ResourceWrapper.h"

#include "vislib/math/Rectangle.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include "UncertaintyColor.h" 

#include <iostream> // DEBUG


#define PI 3.1415926535


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
            resCountPerRowParam( "ResiduesPerRow", "The number of residues per row." ),
            colorTableFileParam( "ColorTableFilename", "The filename of the color table."),
            toggleLegendParam( "LegendDrawing", "Show/hide row legend/key on the left."),
            clearResSelectionParam( "clearResidueSelection", "Clears the current selection (everything will be deselected)."),
            togglePdbParam("Author", "Show/hide PDB secondary structure row." ),
            toggleStrideParam("Stride", "Show/hide STRIDE secondary structure row."),
            toggleDsspParam("Dssp", "Show/hide DSSP secondary structure row."),
            toggleDiffParam("Difference", "Show/hide row with disagreements in secondary structure assignment."),
            toggleUncertaintyParam("Uncertainty", "Show/hide row with uncertainty of secondary structure assignment."), 
            uncertaintyVisualizationParam("Visualization", "Choose uncertainty Visualization."),   
            toggleTooltipParam("ToolTip", "Show/hide tooltip information."),
            animateMorphing("AnimateMorphing", "Turn animation of morphing ON or OFF."),
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
    
    
    // param slot for animation
    this->animateMorphing.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->animateMorphing);

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
    
    // param slot for uncertainty toggling
    this->toggleTooltipParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->toggleTooltipParam);

    // param for uncertainty Visualization
    this->currentVisualization = STACK;
    param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(this->currentVisualization));
    
    tmpEnum->SetTypePair(STACK,    "STACK");
    tmpEnum->SetTypePair(MORPHING, "MORPHING");
    // tmpEnum->SetTypePair(..., "...");
    
    this->uncertaintyVisualizationParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertaintyVisualizationParam);       


    // setting initial row height, as if all secondary structure rows would be shown
    this->secStructRows = 5;
    this->rowHeight = 2.0f + static_cast<float>(this->secStructRows);   

    // init animation timer
    this->animTimer = clock();
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
    
    
    // secondary strucutre vertices
    for (unsigned int i = 0; i < this->secStructVertices.Count(); i++) {
        this->secStructVertices[i].Clear();
    }
    this->secStructVertices.Clear();
        
    // B_BRIDGE     = arrow    => end of E_EXT_STRAND, too!
    // E_EXT_STRAND = quadrat
    for (unsigned int i = 0; i < UncertaintyDataCall::secStructure::NOE; i++) {

        UncertaintyDataCall::secStructure s = static_cast<UncertaintyDataCall::secStructure>(i);
        vislib::math::Vector<float, 2 > pos;
        float flip, dS, offset;
        // default
        unsigned int samples = 50;
        float        height  = 0.075f;  // in range 0.0 - 1.0

        this->secStructVertices.Add(vislib::Array<vislib::math::Vector<float, 2>>());

        if ((s == UncertaintyDataCall::secStructure::E_EXT_STRAND) ||
            (s == UncertaintyDataCall::secStructure::B_BRIDGE)) {
            height = 0.6f;
        }

        pos.SetX(0.0f);
        pos.SetY(-height/2.0f);
        this->secStructVertices.Last().Add(pos);

        for (unsigned int j = 0; j < (samples + 1); j++) { // loop over sample count
            flip = ((j % 2 == 0) ? (1.0f) : (-1.0f));
            offset = flip*(height / 2.0f);
            dS = static_cast<float>(j) / static_cast<float>(samples - 1);

            if (j == samples) // for last vertex
                dS = 1.0f;

            switch (s) {
            case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) :
            case (UncertaintyDataCall::secStructure::G_310_HELIX) :
            case (UncertaintyDataCall::secStructure::I_PI_HELIX) :
                 offset = flip*(height / 2.0f) + sin(dS*2.0f*PI)*0.25f;
                break;
            case (UncertaintyDataCall::secStructure::B_BRIDGE) :
                // 03525f = default height/2.0 of coil
                offset = ((dS > 1.0f/7.0f) ? (flip*(0.5f-0.03525f)*(7.0f-dS*7.0f)/6.0f + flip*(0.03525f)) : (flip*(height/2.0f))); 
                break;
            default: break;
            }
            pos.SetX(dS);
            pos.SetY(offset);
            this->secStructVertices.Last().Add(pos);
        }
    }        
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
        this->dataPrepared = false;
    }
       
    // prepare the data
    if(!this->dataPrepared) {
      
        // reset number of shown secondary structure rows
        this->secStructRows = 0;
        
        // calculate new row height
        if (this->togglePdbParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
        if (this->toggleStrideParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
        if (this->toggleDsspParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
        if (this->toggleDiffParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
        if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
                
        // set new row height
        this->rowHeight = 2.0f + static_cast<float>(this->secStructRows);
              
        unsigned int oldRowHeight = (unsigned int)this->rowHeight;

        this->dataPrepared = this->PrepareData(udc, bs);

        // row height might change because of bindinng sites
        if (oldRowHeight != this->rowHeight) {
            this->dataPrepared = this->PrepareData(udc, bs);
        }

        if (this->dataPrepared) {
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

            udc->SetRecalcFlag(false);
            this->pdbID = udc->GetPdbID();
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
    if(ud == NULL) return false;
    // execute the call
    if( !(*ud)(UncertaintyDataCall::CallForGetData)) return false;
  
    // check if new Visualization was chosen
	if (this->uncertaintyVisualizationParam.IsDirty()) {
        this->uncertaintyVisualizationParam.ResetDirty();  
        this->currentVisualization = static_cast<visualization>(this->uncertaintyVisualizationParam.Param<core::param::EnumParam>()->Value());
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
    if (this->colorTableFileParam.IsDirty()) {
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

    if(this->dataPrepared) {
        
        // temporary variables and constants
        vislib::StringA tmpStr;
        vislib::StringA tmpStr2;
        float yPos;
        float xPos;
        float wordlength;
        float rowPos;            
        float fontSize;
        vislib::math::Vector<float, 2> pos;
        float resRow;
        float diff;
        UncertaintyDataCall::secStructure s;
        float perCentRow;
        float start;
        float end;

        yPos = 0.0f; //!

        // draw uncertainty difference
        if (this->toggleDiffParam.Param<param::BoolParam>()->Value()) {
            glColor3fv(fgColor);
            glBegin(GL_QUADS);
            for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
                if (!this->missingAminoAcids[i]) {
                    // uncertainty Max - Min:
                    // three assignment methods -> max three different sec structures: max=[0] - middle=[1] - min=[2]
                    // assuming values are in range 0.0-1.0
                    diff = this->secUncertainty[i][this->sortedUncertainty[i][0]] - this->secUncertainty[i][this->sortedUncertainty[i][2]];

                    if (diff < 1.0f) {
                        glVertex2f(this->vertices[2 * i], -(this->vertices[2 * i + 1] + yPos) - diff);
                        glVertex2f(this->vertices[2 * i], -(this->vertices[2 * i + 1] + yPos) - 1.0f);
                        glVertex2f(this->vertices[2 * i] + 1.0f, -(this->vertices[2 * i + 1] + yPos) - 1.0f);
                        glVertex2f(this->vertices[2 * i] + 1.0f, -(this->vertices[2 * i + 1] + yPos) - diff);
                    }
                }
            }
            glEnd();
            yPos += 1.0f;
        }
        
        //draw texture tiles for secondary structure
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);      
        //STRIDE
        if(this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
               this->drawSecStructTextureTiles(((i > 0) ? (this->secStructAssignment[UncertaintyDataCall::assMethod::STRIDE][i - 1]) : 
                                               (this->secStructAssignment[UncertaintyDataCall::assMethod::STRIDE][i])), 
                                               this->secStructAssignment[UncertaintyDataCall::assMethod::STRIDE][i],
                                               ((i < (this->aminoAcidCount-1)) ? (this->secStructAssignment[UncertaintyDataCall::assMethod::STRIDE][i + 1]) : 
                                               (this->secStructAssignment[UncertaintyDataCall::assMethod::STRIDE][i])),
                                               this->missingAminoAcids[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);
            } 
            yPos += 1.0f;
        }
        // DSSP
        if(this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
                this->drawSecStructTextureTiles(((i > 0) ? (this->secStructAssignment[UncertaintyDataCall::assMethod::DSSP][i - 1]) : 
                                                (this->secStructAssignment[UncertaintyDataCall::assMethod::DSSP][i])), 
                                                this->secStructAssignment[UncertaintyDataCall::assMethod::DSSP][i],
                                                ((i < (this->aminoAcidCount-1)) ? (this->secStructAssignment[UncertaintyDataCall::assMethod::DSSP][i + 1]) : 
                                                (this->secStructAssignment[UncertaintyDataCall::assMethod::DSSP][i])),
                                                this->missingAminoAcids[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);                
            }
            yPos += 1.0f;
        }  
        // PDB 
        if(this->togglePdbParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
                this->drawSecStructTextureTiles(((i > 0) ? (this->secStructAssignment[UncertaintyDataCall::assMethod::PDB][i - 1]) : 
                                                (this->secStructAssignment[UncertaintyDataCall::assMethod::PDB][i])), 
                                                this->secStructAssignment[UncertaintyDataCall::assMethod::PDB][i],
                                                ((i < (this->aminoAcidCount-1)) ? (this->secStructAssignment[UncertaintyDataCall::assMethod::PDB][i + 1]) : 
                                                (this->secStructAssignment[UncertaintyDataCall::assMethod::PDB][i])),
                                                this->missingAminoAcids[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);                
            }
            yPos += 1.0f;
        }
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);

        // uncertainty visualization
        if(this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
            switch (this->currentVisualization) {
            case (STACK) : 
              
                this->renderUncertaintyStack(yPos); 
                break;

            case (MORPHING) :
            
                if (this->toggleDiffParam.IsDirty()) {
                    this->toggleDiffParam.ResetDirty();
                    if (this->animateMorphing.Param<param::BoolParam>()->Value())
                        this->animTimer = std::clock();
                }
                this->renderUncertaintyMorphing(yPos); 
                break;

            default: break;
            }
            yPos += 1.0f;
        }
    
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
        
        // draw type legend textures and lines
        xPos = 6.0f;
        if (this->toggleLegendParam.Param<param::BoolParam>()->Value()) {

            glEnable(GL_TEXTURE_2D);
            glEnable(GL_BLEND);
            for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); i++) {
                s = static_cast<UncertaintyDataCall::secStructure>(i);
                this->drawSecStructTextureTiles(s, s, UncertaintyDataCall::secStructure::NOTDEFINED, false, 
                                                (static_cast<float>(this->resCols) + 1.0f + xPos), (static_cast<float>(i)+1.5f), bgColor);
            }
            glDisable(GL_BLEND);
            glDisable(GL_TEXTURE_2D);

            glDisable(GL_DEPTH_TEST);

            glColor3fv(fgColor);
            for (unsigned int i = 0; i < (static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE)+1); i++) {
                glBegin(GL_LINES);
                glVertex2f((static_cast<float>(this->resCols) + 1.0f), -(static_cast<float>(i)+1.5f));
                glVertex2f((static_cast<float>(this->resCols) + 2.0f + xPos), -(static_cast<float>(i)+1.5f));
                glEnd();
            }

            glEnable(GL_DEPTH_TEST);
        }

        // INFO: DrawString(float x, float y, float w, float h, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) 
        glColor3fv(fgColor);
        resRow = 0.0f;
        fontSize = 1.5f;
        if( theFont.Initialise() ) {
            
            // pdb id as info on top
            tmpStr = this->pdbID;
            tmpStr.Prepend("PDB-ID: ");
            wordlength = theFont.LineWidth(fontSize, tmpStr);
            theFont.DrawString(0.0f, 0.5f, wordlength+1.0f, 1.0f, 
                               fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
            
            // row labeling
            for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
                // new row after resCols amino-acids
                if (((i%this->resCols) == 0) && (i != 0))
                    resRow += 1.0f;
                    
                // draw the one-letter amino acid code
                tmpStr.Format("%c", this->aminoAcidName[i]);
                theFont.DrawString(static_cast<float>(i%this->resCols), -(resRow*this->rowHeight + (static_cast<float>(this->secStructRows)+1.0f)), 
                                   1.0f, 1.0f, 1.0f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);

                // draw the chain name and amino acid index                  
                tmpStr.Format("%c %i", this->chainID[i], this->aminoAcidIndex[i]);
                theFont.DrawString(static_cast<float>(i%this->resCols), -(resRow*this->rowHeight + (static_cast<float>(this->secStructRows)+1.5f)), 
                                   1.0f, 0.5f, 0.35f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
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
                                   
                for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); i++) {
                    tmpStr = ud->GetSecStructDesc(static_cast<UncertaintyDataCall::secStructure>(i));
                    wordlength = theFont.LineWidth( fontSize, tmpStr);
                    theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -(static_cast<float>(i) + 2.5f), wordlength, 1.0f,
                                       fontSize * 0.75f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                }
                
                // draw binding site legend
                if( !this->bindingSiteNames.IsEmpty() ) {
                    
                    yPos = static_cast<float>(UncertaintyDataCall::secStructure::NOE) + 3.5f;
                    
                    tmpStr = "Binding Sites: ";
                    wordlength = theFont.LineWidth( fontSize, tmpStr);
                    theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -yPos, wordlength, 1.0f,
                                       fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                    for( unsigned int i = 0; i < this->bindingSiteNames.Count(); i++ ) {
                        // draw the binding site names
                        glColor3fv( this->bsColors[i].PeekComponents());
                        theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -(yPos + static_cast<float>(i) * 2.0f + 1.5f), wordlength, 1.0f,
                                           fontSize, true, this->bindingSiteNames[i], vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                        wordlength = theFont.LineWidth( fontSize, this->bindingSiteDescription[i]);
                        theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -(yPos + static_cast<float>(i) * 2.0f + 2.5f), wordlength, 1.0f,
                                           fontSize * 0.5f, true, this->bindingSiteDescription[i], vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                    }
                }
                
                // LEFT SIDE OF DIAGRAM //
                
                // draw diagram row legend for every row
                glColor3fv( fgColor);
                fontSize = 0.5f;
                yPos = 1.0f;
                rowPos = 0.0f;
                for (unsigned int i = 0; i < this->resRows; i++) {
                    if (this->toggleDiffParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Differences";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f,
                            fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "STRIDE";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f, 
                                           fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "DSSP";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f, 
                                           fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->togglePdbParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "PDB";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f, 
                                           fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Uncertainty";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f, 
                                           fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    tmpStr = "Amino Acid";
                    wordlength = theFont.LineWidth( fontSize, tmpStr) + fontSize;
                    theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f, 
                                       fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                    tmpStr = "Chain-ID + PDB-Index";
                    wordlength = theFont.LineWidth( fontSize, tmpStr) + fontSize;
                    theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos) - 0.5f, wordlength, 0.5f, 
                                       fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                    tmpStr = "Chain";
                    wordlength = theFont.LineWidth( fontSize, tmpStr) + fontSize;
                    theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos) - 1.0f, wordlength, 0.5f, 
                                       fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                    if( !this->bindingSiteNames.IsEmpty() ) {
                        tmpStr = "Binding Sites";
                        wordlength = theFont.LineWidth( fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i + 1)*this->rowHeight), wordlength, 
                                           (this->rowHeight - (2.0f + static_cast<float>(this->secStructRows))),
                                           fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                    }
                    
                    yPos = 1.0f;
                }
            }
        }

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
        if( (this->mousePos.X() > -1.0f) && (this->mousePos.X() < static_cast<float>(this->resCols)) &&
            (this->mousePos.Y() >  0.0f) && (this->mousePos.Y() < static_cast<float>(this->resRows+1)) &&
            (this->mousePosResIdx > -1) && (this->mousePosResIdx < (int)this->aminoAcidCount)) {

            glColor3f( 1.0f, 0.75f, 0.0f);
            glBegin( GL_LINE_STRIP);
                glVertex2f( this->mousePos.X()       , -this->rowHeight * (this->mousePos.Y() - 1.0f));
                glVertex2f( this->mousePos.X()       , -this->rowHeight * (this->mousePos.Y()));
                glVertex2f( this->mousePos.X() + 1.0f, -this->rowHeight * (this->mousePos.Y()));
                glVertex2f( this->mousePos.X() + 1.0f, -this->rowHeight * (this->mousePos.Y() - 1.0f));
                glVertex2f( this->mousePos.X()       , -this->rowHeight * (this->mousePos.Y() - 1.0f));
            glEnd();

            // draw tooltip  
            if (this->toggleTooltipParam.Param<param::BoolParam>()->Value()) {
                glColor3fv(fgColor);
                fontSize = 0.5f;
                perCentRow = 1.0f / this->rowHeight;
                start = 0.0;
                end = perCentRow;
                if (theFont.Initialise()) {
                    if (this->toggleDiffParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Difference:";
                        tmpStr2.Format("%.0f %%", (1.0f - (this->secUncertainty[mousePosResIdx][this->sortedUncertainty[mousePosResIdx][0]] 
                                                   - this->secUncertainty[mousePosResIdx][this->sortedUncertainty[mousePosResIdx][2]]))*100.0f);
                        this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Stride:";
                        tmpStr2 = ud->GetSecStructDesc(this->secStructAssignment[UncertaintyDataCall::assMethod::STRIDE][mousePosResIdx]);
                        this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Dssp:";
                        tmpStr2 = ud->GetSecStructDesc(this->secStructAssignment[UncertaintyDataCall::assMethod::DSSP][mousePosResIdx]);
                        this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->togglePdbParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Pdb:";
                        tmpStr2 = ud->GetSecStructDesc(this->secStructAssignment[UncertaintyDataCall::assMethod::PDB][mousePosResIdx]);
                        this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Uncertainty:";
                        tmpStr2.Format("%s: %.0f %% | %s: %.0f %% | %s: %.0f %%", ud->GetSecStructDesc(this->sortedUncertainty[mousePosResIdx][0]), 
                                       this->secUncertainty[mousePosResIdx][this->sortedUncertainty[mousePosResIdx][0]]*100.0f, 
                                       ud->GetSecStructDesc(this->sortedUncertainty[mousePosResIdx][1]), 
                                       this->secUncertainty[mousePosResIdx][this->sortedUncertainty[mousePosResIdx][1]]*100.0f,
                                       ud->GetSecStructDesc(this->sortedUncertainty[mousePosResIdx][2]), 
                                       this->secUncertainty[mousePosResIdx][this->sortedUncertainty[mousePosResIdx][2]]*100.0f);
                        this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                    }
                }
            }
        }

        glEnable(GL_DEPTH_TEST);
        
    } // dataPrepared
    
    return true;
}



/*
 * UncertaintySequenceRenderer::renderUncertaintyMorphing
 */
void UncertaintySequenceRenderer::renderUncertaintyMorphing(float yPos) {

	// get pointer to UncertaintyDataCall
	UncertaintyDataCall *ud = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if(ud == NULL) return false;
    // execute the call
    if( !(*ud)(UncertaintyDataCall::CallForGetData)) return false;

    UncertaintyDataCall::secStructure sMax, sTemp;  // structure types 
    float uMax, uDelta, uTemp;                      // uncertainty
    vislib::math::Vector<float, 2> posOffset, posTemp;
    float timer, deltaT;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 

    for (unsigned int i = 0; i < this->aminoAcidCount; i++) { // loop over all amino-acids
        if (!this->missingAminoAcids[i] && (this->sortedUncertainty[i][0] < 1.0f)) { // skip missing and "sure" amino-acids

            // default
            posOffset.SetX(this->vertices[2 * i]);
            posOffset.SetY(this->vertices[2 * i + 1] + yPos + 0.5f);

            sMax = this->sortedUncertainty[i][0];
            uMax = this->secUncertainty[i][sMax];

            if (uMax == 1.0f) {
                /*
                glColor3fv(ud->GetSecStructColor(sMax).PeekComponents());
                // check if end of strand is an arrow (the arrows vertices are stored in BRIDGE )
                if ((sMax == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sMax)) {
                    sMax = UncertaintyDataCall::secStructure::B_BRIDGE;
                }
                glBegin(GL_TRIANGLE_STRIP);
                for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                    glVertex2f(x + this->secStructVertices[sMax][j].X(), -(y + this->secStructVertices[sMax][j].Y()));
                }
                glEnd();
                */
            }
            else {
                
                glDisable(GL_DEPTH_TEST);

                // draw background quads
                glBegin(GL_QUADS);
                uDelta = 0.0f;
                for (unsigned int j = 0; j < UncertaintyDataCall:assMethod::NOE; j++) {
                    glColor3fv(ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(this->sortedUncertainty[i][j])).PeekComponents());
                    uTemp = this->secUncertainty[i][this->sortedUncertainty[i][j]];
                    glVertex2f(x,        -(y+0.5f-uDelta-uTemp));
                    glVertex2f(x,        -(y+0.5f-uDelta));
                    glVertex2f(x + 1.0f, -(y+0.5f-uDelta));
                    glVertex2f(x + 1.0f, -(y+0.5f-uDelta-uTemp));
                    uDelta += uTemp;            
                }                
                glEnd();
                
                /*
                // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                if ((sLeft == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sLeft)) {
                    sLeft = UncertaintyDataCall::secStructure::B_BRIDGE;
                }
                if ((sMiddle == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sMiddle)) {
                    sMiddle = UncertaintyDataCall::secStructure::B_BRIDGE;
                }
                if ((sRight == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sRight)) {
                    sRight = UncertaintyDataCall::secStructure::B_BRIDGE;
                }
                */

                // draw structure type geometry
                glBegin(GL_TRIANGLE_STRIP);
                glColor3f(0.0f, 0.0f, 0.0f);
                if (this->animateMorphing.Param<param::BoolParam>()->Value()) {
                    // animation
                    /*
                    deltaT = 1.0f; // animation speed: greater value means slower (= time interval)
                    timer = static_cast<float>((clock() - this->animTimer) / (deltaXf*CLOCKS_PER_SEC));
                    if (timer > deltaT) {
                        this->animTimer = clock();
                    }

                    for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                        xTemp = this->secStructVertices[sMax][j].X();
                        
                        yTemp = ; 
                        
                        
                        
                        
                        glVertex2f(x + xTemp, -(y + yTemp));
                    }
                    */
                    // OLD stuff
                    /*
                    for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                        xTemp = this->secStructVertices[sMax][j].X();
                        if (timer< 0.5f) {
                            yTemp = (1.0f - timer*2.0f)*this->secStructVertices[sLeft][j].Y() + (timer*2.0f)*this->secStructVertices[sMiddle][j].Y();
                        }
                        else if (timer< 1.0f){
                            yTemp = (1.0f - (timer-0.5f)*2.0f)*this->secStructVertices[sMiddle][j].Y() + ((timer-0.5f)*2.0f)*this->secStructVertices[sRight][j].Y();
                        }
                        else if (timer< 1.5f){
                            yTemp = (1.0f - (timer-1.0f)*2.0f)*this->secStructVertices[sRight][j].Y() + ((timer-1.0f)*2.0f)*this->secStructVertices[sMiddle][j].Y();
                        }
                        else if (timer< 2.0f){
                            yTemp = (1.0f - (timer-1.5f)*2.0f)*this->secStructVertices[sMiddle][j].Y() + ((timer-1.5f)*2.0f)*this->secStructVertices[sLeft][j].Y();
                        }
                        glVertex2f(x + xTemp, -(y + yTemp));
                    }
                    */
                }
                else { // without animation
                    for (unsigned int j = 0; j < UncertaintyDataCall:assMethod::NOE; j++) {
                        for (unsigned int k = 0; k < this->secStructVertices[sMax].Count(); k++) {
                            posTemp = this->secStructVertices[static_cast<int>(this->sortedUncertainty[i][j])][k];
                            glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() + posTemp.Y()*this->secUncertainty[i][this->sortedUncertainty[i][j]]);
                        }
                        posOffset.SetY(posOffset.Y() + this->secUncertainty[i][this->sortedUncertainty[i][j]]);
                    }
                }
                glEnd();

                glEnable(GL_DEPTH_TEST);
            }
        }
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); //reset
}


// OLD STUFF
/*
 * UncertaintySequenceRenderer::renderUncertaintyMorphing
 */
 /*
void UncertaintySequenceRenderer::renderUncertaintyMorphing(float yPos) {

	// get pointer to UncertaintyDataCall
	UncertaintyDataCall *ud = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if(ud == NULL) return false;
    // execute the call
    if( !(*ud)(UncertaintyDataCall::CallForGetData)) return false;

    UncertaintyDataCall::secStructure sMax, sMid, sMin;       // structure types 
    UncertaintyDataCall::secStructure sLeft, sRight, sMiddle; // structure types 
    float uMax, uLeft, uMiddle, uRight;                       // uncertainty
    vislib::math::Vector<float, 4> cLeft, cMiddle, cRight;    // colors
    float x, y, xTemp, yTemp;
    float deltaXf;
    float timer;
    float startTime, deltaTime;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 

    for (unsigned int i = 0; i < this->aminoAcidCount; i++) { // loop over all amino-acids
        if (!this->missingAminoAcids[i]) { // skip missing

            // default
            x = this->vertices[2 * i];
            y = this->vertices[2 * i + 1] + yPos + 0.5f;

            sMax = this->sortedUncertainty[i][0];
            sMid = this->sortedUncertainty[i][1];
            sMin = this->sortedUncertainty[i][2];

            uMax = this->secUncertainty[i][sMax];

            if (uMax == 1.0f) {
                sRight = sMax;
               glColor3fv(ud->GetSecStructColor(sMax).PeekComponents());
                // check if end of strand is an arrow (the arrows vertices are stored in BRIDGE )
                if ((sMax == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sMax)) {
                    sMax = UncertaintyDataCall::secStructure::B_BRIDGE;
                }
                glBegin(GL_TRIANGLE_STRIP);
                for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                    glVertex2f(x + this->secStructVertices[sMax][j].X(), -(y + this->secStructVertices[sMax][j].Y()));
                }
                glEnd();
            }
            else {
                             
                //assign structure type for left side
                if (i > 0) {
                    if (this->secUncertainty[i-1][sRight] > 0.0f)
                        sLeft = sRight;
                    else if (this->secUncertainty[i-1][sMiddle] > 0.0f)
                        sLeft = sMiddle;
                }
                else {
                    sLeft = this->sortedUncertainty[0][0];
                    sMiddle = this->sortedUncertainty[0][0];
                }
                //assign structure type for right side
                if (i < this->aminoAcidCount - 1)
                    sRight = this->sortedUncertainty[i+1][0];
                else
                    sRight = sMax;

                // check where structure types match with left or right side
                if (sMax == sLeft){
                    if (sMid == sRight)
                        sMiddle = sMin;
                    else {
                        sMiddle = sMid;
                        sRight = sMin;
                    }
                }
                else if (sMid == sLeft) {
                    if (sMax == sRight)
                        sMiddle = sMin;
                    else {
                        sMiddle = sMax;
                        sRight = sMin;
                    }
                }
                else if (sMin == sLeft) {
                    if (sMax == sRight)
                        sMiddle = sMid;
                    else {
                        sMiddle = sMax;
                        sRight = sMid;
                    }
                }
                else if (sMax == sRight) {
                    if (sMin == sLeft)
                        sMiddle = sMid;
                    else {
                        sLeft = sMid;
                        sMiddle = sMin;
                    }
                }
                else if (sMid == sRight) {
                    if (sMin == sLeft)
                        sMiddle = sMax;
                    else {
                        sLeft = sMax;
                        sMiddle = sMin;
                    }
                }
                else if (sMin == sRight) {
                    sLeft = sMax;
                    sMiddle = sMin;
                }
                else {
                    sLeft = sMax;
                    sMiddle = sMid;
                    sRight = sMin;
                }

               cLeft = ud->GetSecStructColor(sLeft);
               cMiddle = ud->GetSecStructColor(sMiddle);
               cRight = ud->GetSecStructColor(sRight);

                glDisable(GL_DEPTH_TEST);

                // draw background quads
                glBegin(GL_QUADS);

                deltaXf = this->secUncertainty[i][sLeft];
                glColor3fv(cLeft.PeekComponents());
                glVertex2f(x,           -(y-0.5f));
                glVertex2f(x,           -(y+0.5f));
                glVertex2f(x + deltaXf, -(y + 0.5f));
                glVertex2f(x + deltaXf, -(y - 0.5f));

                glColor3fv(cMiddle.PeekComponents());
                glVertex2f(x + deltaXf, -(y - 0.5f));
                glVertex2f(x + deltaXf, -(y + 0.5f));

                deltaXf += this->secUncertainty[i][sMiddle];
                glVertex2f(x + deltaXf, -(y + 0.5f));
                glVertex2f(x + deltaXf, -(y - 0.5f));

                glColor3fv(cRight.PeekComponents());
                glVertex2f(x + deltaXf, -(y - 0.5f));
                glVertex2f(x + deltaXf, -(y + 0.5f));
                glVertex2f(x + 1.0f,    -(y + 0.5f));
                glVertex2f(x + 1.0f,    -(y - 0.5f));
                
                glEnd();
                

                uLeft = this->secUncertainty[i][sLeft];
                uMiddle = this->secUncertainty[i][sMiddle];
                uRight = this->secUncertainty[i][sRight];

                // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                if ((sLeft == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sLeft)) {
                    sLeft = UncertaintyDataCall::secStructure::B_BRIDGE;
                }
                if ((sMiddle == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sMiddle)) {
                    sMiddle = UncertaintyDataCall::secStructure::B_BRIDGE;
                }
                if ((sRight == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sRight)) {
                    sRight = UncertaintyDataCall::secStructure::B_BRIDGE;
                }

                // draw structure type geometry
                glBegin(GL_TRIANGLE_STRIP);
                glColor3f(0.0f, 0.0f, 0.0f);
                if (this->animateMorphing.Param<param::BoolParam>()->Value()) {
                    // animation

                    deltaXf = 1.0f; // animation speed: greater value means slower (= time interval)

                    timer = static_cast<float>((clock() - this->animTimer) / (deltaXf*CLOCKS_PER_SEC));
                    if (timer > deltaXf) {
                        this->animTimer = clock();
                    }
                    
                    uLeft = (uLeft*deltaXf)/2.0f;
                    uMiddle = (uMiddle*deltaXf)/2.0f;
                    uRight = (uRight*deltaXf)/2.0f;

                    for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                        xTemp = this->secStructVertices[sMax][j].X();

                        if ((uLeft > 0.0f) && (uMiddle > 0.0f) && (uRight > 0.0f)) {
                            if (timer < uLeft + uMiddle) {
                                startTime = 0.0f;
                                deltaTime = uLeft + uMiddle;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sLeft][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sMiddle][j].Y();
                            }
                            else if (timer < uLeft + uMiddle + uRight) {
                                startTime = uLeft + uMiddle;
                                deltaTime = uRight;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sMiddle][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sRight][j].Y();
                            }
                            else if (timer < uLeft + uMiddle + uRight + uRight) {
                                startTime = uLeft + uMiddle + uRight;
                                deltaTime = uRight;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sRight][j].Y() 
                                        + ((timer - startTime) / (deltaTime))*this->secStructVertices[sMiddle][j].Y();
                            }
                            else {
                                startTime = uLeft + uMiddle + uRight + uRight;
                                deltaTime = deltaXf - startTime;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sMiddle][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sLeft][j].Y();
                            }
                        else if ((uLeft > 0.0f) && (uMiddle > 0.0f) && (uRight == 0.0f)) {
                            if (timer < uLeft + uMiddle) {
                                startTime = 0.0f;
                                deltaTime = uLeft + uMiddle;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sLeft][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sMiddle][j].Y();
                            }
                            else {
                                startTime = uLeft + uMiddle;
                                deltaTime = deltaXf - startTime;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sMiddle][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sLeft][j].Y();
                            }
                        }
                        else if ((uLeft > 0.0f) && (uMiddle == 0.0f) && (uRight > 0.0f)) {
                            if (timer < uLeft+uRight) {
                                startTime = 0.0f;
                                deltaTime = uLeft + uRight;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sLeft][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sRight][j].Y();
                            }
                            else{
                                startTime = uLeft + uRight;
                                deltaTime = deltaXf - startTime;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sRight][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sLeft][j].Y();
                            }
                        }
                        else if ((uLeft == 0.0f) && (uMiddle > 0.0f) && (uRight > 0.0f)) {
                            if (timer < uMiddle+uRight) {
                                startTime = 0.0f;
                                deltaTime = uMiddle + uRight;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sMiddle][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sRight][j].Y();
                            }
                            else {
                                startTime = uMiddle + uRight;
                                deltaTime = deltaXf - startTime;
                                yTemp = (1.0f - (timer - startTime) / deltaTime)*this->secStructVertices[sRight][j].Y() 
                                         + ((timer - startTime) / (deltaTime))*this->secStructVertices[sMiddle][j].Y();
                            }
                        }
                        glVertex2f(x + xTemp, -(y + yTemp));
                    }
                    
                    // OLD stuff
                    //for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                    //    xTemp = this->secStructVertices[sMax][j].X();
                    //    if (timer< 0.5f) {
                    //        yTemp = (1.0f - timer*2.0f)*this->secStructVertices[sLeft][j].Y() + (timer*2.0f)*this->secStructVertices[sMiddle][j].Y();
                    //    }
                    //    else if (timer< 1.0f){
                    //        yTemp = (1.0f - (timer-0.5f)*2.0f)*this->secStructVertices[sMiddle][j].Y() + ((timer-0.5f)*2.0f)*this->secStructVertices[sRight][j].Y();
                    //    }
                    //    else if (timer< 1.5f){
                    //        yTemp = (1.0f - (timer-1.0f)*2.0f)*this->secStructVertices[sRight][j].Y() + ((timer-1.0f)*2.0f)*this->secStructVertices[sMiddle][j].Y();
                    //    }
                    //    else if (timer< 2.0f){
                    //        yTemp = (1.0f - (timer-1.5f)*2.0f)*this->secStructVertices[sMiddle][j].Y() + ((timer-1.5f)*2.0f)*this->secStructVertices[sLeft][j].Y();
                    //    }
                    //    glVertex2f(x + xTemp, -(y + yTemp));
                    //}
                    
                }
                else { // without animation
                    for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                        xTemp = this->secStructVertices[sLeft][j].X() * uLeft;
                        yTemp = this->secStructVertices[sLeft][j].Y(); // * uLeft;
                        glVertex2f(x + xTemp, -(y + yTemp));
                    }
                    for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                        xTemp = this->secStructVertices[sMiddle][j].X()  * uMiddle + uLeft;
                        yTemp = this->secStructVertices[sMiddle][j].Y(); // * uMiddle;
                        glVertex2f(x + xTemp, -(y + yTemp));
                    }
                    for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                        xTemp = this->secStructVertices[sRight][j].X() * uRight + (uLeft + uMiddle);
                        yTemp = this->secStructVertices[sRight][j].Y(); // * uRight;
                        glVertex2f(x + xTemp, -(y + yTemp));
                    }

                    // OLD stuff
                    
                    //for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                    //    if ((static_cast<float>(j) / static_cast<float>(this->secStructVertices[sMax].Count())) < uLeft)
                    //        glColor3fv(cLeft.PeekComponents());
                    //    else if ((static_cast<float>(j) / static_cast<float>(this->secStructVertices[sMax].Count())) > (uLeft + uMiddle))
                    //        glColor3fv(cRight.PeekComponents());
                    //    else
                    //        glColor3fv(cMiddle.PeekComponents());
                    //
                    //    xTemp = this->secStructVertices[sMax][j].X();
                    //    yTemp = uLeft*this->secStructVertices[sLeft][j].Y() + uRight*this->secStructVertices[sRight][j].Y() + uMiddle*this->secStructVertices[sMiddle][j].Y();
                    //    glVertex2f(x + xTemp, -(y + yTemp));
                    //} 
                }
                glEnd();

                glEnable(GL_DEPTH_TEST);
            }
        }
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); //reset
}
*/


/*
 * UncertaintySequenceRenderer::renderUncertaintyStack
 */
void UncertaintySequenceRenderer::renderUncertaintyStack(float yPos) {

	// get pointer to UncertaintyDataCall
	UncertaintyDataCall *ud = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if(ud == NULL) return false;
    // execute the call
    if( !(*ud)(UncertaintyDataCall::CallForGetData)) return false;
    
    float uTemp, uDelta;
   
    // assuming sum of uncertainty values equals always 1.0   
    glBegin(GL_QUADS);
    for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
        if (!this->missingAminoAcids[i] ) { // && (this->sortedUncertainty[i][0] < 1.0f)) {
            uDelta = 0.0f;
            for (unsigned int j = 0; j < UncertaintyDataCall:assMethod::NOE; j++) {
                glColor3fv(ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(this->sortedUncertainty[i][j])).PeekComponents());
                uTemp = this->secUncertainty[i][this->sortedUncertainty[i][j]];
                glVertex2f(this->vertices[2*i],        -(this->vertices[2*i+1] + yPos + 1.0f - uDelta - uTemp));
                glVertex2f(this->vertices[2*i],        -(this->vertices[2*i+1] + yPos + 1.0f - uDelta));
                glVertex2f(this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1] + yPos + 1.0f - uDelta));
                glVertex2f(this->vertices[2*i] + 1.0f, -(this->vertices[2*i+1] + yPos + 1.0f - uDelta - uTemp));
                uDelta += uTemp;            
            }               
        }
    }
    glEnd();
}


/*
* UncertaintySequenceRenderer::renderToolTip
*/
void UncertaintySequenceRenderer::renderToolTip(float start, float end, vislib::StringA str1, vislib::StringA str2, float fgColor[4], float bgColor[4]) {

    float fontSize = 0.5f;
    float wordlength;

    if (this->mousePosDetail.Y() > start && this->mousePosDetail.Y() < end) {

        wordlength = theFont.LineWidth(fontSize, str1) + 0.25f;
        if ((theFont.LineWidth(fontSize, str2) + 0.25f) > wordlength)
            wordlength = theFont.LineWidth(fontSize, str2) + 0.25f;

        glColor3fv(bgColor);
        glBegin(GL_QUADS);
        for (unsigned int i = 0; i < this->chainVertices.Count() / 2; i++) {
            glVertex2f(this->mousePos.X() + 1.05f, -this->rowHeight*(this->mousePos.Y() - (1.0f - start)));
            glVertex2f(this->mousePos.X() + 1.05f, -this->rowHeight*(this->mousePos.Y() - (1.0f - end)));
            glVertex2f(this->mousePos.X() + 1.05f + wordlength, -this->rowHeight*(this->mousePos.Y() - (1.0f - end)));
            glVertex2f(this->mousePos.X() + 1.05f + wordlength, -this->rowHeight*(this->mousePos.Y() - (1.0f - start)));
        }
        glEnd();

        glColor3fv(fgColor);
        theFont.DrawString(this->mousePos.X() + 1.05f, -this->rowHeight*(this->mousePos.Y() - (1.0f - start)) - 0.5f, wordlength, 0.5f,
            fontSize, true, str1, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
        theFont.DrawString(this->mousePos.X() + 1.05f, -this->rowHeight*(this->mousePos.Y() - (1.0f - start)) - 1.0f, wordlength, 0.5f,
            fontSize, true, str2, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
    }
}


/*
 * UncertaintySequenceRenderer::drawSecStructTextureTiles
 */
void UncertaintySequenceRenderer::drawSecStructTextureTiles(UncertaintyDataCall::secStructure secStructPre, 
                                                            UncertaintyDataCall::secStructure secStructi, 
                                                            UncertaintyDataCall::secStructure secStructSuc, bool m, float x, float y, float defColor[4]) {    
    
    // get pointer to UncertaintyDataCall
	UncertaintyDataCall *ud = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if(ud == NULL) return false;
    // execute the call
    if( !(*ud)(UncertaintyDataCall::CallForGetData)) return false;
    
    const float eps = 0.0f;
        
    if(m) {
        glColor4fv(defColor);
        this->markerTextures[0]->Bind(); 
    }
    else {    
        glColor3fv(ud->GetSecStructColor(secStructi).PeekComponents()); 
        switch (secStructi) {
            case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) :
                if (secStructPre != secStructi) {
                    this->markerTextures[4]->Bind();
                } 
                else if (secStructSuc != secStructi) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::G_310_HELIX) :        
                if (secStructPre != secStructi) {
                    this->markerTextures[4]->Bind();
                } 
                else if (secStructSuc != secStructi) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::I_PI_HELIX) :        
                if (secStructPre != secStructi) {
                    this->markerTextures[4]->Bind();
                } 
                else if (secStructSuc != secStructi) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  
                if (secStructSuc != secStructi) {
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
                if (secStructSuc != secStructi) {
                    this->markerTextures[3]->Bind();
                }
                else {
                    this->markerTextures[2]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::S_BEND) :   
                this->markerTextures[1]->Bind(); 
                break;
            case (UncertaintyDataCall::secStructure::C_COIL) :  
                this->markerTextures[1]->Bind();        
                break;
            case (UncertaintyDataCall::secStructure::NOTDEFINED) :  
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
        glVertex2f(  x + 1.0f , -y -1.0f);
        glTexCoord2f(1.0f -eps, eps);
        glVertex2f(  x + 1.0f , -y );
    glEnd();
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool UncertaintySequenceRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;
    this->mousePos.Set(floorf(x), -(floorf(y / this->rowHeight))); // this->mousePos.Set(floorf(x), fabsf(floorf(y / this->rowHeight)));
    this->mousePosDetail.Set(x - floorf(x), ((-y/this->rowHeight) - floorf(-y/this->rowHeight)));

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

    this->secUncertainty.Clear();
    this->secUncertainty.AssertCapacity(this->aminoAcidCount);
    
    this->sortedUncertainty.Clear();
    this->sortedUncertainty.AssertCapacity(this->aminoAcidCount);

    for (unsigned int i = 0; i < this->secStructAssignment.Count(); i++) {
        this->secStructAssignment.Clear();
    }
    this->secStructAssignment.Clear();
    
    for (unsigned int i = 0; i < UncertaintyDataCall::assMethod::NOE; i++) {
        this->secStructAssignment.Add(vislib::Array<secStruture> );
        this->secStructAssignment.Last().AssertCapacity(this->aminoAcidCount);
    }

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

    // temporary variables
    vislib::StringA tmpStr;
    unsigned int maxNumBindingSitesPerRes = 0;
    unsigned int cCnt = 0;
    char currentChainID = udc->GetChainID(0);

    // handling binding sites
    if (bs) {
        this->bindingSiteDescription.SetCount(bs->GetBindingSiteCount());
        this->bindingSiteNames.AssertCapacity(bs->GetBindingSiteCount());
        this->bsColors.SetCount(bs->GetBindingSiteCount());
        // copy binding site names
        for (unsigned int i = 0; i < bs->GetBindingSiteCount(); i++) {
            this->bindingSiteDescription[i].Clear();
            this->bindingSiteNames.Add(bs->GetBindingSiteName(i));
            this->bsColors[i] = bs->GetBindingSiteColor(i);
        }
    }
        
    // collect data from call
    for (unsigned int aa = 0; aa < this->aminoAcidCount; aa++) {

        // store the pdb index of the current amino-acid
        this->aminoAcidIndex.Add(udc->GetPDBAminoAcidIndex(aa));
        
        // store the secondary structure element type of the current amino-acid
        this->secStructAssignment[UncertaintyDataCall::assMethod::DSSP].Add(udc->GetSecStructure(UncertaintyDataCall::assMethod::DSSP, aa));
        this->secStructAssignment[UncertaintyDataCall::assMethod::PDB].Add(udc->GetSecStructure(UncertaintyDataCall::assMethod::PDB, aa));
        this->secStructAssignment[UncertaintyDataCall::assMethod::STRIDE].Add(udc->GetSecStructure(UncertaintyDataCall::assMethod::STRIDE, aa));
        
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

        if (bs) {
            // try to match binding sites
            vislib::Pair<char, unsigned int> bsRes;
            unsigned int numBS = 0;
            // loop over all binding sites
            for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
                for (unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count(); bsResCnt++) {
                    bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
                    if (udc->GetChainID(aa) == bsRes.First() && udc->GetPDBAminoAcidIndex(aa) == bsRes.Second()) {
                        this->bsVertices.Add(this->vertices[this->vertices.Count() - 2]);
                        this->bsVertices.Add(this->vertices[this->vertices.Count() - 1] + (2.0f + static_cast<float>(this->secStructRows)) + numBS*0.5f);
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

    if (bs) {
        // loop over all binding sites and add binding site descriptions
        for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
            this->bindingSiteDescription[bsCnt].Prepend("\n");
            this->bindingSiteDescription[bsCnt].Prepend(bs->GetBindingSiteDescription(bsCnt));
        }
    }

    // set new row height
    this->rowHeight = 2.0f + static_cast<float>(this->secStructRows) + maxNumBindingSitesPerRes * 0.5f;

    // set the number of columns
    this->resCols = vislib::math::Min(this->aminoAcidCount,
        static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value()));
    this->resCountPerRowParam.Param<param::IntParam>()->SetValue( this->resCols);
    // compute the number of rows
    this->resRows = static_cast<unsigned int>( ceilf(static_cast<float>(this->aminoAcidCount) / static_cast<float>(this->resCols)));
    
    return true;
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


/*
* Check if the residue is an amino acid.
*/
char UncertaintySequenceRenderer::GetAminoAcidOneLetterCode(vislib::StringA resName) {
    
    if      (resName.Equals("ALA")) return 'A';
    else if (resName.Equals("ARG")) return 'R';
    else if (resName.Equals("ASN")) return 'N';
    else if (resName.Equals("ASP")) return 'D';
    else if (resName.Equals("CYS")) return 'C';
    else if (resName.Equals("GLN")) return 'Q';
    else if (resName.Equals("GLU")) return 'E';
    else if (resName.Equals("GLY")) return 'G';
    else if (resName.Equals("HIS")) return 'H';
    else if (resName.Equals("ILE")) return 'I';
    else if (resName.Equals("LEU")) return 'L';
    else if (resName.Equals("LYS")) return 'K';
    else if (resName.Equals("MET")) return 'M';
    else if (resName.Equals("PHE")) return 'F';
    else if (resName.Equals("PRO")) return 'P';
    else if (resName.Equals("SER")) return 'S';
    else if (resName.Equals("THR")) return 'T';
    else if (resName.Equals("TRP")) return 'W';
    else if (resName.Equals("TYR")) return 'Y';
    else if (resName.Equals("VAL")) return 'V';
    else if (resName.Equals("ASH")) return 'D';
    else if (resName.Equals("CYX")) return 'C';
    else if (resName.Equals("CYM")) return 'C';
    else if (resName.Equals("GLH")) return 'E';
    else if (resName.Equals("HID")) return 'H';
    else if (resName.Equals("HIE")) return 'H';
    else if (resName.Equals("HIP")) return 'H';
    else if (resName.Equals("MSE")) return 'M';
    else if (resName.Equals("LYN")) return 'K';
    else if (resName.Equals("TYM")) return 'Y';
    else return '?';
    
}

