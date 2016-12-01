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
            toggleDiffParam(                    "01 Difference", "Show/hide row with disagreements in secondary structure assignment."),
            toggleStrideParam(                  "02 Stride", "Show/hide STRIDE secondary structure row."),
            toggleDsspParam(                    "03 Dssp", "Show/hide DSSP secondary structure row."),
            togglePdbParam(                     "04 Author", "Show/hide PDB secondary structure row."),
            toggleUncertaintyParam(             "05 Uncertainty", "Show/hide row with uncertainty of secondary structure assignment."),
            certainBlockChartColorParam(        "06 Certain: Block chart color", "Choose color of block chart for certain structure."),
            certainStructColorParam(             "07 Certain: Structure color", "Choose color of structure for certain structure."),
            uncertainBlockChartColorParam(      "08 Uncertain: Block chart color", "Choose color of block chart for uncertain structure."),
            uncertainBlockChartOrientationParam("09 Uncertain: Block orientation", "Choose orientation of block chart for uncertain structure."),
            uncertainStructColorParam(           "10 Uncertain: Structure color", "Choose color of structure for uncertain structure."),
            uncertainStructGeometryParam(        "11 Uncertain: Structure geometry", "Choose geometry of structure for uncertain structure."),
            toggleLegendParam(                  "12 Legend Drawing", "Show/hide row legend/key on the left."),
            toggleTooltipParam(                 "13 ToolTip", "Show/hide tooltip information."),
            resCountPerRowParam(                "14 Residues Per Row", "The number of residues per row."),
            clearResSelectionParam(             "15 Clear Residue Selection", "Clears the current selection (everything will be deselected)."),
            colorTableFileParam(                "16 Color Table Filename", "The filename of the color table."),
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
    
    // param slot for tooltip toggling
    this->toggleTooltipParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleTooltipParam);

    // parameter for viualization options
    this->currentCertainBlockChartColor         = CERTAIN_BC_NONE;
    this->currentCertainStructColor              = CERTAIN_STRUCT_GRAY;
    this->currentUncertainBlockChartColor       = UNCERTAIN_BC_COLORED;
    this->currentUncertainBlockChartOrientation = UNCERTAIN_BC_VERTI;
    this->currentUncertainStructColor            = UNCERTAIN_STRUCT_GRAY;
    this->currentUncertainStructGeometry         = UNCERTAIN_STRUCT_DYN_EQUAL;
        
    param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(this->currentCertainBlockChartColor));
    tmpEnum->SetTypePair(CERTAIN_BC_NONE,    "No coloring");
    tmpEnum->SetTypePair(CERTAIN_BC_COLORED, "Colored");
    this->certainBlockChartColorParam << tmpEnum;
    this->MakeSlotAvailable(&this->certainBlockChartColorParam);       

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentCertainStructColor));
    tmpEnum->SetTypePair(CERTAIN_STRUCT_NONE,    "No coloring");
    tmpEnum->SetTypePair(CERTAIN_STRUCT_COLORED, "Colored");
    tmpEnum->SetTypePair(CERTAIN_STRUCT_GRAY,    "Gray");
    this->certainStructColorParam << tmpEnum;
    this->MakeSlotAvailable(&this->certainStructColorParam); 

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainBlockChartColor));
    tmpEnum->SetTypePair(UNCERTAIN_BC_NONE,    "No coloring");
    tmpEnum->SetTypePair(UNCERTAIN_BC_COLORED, "Colored");
    this->uncertainBlockChartColorParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainBlockChartColorParam); 
    
    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainBlockChartOrientation));
    tmpEnum->SetTypePair(UNCERTAIN_BC_VERTI, "vertical");
    tmpEnum->SetTypePair(UNCERTAIN_BC_HORIZ, "horizontal");
    this->uncertainBlockChartOrientationParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainBlockChartOrientationParam);     

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainStructColor));
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_NONE,    "No coloring");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_COLORED, "Colored");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_GRAY,    "Gray");
    this->uncertainStructColorParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainStructColorParam); 
    
    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainStructGeometry));
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_STAT_VERTI, "Static - vertical");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_STAT_HORIZ, "Static - horizontal");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_STAT_MORPH, "Static - morphed");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_DYN_TIME,   "Dynamic - time");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_DYN_SPACE,  "Dynamic - space");    
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_DYN_EQUAL,  "Dynamic - equal");     
    this->uncertainStructGeometryParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainStructGeometryParam); 

    // setting initial row height, as if all secondary structure rows would be shown
    this->secStructRows = 5;
    this->rowHeight = 2.0f + static_cast<float>(this->secStructRows);   

    // init animation timer
    this->animTimer = std::clock();
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
    
    // load textures
    this->LoadTexture("secStruct-white.png");       // markerTextures[0]
    this->LoadTexture("secStruct-coil.png");        // markerTextures[1]
    this->LoadTexture("secStruct-sheet.png");       // markerTextures[2]
    this->LoadTexture("secStruct-arrow.png");       // markerTextures[3]
    this->LoadTexture("secStruct-helix-left.png");  // markerTextures[4]
    this->LoadTexture("secStruct-helix2.png");      // markerTextures[5]
    this->LoadTexture("secStruct-helix-right.png"); // markerTextures[6]
    
    // calculate secondary structure vertices
    for (unsigned int i = 0; i < this->secStructVertices.Count(); i++) {
        this->secStructVertices[i].Clear();
    }
    this->secStructVertices.Clear();
        
    // B_BRIDGE     = arrow    => end of E_EXT_STRAND, too!
    // E_EXT_STRAND = quadrat
    for (unsigned int i = 0; i < UncertaintyDataCall::secStructure::NOE; i++) {

        UncertaintyDataCall::secStructure s = static_cast<UncertaintyDataCall::secStructure>(i);
        vislib::math::Vector<float, 2 > pos;
        float flip, dS, offset, heightStrand;
        // default
        unsigned int samples = 50;
        float        height  = 0.1f;  // in range 0.0 - 1.0

        this->secStructVertices.Add(vislib::Array<vislib::math::Vector<float, 2>>());

        pos.SetX(0.0f);
        pos.SetY(-height / 2.0f);

        if ((s == UncertaintyDataCall::secStructure::E_EXT_STRAND) ||
            (s == UncertaintyDataCall::secStructure::B_BRIDGE)) {
            heightStrand = 0.6f;
            pos.SetY(-heightStrand / 2.0f);
        }

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
                offset = ((dS > 1.0f/7.0f) ? (flip*(0.5f-(height/2.0f))*(7.0f-dS*7.0f)/6.0f + flip*(height/2.0f)) : (flip*(heightStrand/2.0f))); 
                break;
            case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :
                offset = flip*(heightStrand / 2.0f);
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
  
      
    // check uncertainty visualization options
    if (this->certainBlockChartColorParam.IsDirty()) {
        this->certainBlockChartColorParam.ResetDirty();
        this->currentCertainBlockChartColor = static_cast<certainBlockChartColor>(this->certainBlockChartColorParam.Param<param::EnumParam>()->Value());
    }
    if (this->certainStructColorParam.IsDirty()) {
        this->certainStructColorParam.ResetDirty();
        this->currentCertainStructColor = static_cast<certainStructColor>(this->certainStructColorParam.Param<param::EnumParam>()->Value());
    }
    if (this->uncertainBlockChartColorParam.IsDirty()) {
        this->uncertainBlockChartColorParam.ResetDirty();
        this->currentUncertainBlockChartColor = static_cast<uncertainBlockChartColor>(this->uncertainBlockChartColorParam.Param<param::EnumParam>()->Value());
    }
    if (this->uncertainBlockChartOrientationParam.IsDirty()) {
        this->uncertainBlockChartOrientationParam.ResetDirty();
        this->currentUncertainBlockChartOrientation = static_cast<uncertainBlockChartOrientation>(this->uncertainBlockChartOrientationParam.Param<param::EnumParam>()->Value());
    }
    if (this->uncertainStructColorParam.IsDirty()) {
        this->uncertainStructColorParam.ResetDirty();
        this->currentUncertainStructColor = static_cast<uncertainStructColor>(this->uncertainStructColorParam.Param<param::EnumParam>()->Value());
    }
    if (this->uncertainStructGeometryParam.IsDirty()) {
        this->uncertainStructGeometryParam.ResetDirty();
        this->currentUncertainStructGeometry = static_cast<uncertainStructGeometry>(this->uncertainStructGeometryParam.Param<param::EnumParam>()->Value());
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
        vislib::StringA tmpStr3;
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
                if (this->residueFlag[i] != UncertaintyDataCall::addFlags::MISSING) {
                    // uncertainty Max - Min:
                    // three assignment methods -> max three different sec structures: max=[0] - middle=[1] - min=[2]
                    // assuming values are in range 0.0-1.0
                    diff = this->secUncertainty[i][(int)this->sortedUncertainty[i][0]] - this->secUncertainty[i][(int)this->sortedUncertainty[i][2]];

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
                                               this->residueFlag[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor, ud);
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
                                                this->residueFlag[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor, ud);                
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
                                                this->residueFlag[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor, ud);                
            }
            yPos += 1.0f;
        }
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);           
            
        // uncertainty visualization
        if(this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
			this->renderUncertainty(yPos, ud, fgColor, bgColor);
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
                this->drawSecStructTextureTiles(s, s, UncertaintyDataCall::secStructure::NOTDEFINED, UncertaintyDataCall::addFlags::NOTHING,
                                                (static_cast<float>(this->resCols) + 1.0f + xPos), (static_cast<float>(i)+1.5f), bgColor, ud);
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
                switch(this->residueFlag[i]) {
                    case(UncertaintyDataCall::addFlags::MISSING):   glColor3f(0.3f, 0.3f, 0.3f); break;
                    case(UncertaintyDataCall::addFlags::HETEROGEN): glColor3f(1.0f, 0.5f, 0.5f); break;
                    // case(UncertaintyDataCall::addFlags::NOTHING):   glColor3fv(fgColor); break;
                    default: break;
                }
                tmpStr.Format("%c", this->aminoAcidName[i]);
                theFont.DrawString(static_cast<float>(i%this->resCols), -(resRow*this->rowHeight + (static_cast<float>(this->secStructRows)+1.0f)), 
                                   1.0f, 1.0f, 1.0f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
                glColor3fv(fgColor); // reset color
                
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
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Difference:";
                            tmpStr2.Format("%.0f %%", (1.0f - (this->secUncertainty[mousePosResIdx][(int)this->sortedUncertainty[mousePosResIdx][0]] 
                                                        - this->secUncertainty[mousePosResIdx][(int)this->sortedUncertainty[mousePosResIdx][2]]))*100.0f);
                            this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Stride:";
                            tmpStr2 = ud->GetSecStructDesc(this->secStructAssignment[UncertaintyDataCall::assMethod::STRIDE][mousePosResIdx]);
                            this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Dssp:";
                            tmpStr2 = ud->GetSecStructDesc(this->secStructAssignment[UncertaintyDataCall::assMethod::DSSP][mousePosResIdx]);
                            this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->togglePdbParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Pdb:";
                            tmpStr2 = ud->GetSecStructDesc(this->secStructAssignment[UncertaintyDataCall::assMethod::PDB][mousePosResIdx]);
                            this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Uncertainty:";
                            tmpStr2 = "";
                            for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
                                if (this->secUncertainty[mousePosResIdx][(int)this->sortedUncertainty[mousePosResIdx][i]] > 0.0f) {
                                    if (i > 0) {
                                        tmpStr2.Append("|");
                                    }
                                    tmpStr3.Format(" %s: %.0f %% ", ud->GetSecStructDesc(this->sortedUncertainty[mousePosResIdx][i]),
                                        this->secUncertainty[mousePosResIdx][(int)this->sortedUncertainty[mousePosResIdx][i]] * 100.0f);
                                    tmpStr2.Append(tmpStr3);
                                }
                            }
                            this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;                        
                    }
                    tmpStr = "Flag:";  
                    switch (this->residueFlag[mousePosResIdx]) {
                        case(UncertaintyDataCall::addFlags::MISSING):   
                            tmpStr2 = "MISSING"; 
                            this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                            break;
                        case(UncertaintyDataCall::addFlags::HETEROGEN): 
                            tmpStr2 = "HETEROGEN"; 
                            this->renderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                            break;
                        case(UncertaintyDataCall::addFlags::NOTHING): 
                            break;
                        default: break;
                    }                  
                } // init font
            } // legend param
        } // mouse pos

        glEnable(GL_DEPTH_TEST);
        
    } // dataPrepared
    
    return true;
}


/*
 * UncertaintySequenceRenderer::renderUncertainty
 */
void UncertaintySequenceRenderer::renderUncertainty(float yPos, UncertaintyDataCall *ud, float fgColor[4], float bgColor[4]) {

    if(ud == NULL) return;

    vislib::math::Vector<float, 4> grayColor = ud->GetSecStructColor(UncertaintyDataCall::secStructure::NOTDEFINED);

    UncertaintyDataCall::secStructure sMax, sTemp;     // structure types 
    float uMax, uDelta, uTemp;                         // uncertainty
    float timer, tDelta, tSpeed;                       // time
    vislib::math::Vector<float, 2> posOffset, posTemp; // position
    unsigned int diffStruct;                           // different strucutres with uncertainty > 0
    vislib::math::Vector<float, 4> colTemp;            // color
    float fTtemp, fDelta;
    bool foundColor;
    
    UncertaintyDataCall::secStructure sLeft, sRight;
    // sorted from left to right
    vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::assMethod::NOM)> sortedStructLR;
    // sorted bottom to top
    vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::assMethod::NOM)> sortedStructBT;
    vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::assMethod::NOM)> lastSortedStructBT;
    

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
    glDisable(GL_DEPTH_TEST);
                
    for (unsigned int i = 0; i < this->aminoAcidCount; i++) { // loop over all amino-acids
        if (this->residueFlag[i] != UncertaintyDataCall::addFlags::MISSING) { // skip missing 

            // default
            posOffset.SetX(this->vertices[2 * i]);
            posOffset.SetY(this->vertices[2 * i + 1] + yPos + 1.0f);

            sMax = this->sortedUncertainty[i][0];
            uMax = this->secUncertainty[i][(int)sMax];
            
// CERTAIN
            if (uMax == 1.0f) {
                // block chart color
                if (this->currentCertainBlockChartColor == CERTAIN_BC_COLORED) {
                    glColor3fv(ud->GetSecStructColor(sMax).PeekComponents());
                }
                else { // this->currentCertainBlockChartColor == CERTAIN_BC_NONE
                    glColor3fv(bgColor);
                }
                
                // draw block chart quad
                glBegin(GL_QUADS);
                glVertex2f(posOffset.X(),        -(posOffset.Y() - 1.0f));
                glVertex2f(posOffset.X(),        -(posOffset.Y()));
                glVertex2f(posOffset.X() + 1.0f, -(posOffset.Y()));
                glVertex2f(posOffset.X() + 1.0f, -(posOffset.Y() - 1.0f));          
                glEnd();
    
                // strucutre color
                if (this->currentCertainStructColor == CERTAIN_STRUCT_COLORED) {
                    glColor3fv(ud->GetSecStructColor(sMax).PeekComponents());
                }
                else if (this->currentCertainStructColor == CERTAIN_STRUCT_GRAY) {
                    glColor3fv(grayColor.PeekComponents());
                }
                else { // this->currentCertainStructColor == CERTAIN_STRUCT_NONE
                    glColor3fv(bgColor);
                }
                    
                // draw structure
                posOffset.SetY(posOffset.Y() - 0.5f);
                // check if end of strand is an arrow (the arrows vertices are stored in BRIDGE )
                if ((sMax == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sMax)) {
                    sMax = UncertaintyDataCall::secStructure::B_BRIDGE;
                }
                glBegin(GL_TRIANGLE_STRIP);
                for (unsigned int j = 0; j < this->secStructVertices[(int)sMax].Count(); j++) {
                    glVertex2f(posOffset.X() + this->secStructVertices[(int)sMax][j].X(), -(posOffset.Y() + this->secStructVertices[(int)sMax][j].Y()));
                }
                glEnd();
            }
// UNCERTAIN
            else { 
                // sort structure from bottom to top
                if ((this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_HORIZ) ||
                    (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_HORIZ)) {
                        
                    if (i > 0) {
                        lastSortedStructBT = sortedStructBT;                        
                    }
                    else {
                        for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
                            lastSortedStructBT[j] = sMax;
                        }
                    }
                    // copy to data sortedStruct
                    for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
                        sortedStructBT[j] = this->sortedUncertainty[i][j];
                    }
                    // search for structure type matching the last sorted 
                    for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {  // lastSortedStruct
                        for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) { // sortedStruct
                            if (lastSortedStructBT[j] == sortedStructBT[k]) {
                                sTemp = sortedStructBT[j];
                                sortedStructBT[j] = sortedStructBT[k];
                                sortedStructBT[k] = sTemp;
                            } 
                        }
                    }
                } 

                // sort structure from left to right 
                if ((this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_VERTI) ||
                    (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_VERTI) ||
                    (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_MORPH)) {

                    //assign structure type for left side
                    if (i > 0) {
                        for (unsigned int j = 0; j < static_cast<int>(UncertaintyDataCall::assMethod::NOM); j++) {
                            if (this->secUncertainty[i-1][(int)sortedStructLR[j]] > 0.0f) {
                                sLeft = sortedStructLR[j]; // take the last one which is the most right one > 0
                            }
                        }
                    }
                    else {
                        sLeft = sMax;
                    }
                    //assign structure type for right side
                    if (i < this->aminoAcidCount - 1)
                        sRight = this->sortedUncertainty[i+1][0]; // next structure type with max certainty
                    else
                        sRight = sMax;
                    // copy to data sortedStruct
                    for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
                        sortedStructLR[j] = this->sortedUncertainty[i][j];
                    }
                    // search for structure type matching left or right ones
                    for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
                        if (sortedStructLR[j] == sRight) {
                            sTemp = sortedStructLR[(int)(UncertaintyDataCall::assMethod::NOM) - 1];
                            sortedStructLR[(int)(UncertaintyDataCall::assMethod::NOM) - 1] = sortedStructLR[j];
                            sortedStructLR[j] = sTemp;
                        }
                        if (sortedStructLR[j] == sLeft) { // left comes last and has priority
                            sTemp = sortedStructLR[0];
                            sortedStructLR[0] = sortedStructLR[j];
                            sortedStructLR[j] = sTemp;
                        } 
                    }
                }
                
                // block chart color
                if (this->currentUncertainBlockChartColor == UNCERTAIN_BC_NONE)
                    glColor3fv(bgColor);
                                       
                // draw block chart in corresponding orientation
                glBegin(GL_QUADS);                
                if (this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_VERTI) {

                    uDelta = 0.0f;
                    for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
                        
                        if (this->currentUncertainBlockChartColor == UNCERTAIN_BC_COLORED)
                            glColor3fv(ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sortedStructLR[j])).PeekComponents());

                        uTemp = this->secUncertainty[i][(int)sortedStructLR[j]];
                        
                        glVertex2f(posOffset.X() + uDelta + uTemp, -(posOffset.Y()));
                        glVertex2f(posOffset.X() + uDelta,         -(posOffset.Y()));
                        glVertex2f(posOffset.X() + uDelta,         -(posOffset.Y() - 1.0f));
                        glVertex2f(posOffset.X() + uDelta + uTemp, -(posOffset.Y() - 1.0f));
                        
                        uDelta += uTemp;            
                    }                
                }
                else if (this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_HORIZ) {

                    uDelta = 0.0f;
                    for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
                        
                        if (this->currentUncertainBlockChartColor == UNCERTAIN_BC_COLORED)
                            glColor3fv(ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sortedStructBT[j])).PeekComponents());

                        uTemp = this->secUncertainty[i][(int)sortedStructBT[j]];
                        
                        glVertex2f(posOffset.X(),        -(posOffset.Y() - uDelta - uTemp));
                        glVertex2f(posOffset.X(),        -(posOffset.Y() - uDelta));
                        glVertex2f(posOffset.X() + 1.0f, -(posOffset.Y() - uDelta));
                        glVertex2f(posOffset.X() + 1.0f, -(posOffset.Y() - uDelta - uTemp));
                        
                        uDelta += uTemp;            
                    }                
                }
                glEnd(); 
                
                // strucutre color
                if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_NONE)
                    glColor3fv(bgColor);                
                else if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_GRAY)
                    glColor3fv(grayColor.PeekComponents());

                // draw structure type geometry
                if ((this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_TIME) || 
                    (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_SPACE) || 
                    (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_EQUAL)) {
                    
                    tSpeed = 2.0f; // animation speed: greater value means slower (= time interval)
                    timer = static_cast<float>((std::clock() - this->animTimer) / (tSpeed*CLOCKS_PER_SEC));
                    if (timer > 1.0f) {
                        this->animTimer = std::clock();
                    }
                    
                    // get the number off structures with uncertainty greater 0
                    diffStruct = 0;
                    for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
                        if (this->secUncertainty[i][(int)this->sortedUncertainty[i][k]] > 0.0f)
                            diffStruct++;
                    }

                    glBegin(GL_TRIANGLE_STRIP);
                    posOffset.SetY(posOffset.Y() - 0.5f);

                    for (unsigned int j = 0; j < this->secStructVertices[(int)sMax].Count(); j++) {
                        
                        if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_TIME) {
                            tDelta = 0.0f;
                            for (unsigned int k = 0; k < diffStruct; k++) {

                                sTemp = this->sortedUncertainty[i][k];
                                uTemp = this->secUncertainty[i][(int)sTemp];

                                if ((tDelta <= timer) && (timer <= (tDelta + uTemp))) {

                                    if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                                        colTemp = (1.0f - (timer - tDelta) / uTemp)*(ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sTemp)));
                                    }
                                    // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                                    if ((sTemp == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sTemp)) {
                                        sTemp = UncertaintyDataCall::secStructure::B_BRIDGE;
                                    }
                                    posTemp = this->secStructVertices[(int)sTemp][j];
                                    posTemp.SetY((1.0f - (timer - tDelta) / uTemp)*this->secStructVertices[(int)sTemp][j].Y());

                                    if (k == diffStruct - 1) {
                                        sTemp = this->sortedUncertainty[i][0];
                                    }
                                    else {
                                        sTemp = this->sortedUncertainty[i][k + 1];
                                    }
                                    if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                                        colTemp = colTemp + ((timer - tDelta) / uTemp)*(ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sTemp)));
                                        glColor3fv(colTemp.PeekComponents());
                                    }
                                    // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                                    if ((sTemp == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sTemp)) {
                                        sTemp = UncertaintyDataCall::secStructure::B_BRIDGE;
                                    }
                                    posTemp.SetY(posTemp.Y() + ((timer - tDelta) / uTemp)*this->secStructVertices[(int)sTemp][j].Y());

                                    glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() + posTemp.Y()));
                                }
                                
                                tDelta += uTemp;
                            }
                        } 
                        else {
                            tDelta = 1.0f / static_cast<float>(diffStruct);
                            for (unsigned int k = 0; k < diffStruct; k++) {
                                
                                if ((static_cast<float>(k)*tDelta <= timer) && (timer <= (static_cast<float>(k)*tDelta + tDelta))) {

                                    sTemp = this->sortedUncertainty[i][k];
                                    uTemp = this->secUncertainty[i][(int)sTemp];

                                    if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                                        colTemp = (1.0f - (timer - static_cast<float>(k)*tDelta) / tDelta)*(ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sTemp)));
                                    }
                                    // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                                    if ((sTemp == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sTemp)) {
                                        sTemp = UncertaintyDataCall::secStructure::B_BRIDGE;
                                    }
                                    posTemp = this->secStructVertices[(int)sTemp][j];
                                    
                                    if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_SPACE) {
                                        posTemp.SetY((1.0f - (timer - static_cast<float>(k)*tDelta) / tDelta)*this->secStructVertices[(int)sTemp][j].Y()*uTemp);
                                    }
                                    if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_EQUAL) {
                                        posTemp.SetY((1.0f - (timer - static_cast<float>(k)*tDelta) / tDelta)*this->secStructVertices[(int)sTemp][j].Y());
                                    }

                                    
                                    if (k == diffStruct - 1) {
                                        sTemp = this->sortedUncertainty[i][0];
                                    }
                                    else {
                                        sTemp = this->sortedUncertainty[i][k+1];
                                    }

                                    uTemp = this->secUncertainty[i][(int)sTemp];

                                    if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                                        colTemp = colTemp + ((timer - static_cast<float>(k)*tDelta) / tDelta)*(ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sTemp)));
                                        glColor3fv(colTemp.PeekComponents());
                                    }
                                    // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                                    if ((sTemp == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sTemp)) {
                                        sTemp = UncertaintyDataCall::secStructure::B_BRIDGE;
                                    }
                                    
                                    if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_SPACE){
                                        posTemp.SetY(posTemp.Y() + ((timer - static_cast<float>(k)*tDelta) / tDelta)*this->secStructVertices[(int)sTemp][j].Y()*uTemp);
                                    }
                                    if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_EQUAL) {
                                        posTemp.SetY(posTemp.Y() + ((timer - static_cast<float>(k)*tDelta) / tDelta)*this->secStructVertices[(int)sTemp][j].Y());
                                    }

                                    glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() + posTemp.Y()));
                                }
                            }
                        }          
                    }
                    glEnd();
                }
                else if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_HORIZ) {
                        
                    for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
                        
                        glBegin(GL_TRIANGLE_STRIP);
                        
                        sTemp = sortedStructBT[j];
                        uTemp = this->secUncertainty[i][(int)sTemp];
                        
                        if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED)
                            glColor3fv(ud->GetSecStructColor(sTemp).PeekComponents());
                        
                        // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                        if ((sTemp == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i+1][0] != sTemp)) {
                            sTemp = UncertaintyDataCall::secStructure::B_BRIDGE;
                        }
                        for (unsigned int k = 0; k < this->secStructVertices[(int)sTemp].Count(); k++) {
                            posTemp = this->secStructVertices[(int)sTemp][k];
                            glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() - uTemp*0.5f - posTemp.Y()*uTemp));
                        }
                        posOffset.SetY(posOffset.Y() - uTemp);
                        
                        glEnd();
                    }
                }
                else if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_VERTI) {

                    posOffset.SetY(posOffset.Y() - 0.5f);
                    
                    for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
                        
                        glBegin(GL_TRIANGLE_STRIP);
                        
                        sTemp = sortedStructLR[j];
                        uTemp = this->secUncertainty[i][(int)sTemp];
                        
                        if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED)
                            glColor3fv(ud->GetSecStructColor(sTemp).PeekComponents());
                        
                        // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                        if ((sTemp == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i+1][0] != sTemp)) {
                            sTemp = UncertaintyDataCall::secStructure::B_BRIDGE;
                        }
                        for (unsigned int k = 0; k < this->secStructVertices[(int)sTemp].Count(); k++) {
                            
                            posTemp = this->secStructVertices[(int)sTemp][k];
                            glVertex2f(posOffset.X() + posTemp.X()*uTemp, -(posOffset.Y() + posTemp.Y()));
                        }
                        posOffset.SetX(posOffset.X() + uTemp);
                        
                        glEnd();
                    }
                }
                else if(this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_MORPH) {
                    
                    posOffset.SetY(posOffset.Y() - 0.5f);

                    glBegin(GL_TRIANGLE_STRIP);
                    
                    for (unsigned int j = 0; j < this->secStructVertices[(int)sMax].Count(); j++) {
                        
                        if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                            foundColor = false;
                            fTtemp = (static_cast<float>(j) / static_cast<float>(this->secStructVertices[(int)sMax].Count()));
                            fDelta = 0.0f;
                            for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
                                sTemp = sortedStructLR[k];
                                uTemp = this->secUncertainty[i][(int)sTemp];

                                if ((uTemp > 0.0f) && (fDelta <= fTtemp) && (fTtemp <= (fDelta + uTemp))) {
                                    colTemp = (1.0f - ((fTtemp - fDelta) / uTemp))*ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sTemp));
                                    for (unsigned int m = k+1; m < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); m++) { // search for next structure with uncertainty > 0
                                        if (this->secUncertainty[i][(int)sortedStructLR[m]] > 0.0f) {
                                            colTemp = colTemp + ((fTtemp - fDelta) / uTemp)*ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sortedStructLR[m]));
                                            foundColor = true;
                                            break;
                                        }
                                    }
                                    if (!foundColor) {
                                        colTemp = colTemp + ((fTtemp - fDelta) / uTemp)*ud->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(sTemp));
                                    }
                                }
                                fDelta += uTemp;
                            }
                            glColor3fv(colTemp.PeekComponents()); 
                        }
                    
                        posTemp.SetX(this->secStructVertices[sMax][j].X());
                        posTemp.SetY(0.0f);

                        for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
                            sTemp = sortedStructLR[k];
                            uTemp = this->secUncertainty[i][(int)sTemp];

                            // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
                            if ((sTemp == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (this->sortedUncertainty[i + 1][0] != sTemp)) {
                                sTemp = UncertaintyDataCall::secStructure::B_BRIDGE;
                            }
                            posTemp.SetY(posTemp.Y() + uTemp*this->secStructVertices[(int)sTemp][j].Y());
                        }

                        glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() + posTemp.Y()));
                    }
                    glEnd();                     
                }
            }
        }
    }
    
    glEnable(GL_DEPTH_TEST);                   // reset
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // reset
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
void UncertaintySequenceRenderer::drawSecStructTextureTiles(UncertaintyDataCall::secStructure secStructPre, UncertaintyDataCall::secStructure secStructi, 
                                                            UncertaintyDataCall::secStructure secStructSuc, UncertaintyDataCall::addFlags f, float x, float y, 
                                                            float defColor[4], UncertaintyDataCall *ud) {    
    if(ud == NULL) return;

    const float eps = 0.0f;
        
    if(f == UncertaintyDataCall::addFlags::MISSING) {
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
                glColor4fv(defColor);
                this->markerTextures[0]->Bind();        
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
    
    for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
        this->secStructAssignment.Add(vislib::Array<UncertaintyDataCall::secStructure>());
        this->secStructAssignment.Last().AssertCapacity(this->aminoAcidCount);
    }

    this->residueFlag.Clear();
    this->residueFlag.AssertCapacity(this->aminoAcidCount);
    
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
        // store residue flag
        this->residueFlag.Add(udc->GetResidueFlag(aa));
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

