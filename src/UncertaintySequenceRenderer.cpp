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
//    
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
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/utility/ResourceWrapper.h"

#include "vislib/math/Rectangle.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"

#include "UncertaintyColor.h" 

#include <iostream> // DEBUG


#define SEQ_PI        3.1415926535897932384626433832795
#define SEQ_FLOAT_EPS 0.00001

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
            uncertaintyDataSlot(    "uncertaintyDataSlot", "Connects the sequence diagram rendering with uncertainty data storage."),
            bindingSiteCallerSlot(  "getBindingSites", "Connects the sequence diagram rendering with binding site storage."),
            resSelectionCallerSlot( "getResSelection", "Connects the sequence diagram rendering with residue selection storage."),
            toggleStrideParam(                  "01 Stride", "Show/hide STRIDE secondary structure row."),
			toggleStrideThreshParam(            "02 Stride Threshold", "Show/hide STRIDE threshold(s)."),
            toggleDsspParam(                    "03 Dssp", "Show/hide DSSP secondary structure row."),
			toggleDsspThreshParam(              "04 Dssp Threshold", "Show/hide DSSP threshold(s)."),
            togglePdbParam(                     "05 PDB", "Show/hide PDB secondary structure row."),
            
            ////////////////////////////////////////////////
            // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
            ////////////////////////////////////////////////     
                   
			toggleUncertaintyParam(             "06 Uncertainty", "Show/hide uncertainty row."),
            toggleUncertainStructParam(         "07 Uncertain Structure", "Show/hide row with detailed uncertainty of secondary structure."),
            certainBlockChartColorParam(        "08 Certain: Block chart color", "Choose color of block chart for certain structure."),
            certainStructColorParam(            "09 Certain: Structure color", "Choose color of structure for certain structure."),
            uncertainBlockChartColorParam(      "10 Uncertain: Block chart color", "Choose color of block chart for uncertain structure."),
            uncertainBlockChartOrientationParam("11 Uncertain: Block orientation", "Choose orientation of block chart for uncertain structure."),
            uncertainStructColorParam(          "12 Uncertain: Structure color", "Choose color of structure for uncertain structure."),
            uncertainStructGeometryParam(       "13 Uncertain: Structure geometry", "Choose geometry of structure for uncertain structure."),
            geometryTessParam(                  "14 Geometry tesselation", "The geometry tesselation level."),
            uncertainColorInterpolParam(        "15 Uncertain: Color interpolation", "Choose method for color interpolation (only available for some uncertainty visualizations)."),
            uncertainGardientIntervalParam(     "16 Uncertain: Gradient interval", "Choose interval width for color interpolation gradient."),
            toggleUncSeparatorParam(            "17 Separator Line", "Show/Hide separator line for amino-acids in uncertainty visualization."),
            toggleLegendParam(                  "18 Legend Drawing", "Show/hide row legend/key on the left."),
            toggleTooltipParam(                 "19 ToolTip", "Show/hide tooltip information."),
            resCountPerRowParam(                "20 Residues Per Row", "The number of residues per row."),
            clearResSelectionParam(             "21 Clear Residue Selection", "Clears the current selection (everything will be deselected)."),
            colorTableFileParam(                "22 Color Table Filename", "The filename of the color table."),
			reloadShaderParam(                  "23 Reload shaders", "Reload the shaders."),
            dataPrepared(false), aminoAcidCount(0), bindingSiteCount(0), resCols(0), resRows(0), rowHeight(2.0f), currentGeometryTess(25),
            markerTextures(0), resSelectionCall(nullptr), rightMouseDown(false), secStructRows(0), pdbID(""), pdbLegend("")
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
    this->resCountPerRowParam.SetParameter(new param::IntParam(50, 1));
    this->MakeSlotAvailable( &this->resCountPerRowParam);
    
    this->geometryTessParam << new core::param::IntParam(this->currentGeometryTess, 4);
    this->MakeSlotAvailable(&this->geometryTessParam);

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

    // param slot for STRIDE toggling
    this->toggleStrideParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleStrideParam);

    // param slot for DSSP toggling
    this->toggleDsspParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleDsspParam);
    
    // param slot for PDB toggling
    this->togglePdbParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->togglePdbParam);

	// param slot for STRIDE threshold toggling
    this->toggleStrideThreshParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleStrideThreshParam);
	
    // param slot for DSSP threshold toggling
    this->toggleDsspThreshParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleDsspThreshParam);
	
    ////////////////////////////////////////////////
    // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
    ////////////////////////////////////////////////    

    // param slot for toggling separator line
    this->toggleUncSeparatorParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleUncSeparatorParam);
            
    // param slot for uncertainty uncertainty toggling
    this->toggleUncertaintyParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleUncertaintyParam);
    
    // param slot for uncertain strucutre toggling
    this->toggleUncertainStructParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleUncertainStructParam);        
    
    // param slot for tooltip toggling
    this->toggleTooltipParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleTooltipParam);

    // parameter for viualization options
    this->currentCertainBlockChartColor         = CERTAIN_BC_NONE;
    this->currentCertainStructColor             = CERTAIN_STRUCT_GRAY;
    this->currentUncertainBlockChartColor       = UNCERTAIN_BC_COLORED;
    this->currentUncertainBlockChartOrientation = UNCERTAIN_BC_VERTI;
    this->currentUncertainStructColor           = UNCERTAIN_STRUCT_GRAY;
    this->currentUncertainStructGeometry        = UNCERTAIN_STRUCT_DYN_EQUAL;
    this->currentUncertainColorInterpol         = UNCERTAIN_COLOR_HSL_HP;
        
    param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(this->currentCertainBlockChartColor));
    tmpEnum->SetTypePair(CERTAIN_BC_NONE,    "no coloring (background)");
    tmpEnum->SetTypePair(CERTAIN_BC_COLORED, "colored");
    this->certainBlockChartColorParam << tmpEnum;
    this->MakeSlotAvailable(&this->certainBlockChartColorParam);       

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentCertainStructColor));
    tmpEnum->SetTypePair(CERTAIN_STRUCT_NONE,    "no coloring (background)");
    tmpEnum->SetTypePair(CERTAIN_STRUCT_COLORED, "colored");
    tmpEnum->SetTypePair(CERTAIN_STRUCT_GRAY,    "gray");
    this->certainStructColorParam << tmpEnum;
    this->MakeSlotAvailable(&this->certainStructColorParam); 

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainBlockChartColor));
    tmpEnum->SetTypePair(UNCERTAIN_BC_NONE,    "no coloring (background)");
    tmpEnum->SetTypePair(UNCERTAIN_BC_COLORED, "colored");
    this->uncertainBlockChartColorParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainBlockChartColorParam); 
    
    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainBlockChartOrientation));
    tmpEnum->SetTypePair(UNCERTAIN_BC_VERTI, "vertical");
    tmpEnum->SetTypePair(UNCERTAIN_BC_HORIZ, "horizontal");
    this->uncertainBlockChartOrientationParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainBlockChartOrientationParam);     

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainStructColor));
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_NONE,    "no coloring (background)");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_COLORED, "colored");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_GRAY,    "gray");
    this->uncertainStructColorParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainStructColorParam); 
    
    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainStructGeometry));
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_STAT_VERTI, "Static - vertical");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_STAT_HORIZ, "Static - horizontal");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_STAT_MORPH, "Static - morphed");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_DYN_TIME,   "Dynamic - time");
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_DYN_SPACE,  "Dynamic - space");    
    tmpEnum->SetTypePair(UNCERTAIN_STRUCT_DYN_EQUAL,  "Dynamic - equal");     
	tmpEnum->SetTypePair(UNCERTAIN_GLYPH,             "Glyph");
	tmpEnum->SetTypePair(UNCERTAIN_GLYPH_WITH_AXES,   "Glyph with axes");
    this->uncertainStructGeometryParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainStructGeometryParam); 

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncertainColorInterpol));
    tmpEnum->SetTypePair(UNCERTAIN_COLOR_RGB,    "RGB - linear");
    tmpEnum->SetTypePair(UNCERTAIN_COLOR_HSL,    "HSL - linear");
    tmpEnum->SetTypePair(UNCERTAIN_COLOR_HSL_HP, "HSL - hue preserving");
    this->uncertainColorInterpolParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncertainColorInterpolParam);

    // param slot for number of residues per row
    this->currentUncertainGardientInterval = 0.2f;
    this->uncertainGardientIntervalParam.SetParameter( new param::FloatParam(this->currentUncertainGardientInterval, 0.0f, 1.0f));
    this->MakeSlotAvailable( &this->uncertainGardientIntervalParam);
    
	this->reloadShaderParam << new core::param::ButtonParam(vislib::sys::KeyCode::KEY_F5);
	this->MakeSlotAvailable(&this->reloadShaderParam);

    // setting initial row height, as if all secondary structure rows would be shown
    this->secStructRows = 5;
    this->rowHeight = 2.0f + static_cast<float>(this->secStructRows);   

	this->strideThresholdCount = 7;
	this->dsspThresholdCount   = 2; // only showing HBondAc0 and HBondDo0
		
    // init animation timer
    this->animTimer = std::clock();
    
    this->showSeparatorLine = true;
}


/*
 * UncertaintySequenceRenderer::~UncertaintySequenceRenderer (DTOR)
 */
UncertaintySequenceRenderer::~UncertaintySequenceRenderer( void ) {
    this->Release(); // DON'T change !
}


/*
* UncertaintySequenceRenderer::release
*/
void UncertaintySequenceRenderer::release() {

    this->shader.Release();
    /** intentionally left empty ... */
}


/*
 * UncertaintySequenceRenderer::create
 */
bool UncertaintySequenceRenderer::create() {

	// load shader
	if (!this->LoadShader()) {
		return false;
	}

    // OLD: load textures
    /*
    this->LoadTexture("secStruct-white.png");       // markerTextures[0]
    this->LoadTexture("secStruct-coil.png");        // markerTextures[1]
    this->LoadTexture("secStruct-sheet.png");       // markerTextures[2]
    this->LoadTexture("secStruct-arrow.png");       // markerTextures[3]
    this->LoadTexture("secStruct-helix-left.png");  // markerTextures[4]
    this->LoadTexture("secStruct-helix2.png");      // markerTextures[5]
    this->LoadTexture("secStruct-helix-right.png"); // markerTextures[6]
    */
    
    // calculate geometry
    this->calculateGeometryVertices(this->currentGeometryTess);

	// compute glyph axes
	unsigned int structCount = 3; // static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); // or: 3-4 for reduced strucutre type set ...
	vislib::math::Vector<float, 2> initVec, tmpVec;
	float angle;
	this->glyphAxis.Clear();
	this->glyphAxis.AssertCapacity(structCount);
	initVec = vislib::math::Vector<float, 2>(0.0f, -0.5f);
	for (unsigned int i = 0; i < structCount; i++) {
		angle = 2.0f*(float)SEQ_PI / static_cast<float>(structCount)* static_cast<float>(i);
		if (i == 0) {
			tmpVec = initVec;
		}
		else {
			tmpVec.SetX(initVec.X()*std::cosf(angle) - initVec.Y()*std::sinf(angle));
			tmpVec.SetY(initVec.X()*std::sinf(angle) + initVec.Y()*std::cosf(angle));
		}
		this->glyphAxis.Add(tmpVec);
	}

    return true;
}


/*
* UncertaintySequenceRenderer::calculateGeometryVertices
*/
void UncertaintySequenceRenderer::calculateGeometryVertices(int samples) {

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

        // default height
        float height = 0.3f;  // in range ]0.0-1.0[

        this->secStructVertices.Add(vislib::Array<vislib::math::Vector<float, 2>>());
        pos.SetX(0.0f);
        pos.SetY(-height / 2.0f);

        if ((s == UncertaintyDataCall::secStructure::E_EXT_STRAND) ||
            (s == UncertaintyDataCall::secStructure::B_BRIDGE)) {
            heightStrand = 0.6f; 
            pos.SetY(-heightStrand / 2.0f);
        }

        this->secStructVertices.Last().Add(pos);

        for (int j = 0; j < (samples + 1); j++) { // loop over sample count

            flip = ((j % 2 == 0) ? (1.0f) : (-1.0f));
            offset = flip*(height / 2.0f);
            dS = static_cast<float>(j) / static_cast<float>(samples - 1);

            if (j == samples) // for last vertex
                dS = 1.0f;

            switch (s) {
            case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) :
            case (UncertaintyDataCall::secStructure::G_310_HELIX) :
            case (UncertaintyDataCall::secStructure::I_PI_HELIX) :
                                                                 offset = flip*(height / 2.0f) + sin(dS*2.0f*(float)SEQ_PI)*0.25f;
                break;
            case (UncertaintyDataCall::secStructure::B_BRIDGE) :
                offset = ((dS > 1.0f / 7.0f) ? (flip*(0.5f - (height / 2.0f))*(7.0f - dS*7.0f) / 6.0f + flip*(height / 2.0f)) : (flip*(heightStrand / 2.0f)));
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
}


/*
* UncertaintySequenceRenderer::LoadShader
*/
bool UncertaintySequenceRenderer::LoadShader(void) {

	if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
		return false;

	instance()->ShaderSourceFactory().LoadBTF("uncertaintysequence", true);

	this->vertex = new ShaderSource();
	this->fragment = new ShaderSource();

	if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintysequence::vertex", *this->vertex)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintysequence::fragment", *this->fragment)) {
		return false;
	}

	try {
		// compile the shader
		if (!this->shader.Compile(this->vertex->Code(), this->vertex->Count(), this->fragment->Code(), this->fragment->Count())) {
			throw vislib::Exception("Could not compile shader. ", __FILE__, __LINE__);
		}
		// link the shader
		if (!this->shader.Link()) {
			throw vislib::Exception("Could not link shader", __FILE__, __LINE__);
		}
	}
	catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile shader (@%s): %s\n",
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
			ce.FailedAction()), ce.GetMsgA());
		return false;
	}
	catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile shader: %s\n", e.GetMsgA());
		return false;
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile shader: Unknown exception\n");
		return false;
	}

	return true;
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
    if(this->toggleStrideParam.IsDirty()) {
        this->dataPrepared = false;
    }    
    if(this->toggleDsspParam.IsDirty()) {
        this->dataPrepared = false;
    }
    if(this->toggleUncertaintyParam.IsDirty()) {
        this->dataPrepared = false;
    }
    if(this->togglePdbParam.IsDirty()) {
        this->dataPrepared = false;
    }    
    if(this->toggleStrideThreshParam.IsDirty()) {
        this->dataPrepared = false;
    }   
	    if(this->toggleDsspThreshParam.IsDirty()) {
        this->dataPrepared = false;
    }   
	
    ////////////////////////////////////////////////
    // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
    ////////////////////////////////////////////////
            
    if(this->toggleUncertainStructParam.IsDirty()) {
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
        if (this->toggleStrideParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
        if (this->toggleDsspParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
        if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
        if (this->togglePdbParam.Param<param::BoolParam>()->Value())
            this->secStructRows++;
        if (this->toggleStrideThreshParam.Param<param::BoolParam>()->Value())
            this->secStructRows += this->strideThresholdCount;
		if (this->toggleDsspThreshParam.Param<param::BoolParam>()->Value())
            this->secStructRows += this->dsspThresholdCount;
                        
        ////////////////////////////////////////////////
        // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
        ////////////////////////////////////////////////
                
        if (this->toggleUncertainStructParam.Param<param::BoolParam>()->Value())
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
            this->toggleStrideParam.ResetDirty();
            this->toggleDsspParam.ResetDirty();
            this->toggleUncertaintyParam.ResetDirty();
            this->toggleUncertainStructParam.ResetDirty();
            this->togglePdbParam.ResetDirty();
            this->toggleStrideThreshParam.ResetDirty();
			this->toggleDsspThreshParam.ResetDirty();	
			
            ////////////////////////////////////////////////
            // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
            ////////////////////////////////////////////////            

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
  
    // check sparator line parameter
    if (this->toggleUncSeparatorParam.IsDirty()) {
        this->toggleUncSeparatorParam.ResetDirty();
        this->showSeparatorLine = this->toggleUncSeparatorParam.Param<param::BoolParam>()->Value();
    }
          
	// reload shader
	if (this->reloadShaderParam.IsDirty()) {
		this->reloadShaderParam.ResetDirty();
		if (!this->LoadShader()) return false;
	}

    // recalculate geometry vertices
    if (this->geometryTessParam.IsDirty()) {
        this->geometryTessParam.ResetDirty();
        this->currentGeometryTess = static_cast<int>(this->geometryTessParam.Param<param::IntParam>()->Value());
        this->calculateGeometryVertices(this->currentGeometryTess);
    }

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
    if (this->uncertainColorInterpolParam.IsDirty()) {
        this->uncertainColorInterpolParam.ResetDirty();
        this->currentUncertainColorInterpol = static_cast<uncertainColorInterpol>(this->uncertainColorInterpolParam.Param<param::EnumParam>()->Value());
    }
    
    if (this->uncertainGardientIntervalParam.IsDirty()) {
        this->uncertainGardientIntervalParam.ResetDirty();
        this->currentUncertainGardientInterval = static_cast<float>(this->uncertainGardientIntervalParam.Param<param::FloatParam>()->Value());
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
    glLineWidth(2.0f);
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
        UncertaintyDataCall::secStructure tmpStruct;
        unsigned int tmpMethod;
        vislib::StringA tmpStr;
        vislib::StringA tmpStr2;
        vislib::StringA tmpStr3;
        float yPos;
        float xPos;
        float wordlength;
        float rowPos;            
        float fontSize;
        vislib::math::Vector<float, 2> pos;
        float perCentRow;
        float start;
        float end;

        yPos = 0.0f; // !  
        
        //draw geometry tiles for secondary structure    
        //STRIDE
        if(this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
                tmpMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::STRIDE);
                this->DrawSecStructGeometryTiles(this->sortedSecStructAssignment[tmpMethod][i][0], ((i < (this->aminoAcidCount - 1)) ? (this->sortedSecStructAssignment[tmpMethod][i + 1][0]) : (this->sortedSecStructAssignment[tmpMethod][i][0])),
                                                this->residueFlag[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);    
			   // draw as texture - deprecated ...
			   // glEnable(GL_TEXTURE_2D); // call only once for all assignment method rows
			   // glEnable(GL_BLEND);
			   // this->DrawSecStructTextureTiles(((i > 0) ? (this->secStructAssignment[tmpMethod][i - 1]) : (this->secStructAssignment[tmpMethod][i])), this->secStructAssignment[tmpMethod][i],
			   //	                              ((i < (this->aminoAcidCount - 1)) ? (this->secStructAssignment[tmpMethod][i + 1]) : (this->secStructAssignment[tmpMethod][i])),
			   //	                              this->residueFlag[i], this->vertices[i * 2], (this->vertices[i * 2 + 1] + yPos), bgColor);
			   // glDisable(GL_BLEND);
			   // glDisable(GL_TEXTURE_2D);
            } 
            yPos += 1.0f;
        }
		// STRIDE thresholds
        if(this->toggleStrideThreshParam.Param<param::BoolParam>()->Value()) {
            float min;
            float max;
            float threshold;

            for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
                tmpStruct = this->sortedSecStructAssignment[static_cast<int>(UncertaintyDataCall::assMethod::STRIDE)][i][0];
				for (unsigned int j = 0; j < this->strideThresholdCount; j++) {
					if (this->strideStructThreshold[i][j] != INFINITY) {
						switch (j) {
						case(0) : threshold = STRIDE_THRESHOLDH1;
							min = -500.0f;
							max = 500.0f;
							break;
						case(1) : threshold = STRIDE_THRESHOLDH3;
							min = -1.0f;
							max = 1.0f;
							break;
						case(2) : threshold = STRIDE_THRESHOLDH4;
							min = -1.0f;
							max = 1.0f;
							break;
						case(3) : threshold = STRIDE_THRESHOLDE2;
							min = -500.0f;
							max = 500.0f;
							break;
						case(4) : threshold = STRIDE_THRESHOLDE2;
							min = -500.0f;
							max = 500.0f;
							break;
						case(5) : threshold = STRIDE_THRESHOLDE1;
							min = -500.0f;
							max = 500.0f;
							break;
						case(6) : threshold = STRIDE_THRESHOLDE1;
							min = -500.0f;
							max = 500.0f;
							break;                            
						default: break;
						}
						this->DrawThresholdEnergyValueTiles(tmpStruct, this->residueFlag[i], this->vertices[i * 2], (this->vertices[i * 2 + 1] + yPos + (float)j),
							this->strideStructThreshold[i][j], min, max, threshold);
					}
			    }
            }
			yPos += this->strideThresholdCount;
        }		
        // DSSP
        if(this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
                tmpMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::DSSP);
                this->DrawSecStructGeometryTiles(this->sortedSecStructAssignment[tmpMethod][i][0], ((i < (this->aminoAcidCount - 1)) ? (this->sortedSecStructAssignment[tmpMethod][i + 1][0]) : (this->sortedSecStructAssignment[tmpMethod][i][0])),
                                                this->residueFlag[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);                
            }
            yPos += 1.0f;
        }  
        // DSSP thresholds
        if(this->toggleDsspThreshParam.Param<param::BoolParam>()->Value()) {
            float min       = -5.0f;
            float max       =  5.0f;
            float threshold = DSSP_HBENERGY;
            
            for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
                tmpStruct = this->sortedSecStructAssignment[static_cast<unsigned int>(UncertaintyDataCall::assMethod::DSSP)][i][0];
                for (unsigned int j = 0; j < this->dsspThresholdCount; j++) {
					if (this->dsspStructEnergy[i][j] != INFINITY) {
						this->DrawThresholdEnergyValueTiles(tmpStruct, this->residueFlag[i], this->vertices[i * 2], (this->vertices[i * 2 + 1] + yPos + (float)j),
															this->dsspStructEnergy[i][j], min, max, threshold);
					}
			    }
            }
			yPos += this->dsspThresholdCount;
        }  		
        // PDB 
        if(this->togglePdbParam.Param<param::BoolParam>()->Value()) {
            for (unsigned int i = 0; i < this->aminoAcidCount; i++ ) {
                tmpMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::PDB);
                this->DrawSecStructGeometryTiles(this->sortedSecStructAssignment[tmpMethod][i][0], ((i < (this->aminoAcidCount - 1)) ? (this->sortedSecStructAssignment[tmpMethod][i + 1][0]) : (this->sortedSecStructAssignment[tmpMethod][i][0])),
                                                this->residueFlag[i], this->vertices[i*2], (this->vertices[i*2+1]+yPos), bgColor);               
            }
            yPos += 1.0f;
        }
        
        ////////////////////////////////////////////////
        // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
        ////////////////////////////////////////////////
                                   
            
        // draw uncertainty 
        if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
			float col;
            glBegin(GL_QUADS);
            for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
				col = 1.0f - this->uncertainty[i];
				glColor3f(col, col, col);
                if (this->residueFlag[i] != UncertaintyDataCall::addFlags::MISSING) {
                    // assuming values are in range 0.0-1.0
					glVertex2f(this->vertices[2 * i], -(this->vertices[2 * i + 1] + yPos + 1.0f - this->uncertainty[i]));
                    glVertex2f(this->vertices[2 * i],        -(this->vertices[2 * i + 1] + yPos + 1.0f));
                    glVertex2f(this->vertices[2 * i] + 1.0f, -(this->vertices[2 * i + 1] + yPos + 1.0f));
					glVertex2f(this->vertices[2 * i] + 1.0f, -(this->vertices[2 * i + 1] + yPos + 1.0f - this->uncertainty[i]));
                }
            }
            glEnd();
            yPos += 1.0f;
        }
        
                    
        // draw structure uncertainty visualization
        if(this->toggleUncertainStructParam.Param<param::BoolParam>()->Value()) {
            
			this->RenderUncertainty(yPos, fgColor, bgColor);

			// draw frames for uncertain amino-acid structure
			if (this->showSeparatorLine) {
               
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

				glColor3fv(fgColor);
				glEnable(GL_VERTEX_ARRAY);
				glVertexPointer(2, GL_FLOAT, 0, this->aminoacidSeparatorVertices.PeekElements());
				glDrawArrays(GL_QUADS, 0, (GLsizei)this->aminoacidSeparatorVertices.Count() / 2);
				glDisable(GL_VERTEX_ARRAY);

				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
            yPos += 1.0f;            
        }


        // draw amino-acid tiles
        glColor3fv(fgColor);
        fontSize = 1.5f;
        if( theFont.Initialise() ) {
            // pdb id as info on top
            tmpStr = this->pdbID;
            tmpStr.Prepend("PDB-ID: ");
            wordlength = theFont.LineWidth(fontSize, tmpStr);
            theFont.DrawString(0.0f, 0.5f, wordlength+1.0f, 1.0f, 
                               fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
			yPos += 1.0f;
            // draw amino-acid and chain name with pdb index tiles
            for (unsigned int i = 0; i < this->aminoAcidCount; i++) {
                // draw the one-letter amino acid code
                switch(this->residueFlag[i]) {
                    case(UncertaintyDataCall::addFlags::MISSING):   glColor3f(0.7f, 0.7f, 0.7f); break;
                    case(UncertaintyDataCall::addFlags::HETEROGEN): glColor3f(1.0f, 0.6f, 0.6f); break;
                    // case(UncertaintyDataCall::addFlags::NOTHING):   glColor3fv(fgColor); break;
                    default: break;
                }
                tmpStr.Format("%c", this->aminoAcidName[i]);
                theFont.DrawString(this->vertices[i*2], -(this->vertices[i*2+1]+yPos), 
                                   1.0f, 1.0f, 1.0f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
                glColor3fv(fgColor); // reset color
                // draw the chain name and amino acid index                  
                tmpStr.Format("%c %s", this->chainID[i], this->aminoAcidIndex[i]);
                theFont.DrawString(this->vertices[i*2], -(this->vertices[i*2+1]+yPos+0.5f), 
                                   1.0f, 0.5f, 0.35f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
            }
        }
        // INFO: DrawString(float x, float y, float w, float h, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) 
            
            
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
            glVertex2f( this->chainVertices[2*i]       , -this->chainVertices[2*i+1] - 0.4f);
            glVertex2f( this->chainVertices[2*i] + 1.0f, -this->chainVertices[2*i+1] - 0.4f);
            glVertex2f( this->chainVertices[2*i] + 1.0f, -this->chainVertices[2*i+1] - 0.2f);
        }
        glEnd();
        // draw chain separators
        glColor3fv(fgColor);
        glEnable( GL_VERTEX_ARRAY);
        glVertexPointer( 2, GL_FLOAT, 0, this->chainSeparatorVertices.PeekElements());
        glDrawArrays( GL_LINES, 0, (GLsizei)this->chainSeparatorVertices.Count() / 2 );
        glDisable( GL_VERTEX_ARRAY);
		
        // draw legend/key
        if(this->toggleLegendParam.Param<param::BoolParam>()->Value()) {
            glColor3fv(fgColor);
            fontSize = 1.0f;
            if( theFont.Initialise() ) {
                 
                // RIGHT SIDE OF DIAGRAM //

                // draw secondary strucure type legend
				xPos = 6.0f;
                tmpStr = "Secondary Structure Types: ";
                wordlength = theFont.LineWidth( fontSize, tmpStr);
                theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -1.0f, wordlength, 1.0f,
                                   fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); i++) {
					glColor3fv(fgColor);
					glBegin(GL_LINES);
						glVertex2f((static_cast<float>(this->resCols) + 1.0f), -(static_cast<float>(i)+1.5f));
						glVertex2f((static_cast<float>(this->resCols) + 2.0f + xPos), -(static_cast<float>(i)+1.5f));
					glEnd();

					tmpStr = this->secStructDescription[i];
                    wordlength = theFont.LineWidth( fontSize, tmpStr);
                    theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -(static_cast<float>(i) + 2.5f), wordlength, 1.0f,
                                       fontSize * 0.75f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);

					// draw type legend geometry
					this->DrawSecStructGeometryTiles(static_cast<UncertaintyDataCall::secStructure>(i), UncertaintyDataCall::secStructure::NOTDEFINED,
						                             UncertaintyDataCall::addFlags::NOTHING, (static_cast<float>(this->resCols) + 1.0f + xPos), (static_cast<float>(i)+1.5f), bgColor);
                }
				// draw sparator lines
				for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE)+1; i++) {  // end index: ...+1!
					glColor3fv(fgColor);
					glBegin(GL_LINES);
						glVertex2f((static_cast<float>(this->resCols) + 1.0f), -(static_cast<float>(i)+1.5f));
						glVertex2f((static_cast<float>(this->resCols) + 2.0f + xPos), -(static_cast<float>(i)+1.5f));
					glEnd();
				}
                
                // draw binding site legend
                if( !this->bindingSiteNames.IsEmpty() ) {
                    
                    yPos = static_cast<float>(UncertaintyDataCall::secStructure::NOE) + 3.5f;
					glColor3fv(fgColor);
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
                for (unsigned int i = 0; i < 1; i++) { // replace 'i < 1' with 'i < this->resRows' to show row legend for every row not just the first one

                    if (this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "STRIDE";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f, 
                                           fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
					if (this->toggleStrideThreshParam.Param<param::BoolParam>()->Value()) {
						for (unsigned int j = 0; j < this->strideThresholdCount; j++) {
                            switch (j) {
                                case (0) : tmpStr = "Stride Threshold T1 a-Helix"; break;
                                case (1) : tmpStr = "Stride Threshold T3 a-Helix"; break;
                                case (2) : tmpStr = "Stride Threshold T4 a-Helix"; break;
                                case (3) : tmpStr = "Stride Threshold T2 Sheet parallel"; break;
                                case (4) : tmpStr = "Stride Threshold T2 Sheet parallel"; break;
                                case (5) : tmpStr = "Stride Threshold T1 Sheet anti-parallel"; break;
                                case (6) : tmpStr = "Stride Threshold T2 Sheet anti-parallel"; break;                                
                                default:  tmpStr = "Stride UNKNOWN Threshold"; break;
                            }
							wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
							theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos + float(j)), wordlength, 1.0f, 
											   fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
						}
                        yPos += (float)this->strideThresholdCount;
                    }
                    if (this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "DSSP";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f, 
                                           fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    if (this->toggleDsspThreshParam.Param<param::BoolParam>()->Value()) {
						for (unsigned int j = 0; j < this->dsspThresholdCount; j++) {
                            switch (j) {
                                case (0) : tmpStr = "Dssp H-Bond Energy as Acceptor"; break;
                                case (1) : tmpStr = "Dssp H-Bond Energy as Donor"; break;
                                default:  tmpStr = "Dssp UNKNOWN Threshold"; break;
                            }
							wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
							theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos + float(j)), wordlength, 1.0f, 
											   fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
						}
                        yPos += (float)this->dsspThresholdCount;
                    }					
                    if (this->togglePdbParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = this->pdbLegend;
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f, 
                                           fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                    
                    ////////////////////////////////////////////////
                    // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
                    ////////////////////////////////////////////////
                          
                    if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Uncertainty";
                        wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                        theFont.DrawString(-wordlength, -(static_cast<float>(i)*this->rowHeight + yPos), wordlength, 1.0f,
                            fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                        yPos += 1.0f;
                    }
                                                           
                    if (this->toggleUncertainStructParam.Param<param::BoolParam>()->Value()) {
                        tmpStr = "Uncertain Structure";
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
        
		
        // draw overlays for selected amino acids
        glDisable(GL_DEPTH_TEST);       
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
                    glVertex2f( pos.X()       , -this->rowHeight * (pos.Y()));
                    glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y()));
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
                    glVertex2f( pos.X()       , -this->rowHeight * (pos.Y()));
                }
                // bottom
                glVertex2f( pos.X()       , -this->rowHeight * (pos.Y()));
                glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y()));
                // right (only draw if right neighbor ist not selected)
                if( i == (this->selection.Count() - 1) || !this->selection[i+1] ) {
                    glVertex2f( pos.X() + 1.0f, -this->rowHeight * (pos.Y()));
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
                    if (this->toggleStrideParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Stride:";
                            tmpStr2 = this->secStructDescription[(int)this->sortedSecStructAssignment[UncertaintyDataCall::assMethod::STRIDE][mousePosResIdx][0]];
                            this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->toggleStrideThreshParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
						    for (unsigned int j = 0; j < this->strideThresholdCount; j++) {
								if (this->strideStructThreshold[mousePosResIdx][j] != INFINITY) {
									switch (j) {
										case (0) : tmpStr = "T1-Helix:";
											tmpStr2.Format("%.3f (< T=%.2f)", this->strideStructThreshold[mousePosResIdx][j], STRIDE_THRESHOLDH1);
											break;
										case (1) : tmpStr = "T3-Helix:";
											tmpStr2.Format("%.3f (> T=%.2f)", this->strideStructThreshold[mousePosResIdx][j], STRIDE_THRESHOLDH3);
											break;
										case (2) : tmpStr = "T4-Helix:";
											tmpStr2.Format("%.3f (> T=%.2f)", this->strideStructThreshold[mousePosResIdx][j], STRIDE_THRESHOLDH4);
											break;
										case (3) : tmpStr = "T1-Sheet-parallel:";
											tmpStr2.Format("%.3f (< T=%.2f)", this->strideStructThreshold[mousePosResIdx][j], STRIDE_THRESHOLDE2);
											break;
										case (4) : tmpStr = "T2-Sheet-parallel:";
											tmpStr2.Format("%.3f (< T=%.2f)", this->strideStructThreshold[mousePosResIdx][j], STRIDE_THRESHOLDE2);
											break;
										case (5) : tmpStr = "T1-Sheet-anti-parallel:";
											tmpStr2.Format("%.3f (< T=%.2f)", this->strideStructThreshold[mousePosResIdx][j], STRIDE_THRESHOLDE1);
											break;
										case (6) : tmpStr = "T2-Sheet-anti-parallel:";
											tmpStr2.Format("%.3f (< T=%.2f)", this->strideStructThreshold[mousePosResIdx][j], STRIDE_THRESHOLDE1);
											break;                                            
										default:  tmpStr = "UNKNOWN value"; break;
									}
									this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
								}
								start += perCentRow;
								end += perCentRow;
							}
						}
						else {
							start += (perCentRow * this->strideThresholdCount);
							end += (perCentRow * this->strideThresholdCount);
						}
                    }					
                    if (this->toggleDsspParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Dssp:";
                            tmpStr2 = this->secStructDescription[(int)this->sortedSecStructAssignment[UncertaintyDataCall::assMethod::DSSP][mousePosResIdx][0]];
                            this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;
                    }
                    if (this->toggleDsspThreshParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
						    for (unsigned int j = 0; j < this->dsspThresholdCount; j++) {
								if (this->dsspStructEnergy[mousePosResIdx][j] != INFINITY) {
									switch (j) {
										case (0) : tmpStr = "Hb-E Acc.: "; break;
										case (1) : tmpStr = "Hb-E Don.: "; break;
										default:  tmpStr = "Dssp UNKNOWN Threshold"; break;
									}
									tmpStr2.Format("%.3f | < T=%.1f[kcal/mol]", this->dsspStructEnergy[mousePosResIdx][j], DSSP_HBENERGY);
									this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
								}
								start += perCentRow;
								end += perCentRow;
                            }
						}
						else {
							start += (perCentRow * this->dsspThresholdCount);
							end += (perCentRow * this->dsspThresholdCount);
						}
                    }
                    if (this->togglePdbParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Pdb:";
                            tmpStr2 = this->secStructDescription[(int)this->sortedSecStructAssignment[UncertaintyDataCall::assMethod::PDB][mousePosResIdx][0]];
                            this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;
                    }
                    
                    ////////////////////////////////////////////////
                    // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
                    ////////////////////////////////////////////////
                    
                    if (this->toggleUncertaintyParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Uncertainty:";
							// tmpStr2.Format("%.0f %%", this->uncertainty[mousePosResIdx] * 100.0f);
							tmpStr2.Format("%.3f", this->uncertainty[mousePosResIdx]);
                            this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;
                    }             
                    if (this->toggleUncertainStructParam.Param<param::BoolParam>()->Value()) {
                        if (this->residueFlag[mousePosResIdx] != UncertaintyDataCall::addFlags::MISSING) {
                            tmpStr = "Structure:";
                            tmpStr2 = "";
							for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); i++) {
                                if (this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::UNCERTAINTY][mousePosResIdx]
                                                              [(int)this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::UNCERTAINTY][mousePosResIdx][i]] > 0.0f) {
                                    if (i > 0) {
                                        tmpStr2.Append("|");
                                   } 
                                    tmpStr3.Format(" %s: %.2f %% ", this->secStructDescription[(int)this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::UNCERTAINTY][mousePosResIdx][i]],
                                        this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::UNCERTAINTY][mousePosResIdx] 
                                                                  [(int)this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::UNCERTAINTY][mousePosResIdx][i]] * 100.0f);
                                    tmpStr2.Append(tmpStr3);
                                }
                            }
                            this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                        }
                        start += perCentRow;
                        end += perCentRow;                        
                    }
                    tmpStr = "Flag:";  
                    switch (this->residueFlag[mousePosResIdx]) {
                        case(UncertaintyDataCall::addFlags::MISSING):   
                            tmpStr2 = "MISSING"; 
                            this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
                            break;
                        case(UncertaintyDataCall::addFlags::HETEROGEN): 
                            tmpStr2 = "HETEROGEN"; 
                            this->RenderToolTip(start, end, tmpStr, tmpStr2, bgColor, fgColor);
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
* UncertaintySequenceRenderer::DrawThresholdEnergyValueTiles
*/
void UncertaintySequenceRenderer::DrawThresholdEnergyValueTiles(UncertaintyDataCall::secStructure str, UncertaintyDataCall::addFlags f, 
                                                                float x, float y, float value, float min, float max, float thresh) {

    float yDMax = 0.49f;

    // ignore as missing flagged amino-acids and NOTDEFINED secondary structure types
    if ((f != UncertaintyDataCall::addFlags::MISSING) && (str != UncertaintyDataCall::secStructure::NOTDEFINED)) {
        // draw middle line
		GLfloat tmpLw;
		glGetFloatv(GL_LINE_WIDTH, &tmpLw);
        glLineWidth(3.0f);
        glColor3f(0.3f, 0.3f, 0.3f);
        glBegin(GL_LINES);
            glVertex2f(x,        -(y + 0.5f));
            glVertex2f(x + 1.0f, -(y + 0.5f));
        glEnd();
        glLineWidth(tmpLw);
        //draw value quad
        float clampVal = (value > max)       ? (max)                               : (value);
        clampVal       = (clampVal < min)    ? (min)                               : (clampVal);
        float yVar     = (clampVal < thresh) ? (-std::abs((clampVal / min)*yDMax)) : (std::abs((clampVal/max)*yDMax));
		float col      = 0.75f; // (0.1f + (0.9f - (yDMax - std::abs(yVar)*0.9f)));
		glColor3f(col, col, col);
        glBegin(GL_QUADS);
            glVertex2f(x,        -(y + 0.5f - yVar));
            glVertex2f(x,        -(y + 0.5f));
            glVertex2f(x + 1.0f, -(y + 0.5f));
            glVertex2f(x + 1.0f, -(y + 0.5f - yVar));
        glEnd();

    }
}


/*
 * UncertaintySequenceRenderer::RenderUncertainty
 */
void UncertaintySequenceRenderer::RenderUncertainty(float yPos, float fgColor[4], float bgColor[4]) {

    vislib::math::Vector<float, 4> grayColor = {0.4f, 0.4f, 0.4f, 1.0f}; //this->secStructColor[(unsigned int)UncertaintyDataCall::secStructure::C_COIL];

    unsigned int sMax, sTemp;                          // structure types 
    float uMax, uDelta, uTemp;                         // uncertainty
    float timer, tDelta, tSpeed;                       // time
    vislib::math::Vector<float, 2> posOffset, posTemp; // position
    unsigned int diffStruct;                           // different structures with uncertainty > 0
    vislib::math::Vector<float, 4> colTemp;            // color
    vislib::math::Vector<float, 4> colStart, colMiddle, colEnd; 
                                    
    UncertaintyDataCall::secStructure sLeft, sRight;
    // sorted from left to right
	const unsigned int structCount = static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);
	vislib::math::Vector<unsigned int, structCount> sortedStructLR;
    // sorted bottom to top
	vislib::math::Vector<unsigned int, structCount> sortedStructBT;
	vislib::math::Vector<unsigned int, structCount> lastSortedStructBT;

    vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > *secUncertainty = 
        &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::UNCERTAINTY];
    vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > *sortedUncertainty = 
        &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::UNCERTAINTY];

	glDisable(GL_DEPTH_TEST); // disabled depth test is necessary because of geometry overlay drawing
    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
    
    for (unsigned int i = 0; i < this->aminoAcidCount; i++) { // loop over all amino-acids
        
        // ignore as missing flagged amino-acids and ignore if all structure types are NOTDEFINED resulting in maximum uncerainty equals 0
        if ((this->residueFlag[i] != UncertaintyDataCall::addFlags::MISSING) || (secUncertainty->operator[](i)[sortedUncertainty->operator[](i)[0]] < SEQ_FLOAT_EPS)) {

            // default
            posOffset.SetX(this->vertices[2 * i]);
            posOffset.SetY(this->vertices[2 * i + 1] + yPos + 1.0f);

            sMax = (unsigned int)sortedUncertainty->operator[](i)[0];
            uMax = secUncertainty->operator[](i)[sMax];
            
            // CERTAIN amino-acids
            if (uMax >= (1.0f)) {
                
                // draw something for certain amino-acids only if they have not both no color assigned (= background color)
                if(!((this->currentCertainBlockChartColor == CERTAIN_BC_NONE) && (this->currentCertainStructColor == CERTAIN_STRUCT_NONE))) {
                    
                    // block chart color
                    if (this->currentCertainBlockChartColor == CERTAIN_BC_COLORED) {
                        glColor3fv(this->secStructColor[sMax].PeekComponents());
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
        
                    // draw colored structure only when block chart is not colored
                    if (!((this->currentCertainBlockChartColor == CERTAIN_BC_COLORED) && (this->currentCertainStructColor == CERTAIN_STRUCT_COLORED))) {
                        
                        // structure color
                        // ALWAYS assign color first (early as possible), because structure can change ... see below: end of strand
                        if (this->currentCertainStructColor == CERTAIN_STRUCT_COLORED) {
                            glColor3fv(this->secStructColor[sMax].PeekComponents());
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
						if (i < this->aminoAcidCount - 1){
                            if ((sMax == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) && (sMax != (unsigned int)sortedUncertainty->operator[](i+1)[0])) {
								sMax = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
							}
						}
                        glBegin(GL_TRIANGLE_STRIP);
                        for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                            glVertex2f(posOffset.X() + this->secStructVertices[sMax][j].X(), -(posOffset.Y() + this->secStructVertices[sMax][j].Y()));
                        }
                        glEnd();
                    }
                }
            }
            // UNCERTAIN amino-acids
            else { 
                // sort structure from bottom to top
                if ((this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_VERTI) ||
                    (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_VERTI)) {
                                            
                    if (i > 0) {
                        if (secUncertainty->operator[](i-1)[(int)sortedUncertainty->operator[](i - 1)[0]] == 1.0f) {
							for (unsigned int j = 0; j < structCount; j++) {
								lastSortedStructBT[j] = sMax;
							}
						}
						else {
							lastSortedStructBT = sortedStructBT;
						}
                    }
                    else {
						for (unsigned int j = 0; j < structCount; j++) {
							lastSortedStructBT[j] = sMax;
                        }
                    }
                    // copy data to sortedStruct
					for (unsigned int j = 0; j < structCount; j++) {
						sortedStructBT[j] = (unsigned int)sortedUncertainty->operator[](i)[j];
                    }
                    // search for structure type matching the last sorted 
					for (unsigned int j = 0; j < structCount; j++) {     // lastSortedStruct
						for (unsigned int k = 0; k < structCount; k++) { // sortedStruct
                            if (lastSortedStructBT[j] == sortedStructBT[k]) {
								sTemp = sortedStructBT[j];
                                sortedStructBT[j] = sortedStructBT[k];
								sortedStructBT[k] = sTemp;
                            } 
                        }
                    }
                }
                
                // sort structure from left to right 
                if  ((this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_HORIZ) ||
                     (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_HORIZ) ||
                     (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_MORPH)) {

                    //assign structure type for left side from previous amino-acid
                    if (i > 0) {
                        if (secUncertainty->operator[](i-1)[(int)sortedUncertainty->operator[](i - 1)[0]] == 1.0f) {
                            sLeft = sortedUncertainty->operator[](i-1)[0];
						}
						else {
							for (unsigned int j = 0; j < structCount; j++) {
                                if (secUncertainty->operator[](i-1)[sortedStructLR[j]] > 0.0f) { // sortedStruct from last iteration
									sLeft = static_cast<UncertaintyDataCall::secStructure>(sortedStructLR[j]); // take the last one which is the most right one > 0
								}
							}
						}
                    }
                    else {
						sLeft = sortedUncertainty->operator[](i)[structCount - 1]; // set minimum on left side because maximum structure type will certainly be on right side
                    }

                    //assign structure type for right side from following amino-acid
                    if (i < this->aminoAcidCount - 1)
                        sRight = sortedUncertainty->operator[](i+1)[0]; // next structure type with max certainty
                    else
						sRight = sortedUncertainty->operator[](i)[structCount - 1]; // set minimum on right side because maximum structure type will certainly be on left side

                    // copy data to sortedStruct
					for (unsigned int j = 0; j < structCount; j++) {
                        sortedStructLR[j] = (unsigned int)sortedUncertainty->operator[](i)[j];
                    }
                    // search for structure type matching on the left and right side
					for (unsigned int j = 0; j < structCount; j++) {

						if (sortedStructLR[j] == sRight) { // check first for right side and ...
							sTemp = sortedStructLR[structCount - 1];
							sortedStructLR[structCount-1] = sortedStructLR[j];
							sortedStructLR[j] = sTemp;
						}

						if (sortedStructLR[j] == sLeft) { // ... than for left side, because otherwise a swapped element wouldn't be handled afterwards
							sTemp = sortedStructLR[0];
							sortedStructLR[0] = sortedStructLR[j];
							sortedStructLR[j] = sTemp;
						}
                    }
                }
                
                                       
                // BLOCK CHART - HORIZONTAL        
				if ((this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_HORIZ) && (this->currentUncertainBlockChartColor != UNCERTAIN_BC_NONE)) {
                                      
					// matrices
					GLfloat modelViewMatrix_column[16];
					glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
					vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
					GLfloat projMatrix_column[16];
					glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
					vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
					// Compute modelviewprojection matrix
					vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;

					this->shader.Enable();

					glUniformMatrix4fv(this->shader.ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
					glUniform2fv(this->shader.ParameterLocation("worldPos"), 1, (GLfloat *)posOffset.PeekComponents());
					glUniform4fv(this->shader.ParameterLocation("structCol"), structCount, (GLfloat *)this->secStructColor.PeekElements());
					glUniform1iv(this->shader.ParameterLocation("sortedStruct"), structCount, (GLint *)sortedStructLR.PeekComponents());
					glUniform1fv(this->shader.ParameterLocation("structUnc"), structCount, (GLfloat *)secUncertainty->operator[](i).PeekComponents());
					glUniform1f(this->shader.ParameterLocation("gradientInt"), this->currentUncertainGardientInterval);
					glUniform1i(this->shader.ParameterLocation("colorInterpol"), (int)this->currentUncertainColorInterpol);
					
					glBegin(GL_QUADS);

						glVertex2f(posOffset.X(), -(posOffset.Y() - 1.0f));
						glVertex2f(posOffset.X(), -(posOffset.Y()));
						glVertex2f(posOffset.X() + 1.0f, -(posOffset.Y()));
						glVertex2f(posOffset.X() + 1.0f, -(posOffset.Y() - 1.0f));

					glEnd();

					this->shader.Disable();

                }
				// BLOCK CHART - VERTICAL 
				else if ((this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_VERTI) && (this->currentUncertainBlockChartColor != UNCERTAIN_BC_NONE)) {
					glBegin(GL_QUADS);
                    uDelta = 0.0f;
					for (unsigned int j = 0; j < structCount; j++) {
                        
                        if (this->currentUncertainBlockChartColor == UNCERTAIN_BC_COLORED)
                            glColor3fv(this->secStructColor[sortedStructBT[j]].PeekComponents());

                        uTemp = secUncertainty->operator[](i)[sortedStructBT[j]];

						if (uTemp > 0.0f) {
							glVertex2f(posOffset.X(), -(posOffset.Y() - uDelta - uTemp));
							glVertex2f(posOffset.X(), -(posOffset.Y() - uDelta));
							glVertex2f(posOffset.X() + 1.0f, -(posOffset.Y() - uDelta));
							glVertex2f(posOffset.X() + 1.0f, -(posOffset.Y() - uDelta - uTemp));
							uDelta += uTemp;
						}
                    }         
					glEnd();
                }
 
                
                // structure color
                if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_NONE)
                    glColor3fv(bgColor);                
                else if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_GRAY)
                    glColor3fv(grayColor.PeekComponents());

                // UNCERTAINTY GLYPHS
				if ((this->currentUncertainStructGeometry == UNCERTAIN_GLYPH) ||
					(this->currentUncertainStructGeometry == UNCERTAIN_GLYPH_WITH_AXES)) {

					// translate position offset from upper left corner to center of "uncertainty" quad
					posOffset += vislib::math::Vector<float, 2>(0.5f, -0.4f); // a little offset for the y-axis to adjust axes visualy in the center

					// TODO color interpolation
					/*
					if (this->currentUncertainColorInterpol == UNCERTAIN_COLOR_RGB) {
						// colTemp = aStart*colStart + aEnd*colEnd;
					}
					else if (this->currentUncertainColorInterpol == UNCERTAIN_COLOR_HSL_HP) {
						// colTemp = this->HuePreservingColorBlending(aStart*colStart, aEnd*colEnd);
					}
					else { // if(this->currentUncertainColorInterpol == UNCERTAIN_COLOR_HSL)
					
					}
					*/

					glBegin(GL_TRIANGLE_FAN);

					uMax = secUncertainty->operator[](i)[(int)sortedUncertainty->operator[](i)[0]];

					// first vertex is center of fan
					glColor3fv(this->secStructColor[(unsigned int)UncertaintyDataCall::secStructure::NOTDEFINED].PeekComponents());
					glVertex2f(posOffset.X(), -posOffset.Y());

					for (unsigned int j = 0; j < this->glyphAxis.Count(); j++) {
						glColor3fv(this->secStructColor[(unsigned int)sortedUncertainty->operator[](i)[j]].PeekComponents());
						uTemp = secUncertainty->operator[](i)[(int)sortedUncertainty->operator[](i)[j]] / uMax;
						posTemp = posOffset + (this->glyphAxis[j] * uTemp);
						glVertex2f(posTemp.X(), -(posTemp.Y()));
					}
					// second vertex of fan once more
					glColor3fv(this->secStructColor[(unsigned int)sortedUncertainty->operator[](i)[0]].PeekComponents());
					uTemp = 1.0f;
					posTemp = posOffset + (this->glyphAxis[0] * uTemp);
					glVertex2f(posTemp.X(), -(posTemp.Y()));

					glEnd();

					// draw axes of glyphs
					if (this->currentUncertainStructGeometry == UNCERTAIN_GLYPH_WITH_AXES) {
						glColor3fv(fgColor);
						glBegin(GL_LINES);
						for (unsigned int j = 0; j < this->glyphAxis.Count(); j++) {
							glVertex2f(posOffset.X(), -posOffset.Y());
							glVertex2f(posOffset.X() + this->glyphAxis[j].X(), -(posOffset.Y() + this->glyphAxis[j].Y()));
						}
						glEnd();
					}
				}
				// UNCERTAINTY ANIMATION - Different TIME Intervalls - Equal Amplitudes: SPACE - EQUAL Time Intervalls
                else if ((this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_TIME) || 
                    (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_SPACE) || 
                    (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_EQUAL)) {
                    
                    tSpeed = 2.0f; // animation speed: greater value means slower (= time interval)
                    timer = static_cast<float>((std::clock() - this->animTimer) / (tSpeed*CLOCKS_PER_SEC));
                    if (timer >= 1.0f) {
                        this->animTimer = std::clock();
                    }
                    
                    // get the number off structures with uncertainty greater 0
                    diffStruct = 0;
					for (unsigned int k = 0; k < structCount; k++) {
                        if (secUncertainty->operator[](i)[(int)sortedUncertainty->operator[](i)[k]] > 0.0f)
                            diffStruct++;
                    }

                    glBegin(GL_TRIANGLE_STRIP);
                    posOffset.SetY(posOffset.Y() - 0.5f);

                    for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                        
                        if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_TIME) {
                            tDelta = 0.0f;
                            for (unsigned int k = 0; k < diffStruct; k++) {

								sTemp = (unsigned int)sortedUncertainty->operator[](i)[k];
                                uTemp = secUncertainty->operator[](i)[sTemp];

                                if ((tDelta <= timer) && (timer <= (tDelta + uTemp))) {

                                    if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                                        colTemp = (1.0f - (timer - tDelta) / uTemp)*(this->secStructColor[sTemp]);
                                    }
                                    // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
									if (i < this->aminoAcidCount - 1){
										if ((sTemp == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) && (sTemp != (unsigned int)sortedUncertainty->operator[](i+1)[0])) {
											sTemp = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
										}
									}
                                    posTemp = this->secStructVertices[sTemp][j];
                                    posTemp.SetY((1.0f - (timer - tDelta) / uTemp)*this->secStructVertices[sTemp][j].Y());

                                    if (k == diffStruct - 1) {
										sTemp = (unsigned int)sortedUncertainty->operator[](i)[0];
                                    }
                                    else {
										sTemp = (unsigned int)sortedUncertainty->operator[](i)[k + 1];
                                    }
                                    if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                                        colTemp = colTemp + ((timer - tDelta) / uTemp)*(this->secStructColor[sTemp]);
                                        glColor3fv(colTemp.PeekComponents());
                                    }
                                    // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
									if (i < this->aminoAcidCount - 1){
										if ((sTemp == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) && (sTemp != (unsigned int)sortedUncertainty->operator[](i+1)[0])) {
											sTemp = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
										}
									}
                                    posTemp.SetY(posTemp.Y() + ((timer - tDelta) / uTemp)*this->secStructVertices[sTemp][j].Y());

                                    glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() + posTemp.Y()));
                                }
                                
                                tDelta += uTemp;
                            }
                        } 
                        else {
                            tDelta = 1.0f / static_cast<float>(diffStruct);
                            for (unsigned int k = 0; k < diffStruct; k++) {
                                
                                if ((static_cast<float>(k)*tDelta <= timer) && (timer <= (static_cast<float>(k)*tDelta + tDelta))) {

									sTemp = (unsigned int)sortedUncertainty->operator[](i)[k];
                                    uTemp = secUncertainty->operator[](i)[sTemp];

                                    if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                                        colTemp = (1.0f - (timer - static_cast<float>(k)*tDelta) / tDelta)*(this->secStructColor[sTemp]);
                                    }
                                    // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
									if (i < this->aminoAcidCount - 1){
										if ((sTemp == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) && (sTemp != (unsigned int)sortedUncertainty->operator[](i+1)[0])) {
											sTemp = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
										}
									}
                                    posTemp = this->secStructVertices[sTemp][j];
                                    
                                    if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_SPACE) {
                                        posTemp.SetY((1.0f - (timer - static_cast<float>(k)*tDelta) / tDelta)*this->secStructVertices[sTemp][j].Y()*uTemp);
                                    }
                                    if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_EQUAL) {
                                        posTemp.SetY((1.0f - (timer - static_cast<float>(k)*tDelta) / tDelta)*this->secStructVertices[sTemp][j].Y());
                                    }

                                    
                                    if (k == diffStruct - 1)
										sTemp = (unsigned int)sortedUncertainty->operator[](i)[0];
                                    else
										sTemp = (unsigned int)sortedUncertainty->operator[](i)[k + 1];

                                    uTemp = secUncertainty->operator[](i)[sTemp];

                                    if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
                                        colTemp = colTemp + ((timer - static_cast<float>(k)*tDelta) / tDelta)*(this->secStructColor[sTemp]);
                                        glColor3fv(colTemp.PeekComponents());
                                    }
                                    // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
									if (i < this->aminoAcidCount - 1){
										if ((sTemp == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) && (sTemp != (unsigned int)sortedUncertainty->operator[](i+1)[0])) {
											sTemp = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
										}
									}

                                    if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_SPACE){
                                        posTemp.SetY(posTemp.Y() + ((timer - static_cast<float>(k)*tDelta) / tDelta)*this->secStructVertices[sTemp][j].Y()*uTemp);
                                    }
                                    if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_DYN_EQUAL) {
                                        posTemp.SetY(posTemp.Y() + ((timer - static_cast<float>(k)*tDelta) / tDelta)*this->secStructVertices[sTemp][j].Y());
                                    }

                                    glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() + posTemp.Y()));
                                }
                            }
                        }          
                    }
                    glEnd();
                }
				// UNCERTAINTY STRUCTURE - VERTICAL
                else if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_VERTI) {
                        
					for (unsigned int j = 0; j < structCount; j++) {
                        
						sTemp = sortedStructBT[j];
                        uTemp = secUncertainty->operator[](i)[sTemp];
                        
						if (uTemp > 0.0f) {
							glBegin(GL_TRIANGLE_STRIP);
							if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED)
								glColor3fv(this->secStructColor[sTemp].PeekComponents());

							// check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
							if (i < this->aminoAcidCount - 1){
								if ((sTemp == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) && (sTemp != (unsigned int)sortedUncertainty->operator[](i+1)[0])) {
									sTemp = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
								}
							}
							for (unsigned int k = 0; k < this->secStructVertices[sTemp].Count(); k++) {
								posTemp = this->secStructVertices[sTemp][k];
								glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() - uTemp*0.5f - posTemp.Y()*uTemp));
							}
							posOffset.SetY(posOffset.Y() - uTemp);
							glEnd();
						}
                    }
                }
				// UNCERTAINTY STRUCTURE - HORIZONTAL
                else if (this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_HORIZ) {

                    posOffset.SetY(posOffset.Y() - 0.5f);
					for (unsigned int j = 0; j < structCount; j++) {

						sTemp = sortedStructLR[j];
                        uTemp = secUncertainty->operator[](i)[sTemp];
                        
						if (uTemp > 0.0f) {
							glBegin(GL_TRIANGLE_STRIP);
							if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED)
								glColor3fv(this->secStructColor[sTemp].PeekComponents());

							// check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
							if (i < this->aminoAcidCount - 1){
								if ((sTemp == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) && (sTemp != (unsigned int)sortedUncertainty->operator[](i+1)[0])) {
									sTemp = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
								}
							}
							for (unsigned int k = 0; k < this->secStructVertices[sTemp].Count(); k++) {

								posTemp = this->secStructVertices[sTemp][k];
								glVertex2f(posOffset.X() + posTemp.X()*uTemp, -(posOffset.Y() + posTemp.Y()));
							}
							posOffset.SetX(posOffset.X() + uTemp);
							glEnd();
						}
                    }
                }
				// UNCERTAINTY STRUCTURE - MORPHED
                else if((this->currentUncertainStructGeometry == UNCERTAIN_STRUCT_STAT_MORPH) && 
					    (!((this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) && ((this->currentUncertainBlockChartOrientation == UNCERTAIN_BC_HORIZ) && 
						                                                                      (this->currentUncertainBlockChartColor == UNCERTAIN_BC_COLORED))))) {

					if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {

						// matricess
						GLfloat modelViewMatrix_column[16];
						glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
						vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
						GLfloat projMatrix_column[16];
						glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
						vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
						// Compute modelviewprojection matrix
						vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;

						this->shader.Enable();

						glUniformMatrix4fv(this->shader.ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
						glUniform2fv(this->shader.ParameterLocation("worldPos"), 1, (GLfloat *)posOffset.PeekComponents());
						glUniform4fv(this->shader.ParameterLocation("structCol"), structCount, (GLfloat *)this->secStructColor.PeekElements());
						glUniform1iv(this->shader.ParameterLocation("sortedStruct"), structCount, (GLint *)sortedStructLR.PeekComponents());
						glUniform1fv(this->shader.ParameterLocation("structUnc"), structCount, (GLfloat *)secUncertainty->operator[](i).PeekComponents());
						glUniform1f(this->shader.ParameterLocation("gradientInt"), this->currentUncertainGardientInterval);
						glUniform1i(this->shader.ParameterLocation("colorInterpol"), (int)this->currentUncertainColorInterpol);
					}

                    posOffset.SetY(posOffset.Y() - 0.5f);
                    glBegin(GL_TRIANGLE_STRIP);
                    for (unsigned int j = 0; j < this->secStructVertices[sMax].Count(); j++) {
                                
                        posTemp.SetX(this->secStructVertices[sMax][j].X());
                        posTemp.SetY(0.0f);

						for (unsigned int k = 0; k < structCount; k++) {
							sTemp = sortedStructLR[k];
                            uTemp = secUncertainty->operator[](i)[sTemp];

                            // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
							if (i < this->aminoAcidCount - 1){
								if ((sTemp == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) && (sTemp != (unsigned int)sortedUncertainty->operator[](i+1)[0])) {
									sTemp = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
								}
							}
                            posTemp.SetY(posTemp.Y() + uTemp*this->secStructVertices[sTemp][j].Y());
                        }

                        glVertex2f(posOffset.X() + posTemp.X(), -(posOffset.Y() + posTemp.Y()));
                    }
                    glEnd(); 

					if (this->currentUncertainStructColor == UNCERTAIN_STRUCT_COLORED) {
						this->shader.Disable();
					}
                }
            }
        }
    }

	glEnable(GL_DEPTH_TEST);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


/*
* UncertaintySequenceRenderer::RenderToolTip
*/
void UncertaintySequenceRenderer::RenderToolTip(float start, float end, vislib::StringA str1, vislib::StringA str2, float fgColor[4], float bgColor[4]) {

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
 * UncertaintySequenceRenderer::DrawSecStructGeometryTiles
 */
void UncertaintySequenceRenderer::DrawSecStructGeometryTiles(UncertaintyDataCall::secStructure cur, UncertaintyDataCall::secStructure fol, 
                                                             UncertaintyDataCall::addFlags f, float x, float y, float bgColor[4]) {

    UncertaintyDataCall::secStructure curTemp = cur;                                                            
    
    // ignore as missing flagged amino-acids and NOTDEFINED secondary structure types
    if((f != UncertaintyDataCall::addFlags::MISSING) && (cur != UncertaintyDataCall::secStructure::NOTDEFINED)) {
        glColor3fv(this->secStructColor[(int)cur].PeekComponents()); 
        // check if end of strand is an arrow (the arrow vertices ae stored in BRIDGE)
        if ((curTemp == UncertaintyDataCall::secStructure::E_EXT_STRAND) && (curTemp != fol)) {
            curTemp = UncertaintyDataCall::secStructure::B_BRIDGE;
        }   
        glBegin(GL_TRIANGLE_STRIP);              
        for (unsigned int i = 0; i < this->secStructVertices[(int)curTemp].Count(); i++) {
            glVertex2f(x + this->secStructVertices[(int)curTemp][i].X(), -(y + 0.5f + this->secStructVertices[(int)curTemp][i].Y()));
        }
        glEnd();
    }                                                       
}                                 


/*
 * UncertaintySequenceRenderer::DrawSecStructTextureTiles
 */
void UncertaintySequenceRenderer::DrawSecStructTextureTiles(UncertaintyDataCall::secStructure pre, UncertaintyDataCall::secStructure cur, UncertaintyDataCall::secStructure fol, 
                                                            UncertaintyDataCall::addFlags f, float x, float y, float bgColor[4]) {    
    const float eps = 0.0f;
        
    if(f == UncertaintyDataCall::addFlags::MISSING) {
        glColor4fv(bgColor);
        this->markerTextures[0]->Bind(); 
    }
    else {    
        glColor3fv(this->secStructColor[(int)cur].PeekComponents()); 
        switch (cur) {
            case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) :
                if (pre != cur) {
                    this->markerTextures[4]->Bind();
                } 
                else if (fol != cur) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::G_310_HELIX) :        
                if (pre != cur) {
                    this->markerTextures[4]->Bind();
                } 
                else if (fol != cur) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::I_PI_HELIX) :        
                if (pre != cur) {
                    this->markerTextures[4]->Bind();
                } 
                else if (fol != cur) {
                    this->markerTextures[6]->Bind();
                } 
                else {
                    this->markerTextures[5]->Bind();
                }
                break;
            case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  
                if (fol != cur) {
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
                if (fol != cur) {
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
                glColor4fv(bgColor);
                this->markerTextures[0]->Bind();        
                break;                
            default: this->markerTextures[0]->Bind(); 
				break;
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

	this->uncertainty.Clear();
	this->uncertainty.AssertCapacity(this->aminoAcidCount);

    for (unsigned int i = 0; i < this->sortedSecStructAssignment.Count(); i++) {
        this->sortedSecStructAssignment.Clear();
    }
    this->sortedSecStructAssignment.Clear();
    for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
        this->sortedSecStructAssignment.Add(vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> >());
        this->sortedSecStructAssignment.Last().AssertCapacity(this->aminoAcidCount);
    }

    for (unsigned int i = 0; i < this->secStructUncertainty.Count(); i++) {
        this->secStructUncertainty.Clear();
    }
    this->secStructUncertainty.Clear();
    for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
        this->secStructUncertainty.Add(vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> >());
        this->secStructUncertainty.Last().AssertCapacity(this->aminoAcidCount);
    }
	
    this->strideStructThreshold.Clear();
    this->strideStructThreshold.AssertCapacity(this->aminoAcidCount);

    this->dsspStructEnergy.Clear();
    this->dsspStructEnergy.AssertCapacity(this->aminoAcidCount);

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

	this->aminoacidSeparatorVertices.Clear();
	this->aminoacidSeparatorVertices.AssertCapacity(this->aminoAcidCount * 8);

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

	this->secStructColor.Clear();
	this->secStructColor.AssertCapacity(UncertaintyDataCall::secStructure::NOE);

	this->secStructDescription.Clear();
	this->secStructDescription.AssertCapacity(UncertaintyDataCall::secStructure::NOE);
	
    // temporary variables
	vislib::StringA tmpStr;
	vislib::StringA letters;
    unsigned int maxNumBindingSitesPerRes = 0;
    unsigned int cCnt = 0;
    char currentChainID = udc->GetChainID(0);
	float aaCur;

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
        

	// get pdb secondary structure assignment methods
	this->pdbLegend = "PDB (HELIX: ";
	switch (udc->GetPdbAssMethodHelix()) {
	case(UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF) : this->pdbLegend.Append("PROMOTIF"); break;
	case(UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR) : this->pdbLegend.Append("Author"); break;
	case(UncertaintyDataCall::pdbAssMethod::PDB_DSSP) : this->pdbLegend.Append("DSSP"); break;
	case(UncertaintyDataCall::pdbAssMethod::PDB_UNKNOWN) : this->pdbLegend.Append("unknown"); break;
	default: break;
	}
	this->pdbLegend.Append(" - SHEET: ");
	switch (udc->GetPdbAssMethodSheet()) {
	case(UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF) : this->pdbLegend.Append("PROMOTIF"); break;
	case(UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR) : this->pdbLegend.Append("Author"); break;
	case(UncertaintyDataCall::pdbAssMethod::PDB_DSSP) : this->pdbLegend.Append("DSSP"); break;
	case(UncertaintyDataCall::pdbAssMethod::PDB_UNKNOWN) : this->pdbLegend.Append("unknown"); break;
	default: break;
	}
	this->pdbLegend.Append(")");

	// get secondary structure type colors and descriptions
	for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); i++) {
		this->secStructColor.Add(udc->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(i)));
		this->secStructDescription.Add(udc->GetSecStructDesc(static_cast<UncertaintyDataCall::secStructure>(i)));
	}
	
    // collect data from call
    for (unsigned int aa = 0; aa < this->aminoAcidCount; aa++) {

        // store the pdb index of the current amino-acid
        this->aminoAcidIndex.Add(udc->GetPDBAminoAcidIndex(aa));
        
        // store the secondary structure element type of the current amino-acid for each assignment method
        for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
            this->sortedSecStructAssignment[k].Add(udc->GetSortedSecStructAssignment(static_cast<UncertaintyDataCall::assMethod>(k), aa));
            this->secStructUncertainty[k].Add(udc->GetSecStructUncertainty(static_cast<UncertaintyDataCall::assMethod>(k), aa));
        }

        // store the amino-acid three letter code and convert it to the one letter code
		this->aminoAcidName.Add(udc->GetAminoAcidOneLetterCode(aa));
        // store the chainID
        this->chainID.Add(udc->GetChainID(aa));
        // init selection
        this->selection.Add(false);
        // store residue flag
        this->residueFlag.Add(udc->GetResidueFlag(aa));
		// store uncertainty value 
		this->uncertainty.Add(udc->GetUncertainty(aa));

        // store stride threshold values
        this->strideStructThreshold.Add(udc->GetStrideThreshold(aa));
        // store dssp energy values
        this->dsspStructEnergy.Add(udc->GetDsspEnergy(aa));


        // count different chains and set seperator vertices and chain color
        if (udc->GetChainID(aa) != currentChainID) {
            currentChainID = udc->GetChainID(aa);
            this->chainSeparatorVertices.Add(this->vertices[this->vertices.Count() - 2] + 1.0f);
			this->chainSeparatorVertices.Add(-(this->vertices[this->vertices.Count() - 1]));
            this->chainSeparatorVertices.Add(this->vertices[this->vertices.Count() - 2] + 1.0f);
            this->chainSeparatorVertices.Add(-(this->chainVertices[this->chainVertices.Count() - 1] + 0.5f));
            cCnt++;
        }

		/*
		// function for chain color assignment shaould be the same as in 'uncertaintycartoontessellation.btf' (~ line 353)
		// -> vec4(1.0 - float(chainIndex % 5) / 4.0, float(chainIndex % 3) / 2.0, float(chainIndex % 5) / 4.0, 1.0);
		this->chainColors.Add(1.0f - float((int)currentChainID % 5) / 4.0f);
		this->chainColors.Add(float((int)currentChainID % 3) / 2.0f);
		this->chainColors.Add(float((int)currentChainID % 5) / 4.0f);
		*/
		
		this->chainColors.Add(this->colorTable[cCnt].X());
		this->chainColors.Add(this->colorTable[cCnt].Y());
		this->chainColors.Add(this->colorTable[cCnt].Z());
		

        // compute the position of the amino-acid icon
        if (aa == 0) {
            // structure vertices
            this->vertices.Add(0.0f);
            this->vertices.Add(0.0f);
            // chain tile vertices and colors
            this->chainVertices.Add(0.0f);
            this->chainVertices.Add(static_cast<float>(this->secStructRows) + 1.5f);
        }
		else {
			if (aa % static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value()) != 0) {
				// structure vertices
				this->vertices.Add(this->vertices[aa * 2 - 2] + 1.0f);
				this->vertices.Add(this->vertices[aa * 2 - 1]);
				// chain tile vertices and colors
				this->chainVertices.Add(this->chainVertices[aa * 2 - 2] + 1.0f);
				this->chainVertices.Add(this->chainVertices[aa * 2 - 1]);
			}
			else {
				// structure vertices
				this->vertices.Add(0.0f);
				this->vertices.Add(this->vertices[aa * 2 - 1] + this->rowHeight);
				// chain tile vertices and colors
				this->chainVertices.Add(0.0f);
				this->chainVertices.Add(this->chainVertices[aa * 2 - 1] + this->rowHeight);
			}
		}

        if (bs) {
            // try to match binding sites
            vislib::Pair<char, unsigned int> bsRes;
            unsigned int numBS = 0;
			bool skipIndexWithLetter;

            // loop over all binding sites
            for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
                for (unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count(); bsResCnt++) {
                    bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
					skipIndexWithLetter = false;

					// remove possible letters in pdb index before converting to integer
                    tmpStr = udc->GetPDBAminoAcidIndex(aa);
					letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
					for (int k = 0; k < letters.Length(); k++) {
						if (tmpStr.Contains(letters[k]))
							skipIndexWithLetter = true;
					}

					if ((!skipIndexWithLetter) && (udc->GetChainID(aa) == bsRes.First()) && (std::atoi(tmpStr) == bsRes.Second())) {
                        this->bsVertices.Add(this->vertices[this->vertices.Count() - 2]);
                        this->bsVertices.Add(this->vertices[this->vertices.Count() - 1] + (2.0f + static_cast<float>(this->secStructRows)) + numBS*0.5f);
                        this->bsIndices.Add(bsCnt);
                        numBS++;
                        maxNumBindingSitesPerRes = vislib::math::Max(maxNumBindingSitesPerRes, numBS);
                    }
                }
            }
        }

    }
	
	// compute the vertices for the quad for each uncertain amino-acid structure
	for (unsigned int aa = 0; aa < this->aminoAcidCount; aa++) {

		if (this->residueFlag[aa] != UncertaintyDataCall::addFlags::MISSING) {
            aaCur = this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::UNCERTAINTY][aa][(int)this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::UNCERTAINTY][aa][0]];

			if (aaCur < 1.0f) {
				this->aminoacidSeparatorVertices.Add(this->vertices[aa * 2]);
				this->aminoacidSeparatorVertices.Add(-(this->vertices[aa * 2 + 1] + this->secStructRows - 1.0f));
                
				this->aminoacidSeparatorVertices.Add(this->vertices[aa * 2]);
				this->aminoacidSeparatorVertices.Add(-(this->vertices[aa * 2 + 1] + this->secStructRows));

				this->aminoacidSeparatorVertices.Add(this->vertices[aa * 2] + 1.0f);
				this->aminoacidSeparatorVertices.Add(-(this->vertices[aa * 2 + 1] + this->secStructRows));
                
				this->aminoacidSeparatorVertices.Add(this->vertices[aa * 2] + 1.0f);
				this->aminoacidSeparatorVertices.Add(-(this->vertices[aa * 2 + 1] + this->secStructRows - 1.0f));
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
* UncertaintySequenceRenderer::HuePreservingColorBlending
*/
vislib::math::Vector<float, 4> UncertaintySequenceRenderer::HuePreservingColorBlending(vislib::math::Vector<float, 4> c1, vislib::math::Vector<float, 4> c2) {
	
	vislib::math::Vector<float, 4> cnew = { 0.0f, 0.0f, 0.0f, 1.0f};
	vislib::math::Vector<float, 4> HSL1 = this->rgb2hsl(c1);
	vislib::math::Vector<float, 4> HSL2 = this->rgb2hsl(c2);
	float tmpHue;
    
	if (HSL1[0] == HSL2[0]) {
		cnew = c1 + c2;
	}
	else {
        if (HSL1[1] > HSL2[1]) { // c1 is dominant
			tmpHue = (HSL1[0] + 180.0f);
			if (tmpHue >= 360.0f) {
				tmpHue -= 360.0f;
			}
			cnew = c1 + this->hsl2rgb(vislib::math::Vector<float, 4>(tmpHue, HSL2[1], HSL2[2], 1.0f));
		}
		else if (HSL1[1] < HSL2[1])  {// c2 is dominant
			tmpHue = (HSL2[0] + 180.0f);
			if (tmpHue >= 360.0f) {
				tmpHue -= 360.0f;
			}
			cnew = c2 + this->hsl2rgb(vislib::math::Vector<float, 4>(tmpHue, HSL1[1], HSL1[2], 1.0f));
		}
        else { // (HSL1[1] == HSL2[1]) - if c1 and c2 have same saturation decide for brighter one 
            if (HSL1[2] >= HSL2[2]) { // c1 is brighter or equal bright as c2
                tmpHue = (HSL1[0] + 180.0f);
                if (tmpHue >= 360.0f) {
                    tmpHue -= 360.0f;
                }
                cnew = c1 + this->hsl2rgb(vislib::math::Vector<float, 4>(tmpHue, HSL2[1], HSL2[2], 1.0f));
            }
            else {// c2 is brighter
                tmpHue = (HSL2[0] + 180.0f);
                if (tmpHue >= 360.0f) {
                    tmpHue -= 360.0f;
                }
                cnew = c2 + this->hsl2rgb(vislib::math::Vector<float, 4>(tmpHue, HSL1[1], HSL1[2], 1.0f));
            }
        }
    }

	return cnew;
}
	
		
/*
* UncertaintySequenceRenderer::rgb2hsl
*/
vislib::math::Vector<float, 4> UncertaintySequenceRenderer::rgb2hsl(vislib::math::Vector<float, 4> rgba) {
	

	vislib::math::Vector<float, 4> out = { 0.0f, 0.0f, 0.0f, 1.0f };
    float min, max, delta;
	float tmpR, tmpG, tmpB;

    min = rgba[0] < rgba[1] ? rgba[0] : rgba[1];
    min = min  < rgba[2] ? min  : rgba[2];

    max = rgba[0] > rgba[1] ? rgba[0] : rgba[1];
    max = max  > rgba[2] ? max  : rgba[2];
	delta = max - min;

    out[2] = (max + min) /2.0f; // L

    if (delta < 0.000001f)
    {
        out[1] = 0.0f; // S
        out[0] = 0.0f; // H 
    }
	else
	{
		if (out[2] < 0.5f) { // L
			out[1] = delta / (max + min); // S
		}
		else {
			out[1] = delta / (2.0f - max - min); // L
		}

		tmpR = (((max - rgba[0]) / 6.0f) + (delta / 2.0f)) / delta;
		tmpG = (((max - rgba[1]) / 6.0f) + (delta / 2.0f)) / delta;
		tmpB = (((max - rgba[2]) / 6.0f) + (delta / 2.0f)) / delta;

		if (rgba[0] == max){
			out[0] = tmpB - tmpG;
		}
		else if (rgba[1] == max) {
			out[0] = (1.0f / 3.0f) + tmpR - tmpB;
		}
		else if (rgba[2] == max) {
			out[0] = (2.0f / 3.0f) + tmpG - tmpR;
		}

		if (out[0] < 0.0f) 
			out[0] += 1.0f;

		if (out[0] >= 1.0f) 
			out[0] -= 1.0f;

		out[0] *= 360.0f;
	}

    return out;
}


/*
* UncertaintySequenceRenderer::hsl2rgb
*/
vislib::math::Vector<float, 4> UncertaintySequenceRenderer::hsl2rgb(vislib::math::Vector<float, 4> hsla) {
	
	float tmp1, tmp2;
	vislib::math::Vector<float, 4> out = { 0.0f, 0.0f, 0.0f, 1.0f};

	if (hsla[1] < 0.000001f) { // S
		out[0] = hsla[2];  // L
		out[1] = hsla[2];
		out[2] = hsla[2];
	}
	else
	{
		if (hsla[2] < 0.5f) {
			tmp2 = hsla[2] * (1.0f + hsla[1]);
		}
		else {
			tmp2 = (hsla[2] + hsla[1]) - (hsla[1] * hsla[2]);
		}

		tmp1 = (2.0f * hsla[2]) - tmp2;

		out[0] = this->hue2rgb(tmp1, tmp2, (hsla[0] / 360.0f) + (1.0f / 3.0f));
		out[1] = this->hue2rgb(tmp1, tmp2, (hsla[0] / 360.0f));
		out[2] = this->hue2rgb(tmp1, tmp2, (hsla[0] / 360.0f) - (1.0f / 3.0f));
	}

    return out;   
}


/*
* UncertaintySequenceRenderer::hue2rgb
*/
float UncertaintySequenceRenderer::hue2rgb(float v1, float v2, float vH) {

	float tmpH = vH;

	if (tmpH < 0.0f) {
		tmpH += 1.0f;
	}

	if (tmpH >= 1.0f) {
		tmpH -= 1.0f;
	}

	if ((6.0f * tmpH) < 1.0f) {
		return (v1 + ((v2 - v1) * 6.0f * tmpH));
	}
	if ((2.0f * tmpH) < 1.0f) {
		return (v2);
	}

	if ((3.0f * tmpH) < 2.0f) {
		return (v1 + ((v2 - v1) * ((2.0f / 3.0f) - tmpH) * 6.0f));
	}

	return (v1);
}
