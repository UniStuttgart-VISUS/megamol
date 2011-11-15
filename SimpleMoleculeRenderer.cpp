/*
 * SimpleMoleculeRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "SimpleMoleculeRenderer.h"
#include "CoreInstance.h"
#include "Color.h"
#include "utility/ShaderSourceFactory.h"
#include "utility/ColourParser.h"
#include "param/StringParam.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "param/BoolParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/ShaderSource.h"
#include "vislib/AbstractOpenGLShader.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <glh/glh_genext.h>
#include <omp.h>



using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * protein::SimpleMoleculeRenderer::SimpleMoleculeRenderer (CTOR)
 */
SimpleMoleculeRenderer::SimpleMoleculeRenderer(void) : Renderer3DModule (),
    molDataCallerSlot( "getData", "Connects the molecule rendering with molecule data storage"),
    colorTableFileParam( "color::colorTableFilename", "The filename of the color table."),
    coloringModeParam0( "color::coloringMode0", "The first coloring mode."),
    coloringModeParam1( "color::coloringMode1", "The second coloring mode."),
    cmWeightParam( "color::colorWeighting", "The weighting of the two coloring modes."),
    renderModeParam( "renderMode", "The rendering mode."),
    stickRadiusParam( "stickRadius", "The radius for stick rendering"),
    probeRadiusParam( "probeRadius", "The probe radius for SAS rendering"),
    minGradColorParam( "color::minGradColor", "The color for the minimum value for gradient coloring" ),
    midGradColorParam( "color::midGradColor", "The color for the middle value for gradient coloring" ),
    maxGradColorParam( "color::maxGradColor", "The color for the maximum value for gradient coloring" ),
    molIdxListParam( "molIdxList", "The list of molecule indices for RS computation:"),
    specialColorParam( "color::specialColor", "The color for the specified molecules" ),
    interpolParam( "posInterpolation", "Enable positional interpolation between frames" )
{
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);

    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    Color::ReadColorTableFromFile( filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(new param::StringParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);

    // coloring mode #0
    this->currentColoringMode0 = Color::CHAIN;
    param::EnumParam *cm0 = new param::EnumParam(int(this->currentColoringMode0));
    cm0->SetTypePair( Color::ELEMENT, "Element");
    cm0->SetTypePair( Color::RESIDUE, "Residue");
    cm0->SetTypePair( Color::STRUCTURE, "Structure");
    cm0->SetTypePair( Color::BFACTOR, "BFactor");
    cm0->SetTypePair( Color::CHARGE, "Charge");
    cm0->SetTypePair( Color::OCCUPANCY, "Occupancy");
    cm0->SetTypePair( Color::CHAIN, "Chain");
    cm0->SetTypePair( Color::MOLECULE, "Molecule");
    cm0->SetTypePair( Color::RAINBOW, "Rainbow");
    this->coloringModeParam0 << cm0;
    this->MakeSlotAvailable( &this->coloringModeParam0);

    // coloring mode #1
    this->currentColoringMode1 = Color::ELEMENT;
    param::EnumParam *cm1 = new param::EnumParam(int(this->currentColoringMode1));
    cm1->SetTypePair( Color::ELEMENT, "Element");
    cm1->SetTypePair( Color::RESIDUE, "Residue");
    cm1->SetTypePair( Color::STRUCTURE, "Structure");
    cm1->SetTypePair( Color::BFACTOR, "BFactor");
    cm1->SetTypePair( Color::CHARGE, "Charge");
    cm1->SetTypePair( Color::OCCUPANCY, "Occupancy");
    cm1->SetTypePair( Color::CHAIN, "Chain");
    cm1->SetTypePair( Color::MOLECULE, "Molecule");
    cm1->SetTypePair( Color::RAINBOW, "Rainbow");
    this->coloringModeParam1 << cm1;
    this->MakeSlotAvailable( &this->coloringModeParam1);

    // Color weighting parameter
    this->cmWeightParam.SetParameter(new param::FloatParam(0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->cmWeightParam);

    // rendering mode
    this->currentRenderMode = LINES;
    //this->currentRenderMode = STICK;
    //this->currentRenderMode = BALL_AND_STICK;
    //this->currentRenderMode = SPACEFILLING;
    //this->currentRenderMode = SAS;
    param::EnumParam *rm = new param::EnumParam(int(this->currentRenderMode));
    rm->SetTypePair( LINES, "Lines");
    rm->SetTypePair( LINES_FILTER, "Lines-Filter");
    rm->SetTypePair( STICK, "Stick");
    rm->SetTypePair( STICK_FILTER, "Stick-Filter");
    rm->SetTypePair( BALL_AND_STICK, "Ball-and-Stick");
    rm->SetTypePair( SPACEFILLING, "Spacefilling");
    rm->SetTypePair( SPACEFILL_FILTER, "Spacefilling-Filter");
    rm->SetTypePair( SAS, "SAS");
    this->renderModeParam << rm;
    this->MakeSlotAvailable( &this->renderModeParam);

    // fill color table with default values and set the filename param
    this->stickRadiusParam.SetParameter(new param::FloatParam( 0.3f, 0.0f));
    this->MakeSlotAvailable( &this->stickRadiusParam);

    // fill color table with default values and set the filename param
    this->probeRadiusParam.SetParameter(new param::FloatParam( 1.4f, 0.0f));
    this->MakeSlotAvailable( &this->probeRadiusParam);

    // the color for the minimum value (gradient coloring
    this->minGradColorParam.SetParameter(new param::StringParam( "#146496"));
    this->MakeSlotAvailable( &this->minGradColorParam);

    // the color for the middle value (gradient coloring
    this->midGradColorParam.SetParameter(new param::StringParam( "#f0f0f0"));
    this->MakeSlotAvailable( &this->midGradColorParam);

    // the color for the maximum value (gradient coloring
    this->maxGradColorParam.SetParameter(new param::StringParam( "#ae3b32"));
    this->MakeSlotAvailable( &this->maxGradColorParam);

    // molecular indices list param
    this->molIdxList.Clear();
    this->molIdxListParam.SetParameter(new param::StringParam( ""));
    this->MakeSlotAvailable( &this->molIdxListParam);

    // the color for the maximum value (gradient coloring
    this->specialColorParam.SetParameter(new param::StringParam( "#228B22"));
    this->MakeSlotAvailable( &this->specialColorParam);

    // en-/disable positional interpolation
    this->interpolParam.SetParameter(new param::BoolParam( true));
    this->MakeSlotAvailable( &this->interpolParam);

    // make the rainbow color table
    Color::MakeRainbowColorTable( 100, this->rainbowColors);
}


/*
 * protein::SimpleMoleculeRenderer::~SimpleMoleculeRenderer (DTOR)
 */
SimpleMoleculeRenderer::~SimpleMoleculeRenderer(void)  {
    this->Release();
}


/*
 * protein::SimpleMoleculeRenderer::release
 */
void SimpleMoleculeRenderer::release(void) {

}


/*
 * protein::SimpleMoleculeRenderer::create
 */
bool SimpleMoleculeRenderer::create(void) {
    if (glh_init_extensions("GL_ARB_vertex_program") == 0) {
        return false;
    }
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;

    // Load sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    try {
        if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load filter sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::filterSphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter sphere shader");
        return false;
    }
    try {
        if (!this->filterSphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create filter sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load cylinder shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cylinder shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cylinder shader");
        return false;
    }
    try {
        if (!this->cylinderShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create cylinder shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load filter cylinder shader
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::filterCylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter cylinder shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::filterCylinderFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter cylinder shader");
        return false;
    }
    try {
        if (!this->filterCylinderShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create filter cylinder shader: %s\n", e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * protein::SimpleMoleculeRenderer::GetCapabilities
 */
bool SimpleMoleculeRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        | view::CallRender3D::CAP_ANIMATION );

    return true;
}


/*
 * protein::SimpleMoleculeRenderer::GetExtents
 */
bool SimpleMoleculeRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;

    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld( scale);
    cr3d->SetTimeFramesCount( mol->FrameCount());

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::SimpleMoleculeRenderer::Render
 */
bool SimpleMoleculeRenderer::Render(Call& call) {

    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL) return false;

    int cnt;

    // set call time
    mol->SetCalltime(callTime);
    // set frame ID and call data
    mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;
    // check if atom count is zero
    if( mol->AtomCount() == 0 ) return true;
    // get positions of the first frame
    float *pos0 = new float[mol->AtomCount() * 3];
    memcpy( pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
    // set next frame ID and get positions of the second frame
    if( ( ( static_cast<int>( callTime) + 1) < int( mol->FrameCount()) ) &&
        this->interpolParam.Param<param::BoolParam>()->Value() )
        mol->SetFrameID(static_cast<int>( callTime) + 1);
    else
        mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) {
        delete[] pos0;
        return false;
    }
    float *pos1 = new float[mol->AtomCount() * 3];
    memcpy( pos1, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));

    // interpolate atom positions between frames
    float *posInter = new float[mol->AtomCount() * 3];
    float inter = callTime - static_cast<float>(static_cast<int>( callTime));
    float threshold = vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
        vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
        mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        if( std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
                std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
                std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
            posInter[3*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
            posInter[3*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
            posInter[3*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
        } else if( inter < 0.5f ) {
            posInter[3*cnt+0] = pos0[3*cnt+0];
            posInter[3*cnt+1] = pos0[3*cnt+1];
            posInter[3*cnt+2] = pos0[3*cnt+2];
        } else {
            posInter[3*cnt+0] = pos1[3*cnt+0];
            posInter[3*cnt+1] = pos1[3*cnt+1];
            posInter[3*cnt+2] = pos1[3*cnt+2];
        }
    }

    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);

    // ---------- update parameters ----------
    this->UpdateParameters( mol);

    // recompute color table, if necessary
    if( this->atomColorTable.Count()/3 < mol->AtomCount() ) {

        // Mix two coloring modes
        Color::MakeColorTable( mol,
          this->currentColoringMode0,
          this->currentColoringMode1,
          cmWeightParam.Param<param::FloatParam>()->Value(),       // weight for the first cm
          1.0 - cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
          this->atomColorTable, this->colorLookupTable, this->rainbowColors,
          this->minGradColorParam.Param<param::StringParam>()->Value(),
          this->midGradColorParam.Param<param::StringParam>()->Value(),
          this->maxGradColorParam.Param<param::StringParam>()->Value(),
          true);

        // Use one coloring mode
        /*Color::MakeColorTable( mol,
          this->currentColoringMode0,
          this->atomColorTable, this->colorLookupTable, this->rainbowColors,
          this->minGradColorParam.Param<param::StringParam>()->Value(),
          this->midGradColorParam.Param<param::StringParam>()->Value(),
          this->maxGradColorParam.Param<param::StringParam>()->Value(),
          true);*/

    }

    // ---------- special color handling ... -----------
    unsigned int midx, ridx, rcnt, aidx, acnt;
    float r, g, b;
    utility::ColourParser::FromString( this->specialColorParam.Param<param::StringParam>()->Value(), r, g, b);
    for( unsigned int mi = 0; mi < this->molIdxList.Count(); ++ mi ) {
        midx = atoi( this->molIdxList[mi]);
        ridx = mol->Molecules()[midx].FirstResidueIndex();
        rcnt = ridx + mol->Molecules()[midx].ResidueCount();
        for( unsigned int ri = ridx; ri < rcnt; ++ri ) {
            aidx = mol->Residues()[ri]->FirstAtomIndex();
            acnt = aidx + mol->Residues()[ri]->AtomCount();
            for( unsigned int ai = aidx; ai < acnt; ++ai ) {
                this->atomColorTable[3*ai+0] = r;
                this->atomColorTable[3*ai+1] = g;
                this->atomColorTable[3*ai+2] = b;
            }
        }
    }
    // ---------- ... special color handling -----------

    // TODO: ---------- render ----------

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);


    // render data using the current rendering mode
    if( this->currentRenderMode == LINES ) {
        this->RenderLines( mol, posInter);
    } else if( this->currentRenderMode == STICK ) {
        this->RenderStick( mol, posInter);
    } else if( this->currentRenderMode == BALL_AND_STICK ) {
        this->RenderBallAndStick( mol, posInter);
    } else if( this->currentRenderMode == SPACEFILLING ) {
        this->RenderSpacefilling( mol, posInter);
    } else if( this->currentRenderMode == SPACEFILL_FILTER ) {
        this->RenderSpacefillingFilter( mol, posInter);
    } else if( this->currentRenderMode == SAS ) {
        this->RenderSAS( mol, posInter);
    } else if( this->currentRenderMode == LINES_FILTER ) {
        this->RenderLinesFilter(mol, posInter);
    } else if( this->currentRenderMode == STICK_FILTER ) {
        this->RenderStickFilter(mol, posInter);
    }

    glPopMatrix();

    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    // unlock the current frame
    mol->Unlock();

    return true;
}

/*
 * render the atom using lines and points
 */
void SimpleMoleculeRenderer::RenderLines( const MolecularDataCall *mol, const float *atomPos) {
    glDisable( GL_LIGHTING);
    glLineWidth( 2.0f);
    // ----- draw atoms as points -----
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set vertex and color pointers and draw them
    glVertexPointer( 3, GL_FLOAT, 0, atomPos);
    glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays( GL_POINTS, 0, mol->AtomCount());
    // disable sphere shader
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    // ----- draw bonds as lines -----
    unsigned int cnt, atomIdx0, atomIdx1;
    glBegin( GL_LINES);
    for( cnt = 0; cnt < mol->ConnectionCount(); ++cnt ) {
        // get atom indices
        atomIdx0 = mol->Connection()[2*cnt+0];
        atomIdx1 = mol->Connection()[2*cnt+1];
        // set colors and vertices of first atom
        glColor3fv( &this->atomColorTable[atomIdx0*3]);
        glVertex3f( atomPos[atomIdx0*3+0], atomPos[atomIdx0*3+1], atomPos[atomIdx0*3+2]);
        // set colors and vertices of second atom
        glColor3fv( &this->atomColorTable[atomIdx1*3]);
        glVertex3f( atomPos[atomIdx1*3+0], atomPos[atomIdx1*3+1], atomPos[atomIdx1*3+2]);
    }
    glEnd(); // GL_LINES
    glEnable( GL_LIGHTING);
}

/*
 * Render the molecular data in stick mode.
 */
void SimpleMoleculeRenderer::RenderStick( const MolecularDataCall *mol, const float *atomPos) {
    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount( mol->AtomCount() * 4 );
    this->vertCylinders.SetCount( mol->ConnectionCount() * 4);
    this->quatCylinders.SetCount( mol->ConnectionCount() * 4);
    this->inParaCylinders.SetCount( mol->ConnectionCount() * 2);
    this->color1Cylinders.SetCount( mol->ConnectionCount() * 3);
    this->color2Cylinders.SetCount( mol->ConnectionCount() * 3);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        this->vertSpheres[4*cnt+0] = atomPos[3*cnt+0];
        this->vertSpheres[4*cnt+1] = atomPos[3*cnt+1];
        this->vertSpheres[4*cnt+2] = atomPos[3*cnt+2];
        this->vertSpheres[4*cnt+3] =
            this->stickRadiusParam.Param<param::FloatParam>()->Value();
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC( 0, 0, 0, 1);
    vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
#pragma omp parallel for private( idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for( cnt = 0; cnt < int( mol->ConnectionCount()); ++cnt ) {
        idx0 = mol->Connection()[2*cnt];
        idx1 = mol->Connection()[2*cnt+1];

        firstAtomPos.SetX( atomPos[3*idx0+0]);
        firstAtomPos.SetY( atomPos[3*idx0+1]);
        firstAtomPos.SetZ( atomPos[3*idx0+2]);

        secondAtomPos.SetX( atomPos[3*idx1+0]);
        secondAtomPos.SetY( atomPos[3*idx1+1]);
        secondAtomPos.SetZ( atomPos[3*idx1+2]);

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set( 1.0f, 0.0f, 0.0f);
        angle = - tmpVec.Angle( dir);
        ortho = tmpVec.Cross( dir);
        ortho.Normalise();
        quatC.Set( angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir/2.0f);

        this->inParaCylinders[2*cnt] = this->stickRadiusParam.Param<param::FloatParam>()->Value();
		this->inParaCylinders[2*cnt+1] = ( firstAtomPos-secondAtomPos).Length();

		// thomasbm: hotfix for jumping molecules near bounding box
		if(this->inParaCylinders[2*cnt+1] > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() + mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius() ) {
			this->inParaCylinders[2*cnt+1] = 0;
		}

        this->quatCylinders[4*cnt+0] = quatC.GetX();
        this->quatCylinders[4*cnt+1] = quatC.GetY();
        this->quatCylinders[4*cnt+2] = quatC.GetZ();
        this->quatCylinders[4*cnt+3] = quatC.GetW();

        this->color1Cylinders[3*cnt+0] = this->atomColorTable[3*idx0+0];
        this->color1Cylinders[3*cnt+1] = this->atomColorTable[3*idx0+1];
        this->color1Cylinders[3*cnt+2] = this->atomColorTable[3*idx0+2];

        this->color2Cylinders[3*cnt+0] = this->atomColorTable[3*idx1+0];
        this->color2Cylinders[3*cnt+1] = this->atomColorTable[3*idx1+1];
        this->color2Cylinders[3*cnt+2] = this->atomColorTable[3*idx1+2];

        this->vertCylinders[4*cnt+0] = position.X();
        this->vertCylinders[4*cnt+1] = position.Y();
        this->vertCylinders[4*cnt+2] = position.Z();
        this->vertCylinders[4*cnt+3] = 0.0f;
    }

    // ---------- actual rendering ----------

    // get viewpoint parameters for raycasting
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays( GL_POINTS, 0, mol->AtomCount());
    // disable sphere shader
    this->sphereShader.Disable();


    // enable cylinder shader
    this->cylinderShader.Enable();
    // set shader variables
    glUniform4fvARB( this->cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB( this->cylinderShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB( this->cylinderShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB( this->cylinderShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // get the attribute locations
    attribLocInParams = glGetAttribLocationARB( this->cylinderShader, "inParams");
    attribLocQuatC = glGetAttribLocationARB( this->cylinderShader, "quatC");
    attribLocColor1 = glGetAttribLocationARB( this->cylinderShader, "color1");
    attribLocColor2 = glGetAttribLocationARB( this->cylinderShader, "color2");
    // enable vertex attribute arrays for the attribute locations
    glDisableClientState( GL_COLOR_ARRAY);
    glEnableVertexAttribArrayARB( this->attribLocInParams);
    glEnableVertexAttribArrayARB( this->attribLocQuatC);
    glEnableVertexAttribArrayARB( this->attribLocColor1);
    glEnableVertexAttribArrayARB( this->attribLocColor2);
    // set vertex and attribute pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());
    glDrawArrays( GL_POINTS, 0, mol->ConnectionCount());
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArrayARB( this->attribLocInParams);
    glDisableVertexAttribArrayARB( this->attribLocQuatC);
    glDisableVertexAttribArrayARB( this->attribLocColor1);
    glDisableVertexAttribArrayARB( this->attribLocColor2);
    glDisableClientState(GL_VERTEX_ARRAY);
    // disable cylinder shader
    this->cylinderShader.Disable();
}

/*
 * Render the molecular data in ball-and-stick mode.
 */
void SimpleMoleculeRenderer::RenderBallAndStick( const MolecularDataCall *mol, const float *atomPos) {
    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount( mol->AtomCount() * 4 );
    this->vertCylinders.SetCount( mol->ConnectionCount() * 4);
    this->quatCylinders.SetCount( mol->ConnectionCount() * 4);
    this->inParaCylinders.SetCount( mol->ConnectionCount() * 2);
    this->color1Cylinders.SetCount( mol->ConnectionCount() * 3);
    this->color2Cylinders.SetCount( mol->ConnectionCount() * 3);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        this->vertSpheres[4*cnt+0] = atomPos[3*cnt+0];
        this->vertSpheres[4*cnt+1] = atomPos[3*cnt+1];
        this->vertSpheres[4*cnt+2] = atomPos[3*cnt+2];
        this->vertSpheres[4*cnt+3] =
            this->stickRadiusParam.Param<param::FloatParam>()->Value();
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC( 0, 0, 0, 1);
    vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
#pragma omp parallel for private( idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for( cnt = 0; cnt < int( mol->ConnectionCount()); ++cnt ) {
        idx0 = mol->Connection()[2*cnt];
        idx1 = mol->Connection()[2*cnt+1];

        firstAtomPos.SetX( atomPos[3*idx0+0]);
        firstAtomPos.SetY( atomPos[3*idx0+1]);
        firstAtomPos.SetZ( atomPos[3*idx0+2]);

        secondAtomPos.SetX( atomPos[3*idx1+0]);
        secondAtomPos.SetY( atomPos[3*idx1+1]);
        secondAtomPos.SetZ( atomPos[3*idx1+2]);

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set( 1.0f, 0.0f, 0.0f);
        angle = - tmpVec.Angle( dir);
        ortho = tmpVec.Cross( dir);
        ortho.Normalise();
        quatC.Set( angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir/2.0f);

        this->inParaCylinders[2*cnt] = this->stickRadiusParam.Param<param::FloatParam>()->Value() / 3.0f;
        this->inParaCylinders[2*cnt+1] = ( firstAtomPos-secondAtomPos).Length();

		// thomasbm: hotfix for jumping molecules near bounding box
		if(this->inParaCylinders[2*cnt+1] > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() + mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius() ) {
			this->inParaCylinders[2*cnt+1] = 0;
		}


        this->quatCylinders[4*cnt+0] = quatC.GetX();
        this->quatCylinders[4*cnt+1] = quatC.GetY();
        this->quatCylinders[4*cnt+2] = quatC.GetZ();
        this->quatCylinders[4*cnt+3] = quatC.GetW();

        this->color1Cylinders[3*cnt+0] = this->atomColorTable[3*idx0+0];
        this->color1Cylinders[3*cnt+1] = this->atomColorTable[3*idx0+1];
        this->color1Cylinders[3*cnt+2] = this->atomColorTable[3*idx0+2];

        this->color2Cylinders[3*cnt+0] = this->atomColorTable[3*idx1+0];
        this->color2Cylinders[3*cnt+1] = this->atomColorTable[3*idx1+1];
        this->color2Cylinders[3*cnt+2] = this->atomColorTable[3*idx1+2];

        this->vertCylinders[4*cnt+0] = position.X();
        this->vertCylinders[4*cnt+1] = position.Y();
        this->vertCylinders[4*cnt+2] = position.Z();
        this->vertCylinders[4*cnt+3] = 0.0f;
    }

    // ---------- actual rendering ----------

    // get viewpoint parameters for raycasting
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays( GL_POINTS, 0, mol->AtomCount());
    // disable sphere shader
    this->sphereShader.Disable();


    // enable cylinder shader
    this->cylinderShader.Enable();
    // set shader variables
    glUniform4fvARB( this->cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB( this->cylinderShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB( this->cylinderShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB( this->cylinderShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // get the attribute locations
    attribLocInParams = glGetAttribLocationARB( this->cylinderShader, "inParams");
    attribLocQuatC = glGetAttribLocationARB( this->cylinderShader, "quatC");
    attribLocColor1 = glGetAttribLocationARB( this->cylinderShader, "color1");
    attribLocColor2 = glGetAttribLocationARB( this->cylinderShader, "color2");
    // enable vertex attribute arrays for the attribute locations
    glDisableClientState( GL_COLOR_ARRAY);
    glEnableVertexAttribArrayARB( this->attribLocInParams);
    glEnableVertexAttribArrayARB( this->attribLocQuatC);
    glEnableVertexAttribArrayARB( this->attribLocColor1);
    glEnableVertexAttribArrayARB( this->attribLocColor2);
    // set vertex and attribute pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());
    glDrawArrays( GL_POINTS, 0, mol->ConnectionCount());
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArrayARB( this->attribLocInParams);
    glDisableVertexAttribArrayARB( this->attribLocQuatC);
    glDisableVertexAttribArrayARB( this->attribLocColor1);
    glDisableVertexAttribArrayARB( this->attribLocColor2);
    glDisableClientState(GL_VERTEX_ARRAY);
    // disable cylinder shader
    this->cylinderShader.Disable();
}


/*
 * Render the molecular data in spacefilling mode.
 */
void SimpleMoleculeRenderer::RenderSpacefilling( const MolecularDataCall *mol, const float *atomPos) {
    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount( mol->AtomCount() * 4 );

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        this->vertSpheres[4*cnt+0] = atomPos[3*cnt+0];
        this->vertSpheres[4*cnt+1] = atomPos[3*cnt+1];
        this->vertSpheres[4*cnt+2] = atomPos[3*cnt+2];
        this->vertSpheres[4*cnt+3] =
            mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
    }

    // ---------- actual rendering ----------

    // get viewpoint parameters for raycasting
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays( GL_POINTS, 0, mol->AtomCount());
    // disable sphere shader
    this->sphereShader.Disable();
}


/*
 * Render the molecular data in solvent accessible surface mode.
 */
void SimpleMoleculeRenderer::RenderSAS( const MolecularDataCall *mol, const float *atomPos) {
    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount( mol->AtomCount() * 4 );

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        this->vertSpheres[4*cnt+0] = atomPos[3*cnt+0];
        this->vertSpheres[4*cnt+1] = atomPos[3*cnt+1];
        this->vertSpheres[4*cnt+2] = atomPos[3*cnt+2];
        this->vertSpheres[4*cnt+3] =
            mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius() +
            this->probeRadiusParam.Param<param::FloatParam>()->Value();

    }

    // ---------- actual rendering ----------

    // get viewpoint parameters for raycasting
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays( GL_POINTS, 0, mol->AtomCount());
    // disable sphere shader
    this->sphereShader.Disable();
}


/*
 * renderPointsFilter
 *
 * Helper function to test the filter module.
 */
void SimpleMoleculeRenderer::RenderPointsFilter(const MolecularDataCall *mol,
    const float *atomPos) {

    vislib::Array<unsigned int> idx;
    unsigned int i, visAtmCnt = 0;

    idx.SetCapacityIncrement(1000);

    // Get indices of visible atoms
    for(i = 0; i < mol->AtomCount(); i++) {
        if(mol->Filter()[i] == 1) {
            idx.Add(i);
            visAtmCnt++;
        }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, atomPos);
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());

    glDrawElements(GL_POINTS, visAtmCnt, GL_UNSIGNED_INT, idx.PeekElements());

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

}

/*
 * renderLinesFilter
 *
 * Helper function to test the filter module.
 */
void SimpleMoleculeRenderer::RenderLinesFilter(const MolecularDataCall *mol,
    const float *atomPos) {

    vislib::Array<unsigned int> visAtmIdx;
    vislib::Array<unsigned int> visConIdx;

    unsigned int visAtmCnt = 0, visConCnt = 0;
    unsigned int m, at, c;
    unsigned int firstAtmIdx, lastAtmIdx, firstConIdx, lastConIdx;

    visAtmIdx.SetCapacityIncrement(2000);
    visConIdx.SetCapacityIncrement(2000);

   // Loop through all molecules
    for(m = 0; m < mol->MoleculeCount(); m++) {

        // If the molecule is visible
        if(mol->Molecules()[m].Filter() == 1) {

            // Get indices of all atoms in this molecule
            firstAtmIdx =
                mol->Residues()[mol->Molecules()[m].FirstResidueIndex()]->FirstAtomIndex();

            lastAtmIdx =
                mol->Residues()[ mol->Molecules()[m].FirstResidueIndex() +
                    mol->Molecules()[m].ResidueCount() - 1]->FirstAtomIndex()
                + mol->Residues()[mol->Molecules()[m].FirstResidueIndex() +
                    mol->Molecules()[m].ResidueCount() - 1]->AtomCount() - 1;

            for(at = firstAtmIdx; at <= lastAtmIdx; at++)  {
                visAtmIdx.Add(at);
            }

            visAtmCnt += (lastAtmIdx - firstAtmIdx + 1);

            // Get indices of all connections in this molecule
            if(mol->Molecules()[m].ConnectionCount() > 0) {

                firstConIdx =
                    mol->Molecules()[m].FirstConnectionIndex();

                lastConIdx =
                    mol->Molecules()[m].FirstConnectionIndex()
                    + (mol->Molecules()[m].ConnectionCount() - 1)*2;

                for(c = firstConIdx; c <= lastConIdx; c += 2) {

                    visConIdx.Add(mol->Connection()[c]);
                    visConIdx.Add(mol->Connection()[c+1]);
                }
                visConCnt += (lastConIdx - firstConIdx + 2);
            }
        }
    }

    glDisable( GL_LIGHTING);
    glLineWidth( 2.0f);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, atomPos);
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());

    // Draw visible atoms
    glDrawElements(GL_POINTS, visAtmCnt, GL_UNSIGNED_INT, visAtmIdx.PeekElements());
    // Draw vivisble bonds
    glDrawElements(GL_LINES, visConCnt, GL_UNSIGNED_INT, visConIdx.PeekElements());

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    glEnable( GL_LIGHTING);
}


/*
 * Render the molecular data in stick mode.
 */
void SimpleMoleculeRenderer::RenderStickFilter( const MolecularDataCall *mol, const float *atomPos) {

    //int n;
    //glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &n);
    //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
    //  "Maximum num of generic vertex attributes: %i\n", n);

    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount( mol->AtomCount() * 4 );
    this->vertCylinders.SetCount( mol->ConnectionCount() * 4);
    this->quatCylinders.SetCount( mol->ConnectionCount() * 4);
    this->inParaCylinders.SetCount( mol->ConnectionCount() * 2);
    this->color1Cylinders.SetCount( mol->ConnectionCount() * 3);
    this->color2Cylinders.SetCount( mol->ConnectionCount() * 3);
    this->conFilter.SetCount( mol->ConnectionCount());

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        this->vertSpheres[4*cnt+0] = atomPos[3*cnt+0];
        this->vertSpheres[4*cnt+1] = atomPos[3*cnt+1];
        this->vertSpheres[4*cnt+2] = atomPos[3*cnt+2];
        this->vertSpheres[4*cnt+3] =
            this->stickRadiusParam.Param<param::FloatParam>()->Value();
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC( 0, 0, 0, 1);
    vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
#pragma omp parallel for private( idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for( cnt = 0; cnt < int( mol->ConnectionCount()); ++cnt ) {
        idx0 = mol->Connection()[2*cnt];
        idx1 = mol->Connection()[2*cnt+1];

        firstAtomPos.SetX( atomPos[3*idx0+0]);
        firstAtomPos.SetY( atomPos[3*idx0+1]);
        firstAtomPos.SetZ( atomPos[3*idx0+2]);

        secondAtomPos.SetX( atomPos[3*idx1+0]);
        secondAtomPos.SetY( atomPos[3*idx1+1]);
        secondAtomPos.SetZ( atomPos[3*idx1+2]);

        // Set filter information for this connection
        if((mol->Filter()[idx0] == 1) && (mol->Filter()[idx1] == 1))
            this->conFilter[cnt] = 1.0f;
        else
            this->conFilter[cnt] = 0.0f;

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set( 1.0f, 0.0f, 0.0f);
        angle = - tmpVec.Angle( dir);
        ortho = tmpVec.Cross( dir);
        ortho.Normalise();
        quatC.Set( angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir/2.0f);

        this->inParaCylinders[2*cnt] = this->stickRadiusParam.Param<param::FloatParam>()->Value();
		this->inParaCylinders[2*cnt+1] = ( firstAtomPos-secondAtomPos).Length();

		// thomasbm: hotfix for jumping molecules near bounding box
		if(this->inParaCylinders[2*cnt+1] > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius()
            + mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius() ) {
			this->inParaCylinders[2*cnt+1] = 0;
		}

        this->quatCylinders[4*cnt+0] = quatC.GetX();
        this->quatCylinders[4*cnt+1] = quatC.GetY();
        this->quatCylinders[4*cnt+2] = quatC.GetZ();
        this->quatCylinders[4*cnt+3] = quatC.GetW();

        this->color1Cylinders[3*cnt+0] = this->atomColorTable[3*idx0+0];
        this->color1Cylinders[3*cnt+1] = this->atomColorTable[3*idx0+1];
        this->color1Cylinders[3*cnt+2] = this->atomColorTable[3*idx0+2];

        this->color2Cylinders[3*cnt+0] = this->atomColorTable[3*idx1+0];
        this->color2Cylinders[3*cnt+1] = this->atomColorTable[3*idx1+1];
        this->color2Cylinders[3*cnt+2] = this->atomColorTable[3*idx1+2];

        this->vertCylinders[4*cnt+0] = position.X();
        this->vertCylinders[4*cnt+1] = position.Y();
        this->vertCylinders[4*cnt+2] = position.Z();
        this->vertCylinders[4*cnt+3] = 0.0f;
    }

    // Set filter information of connections according to molecules
    /*
    unsigned int c, m, firstConIdx, lastConIdx;

    for(m = 0; m < mol->MoleculeCount(); m++) {
        if(mol->Molecules()[m].ConnectionCount() > 0) {

            firstConIdx =
                mol->Molecules()[m].FirstConnectionIndex();

            lastConIdx =
                mol->Molecules()[m].FirstConnectionIndex()
                + mol->Molecules()[m].ConnectionCount() - 1;

            if(mol->Molecules()[m].Filter() == 1) {
                for(c = firstConIdx; c <= lastConIdx; c ++) {
                    //conFilter[c] = 1.0;
                    conFilter.Add(1.0);
                }
            }
            else {
                for(c = firstConIdx; c <= lastConIdx; c ++) {
                    conFilter.Add(0.0);
                    //conFilter[c] = 0.0;
                }
            }
        }
    }
    */

    // ---------- actual rendering ----------

    // get viewpoint parameters for raycasting
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->filterSphereShader.Enable();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // set shader variables
    glUniform4fvARB(this->filterSphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->filterSphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->filterSphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->filterSphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

    // Set filter attribute
    this->attribLocAtomFilter = glGetAttribLocationARB(this->filterSphereShader.ProgramHandle(), "filter");
    glEnableVertexAttribArrayARB(this->attribLocAtomFilter);

    // Set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements());

    // Set attribute pointer
    glVertexAttribPointerARB(this->attribLocAtomFilter, 1, GL_INT, 0, 0, mol->Filter());

    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    glDisableVertexAttribArrayARB(this->attribLocAtomFilter);

    // disable sphere shader
    this->filterSphereShader.Disable();

    // enable cylinder shader
    this->filterCylinderShader.Enable();
    // set shader variables
    glUniform4fvARB( this->filterCylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB( this->filterCylinderShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB( this->filterCylinderShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB( this->filterCylinderShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

    // get the attribute locations
    attribLocInParams = glGetAttribLocationARB( this->filterCylinderShader, "inParams");
    attribLocQuatC = glGetAttribLocationARB( this->filterCylinderShader, "quatC");
    attribLocColor1 = glGetAttribLocationARB( this->filterCylinderShader, "color1");
    attribLocColor2 = glGetAttribLocationARB( this->filterCylinderShader, "color2");
    this->attribLocConFilter = glGetAttribLocationARB( this->filterCylinderShader, "filter");
    // enable vertex attribute arrays for the attribute locations
    glDisableClientState( GL_COLOR_ARRAY);
    glEnableVertexAttribArrayARB( this->attribLocInParams);
    glEnableVertexAttribArrayARB( this->attribLocQuatC);
    glEnableVertexAttribArrayARB( this->attribLocColor1);
    glEnableVertexAttribArrayARB( this->attribLocColor2);
    glEnableVertexAttribArrayARB( this->attribLocConFilter);
    // set vertex and attribute pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());
    glVertexAttribPointerARB( this->attribLocConFilter, 1, GL_FLOAT, 0, 0, this->conFilter.PeekElements());

    glDrawArrays( GL_POINTS, 0, mol->ConnectionCount());
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArrayARB( this->attribLocInParams);
    glDisableVertexAttribArrayARB( this->attribLocQuatC);
    glDisableVertexAttribArrayARB( this->attribLocColor1);
    glDisableVertexAttribArrayARB( this->attribLocColor2);
    glDisableVertexAttribArrayARB( this->attribLocConFilter);
    glDisableClientState(GL_VERTEX_ARRAY);
    // disable cylinder shader
    this->filterCylinderShader.Disable();

   /* GLenum errCode;
    const GLubyte *errString;

    if ((errCode = glGetError()) != GL_NO_ERROR) {
        errString = gluErrorString(errCode);
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "OpenGL Error: %s\n", errString);
        //fprintf (stderr, "OpenGL Error: %s\n", errString);
    }*/
}

/*
 * Render the molecular data in stick mode.
 */
void SimpleMoleculeRenderer::RenderSpacefillingFilter( const MolecularDataCall *mol, const float *atomPos) {

    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount( mol->AtomCount() * 4 );

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        this->vertSpheres[4*cnt+0] = atomPos[3*cnt+0];
        this->vertSpheres[4*cnt+1] = atomPos[3*cnt+1];
        this->vertSpheres[4*cnt+2] = atomPos[3*cnt+2];
        this->vertSpheres[4*cnt+3] =
            mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
    }

    // ---------- actual rendering ----------

    // get viewpoint parameters for raycasting
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->filterSphereShader.Enable();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // set shader variables
    glUniform4fvARB(this->filterSphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->filterSphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->filterSphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->filterSphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

    // Set filter attribute
    this->attribLocAtomFilter = glGetAttribLocationARB(this->filterSphereShader.ProgramHandle(), "filter");
    glEnableVertexAttribArrayARB(this->attribLocAtomFilter);

    // Set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements());

    // Set attribute pointer
    glVertexAttribPointerARB(this->attribLocAtomFilter, 1, GL_INT, 0, 0, mol->Filter());

    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    glDisableVertexAttribArrayARB(this->attribLocAtomFilter);

    // disable sphere shader
    this->filterSphereShader.Disable();

}


/*
 * update parameters
 */
void SimpleMoleculeRenderer::UpdateParameters( const MolecularDataCall *mol) {
    // color table param
    if( this->colorTableFileParam.IsDirty() ) {
        Color::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::StringParam>()->Value(),
            this->colorLookupTable);
        this->colorTableFileParam.ResetDirty();
    }
    // Recompute color table
    if((this->coloringModeParam0.IsDirty())
            ||(this->coloringModeParam1.IsDirty())
            ||(this->cmWeightParam.IsDirty())) {

        this->currentColoringMode0 = static_cast<Color::ColoringMode>(int(
            this->coloringModeParam0.Param<param::EnumParam>()->Value()));

        this->currentColoringMode1 = static_cast<Color::ColoringMode>(int(
            this->coloringModeParam1.Param<param::EnumParam>()->Value()));

        // Mix two coloring modes
        Color::MakeColorTable( mol,
          this->currentColoringMode0,
          this->currentColoringMode1,
          cmWeightParam.Param<param::FloatParam>()->Value(),       // weight for the first cm
          1.0 - cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
          this->atomColorTable, this->colorLookupTable, this->rainbowColors,
          this->minGradColorParam.Param<param::StringParam>()->Value(),
          this->midGradColorParam.Param<param::StringParam>()->Value(),
          this->maxGradColorParam.Param<param::StringParam>()->Value(),
          true);

        // Use one coloring mode
        /*Color::MakeColorTable( mol,
          this->currentColoringMode0,
          this->atomColorTable, this->colorLookupTable, this->rainbowColors,
          this->minGradColorParam.Param<param::StringParam>()->Value(),
          this->midGradColorParam.Param<param::StringParam>()->Value(),
          this->maxGradColorParam.Param<param::StringParam>()->Value(),
          true);*/

        this->coloringModeParam0.ResetDirty();
        this->coloringModeParam1.ResetDirty();
        this->cmWeightParam.ResetDirty();
    }
    // rendering mode param
    if( this->renderModeParam.IsDirty() ) {
        this->currentRenderMode = static_cast<RenderMode>( int(
            this->renderModeParam.Param<param::EnumParam>()->Value() ) );
    }
    // get molecule lust
    if( this->molIdxListParam.IsDirty() ) {
        vislib::StringA tmpStr( this->molIdxListParam.Param<param::StringParam>()->Value());
        this->molIdxList = vislib::StringTokeniser<vislib::CharTraitsA>::Split( tmpStr, ';', true);
        this->molIdxListParam.ResetDirty();
    }
}

