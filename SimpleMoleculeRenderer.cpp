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
#include "utility/ShaderSourceFactory.h"
#include "utility/ColourParser.h"
#include "param/StringParam.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/ShaderSource.h"
#include "vislib/AbstractOpenGLShader.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/stringconverter.h"
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
    molRendererCallerSlot( "renderMolecule", "Connects the molecule rendering with another renderer" ),
    colorTableFileParam( "colorTableFilename", "The filename of the color table."),
    coloringModeParam( "coloringMode", "The coloring mode."),
    renderModeParam( "renderMode", "The rendering mode."),
    stickRadiusParam( "stickRadius", "The radius for stick rendering"),
    probeRadiusParam( "probeRadius", "The probe radius for SAS rendering"),
    minGradColorParam( "minGradColor", "The color for the minimum value for gradient coloring" ),
    midGradColorParam( "midGradColor", "The color for the middle value for gradient coloring" ),
    maxGradColorParam( "maxGradColor", "The color for the maximum value for gradient coloring" ),
    molIdxListParam( "molIdxList", "The list of molecule indices for RS computation:"),
    specialColorParam( "specialColor", "The color for the specified molecules" )
{
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);

    this->molRendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable( &this->molRendererCallerSlot);

    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    this->ReadColorTableFromFile( filename);
    this->colorTableFileParam.SetParameter(new param::StringParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);

    // coloring mode
    //this->currentColoringMode = ELEMENT;
    this->currentColoringMode = RESIDUE;
    param::EnumParam *cm = new param::EnumParam(int(this->currentColoringMode));
    cm->SetTypePair( ELEMENT, "Element");
    cm->SetTypePair( RESIDUE, "Residue");
    cm->SetTypePair( STRUCTURE, "Structure");
    cm->SetTypePair( BFACTOR, "BFactor");
    cm->SetTypePair( CHARGE, "Charge");
    cm->SetTypePair( OCCUPANCY, "Occupancy");
    cm->SetTypePair( CHAIN, "Chain");
    cm->SetTypePair( MOLECULE, "Molecule");
    cm->SetTypePair( RAINBOW, "Rainbow");
    this->coloringModeParam << cm;
    this->MakeSlotAvailable( &this->coloringModeParam);
    
    // rendering mode
    this->currentRenderMode = LINES;
    //this->currentRenderMode = STICK;
    //this->currentRenderMode = BALL_AND_STICK;
    //this->currentRenderMode = SPACEFILLING;
    //this->currentRenderMode = SAS;
    param::EnumParam *rm = new param::EnumParam(int(this->currentRenderMode));
    rm->SetTypePair( LINES, "Lines");
    rm->SetTypePair( STICK, "Stick");
    rm->SetTypePair( BALL_AND_STICK, "Ball-and-Stick");
    rm->SetTypePair( SPACEFILLING, "Spacefilling");
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
    this->molIdxList.Add( "0");
    this->molIdxListParam.SetParameter(new param::StringParam( "0"));
    this->MakeSlotAvailable( &this->molIdxListParam);

    // the color for the maximum value (gradient coloring
    this->specialColorParam.SetParameter(new param::StringParam( "#228B22"));
    this->MakeSlotAvailable( &this->specialColorParam);

    // make the rainbow color table
    this->MakeRainbowColorTable( 100);
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

    // get the pointer to CallRender3D (molecule renderer)
    view::CallRender3D *molcr3d = this->molRendererCallerSlot.CallAs<view::CallRender3D>();
    if( molcr3d ) {
        (*molcr3d)(MolecularDataCall::CallForGetExtent); // GetExtents
    }

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

    // set frame ID and call data
    mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;
    // check if atom count is zero
    if( mol->AtomCount() == 0 ) return true;
    // get positions of the first frame
    float *pos0 = new float[mol->AtomCount() * 3];
    memcpy( pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
    // set next frame ID and get positions of the second frame
    if( ( static_cast<int>( callTime) + 1) < mol->FrameCount() ) 
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
    for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
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

    // =============== Molecule Rendering ===============
    // get the pointer to CallRender3D
    view::CallRender3D *molcr3d = this->molRendererCallerSlot.CallAs<view::CallRender3D>();
    if( molcr3d ) {
        // setup and call molecule renderer
        glPushMatrix();
        //glTranslatef( this->protrenTranslate.X(), this->protrenTranslate.Y(), this->protrenTranslate.Z());
        //glScalef( this->protrenScale, this->protrenScale, this->protrenScale);
        *molcr3d = *cr3d;
        (*molcr3d)();
        glPopMatrix();
	}

    // ---------- update parameters ----------
    this->UpdateParameters( mol);

    // recompute color table, if necessary
    if( this->atomColorTable.Count() < mol->AtomCount() ) {
        this->MakeColorTable( mol, true);
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

    glPushMatrix();

    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);

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
    } else if( this->currentRenderMode == SAS ) {
        this->RenderSAS( mol, posInter);
    }

    //glDisable(GL_DEPTH_TEST);

    glPopMatrix();

    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    return true;
}

/*
 * render the atom using lines and points
 */
void SimpleMoleculeRenderer::RenderLines( const MolecularDataCall *mol, const float *atomPos) {
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
    for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
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
    for( cnt = 0; cnt < mol->ConnectionCount(); ++cnt ) {
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
    for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
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
    for( cnt = 0; cnt < mol->ConnectionCount(); ++cnt ) {
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
    for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
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
    for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
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
 * update parameters
 */
void SimpleMoleculeRenderer::UpdateParameters( const MolecularDataCall *mol) {
    // color table param
    if( this->colorTableFileParam.IsDirty() ) {
        this->ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::StringParam>()->Value());
        this->colorTableFileParam.ResetDirty();
    }
    // coloring mode param
    if( this->coloringModeParam.IsDirty() ) {
        this->currentColoringMode = static_cast<ColoringMode>( int(
            this->coloringModeParam.Param<param::EnumParam>()->Value() ) );
        this->MakeColorTable( mol, true);
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

/*
 * Read color table from file
 */
void SimpleMoleculeRenderer::ReadColorTableFromFile( vislib::StringA filename) {
    // file buffer variable
    vislib::sys::ASCIIFileBuffer file;
    // delete old color table
    this->colorLookupTable.SetCount( 0);
    // try to load the color table file
    if( file.LoadFile( filename) ) {
        float r, g, b;
        this->colorLookupTable.AssertCapacity( file.Count());
        // get colors from file
        for( unsigned int cnt = 0; cnt < file.Count(); ++cnt ) {
            if( utility::ColourParser::FromString( file.Line( cnt), r, g, b) ) {
                this->colorLookupTable.Add( vislib::math::Vector<float, 3>( r, g, b));
            }
        }
    }
    // if the file could not be loaded or contained no valid colors
    if( this->colorLookupTable.Count() == 0 ) {
        // set default color table
        this->colorLookupTable.SetCount( 25);
        this->colorLookupTable[0].Set( 0.5f, 0.5f, 0.5f);
        this->colorLookupTable[1].Set( 1.0f, 0.0f, 0.0f);
        this->colorLookupTable[2].Set( 1.0f, 1.0f, 0.0f);
        this->colorLookupTable[3].Set( 0.0f, 1.0f, 0.0f);
        this->colorLookupTable[4].Set( 0.0f, 1.0f, 1.0f);
        this->colorLookupTable[5].Set( 0.0f, 0.0f, 1.0f);
        this->colorLookupTable[6].Set( 1.0f, 0.0f, 1.0f);
        this->colorLookupTable[7].Set( 0.5f, 0.0f, 0.0f);
        this->colorLookupTable[8].Set( 0.5f, 0.5f, 0.0f);
        this->colorLookupTable[9].Set( 0.0f, 0.5f, 0.0f);
        this->colorLookupTable[10].Set( 0.00f, 0.50f, 0.50f);
        this->colorLookupTable[11].Set( 0.00f, 0.00f, 0.50f);
        this->colorLookupTable[12].Set( 0.50f, 0.00f, 0.50f);
        this->colorLookupTable[13].Set( 1.00f, 0.50f, 0.00f);
        this->colorLookupTable[14].Set( 0.00f, 0.50f, 1.00f);
        this->colorLookupTable[15].Set( 1.00f, 0.50f, 1.00f);
        this->colorLookupTable[16].Set( 0.50f, 0.25f, 0.00f);
        this->colorLookupTable[17].Set( 1.00f, 1.00f, 0.50f);
        this->colorLookupTable[18].Set( 0.50f, 1.00f, 0.50f);
        this->colorLookupTable[19].Set( 0.75f, 1.00f, 0.00f);
        this->colorLookupTable[20].Set( 0.50f, 0.00f, 0.75f);
        this->colorLookupTable[21].Set( 1.00f, 0.50f, 0.50f);
        this->colorLookupTable[22].Set( 0.75f, 1.00f, 0.75f);
        this->colorLookupTable[23].Set( 0.75f, 0.75f, 0.50f);
        this->colorLookupTable[24].Set( 1.00f, 0.75f, 0.50f);
    }
}


/*
 * protein::ProteinRenderer::MakeColorTable
 */
void SimpleMoleculeRenderer::MakeColorTable( const MolecularDataCall *mol, bool forceRecompute) {
    // temporary variables
    unsigned int cnt, idx, cntAtom, cntRes, cntChain, cntMol, cntSecS, atomIdx, atomCnt;
    vislib::math::Vector<float, 3> color;
    float r, g, b;

    // if recomputation is forced: clear current color table
    if( forceRecompute ) {
        this->atomColorTable.Clear();
    }
    // reserve memory for all atoms
    this->atomColorTable.AssertCapacity( mol->AtomCount() * 3 );

    // only compute color table if necessary
    if( this->atomColorTable.IsEmpty() ) {
        if( this->currentColoringMode == ELEMENT ) {
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                this->atomColorTable.Add( float( mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[0]) / 255.0f );
                this->atomColorTable.Add( float( mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[1]) / 255.0f );
                this->atomColorTable.Add( float( mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[2]) / 255.0f );
            }
        } // ... END coloring mode ELEMENT
        else if( this->currentColoringMode == RESIDUE ) {
            unsigned int resTypeIdx;
            // loop over all residues
            for( cntRes = 0; cntRes < mol->ResidueCount(); ++cntRes ) {
                // loop over all atoms of the current residue
                idx = mol->Residues()[cntRes]->FirstAtomIndex();
                cnt = mol->Residues()[cntRes]->AtomCount();
                // get residue type index
                resTypeIdx = mol->Residues()[cntRes]->Type();
                for( cntAtom = idx; cntAtom < idx + cnt; ++cntAtom ) {
                    this->atomColorTable.Add( this->colorLookupTable[resTypeIdx%this->colorLookupTable.Count()].X());
                    this->atomColorTable.Add( this->colorLookupTable[resTypeIdx%this->colorLookupTable.Count()].Y());
                    this->atomColorTable.Add( this->colorLookupTable[resTypeIdx%this->colorLookupTable.Count()].Z());
                }
            }
        } // ... END coloring mode RESIDUE
        else if( this->currentColoringMode == STRUCTURE ) {
            utility::ColourParser::FromString( "#00ff00", r, g, b);
            vislib::math::Vector<float, 3> colNone( r, g, b);
            utility::ColourParser::FromString( "#ff0000", r, g, b);
            vislib::math::Vector<float, 3> colHelix( r, g, b);
            utility::ColourParser::FromString( "#0000ff", r, g, b);
            vislib::math::Vector<float, 3> colSheet( r, g, b);
            utility::ColourParser::FromString( "#ffffff", r, g, b);
            vislib::math::Vector<float, 3> colRCoil( r, g, b);
            // loop over all atoms and fill the table with the default color
            for( cntAtom = 0; cntAtom < mol->AtomCount(); ++cntAtom ) {
                this->atomColorTable.Add( colNone.X());
                this->atomColorTable.Add( colNone.Y());
                this->atomColorTable.Add( colNone.Z());
            }
            // write colors for sec structure elements
            MolecularDataCall::SecStructure::ElementType elemType;
            for( cntSecS = 0; cntSecS < mol->SecondaryStructureCount(); ++cntSecS ) {
                idx = mol->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
                cnt = idx + mol->SecondaryStructures()[cntSecS].AminoAcidCount();
                elemType = mol->SecondaryStructures()[cntSecS].Type();
                for( cntRes = idx; cntRes < cnt; ++cntRes ) {
                    atomIdx = mol->Residues()[cntRes]->FirstAtomIndex();
                    atomCnt = atomIdx + mol->Residues()[cntRes]->AtomCount();
                    for( cntAtom = atomIdx; cntAtom < atomCnt; ++cntAtom ) {
                        if( elemType == MolecularDataCall::SecStructure::TYPE_HELIX ) {
                            this->atomColorTable[3*cntAtom+0] = colHelix.X();
                            this->atomColorTable[3*cntAtom+1] = colHelix.Y();
                            this->atomColorTable[3*cntAtom+2] = colHelix.Z();
                        } else if( elemType == MolecularDataCall::SecStructure::TYPE_SHEET ) {
                            this->atomColorTable[3*cntAtom+0] = colSheet.X();
                            this->atomColorTable[3*cntAtom+1] = colSheet.Y();
                            this->atomColorTable[3*cntAtom+2] = colSheet.Z();
                        } else if( elemType == MolecularDataCall::SecStructure::TYPE_COIL ) {
                            this->atomColorTable[3*cntAtom+0] = colRCoil.X();
                            this->atomColorTable[3*cntAtom+1] = colRCoil.Y();
                            this->atomColorTable[3*cntAtom+2] = colRCoil.Z();
                        }
                    }
                }
            }
        } // ... END coloring mode STRUCTURE
        else if( this->currentColoringMode == BFACTOR ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString( 
                this->minGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString( 
                this->midGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString( 
                this->maxGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min( mol->MinimumBFactor());
            float max( mol->MaximumBFactor());
            float mid( ( max - min)/2.0f + min );
            float val;
            
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    this->atomColorTable.Add( colMid.GetX() );
                    this->atomColorTable.Add( colMid.GetY() );
                    this->atomColorTable.Add( colMid.GetZ() );
                    continue;
                }
                
                val = mol->AtomBFactors()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( val - min );
                    this->atomColorTable.Add( col.GetX() );
                    this->atomColorTable.Add( col.GetY() );
                    this->atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( val - mid );
                    this->atomColorTable.Add( col.GetX() );
                    this->atomColorTable.Add( col.GetY() );
                    this->atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    this->atomColorTable.Add( colMid.GetX() );
                    this->atomColorTable.Add( colMid.GetY() );
                    this->atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode BFACTOR
        else if( this->currentColoringMode == CHARGE ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString( 
                this->minGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString( 
                this->midGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString( 
                this->maxGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min( mol->MinimumCharge());
            float max( mol->MaximumCharge());
            float mid( ( max - min)/2.0f + min );
            float val;
            
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    this->atomColorTable.Add( colMid.GetX() );
                    this->atomColorTable.Add( colMid.GetY() );
                    this->atomColorTable.Add( colMid.GetZ() );
                    continue;
                }
                
                val = mol->AtomCharges()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( val - min );
                    this->atomColorTable.Add( col.GetX() );
                    this->atomColorTable.Add( col.GetY() );
                    this->atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( val - mid );
                    this->atomColorTable.Add( col.GetX() );
                    this->atomColorTable.Add( col.GetY() );
                    this->atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    this->atomColorTable.Add( colMid.GetX() );
                    this->atomColorTable.Add( colMid.GetY() );
                    this->atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode CHARGE
        else if( this->currentColoringMode == OCCUPANCY ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString( 
                this->minGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString( 
                this->midGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString( 
                this->maxGradColorParam.Param<param::StringParam>()->Value(),
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min( mol->MinimumOccupancy());
            float max( mol->MaximumOccupancy());
            float mid( ( max - min)/2.0f + min );
            float val;
            
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    this->atomColorTable.Add( colMid.GetX() );
                    this->atomColorTable.Add( colMid.GetY() );
                    this->atomColorTable.Add( colMid.GetZ() );
                    continue;
                }
                
                val = mol->AtomOccupancies()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( val - min );
                    this->atomColorTable.Add( col.GetX() );
                    this->atomColorTable.Add( col.GetY() );
                    this->atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( val - mid );
                    this->atomColorTable.Add( col.GetX() );
                    this->atomColorTable.Add( col.GetY() );
                    this->atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    this->atomColorTable.Add( colMid.GetX() );
                    this->atomColorTable.Add( colMid.GetY() );
                    this->atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode OCCUPANCY
        else if( this->currentColoringMode == CHAIN ) {
            // get the last atom of the last res of the last mol of the first chain
            cntChain = 0;
            cntMol = mol->Chains()[cntChain].MoleculeCount() - 1;
            cntRes = mol->Molecules()[cntMol].FirstResidueIndex() + mol->Molecules()[cntMol].ResidueCount() - 1;
            cntAtom = mol->Residues()[cntRes]->FirstAtomIndex() + mol->Residues()[cntRes]->AtomCount() - 1;
            // get the first color
            idx = 0;
            color = this->colorLookupTable[idx%this->colorLookupTable.Count()];
            // loop over all atoms
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                // check, if the last atom of the current chain is reached
                if( cnt > cntAtom ) {
                    // get the last atom of the last res of the last mol of the next chain
                    cntChain++;
                    cntMol = mol->Chains()[cntChain].FirstMoleculeIndex() + mol->Chains()[cntChain].MoleculeCount() - 1;
                    cntRes = mol->Molecules()[cntMol].FirstResidueIndex() + mol->Molecules()[cntMol].ResidueCount() - 1;
                    cntAtom = mol->Residues()[cntRes]->FirstAtomIndex() + mol->Residues()[cntRes]->AtomCount() - 1;
                    // get the next color
                    idx++;
                    color = this->colorLookupTable[idx%this->colorLookupTable.Count()];
                    
                }
                this->atomColorTable.Add( color.X());
                this->atomColorTable.Add( color.Y());
                this->atomColorTable.Add( color.Z());
            }
        } // ... END coloring mode CHAIN
        else if( this->currentColoringMode == MOLECULE ) {
            // get the last atom of the last res of the first mol
            cntMol = 0;
            cntRes = mol->Molecules()[cntMol].FirstResidueIndex() + mol->Molecules()[cntMol].ResidueCount() - 1;
            cntAtom = mol->Residues()[cntRes]->FirstAtomIndex() + mol->Residues()[cntRes]->AtomCount() - 1;
            // get the first color
            idx = 0;
            color = this->colorLookupTable[idx%this->colorLookupTable.Count()];
            // loop over all atoms
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                // check, if the last atom of the current chain is reached
                if( cnt > cntAtom ) {
                    // get the last atom of the last res of the next mol
                    cntMol++;
                    cntRes = mol->Molecules()[cntMol].FirstResidueIndex() + mol->Molecules()[cntMol].ResidueCount() - 1;
                    cntAtom = mol->Residues()[cntRes]->FirstAtomIndex() + mol->Residues()[cntRes]->AtomCount() - 1;
                    // get the next color
                    idx++;
                    color = this->colorLookupTable[idx%this->colorLookupTable.Count()];
                    
                }
                this->atomColorTable.Add( color.X());
                this->atomColorTable.Add( color.Y());
                this->atomColorTable.Add( color.Z());
            }
        } // ... END coloring mode MOLECULE
        else if( this->currentColoringMode == RAINBOW ) {
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                idx = int( ( float( cnt) / float( mol->AtomCount())) * float( rainbowColors.Count()));
                color = this->rainbowColors[idx];
                this->atomColorTable.Add( color.GetX());
                this->atomColorTable.Add( color.GetY());
                this->atomColorTable.Add( color.GetZ());
            }
        } // ... END coloring mode RAINBOW
    }
}


/*
 * Creates a rainbow color table with 'num' entries.
 */
void SimpleMoleculeRenderer::MakeRainbowColorTable( unsigned int num) {
    unsigned int n = (num/4);
    // the color table should have a minimum size of 16
    if( n < 4 )
        n = 4;
    this->rainbowColors.Clear();
    this->rainbowColors.AssertCapacity( num);
    float f = 1.0f/float(n);
    vislib::math::Vector<float,3> color;
    color.Set( 1.0f, 0.0f, 0.0f);
    for( unsigned int i = 0; i < n; i++) {
        color.SetY( vislib::math::Min( color.GetY() + f, 1.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetX( vislib::math::Max( color.GetX() - f, 0.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetZ( vislib::math::Min( color.GetZ() + f, 1.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetY( vislib::math::Max( color.GetY() - f, 0.0f));
        rainbowColors.Add( color);
    }
}

