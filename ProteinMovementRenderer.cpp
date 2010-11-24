/*
 * ProteinMovementRenderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "ProteinMovementRenderer.h"
#include "CoreInstance.h"
#include "Color.h"
#include "param/EnumParam.h"
#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "utility/ShaderSourceFactory.h"
#include "vislib/assert.h"
#include "vislib/File.h"
#include "vislib/String.h"
#include "vislib/Point.h"
#include "vislib/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/ShaderSource.h"
#include "vislib/AbstractOpenGLShader.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <glh/glh_genext.h>
#include <math.h>
#include <time.h>

using namespace megamol;
using namespace megamol::core;

/*
 * protein::ProteinMovementRenderer::ProteinMovementRenderer (CTOR)
 */
protein::ProteinMovementRenderer::ProteinMovementRenderer ( void ) : Renderer3DModule (),
        protDataCallerSlot ( "getData", "Connects the protein rendering with protein data storage" ),
        renderingModeParam ( "renderingMode", "Rendering Mode" ),
        coloringModeParam ( "coloringMode", "Coloring Mode" ),
        stickRadiusParam ( "stickRadius", "Stick Radius for spheres and sticks with STICK_ render modes" ),
        arrowRadiusParam ( "arrowRadius", "Radius for movement arrows" ),
        arrowScaleParam ( "arrowScale", "Scale factor for movement arrows" ),
        arrowMinLengthParam ( "arrowMinLength", "Minimum length for movement arrows" ),
        arrowRadiusScaleParam ( "arrowRadiusScale", "Scale factor for radius of movement arrows" ),
        arrowScaleLogParam( "arrowScaleLog", "Scale factor for logarithmic component" ),
        arrowBaseLengthParam( "arrowBaseLength", "Base length of all arrows" ),
        arrowColoringModeParam( "arrowColoringMode", "Coloring Mode for movement arrows"),
        atomCount( 0 )
{
    this->protDataCallerSlot.SetCompatibleCall<CallProteinMovementDataDescription>();
    this->MakeSlotAvailable ( &this->protDataCallerSlot );

    // --- set the coloring mode ---

    this->SetColoringMode ( Color::ELEMENT );
    //this->SetColoringMode(AMINOACID);
    //this->SetColoringMode(STRUCTURE);
    //this->SetColoringMode(VALUE);
    //this->SetColoringMode(CHAIN_ID);
    //this->SetColoringMode(RAINBOW);
	param::EnumParam *cm = new param::EnumParam ( int ( this->currentColoringMode ) );

    cm->SetTypePair ( Color::ELEMENT, "Element" );
    cm->SetTypePair ( Color::AMINOACID, "AminoAcid" );
    cm->SetTypePair ( Color::STRUCTURE, "SecondaryStructure" );
    cm->SetTypePair ( Color::CHAIN_ID, "ChainID" );
    cm->SetTypePair( Color::RAINBOW, "Rainbow");
    cm->SetTypePair( Color::MOVEMENT, "MovementDistance");

    this->coloringModeParam << cm;

    // --- set the render mode ---

    //SetRenderMode ( LINES );
    SetRenderMode(STICK_RAYCASTING);
    //SetRenderMode(BALL_AND_STICK);
    param::EnumParam *rm = new param::EnumParam ( int ( this->currentRenderMode ) );

    rm->SetTypePair ( LINES, "Lines" );
    rm->SetTypePair ( STICK_RAYCASTING, "StickRaycasting" );
    rm->SetTypePair ( BALL_AND_STICK, "BallAndStick" );

    this->renderingModeParam << rm;

    // --- set the radius for the stick rendering mode ---
    this->radiusStick = 0.3f;
    this->stickRadiusParam.SetParameter ( new param::FloatParam ( this->radiusStick, 0.0f ) );

    // --- set the radius for the arrows ---
    this->radiusArrow = 0.3f;
    this->arrowRadiusParam.SetParameter ( new param::FloatParam ( this->radiusArrow, 0.0f ) );

    // --- set the scale factor for the arrows ---
    this->scaleArrow = 1.0f;
    this->arrowScaleParam.SetParameter ( new param::FloatParam ( this->scaleArrow, 0.0f ) );

    // --- set the minimum length for the arrows ---
    this->minLenArrow = 1.0f;
    this->arrowMinLengthParam.SetParameter ( new param::FloatParam ( this->minLenArrow, 0.0f ) );

    // --- set the scale factor for the arrow radius ---
    this->scaleRadiusArrow = 1.0f;
    this->arrowRadiusScaleParam.SetParameter ( new param::FloatParam ( this->scaleRadiusArrow, 0.0f ) );

    // --- set the scale factor for the logarithmic component of the arrow ---
    this->scaleLogArrow = 1.0f;
    this->arrowScaleLogParam.SetParameter ( new param::FloatParam ( this->scaleLogArrow, 0.0f ) );

    // --- set the base lenght of the arrow ---
    this->baseLengthArrow = 0.5f;
    this->arrowBaseLengthParam.SetParameter ( new param::FloatParam ( this->baseLengthArrow, 0.0f ) );

    // --- set the render mode ---

    this->arrowColorMode = UNIFORM_COLOR;
	param::EnumParam *acm = new param::EnumParam ( int ( this->currentRenderMode ) );
    acm->SetTypePair ( PROTEIN_COLOR, "ProteinColor" );
    acm->SetTypePair ( UNIFORM_COLOR, "UniformColor" );
    acm->SetTypePair ( DISTANCE, "Distance" );
    this->arrowColoringModeParam << acm;

    this->MakeSlotAvailable( &this->coloringModeParam );
    this->MakeSlotAvailable( &this->renderingModeParam );
    this->MakeSlotAvailable( &this->stickRadiusParam );
    this->MakeSlotAvailable( &this->arrowRadiusParam );
    this->MakeSlotAvailable( &this->arrowScaleParam );
    this->MakeSlotAvailable( &this->arrowMinLengthParam );
    this->MakeSlotAvailable( &this->arrowRadiusScaleParam );
    this->MakeSlotAvailable( &this->arrowScaleLogParam );
    this->MakeSlotAvailable( &this->arrowBaseLengthParam );
    this->MakeSlotAvailable( &this->arrowColoringModeParam );

    // set empty display list to zero
    this->proteinDisplayListLines = 0;
    // STICK_RAYCASTING render mode was not prepared yet
    this->prepareStickRaycasting = true;
    // BALL_AND_STICK render mode was not prepared yet
    this->prepareBallAndStick = true;

    // fill amino acid color table
    Color::FillAminoAcidColorTable( this->aminoAcidColorTable);
    // fill rainbow color table
    Color::MakeRainbowColorTable( 100, this->rainbowColors);

    // set minimum, maximum and middle value colors
	colMax.Set( 255,   0,   0);
	colMid.Set( 255, 255, 255);
	colMin.Set(   0,   0, 255);

    // draw dots for atoms when using LINES mode
    this->drawDotsWithLine = true;

}


/*
 * protein::ProteinMovementRenderer::~ProteinMovementRenderer (DTOR)
 */
protein::ProteinMovementRenderer::~ProteinMovementRenderer ( void )
{
    this->Release ();
}


/*
 * protein::ProteinMovementRenderer::release
 */
void protein::ProteinMovementRenderer::release ( void )
{

}


/*
 * protein::ProteinMovementRenderer::create
 */
bool protein::ProteinMovementRenderer::create ( void )
{
    if ( glh_init_extensions ( "GL_ARB_vertex_program" ) == 0 )
    {
        return false;
    }
    if ( !vislib::graphics::gl::GLSLShader::InitialiseExtensions() )
    {
        return false;
    }

    glEnable ( GL_DEPTH_TEST );
    glDepthFunc ( GL_LEQUAL );
    glHint ( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    glEnable ( GL_VERTEX_PROGRAM_TWO_SIDE );
    glEnable ( GL_VERTEX_PROGRAM_POINT_SIZE_ARB );

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;

    // Load sphere shader
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::sphereVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere shader", this->ClassName() );
        return false;
    }
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::sphereFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere shader", this->ClassName() );
        return false;
    }
    try
    {
        if ( !this->sphereShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch ( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create sphere shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    // Load cylinder shader
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::cylinderVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for cylinder shader", this->ClassName() );
        return false;
    }
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::cylinderFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for cylinder shader", this->ClassName() );
        return false;
    }
    try
    {
        if ( !this->cylinderShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch ( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create cylinder shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    // Load arrow shader
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::arrowVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for arrow shader", this->ClassName() );
        return false;
    }
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::arrowFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for arrow shader", this->ClassName() );
        return false;
    }
    try
    {
        if ( !this->arrowShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch ( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create arrow shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::ProteinMovementRenderer::GetCapabilities
 */
bool protein::ProteinMovementRenderer::GetCapabilities( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING);

    return true;
}


/*
 * protein::ProteinMovementRenderer::GetExtents
 */
bool protein::ProteinMovementRenderer::GetExtents( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    protein::CallProteinMovementData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinMovementData>();
    if( protein == NULL ) return false;
    if( !(*protein)() ) return false;

    float scale, xoff, yoff, zoff;
    vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    scale = 2.0f / vislib::math::Max(vislib::math::Max( protein->BoundingBox().Width(),
        protein->BoundingBox().Height()), protein->BoundingBox().Depth());

    BoundingBoxes &bbox = cr3d->AccessBoundingBoxes();
    bbox.SetObjectSpaceBBox(protein->BoundingBox());
    bbox.SetWorldSpaceBBox(
        (protein->BoundingBox().Left() + xoff) * scale,
        (protein->BoundingBox().Bottom() + yoff) * scale,
        (protein->BoundingBox().Back() + zoff) * scale,
        (protein->BoundingBox().Right() + xoff) * scale,
        (protein->BoundingBox().Top() + yoff) * scale,
        (protein->BoundingBox().Front() + zoff) * scale);
    bbox.SetObjectSpaceClipBox( bbox.ObjectSpaceBBox());
    bbox.SetWorldSpaceClipBox( bbox.WorldSpaceBBox());

    return true;
}


/*
 * protein::ProteinMovementRenderer::Render
 */
bool protein::ProteinMovementRenderer::Render ( megamol::core::Call& call )
{
    // get pointer to CallProteinMovementData
    protein::CallProteinMovementData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinMovementData>();

    if ( protein == NULL )
        return false;

    if ( ! ( *protein ) () )
        return false;

    // check last atom count with current atom count
    if( this->atomCount != protein->ProteinAtomCount() ) {
        this->atomCount = protein->ProteinAtomCount();
        this->RecomputeAll();
    }

    // get camera information
	this->cameraInfo = dynamic_cast<view::CallRender3D*> ( &call )->GetCameraParameters();

    // parameter refresh
    if ( this->renderingModeParam.IsDirty() ) {
        this->SetRenderMode ( static_cast<RenderMode> ( int ( this->renderingModeParam.Param<param::EnumParam>()->Value() ) ) );
        this->renderingModeParam.ResetDirty();
    }
    if ( this->coloringModeParam.IsDirty() ) {
        this->SetColoringMode ( static_cast<Color::ColoringMode> ( int ( this->coloringModeParam.Param<param::EnumParam>()->Value() ) ) );
        this->coloringModeParam.ResetDirty();
    }
    if ( this->stickRadiusParam.IsDirty() ) {
        this->SetRadiusStick ( this->stickRadiusParam.Param<param::FloatParam>()->Value() );
        this->stickRadiusParam.ResetDirty();
    }
    if ( this->arrowRadiusParam.IsDirty() ) {
        this->SetRadiusArrow( this->arrowRadiusParam.Param<param::FloatParam>()->Value() );
        this->arrowRadiusParam.ResetDirty();
    }
    if ( this->arrowMinLengthParam.IsDirty() ) {
        this->SetMinLenghtArrow( this->arrowMinLengthParam.Param<param::FloatParam>()->Value() );
        this->arrowMinLengthParam.ResetDirty();
    }
    if ( this->arrowRadiusScaleParam.IsDirty() ) {
        this->scaleRadiusArrow = this->arrowRadiusScaleParam.Param<param::FloatParam>()->Value();
        this->arrowRadiusScaleParam.ResetDirty();
    }
    if( this->arrowScaleParam.IsDirty() ) {
        this->scaleArrow = this->arrowScaleParam.Param<param::FloatParam>()->Value();
        this->arrowScaleParam.ResetDirty();
    }
    if( this->arrowScaleLogParam.IsDirty() ) {
        this->scaleLogArrow = this->arrowScaleLogParam.Param<param::FloatParam>()->Value();
        this->arrowScaleLogParam.ResetDirty();
    }
    if( this->arrowBaseLengthParam.IsDirty() ) {
        this->baseLengthArrow = this->arrowBaseLengthParam.Param<param::FloatParam>()->Value();
        this->arrowBaseLengthParam.ResetDirty();
    }
    if( this->arrowColoringModeParam.IsDirty() ) {
        this->arrowColorMode = static_cast<ArrowColoringMode>( int( this->arrowColoringModeParam.Param<param::EnumParam>()->Value() ) );
        this->arrowColoringModeParam.ResetDirty();
    }

    // make the atom color table if necessary
    Color::MakeColorTable ( protein,
        this->currentColoringMode,
        this->colMax,
        this->colMid,
        this->colMin,
        this->col,
        this->protAtomColorTable,
        this->aminoAcidColorTable,
        this->rainbowColors );

    glEnable ( GL_DEPTH_TEST );
    glEnable ( GL_LIGHTING );
    glEnable ( GL_VERTEX_PROGRAM_POINT_SIZE );

    glPushMatrix();

    float scale, xoff, yoff, zoff;
    vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();

    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();

    scale = 2.0f / vislib::math::Max ( vislib::math::Max ( protein->BoundingBox().Width(),
            protein->BoundingBox().Height() ), protein->BoundingBox().Depth() );
    
    glScalef ( scale, scale, scale );
    glTranslatef ( xoff, yoff, zoff );

    if ( currentRenderMode == LINES ) {
        // -----------------------------------------------------------------------
        // --- LINES                                                           ---
        // --- render the sceleton of the protein using GL_POINTS and GL_LINES ---
        // -----------------------------------------------------------------------
        this->RenderLines ( protein );
    }

    if ( currentRenderMode == STICK_RAYCASTING ) {
        // ------------------------------------------------------------
        // --- STICK                                                ---
        // --- render the protein using shaders / raycasting (glsl) ---
        // ------------------------------------------------------------
        this->RenderStickRaycasting ( protein );
    }

    if ( currentRenderMode == BALL_AND_STICK ) {
        // ------------------------------------------------------------
        // --- BALL & STICK                                         ---
        // --- render the protein using shaders / raycasting (glsl) ---
        // ------------------------------------------------------------
        this->RenderBallAndStick ( protein );
    }

    // ----------------------------------------
    // --- render arrows to depict movement ---
    // ----------------------------------------
    unsigned int cnt;
    vislib::math::Vector<float, 3> tmpVec1, tmpVec2, diffVec;
    float lenDiffVec;

    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()
    };
    if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
    if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable ( GL_BLEND );

    // enable arrow shader
    this->arrowShader.Enable();
    // set shader variables
    glUniform4fvARB ( this->arrowShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
    glUniform3fvARB ( this->arrowShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
    glUniform3fvARB ( this->arrowShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
    glUniform3fvARB ( this->arrowShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
    glUniform1fARB ( this->arrowShader.ParameterLocation ( "radScale" ), this->scaleRadiusArrow );
    
    glBegin( GL_POINTS);
    for( cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {
        tmpVec1.Set( protein->ProteinAtomPositions()[cnt*3+0],
            protein->ProteinAtomPositions()[cnt*3+1],
            protein->ProteinAtomPositions()[cnt*3+2]);
        tmpVec2.Set( protein->ProteinAtomMovementPositions()[cnt*3+0],
            protein->ProteinAtomMovementPositions()[cnt*3+1],
            protein->ProteinAtomMovementPositions()[cnt*3+2]);
        diffVec = tmpVec2 - tmpVec1;
        lenDiffVec = diffVec.Normalise();
        if( lenDiffVec < this->minLenArrow )
            continue;
        //tmpVec2 = tmpVec1 + diffVec * ( 1.0f + log10f( 1.0f + lenDiffVec)) * this->scaleArrow;
        tmpVec2 = tmpVec1 + diffVec * ( this->baseLengthArrow +
            1.0f / this->scaleLogArrow * log( lenDiffVec * this->scaleLogArrow + 1.0f ))
            * this->scaleArrow;
        if( this->arrowColorMode == UNIFORM_COLOR ) {
            glColor3f( 0.0f, 1.0f, 1.0f);
        } else if( this->arrowColorMode == PROTEIN_COLOR ) {
            glColor3ubv( this->GetProteinAtomColor( cnt));
        } else {
			float maxVal( protein->GetMaxMovementDistance() );
			float mid( maxVal/2.0f );

            if( fabs( maxVal) < vislib::math::FLOAT_EPSILON ) {
				col = colMid;
            } else {
			    if( lenDiffVec < mid )
				    col = colMin + ( ( colMid - colMin ) * ( lenDiffVec / mid) );
			    else if( lenDiffVec > mid )
			        col = colMid + ( ( colMax - colMid ) * ( ( lenDiffVec - mid ) / ( maxVal - mid) ) );
			    else
                    col = colMid;
            }
            glColor3ub( col.GetX(), col.GetY(), col.GetZ());
        }
        glTexCoord3fv( tmpVec1.PeekComponents() );
        glVertex4f( tmpVec2.GetX(), tmpVec2.GetY(), tmpVec2.GetZ(), this->radiusArrow);
    }
    glEnd();
    this->arrowShader.Disable();

    glDisable ( GL_VERTEX_PROGRAM_POINT_SIZE );

    glPopMatrix();

    return true;
}


/**
 * protein::ProteinMovementRenderer::RenderLines
 */
void protein::ProteinMovementRenderer::RenderLines ( const CallProteinMovementData *prot )
{
    // built the display list if it was not yet created
    if ( !glIsList ( this->proteinDisplayListLines ) ) {
        // generate a new display list
        this->proteinDisplayListLines = glGenLists ( 1 );
        // compile new display list
        glNewList ( this->proteinDisplayListLines, GL_COMPILE );

        unsigned int i;
        unsigned int currentChain, currentAminoAcid, currentConnection;
        unsigned int first, second;
        // lines can not be lighted --> turn light off
        glDisable ( GL_LIGHTING );

        protein::CallProteinMovementData::Chain chain;
        const float *protAtomPos = prot->ProteinAtomPositions();

        glPushAttrib ( GL_ENABLE_BIT | GL_POINT_BIT | GL_LINE_BIT | GL_POLYGON_BIT );
        glEnable ( GL_BLEND );
        glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable ( GL_LINE_SMOOTH );
        glEnable ( GL_LINE_WIDTH );
        glEnable ( GL_POINT_SMOOTH );
        glEnable ( GL_POINT_SIZE );
        glLineWidth ( 3.0f );
        glPointSize ( 6.0f );

        if ( this->drawDotsWithLine ) {
            // draw atoms as points
            glBegin ( GL_POINTS );
            for ( i = 0; i < prot->ProteinAtomCount(); i++ ) {
                glColor3ubv ( this->GetProteinAtomColor ( i ) );
                glVertex3f ( protAtomPos[i*3+0], protAtomPos[i*3+1], protAtomPos[i*3+2] );
            }
            glEnd(); // GL_POINTS
        }

        // draw connections as lines
        glBegin ( GL_LINES );
        // loop over all chains
        for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ ) {
            chain = prot->ProteinChain ( currentChain );
            // loop over all amino acids in the current chain
            for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ ) {
                // loop over all connections of the current amino acid
                for ( currentConnection = 0;
                        currentConnection < chain.AminoAcid() [currentAminoAcid].Connectivity().Count();
                        currentConnection++ ) {
                    first = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].First();
                    first += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
                    second = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].Second();
                    second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
                    glColor3ubv( this->GetProteinAtomColor( first ) );
                    glVertex3fv( &protAtomPos[first*3]);
                    glColor3ubv( this->GetProteinAtomColor( second ) );
                    glVertex3fv( &protAtomPos[second*3]);
                }
                // try to make the connection between this amino acid and its predecessor
                // --> only possible if the current amino acid is not the first in this chain
                if ( currentAminoAcid > 0 &&
                    chain.AminoAcid()[currentAminoAcid-1].NameIndex() != 0 &&
                    chain.AminoAcid()[currentAminoAcid].NameIndex() != 0 ) {
                    first = chain.AminoAcid() [currentAminoAcid-1].CCarbIndex();
                    first += chain.AminoAcid() [currentAminoAcid-1].FirstAtomIndex();
                    second = chain.AminoAcid() [currentAminoAcid].NIndex();
                    second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
                    glColor3ubv( this->GetProteinAtomColor ( first ) );
                    glVertex3fv( &protAtomPos[first*3]);
                    glColor3ubv( this->GetProteinAtomColor ( second ) );
                    glVertex3fv( &protAtomPos[second*3+0]);
                }
            }
        }
        glEnd(); // GL_LINES

        glPopAttrib();

        glEndList();
        vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO+200, "%s: Display list for LINES render mode built.", this->ClassName() );
    } else {
        //draw the display list
        glCallList ( this->proteinDisplayListLines );
    }
    // turn light on after rendering
    glEnable ( GL_LIGHTING );
    glDisable ( GL_BLEND );

}


/*
 * protein::ProteinMovementRenderer::RenderStickRaycasting
 */
void protein::ProteinMovementRenderer::RenderStickRaycasting (
    const CallProteinMovementData *prot )
{
    if ( this->prepareStickRaycasting ) {
        unsigned int i1;
        unsigned int first, second;
        unsigned int currentChain, currentAminoAcid, currentConnection;
        const unsigned char *color1;
        const unsigned char *color2;

        // -----------------------------
        // -- computation for spheres --
        // -----------------------------

        // clear vertex array for spheres
        this->vertSphereStickRay.Clear();
        // clear color array for sphere colors
        this->colorSphereStickRay.Clear();

        // store the points (will be rendered as spheres by the shader)
        for ( i1 = 0; i1 < prot->ProteinAtomCount(); i1++ ) {
            this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+0] );
            this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+1] );
            this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+2] );
            this->vertSphereStickRay.Add ( radiusStick );

            color1 = this->GetProteinAtomColor ( i1 );
            this->colorSphereStickRay.Add ( color1[0] );
            this->colorSphereStickRay.Add ( color1[1] );
            this->colorSphereStickRay.Add ( color1[2] );
        }

        // -------------------------------
        // -- computation for cylinders --
        // -------------------------------
        protein::CallProteinMovementData::Chain chain;
        vislib::math::Quaternion<float> quatC;
        quatC.Set ( 0, 0, 0, 1 );
        vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
        vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
        float angle;
        // vertex array for cylinders
        this->vertCylinderStickRay.Clear();
        // color array for first cylinder colors
        this->color1CylinderStickRay.Clear();
        // color array for second cylinder colors
        this->color2CylinderStickRay.Clear();
        // attribute array for quaterions
        this->quatCylinderStickRay.Clear();
        // attribute array for in-parameters
        this->inParaCylStickRaycasting.Clear();

        // loop over all chains
        for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ ) {
            chain = prot->ProteinChain ( currentChain );
            // loop over all amino acids in the current chain
            for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ ) {
                // loop over all connections of the current amino acid
                for ( currentConnection = 0;
                        currentConnection < chain.AminoAcid() [currentAminoAcid].Connectivity().Count();
                        currentConnection++ ) {
                    first = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].First();
                    first += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
                    second = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].Second();
                    second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();

                    firstAtomPos.SetX ( prot->ProteinAtomPositions() [first*3+0] );
                    firstAtomPos.SetY ( prot->ProteinAtomPositions() [first*3+1] );
                    firstAtomPos.SetZ ( prot->ProteinAtomPositions() [first*3+2] );
                    color1 = this->GetProteinAtomColor ( first );

                    secondAtomPos.SetX ( prot->ProteinAtomPositions() [second*3+0] );
                    secondAtomPos.SetY ( prot->ProteinAtomPositions() [second*3+1] );
                    secondAtomPos.SetZ ( prot->ProteinAtomPositions() [second*3+2] );
                    color2 = this->GetProteinAtomColor ( second );

                    // compute the quaternion for the rotation of the cylinder
                    dir = secondAtomPos - firstAtomPos;
                    tmpVec.Set ( 1.0f, 0.0f, 0.0f );
                    angle = - tmpVec.Angle ( dir );
                    ortho = tmpVec.Cross ( dir );
                    ortho.Normalise();
                    quatC.Set ( angle, ortho );
                    // compute the absolute position 'position' of the cylinder (center point)
                    position = firstAtomPos + ( dir/2.0f );

                    this->inParaCylStickRaycasting.Add ( radiusStick );
                    this->inParaCylStickRaycasting.Add ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) );

                    this->quatCylinderStickRay.Add ( quatC.GetX() );
                    this->quatCylinderStickRay.Add ( quatC.GetY() );
                    this->quatCylinderStickRay.Add ( quatC.GetZ() );
                    this->quatCylinderStickRay.Add ( quatC.GetW() );

                    this->color1CylinderStickRay.Add ( float ( int ( color1[0] ) ) /255.0f );
                    this->color1CylinderStickRay.Add ( float ( int ( color1[1] ) ) /255.0f );
                    this->color1CylinderStickRay.Add ( float ( int ( color1[2] ) ) /255.0f );

                    this->color2CylinderStickRay.Add ( float ( int ( color2[0] ) ) /255.0f );
                    this->color2CylinderStickRay.Add ( float ( int ( color2[1] ) ) /255.0f );
                    this->color2CylinderStickRay.Add ( float ( int ( color2[2] ) ) /255.0f );

                    this->vertCylinderStickRay.Add ( position.GetX() );
                    this->vertCylinderStickRay.Add ( position.GetY() );
                    this->vertCylinderStickRay.Add ( position.GetZ() );
                    this->vertCylinderStickRay.Add ( 1.0f );
                }
                // try to make the connection between this amino acid and its predecessor
                // --> only possible if the current amino acid is not the first in this chain
                
                if ( currentAminoAcid > 0 &&
                    chain.AminoAcid()[currentAminoAcid-1].NameIndex() != 0 &&
                    chain.AminoAcid()[currentAminoAcid].NameIndex() != 0 ) {
                    first = chain.AminoAcid() [currentAminoAcid-1].CCarbIndex();
                    first += chain.AminoAcid() [currentAminoAcid-1].FirstAtomIndex();
                    second = chain.AminoAcid() [currentAminoAcid].NIndex();
                    second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();

                    firstAtomPos.SetX ( prot->ProteinAtomPositions() [first*3+0] );
                    firstAtomPos.SetY ( prot->ProteinAtomPositions() [first*3+1] );
                    firstAtomPos.SetZ ( prot->ProteinAtomPositions() [first*3+2] );
                    color1 = this->GetProteinAtomColor ( first );

                    secondAtomPos.SetX ( prot->ProteinAtomPositions() [second*3+0] );
                    secondAtomPos.SetY ( prot->ProteinAtomPositions() [second*3+1] );
                    secondAtomPos.SetZ ( prot->ProteinAtomPositions() [second*3+2] );
                    color2 = this->GetProteinAtomColor ( second );

                    // compute the quaternion for the rotation of the cylinder
                    dir = secondAtomPos - firstAtomPos;
                    tmpVec.Set ( 1.0f, 0.0f, 0.0f );
                    angle = - tmpVec.Angle ( dir );
                    ortho = tmpVec.Cross ( dir );
                    ortho.Normalise();
                    quatC.Set ( angle, ortho );
                    // compute the absolute position 'position' of the cylinder (center point)
                    position = firstAtomPos + ( dir/2.0f );

                    // don't draw bonds that are too long
                    if ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) > 3.5f )
                        continue;
                    
                    this->inParaCylStickRaycasting.Add ( radiusStick );
                    this->inParaCylStickRaycasting.Add ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) );

                    this->quatCylinderStickRay.Add ( quatC.GetX() );
                    this->quatCylinderStickRay.Add ( quatC.GetY() );
                    this->quatCylinderStickRay.Add ( quatC.GetZ() );
                    this->quatCylinderStickRay.Add ( quatC.GetW() );

                    this->color1CylinderStickRay.Add ( float ( int ( color1[0] ) ) /255.0f );
                    this->color1CylinderStickRay.Add ( float ( int ( color1[1] ) ) /255.0f );
                    this->color1CylinderStickRay.Add ( float ( int ( color1[2] ) ) /255.0f );

                    this->color2CylinderStickRay.Add ( float ( int ( color2[0] ) ) /255.0f );
                    this->color2CylinderStickRay.Add ( float ( int ( color2[1] ) ) /255.0f );
                    this->color2CylinderStickRay.Add ( float ( int ( color2[2] ) ) /255.0f );

                    this->vertCylinderStickRay.Add ( position.GetX() );
                    this->vertCylinderStickRay.Add ( position.GetY() );
                    this->vertCylinderStickRay.Add ( position.GetZ() );
                    this->vertCylinderStickRay.Add ( 1.0f );
                }
            }
        }

        this->prepareStickRaycasting = false;
    }

    // -----------
    // -- draw  --
    // -----------
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()
    };
    if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
    if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable ( GL_BLEND );

    // enable sphere shader
    this->sphereShader.Enable();
    glEnableClientState ( GL_VERTEX_ARRAY );
    glEnableClientState ( GL_COLOR_ARRAY );
    // set shader variables
    glUniform4fvARB ( this->sphereShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
    glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
    glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
    glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
    // set vertex and color pointers and draw them
    glVertexPointer ( 4, GL_FLOAT, 0, this->vertSphereStickRay.PeekElements() );
    glColorPointer ( 3, GL_UNSIGNED_BYTE, 0, this->colorSphereStickRay.PeekElements() );
    glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertSphereStickRay.Count() /4 ) );
    // disable sphere shader
    this->sphereShader.Disable();

    // enable cylinder shader
    this->cylinderShader.Enable();
    // set shader variables
    glUniform4fvARB ( this->cylinderShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
    glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
    glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
    glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
    // get the attribute locations
    attribLocInParams = glGetAttribLocationARB ( this->cylinderShader, "inParams" );
    attribLocQuatC = glGetAttribLocationARB ( this->cylinderShader, "quatC" );
    attribLocColor1 = glGetAttribLocationARB ( this->cylinderShader, "color1" );
    attribLocColor2 = glGetAttribLocationARB ( this->cylinderShader, "color2" );
    // enable vertex attribute arrays for the attribute locations
    glDisableClientState ( GL_COLOR_ARRAY );
    glEnableVertexAttribArrayARB ( this->attribLocInParams );
    glEnableVertexAttribArrayARB ( this->attribLocQuatC );
    glEnableVertexAttribArrayARB ( this->attribLocColor1 );
    glEnableVertexAttribArrayARB ( this->attribLocColor2 );
    // set vertex and attribute pointers and draw them
    glVertexPointer ( 4, GL_FLOAT, 0, this->vertCylinderStickRay.PeekElements() );
    glVertexAttribPointerARB ( this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylStickRaycasting.PeekElements() );
    glVertexAttribPointerARB ( this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinderStickRay.PeekElements() );
    glVertexAttribPointerARB ( this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1CylinderStickRay.PeekElements() );
    glVertexAttribPointerARB ( this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2CylinderStickRay.PeekElements() );
    glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertCylinderStickRay.Count() /4 ) );
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArrayARB ( this->attribLocInParams );
    glDisableVertexAttribArrayARB ( this->attribLocQuatC );
    glDisableVertexAttribArrayARB ( this->attribLocColor1 );
    glDisableVertexAttribArrayARB ( this->attribLocColor2 );
    glDisableClientState ( GL_VERTEX_ARRAY );
    // disable cylinder shader
    this->cylinderShader.Disable();

}


/*
 * protein::ProteinMovementRenderer::RenderBallAndStick
 */
void protein::ProteinMovementRenderer::RenderBallAndStick (
    const CallProteinMovementData *prot )
{
    if ( this->prepareBallAndStick ) {
        unsigned int i1;
        unsigned int first, second;
        unsigned int currentChain, currentAminoAcid, currentConnection;
        const unsigned char *color1;
        const unsigned char *color2;

        // -----------------------------
        // -- computation for spheres --
        // -----------------------------

        // clear vertex array for spheres
        this->vertSphereStickRay.Clear();
        // clear color array for sphere colors
        this->colorSphereStickRay.Clear();

        // store the points (will be rendered as spheres by the shader)
        for ( i1 = 0; i1 < prot->ProteinAtomCount(); i1++ ) {
            this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+0] );
            this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+1] );
            this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+2] );
            this->vertSphereStickRay.Add ( radiusStick );

            color1 = this->GetProteinAtomColor ( i1 );
            this->colorSphereStickRay.Add ( color1[0] );
            this->colorSphereStickRay.Add ( color1[1] );
            this->colorSphereStickRay.Add ( color1[2] );
        }

        // -------------------------------
        // -- computation for cylinders --
        // -------------------------------
        protein::CallProteinMovementData::Chain chain;
        vislib::math::Quaternion<float> quatC;
        quatC.Set ( 0, 0, 0, 1 );
        vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
        vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
        float angle;
        // vertex array for cylinders
        this->vertCylinderStickRay.Clear();
        // color array for first cylinder colors
        this->color1CylinderStickRay.Clear();
        // color array for second cylinder colors
        this->color2CylinderStickRay.Clear();
        // attribute array for quaterions
        this->quatCylinderStickRay.Clear();
        // attribute array for in-parameters
        this->inParaCylStickRaycasting.Clear();

        // loop over all chains
        for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ ) {
            chain = prot->ProteinChain ( currentChain );
            // loop over all amino acids in the current chain
            for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ ) {
                // loop over all connections of the current amino acid
                for ( currentConnection = 0;
                        currentConnection < chain.AminoAcid() [currentAminoAcid].Connectivity().Count();
                        currentConnection++ ) {
                    first = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].First();
                    first += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
                    second = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].Second();
                    second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();

                    firstAtomPos.SetX ( prot->ProteinAtomPositions() [first*3+0] );
                    firstAtomPos.SetY ( prot->ProteinAtomPositions() [first*3+1] );
                    firstAtomPos.SetZ ( prot->ProteinAtomPositions() [first*3+2] );
                    color1 = this->GetProteinAtomColor ( first );

                    secondAtomPos.SetX ( prot->ProteinAtomPositions() [second*3+0] );
                    secondAtomPos.SetY ( prot->ProteinAtomPositions() [second*3+1] );
                    secondAtomPos.SetZ ( prot->ProteinAtomPositions() [second*3+2] );
                    color2 = this->GetProteinAtomColor ( second );

                    // compute the quaternion for the rotation of the cylinder
                    dir = secondAtomPos - firstAtomPos;
                    tmpVec.Set ( 1.0f, 0.0f, 0.0f );
                    angle = - tmpVec.Angle ( dir );
                    ortho = tmpVec.Cross ( dir );
                    ortho.Normalise();
                    quatC.Set ( angle, ortho );
                    // compute the absolute position 'position' of the cylinder (center point)
                    position = firstAtomPos + ( dir/2.0f );

                    this->inParaCylStickRaycasting.Add ( radiusStick/3.0f );
                    this->inParaCylStickRaycasting.Add ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) );

                    this->quatCylinderStickRay.Add ( quatC.GetX() );
                    this->quatCylinderStickRay.Add ( quatC.GetY() );
                    this->quatCylinderStickRay.Add ( quatC.GetZ() );
                    this->quatCylinderStickRay.Add ( quatC.GetW() );

                    this->color1CylinderStickRay.Add ( float ( int ( color1[0] ) ) /255.0f );
                    this->color1CylinderStickRay.Add ( float ( int ( color1[1] ) ) /255.0f );
                    this->color1CylinderStickRay.Add ( float ( int ( color1[2] ) ) /255.0f );

                    this->color2CylinderStickRay.Add ( float ( int ( color2[0] ) ) /255.0f );
                    this->color2CylinderStickRay.Add ( float ( int ( color2[1] ) ) /255.0f );
                    this->color2CylinderStickRay.Add ( float ( int ( color2[2] ) ) /255.0f );

                    this->vertCylinderStickRay.Add ( position.GetX() );
                    this->vertCylinderStickRay.Add ( position.GetY() );
                    this->vertCylinderStickRay.Add ( position.GetZ() );
                    this->vertCylinderStickRay.Add ( 1.0f );
                }
                // try to make the connection between this amino acid and its predecessor
                // --> only possible if the current amino acid is not the first in this chain
                if ( currentAminoAcid > 0 &&
                    chain.AminoAcid()[currentAminoAcid-1].NameIndex() != 0 &&
                    chain.AminoAcid()[currentAminoAcid].NameIndex() != 0 ) {
                    first = chain.AminoAcid() [currentAminoAcid-1].CCarbIndex();
                    first += chain.AminoAcid() [currentAminoAcid-1].FirstAtomIndex();
                    second = chain.AminoAcid() [currentAminoAcid].NIndex();
                    second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();

                    firstAtomPos.SetX ( prot->ProteinAtomPositions() [first*3+0] );
                    firstAtomPos.SetY ( prot->ProteinAtomPositions() [first*3+1] );
                    firstAtomPos.SetZ ( prot->ProteinAtomPositions() [first*3+2] );
                    color1 = this->GetProteinAtomColor ( first );

                    secondAtomPos.SetX ( prot->ProteinAtomPositions() [second*3+0] );
                    secondAtomPos.SetY ( prot->ProteinAtomPositions() [second*3+1] );
                    secondAtomPos.SetZ ( prot->ProteinAtomPositions() [second*3+2] );
                    color2 = this->GetProteinAtomColor ( second );

                    // compute the quaternion for the rotation of the cylinder
                    dir = secondAtomPos - firstAtomPos;
                    tmpVec.Set ( 1.0f, 0.0f, 0.0f );
                    angle = - tmpVec.Angle ( dir );
                    ortho = tmpVec.Cross ( dir );
                    ortho.Normalise();
                    quatC.Set ( angle, ortho );
                    // compute the absolute position 'position' of the cylinder (center point)
                    position = firstAtomPos + ( dir/2.0f );

                    // don't draw bonds that are too long
                    if ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) > 3.5f )
                        continue;

                    this->inParaCylStickRaycasting.Add ( radiusStick/3.0f );
                    this->inParaCylStickRaycasting.Add ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) );

                    this->quatCylinderStickRay.Add ( quatC.GetX() );
                    this->quatCylinderStickRay.Add ( quatC.GetY() );
                    this->quatCylinderStickRay.Add ( quatC.GetZ() );
                    this->quatCylinderStickRay.Add ( quatC.GetW() );

                    this->color1CylinderStickRay.Add ( float ( int ( color1[0] ) ) /255.0f );
                    this->color1CylinderStickRay.Add ( float ( int ( color1[1] ) ) /255.0f );
                    this->color1CylinderStickRay.Add ( float ( int ( color1[2] ) ) /255.0f );

                    this->color2CylinderStickRay.Add ( float ( int ( color2[0] ) ) /255.0f );
                    this->color2CylinderStickRay.Add ( float ( int ( color2[1] ) ) /255.0f );
                    this->color2CylinderStickRay.Add ( float ( int ( color2[2] ) ) /255.0f );

                    this->vertCylinderStickRay.Add ( position.GetX() );
                    this->vertCylinderStickRay.Add ( position.GetY() );
                    this->vertCylinderStickRay.Add ( position.GetZ() );
                    this->vertCylinderStickRay.Add ( 1.0f );
                }
            }
        }
        this->prepareBallAndStick = false;
    }

    // -----------
    // -- draw  --
    // -----------
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()
    };
    if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
    if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable ( GL_BLEND );

    // enable sphere shader
    this->sphereShader.Enable();
    glEnableClientState ( GL_VERTEX_ARRAY );
    glEnableClientState ( GL_COLOR_ARRAY );
    // set shader variables
    glUniform4fvARB ( this->sphereShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
    glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
    glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
    glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
    // set vertex and color pointers and draw them
    glVertexPointer ( 4, GL_FLOAT, 0, this->vertSphereStickRay.PeekElements() );
    glColorPointer ( 3, GL_UNSIGNED_BYTE, 0, this->colorSphereStickRay.PeekElements() );
    glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertSphereStickRay.Count() /4 ) );
    // disable sphere shader
    glDisableClientState ( GL_COLOR_ARRAY );
    this->sphereShader.Disable();

    // enable cylinder shader
    this->cylinderShader.Enable();
    // set shader variables
    glUniform4fvARB ( this->cylinderShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
    glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
    glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
    glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
    // get the attribute locations
    attribLocInParams = glGetAttribLocationARB ( this->cylinderShader, "inParams" );
    attribLocQuatC = glGetAttribLocationARB ( this->cylinderShader, "quatC" );
    attribLocColor1 = glGetAttribLocationARB ( this->cylinderShader, "color1" );
    attribLocColor2 = glGetAttribLocationARB ( this->cylinderShader, "color2" );
    // enable vertex attribute arrays for the attribute locations
    glEnableVertexAttribArrayARB ( this->attribLocInParams );
    glEnableVertexAttribArrayARB ( this->attribLocQuatC );
    glEnableVertexAttribArrayARB ( this->attribLocColor1 );
    glEnableVertexAttribArrayARB ( this->attribLocColor2 );
    // set vertex and attribute pointers and draw them
    glVertexPointer ( 4, GL_FLOAT, 0, this->vertCylinderStickRay.PeekElements() );
    glVertexAttribPointerARB ( this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylStickRaycasting.PeekElements() );
    glVertexAttribPointerARB ( this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinderStickRay.PeekElements() );
    glVertexAttribPointerARB ( this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1CylinderStickRay.PeekElements() );
    glVertexAttribPointerARB ( this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2CylinderStickRay.PeekElements() );
    glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertCylinderStickRay.Count() /4 ) );
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArrayARB ( this->attribLocInParams );
    glDisableVertexAttribArrayARB ( this->attribLocQuatC );
    glDisableVertexAttribArrayARB ( this->attribLocColor1 );
    glDisableVertexAttribArrayARB ( this->attribLocColor2 );
    glDisableClientState ( GL_VERTEX_ARRAY );
    // disable cylinder shader
    this->cylinderShader.Disable();

}


/*
 * protein::ProteinMovementRenderer::MakeColorTable
 */
 /*
void protein::ProteinMovementRenderer::MakeColorTable( const CallProteinMovementData *prot, bool forceRecompute )
{
    unsigned int i;
    unsigned int currentChain, currentAminoAcid, currentAtom, currentSecStruct;
    unsigned int cntCha, cntRes, cntAto;
    protein::CallProteinMovementData::Chain chain;
    vislib::math::Vector<float, 3> color;
    // if recomputation is forced: clear current color table
    if ( forceRecompute ) {
        this->protAtomColorTable.Clear();
    }
    // reserve memory for all atoms
    this->protAtomColorTable.AssertCapacity( prot->ProteinAtomCount()*3 );
    // only compute color table if necessary
    if ( this->protAtomColorTable.IsEmpty() ) {
        if ( this->currentColoringMode == ELEMENT ) {
            for ( i = 0; i < prot->ProteinAtomCount(); i++ )
            {
                this->protAtomColorTable.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i].TypeIndex() ].Colour() [0] );
                this->protAtomColorTable.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i].TypeIndex() ].Colour() [1] );
                this->protAtomColorTable.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i].TypeIndex() ].Colour() [2] );
            }
            // ... END coloring mode ELEMENT
        } else if ( this->currentColoringMode == AMINOACID ) {
            // loop over all chains
            for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ ) {
                chain = prot->ProteinChain ( currentChain );
                // loop over all amino acids in the current chain
                for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ ) {
                    // loop over all connections of the current amino acid
                    for ( currentAtom = 0;
                            currentAtom < chain.AminoAcid() [currentAminoAcid].AtomCount();
                            currentAtom++ ) {
                        i = chain.AminoAcid() [currentAminoAcid].NameIndex() +1;
                        i = i % ( unsigned int ) ( this->aminoAcidColorTable.Count() );
                        this->protAtomColorTable.Add (
                            this->aminoAcidColorTable[i].GetX() );
                        this->protAtomColorTable.Add (
                            this->aminoAcidColorTable[i].GetY() );
                        this->protAtomColorTable.Add (
                            this->aminoAcidColorTable[i].GetZ() );
                    }
                }
            }
            // ... END coloring mode AMINOACID
        } else if ( this->currentColoringMode == STRUCTURE ) {
            // loop over all chains
            for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ ) {
                chain = prot->ProteinChain ( currentChain );
                // loop over all secondary structure elements in this chain
                for ( currentSecStruct = 0;
                        currentSecStruct < chain.SecondaryStructureCount();
                        currentSecStruct++ ) {
                    i = chain.SecondaryStructure() [currentSecStruct].AtomCount();
                    // loop over all atoms in this secondary structure element
                    for ( currentAtom = 0; currentAtom < i; currentAtom++ ) {
                        if ( chain.SecondaryStructure() [currentSecStruct].Type() ==
                                protein::CallProteinMovementData::SecStructure::TYPE_HELIX ) {
                            this->protAtomColorTable.Add ( 255 );
                            this->protAtomColorTable.Add ( 0 );
                            this->protAtomColorTable.Add ( 0 );
                        } else if ( chain.SecondaryStructure() [currentSecStruct].Type() ==
                                  protein::CallProteinMovementData::SecStructure::TYPE_SHEET ) {
                            this->protAtomColorTable.Add ( 0 );
                            this->protAtomColorTable.Add ( 0 );
                            this->protAtomColorTable.Add ( 255 );
                        } else if ( chain.SecondaryStructure() [currentSecStruct].Type() ==
                                  protein::CallProteinMovementData::SecStructure::TYPE_TURN ) {
                            this->protAtomColorTable.Add ( 255 );
                            this->protAtomColorTable.Add ( 255 );
                            this->protAtomColorTable.Add ( 0 );
                        } else {
                            this->protAtomColorTable.Add ( 230 );
                            this->protAtomColorTable.Add ( 230 );
                            this->protAtomColorTable.Add ( 230 );
                        }
                    }
                }
            }
            // add missing atom colors
            if ( prot->ProteinAtomCount() > ( this->protAtomColorTable.Count() / 3 ) ) {
                currentAtom = this->protAtomColorTable.Count() / 3;
                for ( ; currentAtom < prot->ProteinAtomCount(); ++currentAtom ) {
                    this->protAtomColorTable.Add ( 200 );
                    this->protAtomColorTable.Add ( 200 );
                    this->protAtomColorTable.Add ( 200 );
                }
            }
            // ... END coloring mode STRUCTURE
        } else if ( this->currentColoringMode == CHAIN_ID ) {
            // loop over all chains
            for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ ) {
                chain = prot->ProteinChain ( currentChain );
                // loop over all amino acids in the current chain
                for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ ) {
                    // loop over all connections of the current amino acid
                    for ( currentAtom = 0;
                            currentAtom < chain.AminoAcid() [currentAminoAcid].AtomCount();
                            currentAtom++ ) {
                        i = ( currentChain + 1 ) % ( unsigned int ) ( this->aminoAcidColorTable.Count() );
                        this->protAtomColorTable.Add( this->aminoAcidColorTable[i].GetX() );
                        this->protAtomColorTable.Add( this->aminoAcidColorTable[i].GetY() );
                        this->protAtomColorTable.Add( this->aminoAcidColorTable[i].GetZ() );
                    }
                }
            }
            // ... END coloring mode CHAIN_ID
        } else if ( this->currentColoringMode == RAINBOW ) {
            for( cntCha = 0; cntCha < prot->ProteinChainCount(); ++cntCha ) {
                for( cntRes = 0; cntRes < prot->ProteinChain( cntCha).AminoAcidCount(); ++cntRes ) {
                    i = int( ( float( cntRes) / float( prot->ProteinChain( cntCha).AminoAcidCount() ) ) * float( rainbowColors.size() ) );
                    color = this->rainbowColors[i];
                    for( cntAto = 0;
                         cntAto < prot->ProteinChain( cntCha).AminoAcid()[cntRes].AtomCount();
                         ++cntAto ) {
                        this->protAtomColorTable.Add( int(color.GetX() * 255.0f) );
                        this->protAtomColorTable.Add( int(color.GetY() * 255.0f) );
                        this->protAtomColorTable.Add( int(color.GetZ() * 255.0f) );
                    }
                }
            }
            // ... END coloring mode RAINBOW
        } else if ( this->currentColoringMode == MOVEMENT ) {
			float minVal( 0.0f );
			float maxVal( prot->GetMaxMovementDistance() );
			float mid( maxVal/2.0f );
			float val;
			
            unsigned int protac = prot->ProteinAtomCount();
            for ( cntAto = 0; cntAto < protac; ++cntAto ) {
                if( fabs( minVal - maxVal) < vislib::math::FLOAT_EPSILON ) {
					this->protAtomColorTable.Add( colMid.GetX() );
					this->protAtomColorTable.Add( colMid.GetY() );
					this->protAtomColorTable.Add( colMid.GetZ() );
                } else {
                    val = sqrt( pow( prot->ProteinAtomPositions()[cntAto*3+0] - prot->ProteinAtomMovementPositions()[cntAto*3+0], 2.0f) +
                         pow( prot->ProteinAtomPositions()[cntAto*3+1] - prot->ProteinAtomMovementPositions()[cntAto*3+1], 2.0f) +
                         pow( prot->ProteinAtomPositions()[cntAto*3+2] - prot->ProteinAtomMovementPositions()[cntAto*3+2], 2.0f) );

				    if( val < mid ) {
                        // below middle value --> blend between minVal and mid color
					    col = colMin + ( ( colMid - colMin ) * ( val / mid) );
					    this->protAtomColorTable.Add( col.GetX() );
					    this->protAtomColorTable.Add( col.GetY() );
					    this->protAtomColorTable.Add( col.GetZ() );
				    } else if( val > mid ) {
				        // above middle value --> blend between maxVal and mid color
				        col = colMid + ( ( colMax - colMid ) * ( ( val - mid ) / ( maxVal - mid) ) );
					    this->protAtomColorTable.Add( col.GetX() );
					    this->protAtomColorTable.Add( col.GetY() );
					    this->protAtomColorTable.Add( col.GetZ() );
				    } else {
				        // middle value --> assign mid color
					    this->protAtomColorTable.Add( colMid.GetX() );
					    this->protAtomColorTable.Add( colMid.GetY() );
					    this->protAtomColorTable.Add( colMid.GetZ() );
				    }
                }
			}
            // ... END coloring mode MOVEMENT
		}
    }
}*/


/*
 * protein::ProteinMovementRenderer::RecomputeAll
 */
void protein::ProteinMovementRenderer::RecomputeAll()
{
    this->prepareBallAndStick = true;
    this->prepareStickRaycasting = true;

    glDeleteLists ( this->proteinDisplayListLines, 1 );
    this->proteinDisplayListLines = 0;

    this->protAtomColorTable.Clear();
}
