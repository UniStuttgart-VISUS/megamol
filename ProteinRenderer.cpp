/*
 * ProteinRenderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "ProteinRenderer.h"
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
 * protein::ProteinRenderer::ProteinRenderer (CTOR)
 */
protein::ProteinRenderer::ProteinRenderer(void) : Renderer3DModule (),
    m_protDataCallerSlot ("getData", "Connects the protein rendering with protein data storage"),
    m_callFrameCalleeSlot("callFrame", "Connects the protein rendering with frame call from RMS renderer"),
    solventRendererCallerSlot ( "renderSolvent", "Connects the protein rendering with a solvent renderer" ),
    m_renderingModeParam("renderingMode", "Rendering Mode"),
    m_coloringModeParam("coloringMode", "Coloring Mode"),
    m_drawBackboneParam("drawBackbone", "Draw Backbone only"),
    m_drawDisulfideBondsParam("drawDisulfideBonds", "Draw Disulfide Bonds"),
    m_stickRadiusParam("stickRadius", "Stick Radius for spheres and sticks with STICK_ render modes"),
        m_probeRadiusParam ( "probeRadius", "Probe Radius for SAS rendering" ),
        m_currentFrameId ( 0 ), atomCount( 0 )
{
    this->m_protDataCallerSlot.SetCompatibleCall<CallProteinDataDescription>();
    this->MakeSlotAvailable(&this->m_protDataCallerSlot);

    protein::CallFrameDescription dfd;
    this->m_callFrameCalleeSlot.SetCallback (dfd.ClassName(), "CallFrame", &ProteinRenderer::ProcessFrameRequest);
    this->MakeSlotAvailable(&this->m_callFrameCalleeSlot);

    this->solventRendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable( &this->solventRendererCallerSlot);

    // --- set the coloring mode ---

    this->SetColoringMode(Color::ELEMENT);
    //this->SetColoringMode(AMINOACID);
    //this->SetColoringMode(STRUCTURE);
    //this->SetColoringMode(VALUE);
    //this->SetColoringMode(CHAIN_ID);
    //this->SetColoringMode(RAINBOW);
    param::EnumParam *cm = new param::EnumParam(int(this->m_currentColoringMode));

    cm->SetTypePair(Color::ELEMENT, "Element");
    cm->SetTypePair(Color::AMINOACID, "AminoAcid");
    cm->SetTypePair(Color::STRUCTURE, "SecondaryStructure");
    cm->SetTypePair(Color::VALUE, "Value");
    cm->SetTypePair(Color::CHAIN_ID, "ChainID");
    cm->SetTypePair(Color::RAINBOW, "Rainbow");
    cm->SetTypePair ( Color::CHARGE, "Charge" );

    this->m_coloringModeParam << cm;

    // --- set the render mode ---

    SetRenderMode(LINES);
    //SetRenderMode(STICK_POLYGON);
    //SetRenderMode(STICK_RAYCASTING);
    //SetRenderMode(BALL_AND_STICK);
    //SetRenderMode(SPACEFILLING);
    param::EnumParam *rm = new param::EnumParam(int(this->m_currentRenderMode));

    rm->SetTypePair(LINES, "Lines");
    //rm->SetTypePair(STICK_POLYGON, "StickPoly gonal");
    rm->SetTypePair(STICK_RAYCASTING, "StickRaycasting");
    rm->SetTypePair(BALL_AND_STICK, "BallAndStick");
    rm->SetTypePair(SPACEFILLING, "SpaceFilling");
    rm->SetTypePair(SAS, "SolventAccessibleSurface");

    this->m_renderingModeParam << rm;

    // --- draw only the backbone, if 'true' ---
    this->m_drawBackbone = false;
    //this->m_drawBackboneParam.SetParameter(new view::BoolParam(this->m_drawBackbone));

    // --- draw disulfide bonds, if 'true' ---
    this->m_drawDisulfideBonds = false;
    this->m_drawDisulfideBondsParam.SetParameter(new param::BoolParam(this->m_drawDisulfideBonds));

    // --- set the radius for the stick rednering mode ---
    this->m_radiusStick = 0.3f;
    this->m_stickRadiusParam.SetParameter(new param::FloatParam(this->m_radiusStick, 0.0f));

    // --- set the probe radius for sas rendering mode ---
    this->m_probeRadius = 1.4f;
    this->m_probeRadiusParam.SetParameter(new param::FloatParam(this->m_probeRadius, 0.0f));

    this->MakeSlotAvailable(&this->m_coloringModeParam);
    this->MakeSlotAvailable(&this->m_renderingModeParam);
    //this->MakeSlotAvailable(&this->m_drawBackboneParam);
    this->MakeSlotAvailable(&this->m_drawDisulfideBondsParam);
    this->MakeSlotAvailable(&this->m_stickRadiusParam);
    this->MakeSlotAvailable(&this->m_probeRadiusParam);

    // set empty display list to zero
    this->m_proteinDisplayListLines = 0;
    // set empty display list to zero
    this->m_disulfideBondsDisplayList = 0;
    // STICK_RAYCASTING render mode was not prepared yet
    this->m_prepareStickRaycasting = true;
    // BALL_AND_STICK render mode was not prepared yet
    this->m_prepareBallAndStick = true;
    // SPACEFILLING render mode was not prepared yet
    this->m_prepareSpacefilling = true;
    // SAS render mode was not prepared yet
    this->m_prepareSAS = true;

    // fill amino acid color table
    Color::FillAminoAcidColorTable( this->m_aminoAcidColorTable);
    // fill rainbow color table
    Color::MakeRainbowColorTable( 100, this->rainbowColors);

    // draw no dots for atoms when using LINES mode
    this->m_drawDotsWithLine = false;

    this->m_renderRMSData = false;
    this->m_frameLabel = NULL;
}


/*
 * protein::ProteinRenderer::~ProteinRenderer (DTOR)
 */
protein::ProteinRenderer::~ProteinRenderer(void)
{
    delete this->m_frameLabel;
    this->Release ();
}


/*
 * protein::ProteinRenderer::release
 */
void protein::ProteinRenderer::release(void)
{

}


/*
 * protein::ProteinRenderer::create
 */
bool protein::ProteinRenderer::create(void)
{
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
        if (!this->m_sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
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
        if (!this->m_cylinderShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create cylinder shader: %s\n", e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * protein::ProteinRenderer::GetCapabilities
 */
bool protein::ProteinRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        | view::CallRender3D::CAP_ANIMATION );

    return true;
}


/*
 * protein::ProteinRenderer::GetExtents
 */
bool protein::ProteinRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    protein::CallProteinData *protein = this->m_protDataCallerSlot.CallAs<protein::CallProteinData>();
    if (protein == NULL) return false;
    // decide to use already loaded frame request from CallFrame or 'normal' rendering
    if (this->m_callFrameCalleeSlot.GetStatus() == AbstractSlot::STATUS_CONNECTED) {
        if (!this->m_renderRMSData) return false;
    } else {
        if (!(*protein)()) return false;
    }

    cr3d->AccessBoundingBoxes() = protein->AccessBoundingBoxes();

    float scale, xoff, yoff, zoff;
    vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    if( !vislib::math::IsEqual( protein->BoundingBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / protein->BoundingBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    BoundingBoxes &bbox = cr3d->AccessBoundingBoxes();
    bbox.SetObjectSpaceBBox(protein->BoundingBox());
    bbox.SetWorldSpaceBBox(
        (protein->BoundingBox().Left() + xoff) * scale,
        (protein->BoundingBox().Bottom() + yoff) * scale,
        (protein->BoundingBox().Back() + zoff) * scale,
        (protein->BoundingBox().Right() + xoff) * scale,
        (protein->BoundingBox().Top() + yoff) * scale,
        (protein->BoundingBox().Front() + zoff) * scale);
    bbox.SetObjectSpaceClipBox(bbox.ObjectSpaceBBox());
    bbox.SetWorldSpaceClipBox(bbox.WorldSpaceBBox());

    // get the pointer to CallRender3D (solvent renderer)
    view::CallRender3D *solrencr3d = this->solventRendererCallerSlot.CallAs<view::CallRender3D>();
    vislib::math::Point<float, 3> solrenbbc;
    if( solrencr3d ) {
        (*solrencr3d)(1); // GetExtents
        BoundingBoxes &solrenbb = solrencr3d->AccessBoundingBoxes();
        //this->solrenScale =  solrenbb.ObjectSpaceBBox().Width() / boundingBox.Width();
        //this->solrenTranslate = ( solrenbb.ObjectSpaceBBox().CalcCenter() - bbc) * scale;
    }

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::ProteinRenderer::Render
 */
bool protein::ProteinRenderer::Render(Call& call)
{
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    // get the pointer to CallRender3D (protein renderer)
    view::CallRender3D *solventrencr3d = this->solventRendererCallerSlot.CallAs<view::CallRender3D>();
    // get pointer to CallProteinData
    CallProteinData *protein = this->m_protDataCallerSlot.CallAs<protein::CallProteinData>();

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    // =============== Sovent Rendering ===============
    if( solventrencr3d ) {
        // setup and call protein renderer
        glPushMatrix();
        //glTranslatef( this->protrenTranslate.X(), this->protrenTranslate.Y(), this->protrenTranslate.Z());
        //glScalef( this->protrenScale, this->protrenScale, this->protrenScale);
        *solventrencr3d = *cr3d;
        (*solventrencr3d)();
        glPopMatrix();
    }

    if( protein == NULL)
        return false;

    if( this->m_currentFrameId != protein->GetCurrentFrameId() ) {
        this->m_currentFrameId = protein->GetCurrentFrameId();
        this->RecomputeAll();
    }

    // decide to use already loaded frame request from CallFrame or 'normal' rendering
    if( this->m_callFrameCalleeSlot.GetStatus() == AbstractSlot::STATUS_CONNECTED) {
        if( !this->m_renderRMSData )
            return false;
    } else {
        //if( !(*protein)() ) return false;
        if (!(*protein)(CallProteinData::CallForGetData)) return false;
    }

    // check last atom count with current atom count
    if( this->atomCount != protein->ProteinAtomCount() ) {
        this->atomCount = protein->ProteinAtomCount();
        this->RecomputeAll();
    }

    // parameter refresh
    if( this->m_renderingModeParam.IsDirty() ) {
        this->SetRenderMode(static_cast<RenderMode>(int(this->m_renderingModeParam.Param<param::EnumParam>()->Value())));
        this->m_renderingModeParam.ResetDirty();
    }
    if (this->m_coloringModeParam.IsDirty())
    {
        this->SetColoringMode(static_cast<Color::ColoringMode>(int(this->m_coloringModeParam.Param<param::EnumParam>()->Value())));
        this->m_coloringModeParam.ResetDirty();
    }
    //if (this->m_drawBackboneParam.IsDirty())
    //{
    //  this->m_drawBackbone = this->m_drawBackboneParam.Param<view::BoolParam>()->Value();
    //  this->m_drawBackboneParam.ResetDirty();
    //}
    if (this->m_drawDisulfideBondsParam.IsDirty())
    {
        this->m_drawDisulfideBonds = this->m_drawDisulfideBondsParam.Param<param::BoolParam>()->Value();
        this->m_drawDisulfideBondsParam.ResetDirty();
    }
    if (this->m_stickRadiusParam.IsDirty())
    {
        this->SetRadiusStick(this->m_stickRadiusParam.Param<param::FloatParam>()->Value());
        this->m_stickRadiusParam.ResetDirty();
    }
    if (this->m_probeRadiusParam.IsDirty())
    {
        this->SetRadiusProbe(this->m_probeRadiusParam.Param<param::FloatParam>()->Value());
        this->m_probeRadiusParam.ResetDirty();
    }

    // make the atom color table if necessary
    Color::MakeColorTable( protein,
                           this->m_currentColoringMode,
                           this->m_protAtomColorTable,
                           this->m_aminoAcidColorTable,
                           this->rainbowColors);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glPushMatrix();

    float scale, xoff, yoff, zoff;
    vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();

    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();

    scale = 2.0f / vislib::math::Max(vislib::math::Max(protein->BoundingBox().Width(),
        protein->BoundingBox().Height()), protein->BoundingBox().Depth());

    glScalef(scale, scale, scale);
    glTranslatef(xoff, yoff, zoff);

    if( this->m_drawDisulfideBonds )
    {
        // ---------------------------------------------------------
        // --- draw disulfide bonds                              ---
        // ---------------------------------------------------------
        this->RenderDisulfideBondsLine(protein);
    }

    if( m_currentRenderMode == LINES )
    {
        // -----------------------------------------------------------------------
        // --- LINES                                                           ---
        // --- render the sceleton of the protein using GL_POINTS and GL_LINES ---
        // -----------------------------------------------------------------------
        this->RenderLines(protein);
    }

    if( m_currentRenderMode == STICK_RAYCASTING )
    {
        // ------------------------------------------------------------
        // --- STICK                                                ---
        // --- render the protein using shaders / raycasting (glsl) ---
        // ------------------------------------------------------------
        this->RenderStickRaycasting(protein);
    }

    if( m_currentRenderMode == BALL_AND_STICK )
    {
        // ------------------------------------------------------------
        // --- BALL & STICK                                         ---
        // --- render the protein using shaders / raycasting (glsl) ---
        // ------------------------------------------------------------
        this->RenderBallAndStick(protein);
    }

    if( m_currentRenderMode == SPACEFILLING )
    {
        // ------------------------------------------------------------
        // --- SPACEFILLING                                         ---
        // --- render the protein using shaders / raycasting (glsl) ---
        // ------------------------------------------------------------
        this->RenderSpacefilling(protein);
    }

    if( m_currentRenderMode == SAS )
    {
        // ------------------------------------------------------------
        // --- SAS (Solvent Accessible Surface)                     ---
        // --- render the protein using shaders / raycasting (glsl) ---
        // ------------------------------------------------------------
        this->RenderSolventAccessibleSurface(protein);
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glPopMatrix();

    // render label if RMS is used
    if(this->m_renderRMSData)
        this->DrawLabel(protein->GetRequestedRMSFrame());

    return true;
}


/*
 * protein::ProteinRenderer::ProcessFrameRequest
 */
bool protein::ProteinRenderer::ProcessFrameRequest(Call& call)
{
    // get pointer to CallProteinData
    protein::CallProteinData *protein = this->m_protDataCallerSlot.CallAs<protein::CallProteinData>();

    // ensure that NetCDFData uses 'RMS' specific frame handling
    protein->SetRMSUse(true);

    // get pointer to frame call
    protein::CallFrame *pcf = dynamic_cast<protein::CallFrame*>(&call);

    if(pcf->NewRequest())
    {
        // pipe frame request from frame call to protein call
        protein->SetRequestedRMSFrame(pcf->GetFrameRequest());
        if(!(*protein)())
        {
            this->m_renderRMSData = false;
            return false;
        }
        this->m_renderRMSData = true;
    }

    // unlock the current frame
    protein->Unlock();

    return true;
}


/**
 * protein::ProteinRenderer::DrawLabel
 */
void protein::ProteinRenderer::DrawLabel(unsigned int frameID)
{
    using namespace vislib::graphics;
    char frameChar[10];

    glPushAttrib( GL_ENABLE_BIT);
    glDisable( GL_CULL_FACE);
    glDisable( GL_LIGHTING);

    glMatrixMode( GL_MODELVIEW);
    glPushMatrix();

    glTranslatef(-1.0f, 1.0f, 1.0f);

    glColor3f( 1.0, 1.0, 1.0);
    if( this->m_frameLabel == NULL) {
        this->m_frameLabel = new vislib::graphics::gl::SimpleFont();
        if( !this->m_frameLabel->Initialise()) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "ProteinRenderer: Problems to initalise the Font");
        }
    }
#ifdef _WIN32
    _itoa_s(frameID, frameChar, 10, 10);
#else  /* _WIN32 */
    vislib::StringA tmp; /* worst idea ever, but linux does not deserve anything better! */
    tmp.Format("%i", frameID);
    memcpy(frameChar, tmp.PeekBuffer(), 10);
#endif /* _WIN32 */

    this->m_frameLabel->DrawString(0.0f, 0.0f, 0.1f, true, (vislib::StringA("Frame: ") + frameChar).PeekBuffer() , AbstractFont::ALIGN_LEFT_TOP);

    glPopMatrix();

    glPopAttrib();
}


/**
 * protein::ProteinRenderer::RenderLines
 */
void protein::ProteinRenderer::RenderLines( const CallProteinData *prot)
{
    // built the display list if it was not yet created
    if( !glIsList( this->m_proteinDisplayListLines ) )
    {
        // generate a new display list
        this->m_proteinDisplayListLines = glGenLists(1);
        // compile new display list
        glNewList( this->m_proteinDisplayListLines, GL_COMPILE);

        unsigned int i;
        unsigned int currentChain, currentAminoAcid, currentConnection;
        unsigned int first, second;
        vislib::math::Vector<float, 3> firstPos, secondPos;
        // lines can not be lighted --> turn light off
        glDisable(GL_LIGHTING);

        protein::CallProteinData::Chain chain;
//      const protein::CallProteinData::AtomData *protAtomData = prot->ProteinAtomData();
        const float *protAtomPos = prot->ProteinAtomPositions();

        glPushAttrib( GL_ENABLE_BIT | GL_POINT_BIT | GL_LINE_BIT | GL_POLYGON_BIT);
        glEnable( GL_BLEND);
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable( GL_LINE_SMOOTH);
        glEnable( GL_LINE_WIDTH);
        glEnable( GL_POINT_SMOOTH);
        glEnable( GL_POINT_SIZE);
        glLineWidth( 3.0f);
        glPointSize( 6.0f);

        if( this->m_drawDotsWithLine )
        {
            // draw atoms as points
            glBegin( GL_POINTS);
            for( i = 0; i < prot->ProteinAtomCount(); i++ )
            {
                glColor3ubv( this->GetProteinAtomColor( i));
                glVertex3f( protAtomPos[i*3+0], protAtomPos[i*3+1], protAtomPos[i*3+2]);
            }
            glEnd(); // GL_POINTS
        }

        // draw connections as lines
        glBegin( GL_LINES);
        // loop over all chains
        for( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
        {
            chain = prot->ProteinChain( currentChain);
            // loop over all amino acids in the current chain
            for( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
            {
                // loop over all connections of the current amino acid
                for( currentConnection = 0;
                     currentConnection < chain.AminoAcid()[currentAminoAcid].Connectivity().Count();
                     currentConnection++ )
                {
                    first = chain.AminoAcid()[currentAminoAcid].Connectivity()[currentConnection].First();
                    first += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();
                    second = chain.AminoAcid()[currentAminoAcid].Connectivity()[currentConnection].Second();
                    second += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();
                    firstPos = vislib::math::Vector<float, 3>( &protAtomPos[first*3]);
                    secondPos = vislib::math::Vector<float, 3>( &protAtomPos[second*3]);
                    if( ( firstPos - secondPos).Length() < 3.5f ) {
                    glColor3ubv( this->GetProteinAtomColor( first));
                        glVertex3fv( firstPos.PeekComponents());
                    glColor3ubv( this->GetProteinAtomColor( second));
                        glVertex3fv( secondPos.PeekComponents());
                    }
                }
                // try to make the connection between this amino acid and its predecessor
                // --> only possible if the current amino acid is not the first in this chain
                if( currentAminoAcid > 0 )
                {
                    first = chain.AminoAcid()[currentAminoAcid-1].CCarbIndex();
                    first += chain.AminoAcid()[currentAminoAcid-1].FirstAtomIndex();
                    second = chain.AminoAcid()[currentAminoAcid].NIndex();
                    second += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();
                    firstPos = vislib::math::Vector<float, 3>( &protAtomPos[first*3]);
                    secondPos = vislib::math::Vector<float, 3>( &protAtomPos[second*3]);
                    if( ( firstPos - secondPos).Length() < 3.5f ) {
                    glColor3ubv( this->GetProteinAtomColor( first));
                        glVertex3fv( firstPos.PeekComponents());
                    glColor3ubv( this->GetProteinAtomColor( second));
                        glVertex3fv( secondPos.PeekComponents());
                    }
                }
            }
        }
        glEnd(); // GL_LINES

        glPopAttrib();

        glEndList();
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO+200, "Display list for LINES render mode built.");
    }
    else
    {
        //draw the display list
        glCallList( this->m_proteinDisplayListLines);
    }
    // turn light on after rendering
    glEnable(GL_LIGHTING);
    glDisable( GL_BLEND);

}


/*
 * protein::ProteinRenderer::RenderStickRaycasting
 */
void protein::ProteinRenderer::RenderStickRaycasting(
    const CallProteinData *prot)
{
    if( this->m_prepareStickRaycasting )
    {
        unsigned int i1;
        unsigned int first, second;
        unsigned int currentChain, currentAminoAcid, currentConnection;
        const unsigned char *color1;
        const unsigned char *color2;

        // -----------------------------
        // -- computation for spheres --
        // -----------------------------

        // clear vertex array for spheres
        this->m_vertSphereStickRay.Clear();
        // clear color array for sphere colors
        this->m_colorSphereStickRay.Clear();

        // store the points (will be rendered as spheres by the shader)
        for( i1 = 0; i1 < prot->ProteinAtomCount(); i1++)
        {
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+0]);
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+1]);
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+2]);
            this->m_vertSphereStickRay.Add( m_radiusStick);

            color1 = this->GetProteinAtomColor( i1);
            this->m_colorSphereStickRay.Add( color1[0]);
            this->m_colorSphereStickRay.Add( color1[1]);
            this->m_colorSphereStickRay.Add( color1[2]);
        }

        // -------------------------------
        // -- computation for cylinders --
        // -------------------------------
        protein::CallProteinData::Chain chain;
        vislib::math::Quaternion<float> quatC;
        quatC.Set( 0, 0, 0, 1);
        vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
        vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
        float angle;
        // vertex array for cylinders
        this->m_vertCylinderStickRay.Clear();
        // color array for first cylinder colors
        this->m_color1CylinderStickRay.Clear();
        // color array for second cylinder colors
        this->m_color2CylinderStickRay.Clear();
        // attribute array for quaterions
        this->m_quatCylinderStickRay.Clear();
        // attribute array for in-parameters
        this->m_inParaCylStickRaycasting.Clear();

        // loop over all chains
        for( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
        {
            chain = prot->ProteinChain( currentChain);
            // loop over all amino acids in the current chain
            for( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
            {
                // loop over all connections of the current amino acid
                for( currentConnection = 0;
                     currentConnection < chain.AminoAcid()[currentAminoAcid].Connectivity().Count();
                     currentConnection++ )
                {
                    first = chain.AminoAcid()[currentAminoAcid].Connectivity()[currentConnection].First();
                    first += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();
                    second = chain.AminoAcid()[currentAminoAcid].Connectivity()[currentConnection].Second();
                    second += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();

                    firstAtomPos.SetX( prot->ProteinAtomPositions()[first*3+0]);
                    firstAtomPos.SetY( prot->ProteinAtomPositions()[first*3+1]);
                    firstAtomPos.SetZ( prot->ProteinAtomPositions()[first*3+2]);
                    color1 = this->GetProteinAtomColor( first);

                    secondAtomPos.SetX( prot->ProteinAtomPositions()[second*3+0]);
                    secondAtomPos.SetY( prot->ProteinAtomPositions()[second*3+1]);
                    secondAtomPos.SetZ( prot->ProteinAtomPositions()[second*3+2]);
                    color2 = this->GetProteinAtomColor( second);

                    // compute the quaternion for the rotation of the cylinder
                    dir = secondAtomPos - firstAtomPos;
                    tmpVec.Set( 1.0f, 0.0f, 0.0f);
                    angle = - tmpVec.Angle( dir);
                    ortho = tmpVec.Cross( dir);
                    ortho.Normalise();
                    quatC.Set( angle, ortho);
                    // compute the absolute position 'position' of the cylinder (center point)
                    position = firstAtomPos + (dir/2.0f);

                    this->m_inParaCylStickRaycasting.Add( m_radiusStick);
                    this->m_inParaCylStickRaycasting.Add( abs( (firstAtomPos-secondAtomPos).Length()));

                    this->m_quatCylinderStickRay.Add( quatC.GetX());
                    this->m_quatCylinderStickRay.Add( quatC.GetY());
                    this->m_quatCylinderStickRay.Add( quatC.GetZ());
                    this->m_quatCylinderStickRay.Add( quatC.GetW());

                    this->m_color1CylinderStickRay.Add( float(int(color1[0]))/255.0f);
                    this->m_color1CylinderStickRay.Add( float(int(color1[1]))/255.0f);
                    this->m_color1CylinderStickRay.Add( float(int(color1[2]))/255.0f);

                    this->m_color2CylinderStickRay.Add( float(int(color2[0]))/255.0f);
                    this->m_color2CylinderStickRay.Add( float(int(color2[1]))/255.0f);
                    this->m_color2CylinderStickRay.Add( float(int(color2[2]))/255.0f);

                    this->m_vertCylinderStickRay.Add( position.GetX());
                    this->m_vertCylinderStickRay.Add( position.GetY());
                    this->m_vertCylinderStickRay.Add( position.GetZ());
                    this->m_vertCylinderStickRay.Add( 1.0f);
                }
                // try to make the connection between this amino acid and its predecessor
                // --> only possible if the current amino acid is not the first in this chain
                if( currentAminoAcid > 0 )
                {
                    first = chain.AminoAcid()[currentAminoAcid-1].CCarbIndex();
                    first += chain.AminoAcid()[currentAminoAcid-1].FirstAtomIndex();
                    second = chain.AminoAcid()[currentAminoAcid].NIndex();
                    second += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();

                    firstAtomPos.SetX( prot->ProteinAtomPositions()[first*3+0]);
                    firstAtomPos.SetY( prot->ProteinAtomPositions()[first*3+1]);
                    firstAtomPos.SetZ( prot->ProteinAtomPositions()[first*3+2]);
                    color1 = this->GetProteinAtomColor( first);

                    secondAtomPos.SetX( prot->ProteinAtomPositions()[second*3+0]);
                    secondAtomPos.SetY( prot->ProteinAtomPositions()[second*3+1]);
                    secondAtomPos.SetZ( prot->ProteinAtomPositions()[second*3+2]);
                    color2 = this->GetProteinAtomColor( second);

                    // compute the quaternion for the rotation of the cylinder
                    dir = secondAtomPos - firstAtomPos;
                    tmpVec.Set( 1.0f, 0.0f, 0.0f);
                    angle = - tmpVec.Angle( dir);
                    ortho = tmpVec.Cross( dir);
                    ortho.Normalise();
                    quatC.Set( angle, ortho);
                    // compute the absolute position 'position' of the cylinder (center point)
                    position = firstAtomPos + (dir/2.0f);

                    // don't draw bonds that are too long
                    if ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) > 3.5f )
                        continue;

                    this->m_inParaCylStickRaycasting.Add( m_radiusStick);
                    this->m_inParaCylStickRaycasting.Add( abs( (firstAtomPos-secondAtomPos).Length()));

                    this->m_quatCylinderStickRay.Add( quatC.GetX());
                    this->m_quatCylinderStickRay.Add( quatC.GetY());
                    this->m_quatCylinderStickRay.Add( quatC.GetZ());
                    this->m_quatCylinderStickRay.Add( quatC.GetW());

                    this->m_color1CylinderStickRay.Add( float(int(color1[0]))/255.0f);
                    this->m_color1CylinderStickRay.Add( float(int(color1[1]))/255.0f);
                    this->m_color1CylinderStickRay.Add( float(int(color1[2]))/255.0f);

                    this->m_color2CylinderStickRay.Add( float(int(color2[0]))/255.0f);
                    this->m_color2CylinderStickRay.Add( float(int(color2[1]))/255.0f);
                    this->m_color2CylinderStickRay.Add( float(int(color2[2]))/255.0f);

                    this->m_vertCylinderStickRay.Add( position.GetX());
                    this->m_vertCylinderStickRay.Add( position.GetY());
                    this->m_vertCylinderStickRay.Add( position.GetZ());
                    this->m_vertCylinderStickRay.Add( 1.0f);
                }
            }
        }

        this->m_prepareStickRaycasting = false;
    }

    // -----------
    // -- draw  --
    // -----------
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable( GL_BLEND);

    // enable sphere shader
    this->m_sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->m_sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->m_vertSphereStickRay.PeekElements());
    glColorPointer( 3, GL_UNSIGNED_BYTE, 0, this->m_colorSphereStickRay.PeekElements());
    glDrawArrays( GL_POINTS, 0, (unsigned int)(this->m_vertSphereStickRay.Count()/4));
    // disable sphere shader
    this->m_sphereShader.Disable();

    // enable cylinder shader
    this->m_cylinderShader.Enable();
    // set shader variables
    glUniform4fvARB(this->m_cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->m_cylinderShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->m_cylinderShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->m_cylinderShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // get the attribute locations
    m_attribLocInParams = glGetAttribLocationARB( this->m_cylinderShader, "inParams");
    m_attribLocQuatC = glGetAttribLocationARB( this->m_cylinderShader, "quatC");
    m_attribLocColor1 = glGetAttribLocationARB( this->m_cylinderShader, "color1");
    m_attribLocColor2 = glGetAttribLocationARB( this->m_cylinderShader, "color2");
    // enable vertex attribute arrays for the attribute locations
    glDisableClientState(GL_COLOR_ARRAY);
    glEnableVertexAttribArrayARB( this->m_attribLocInParams);
    glEnableVertexAttribArrayARB( this->m_attribLocQuatC);
    glEnableVertexAttribArrayARB( this->m_attribLocColor1);
    glEnableVertexAttribArrayARB( this->m_attribLocColor2);
    // set vertex and attribute pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->m_vertCylinderStickRay.PeekElements());
    glVertexAttribPointerARB( this->m_attribLocInParams, 2, GL_FLOAT, 0, 0, this->m_inParaCylStickRaycasting.PeekElements());
    glVertexAttribPointerARB( this->m_attribLocQuatC, 4, GL_FLOAT, 0, 0, this->m_quatCylinderStickRay.PeekElements());
    glVertexAttribPointerARB( this->m_attribLocColor1, 3, GL_FLOAT, 0, 0, this->m_color1CylinderStickRay.PeekElements());
    glVertexAttribPointerARB( this->m_attribLocColor2, 3, GL_FLOAT, 0, 0, this->m_color2CylinderStickRay.PeekElements());
    glDrawArrays( GL_POINTS, 0, (unsigned int)(this->m_vertCylinderStickRay.Count()/4));
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArrayARB( this->m_attribLocInParams);
    glDisableVertexAttribArrayARB( this->m_attribLocQuatC);
    glDisableVertexAttribArrayARB( this->m_attribLocColor1);
    glDisableVertexAttribArrayARB( this->m_attribLocColor2);
    glDisableClientState(GL_VERTEX_ARRAY);
    // disable cylinder shader
    this->m_cylinderShader.Disable();

}


/*
 * protein::ProteinRenderer::RenderBallAndStick
 */
void protein::ProteinRenderer::RenderBallAndStick(
    const CallProteinData *prot)
{
    if( this->m_prepareBallAndStick )
    {
        unsigned int i1;
        unsigned int first, second;
        unsigned int currentChain, currentAminoAcid, currentConnection;
        const unsigned char *color1;
        const unsigned char *color2;

        // -----------------------------
        // -- computation for spheres --
        // -----------------------------

        // clear vertex array for spheres
        this->m_vertSphereStickRay.Clear();
        // clear color array for sphere colors
        this->m_colorSphereStickRay.Clear();

        this->m_vertSphereStickRay.AssertCapacity( 4 * prot->ProteinAtomCount() );
        this->m_colorSphereStickRay.AssertCapacity( 3 * prot->ProteinAtomCount() );
        // store the points (will be rendered as spheres by the shader)
        for( i1 = 0; i1 < prot->ProteinAtomCount(); i1++)
        {
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+0]);
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+1]);
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+2]);
            this->m_vertSphereStickRay.Add( m_radiusStick);

            color1 = this->GetProteinAtomColor( i1);
            this->m_colorSphereStickRay.Add( color1[0]);
            this->m_colorSphereStickRay.Add( color1[1]);
            this->m_colorSphereStickRay.Add( color1[2]);
        }

        // -------------------------------
        // -- computation for cylinders --
        // -------------------------------
        protein::CallProteinData::Chain chain;
        vislib::math::Quaternion<float> quatC;
        quatC.Set( 0, 0, 0, 1);
        vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
        vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
        float angle;
        // vertex array for cylinders
        this->m_vertCylinderStickRay.Clear();
        // color array for first cylinder colors
        this->m_color1CylinderStickRay.Clear();
        // color array for second cylinder colors
        this->m_color2CylinderStickRay.Clear();
        // attribute array for quaterions
        this->m_quatCylinderStickRay.Clear();
        // attribute array for in-parameters
        this->m_inParaCylStickRaycasting.Clear();

        // loop over all chains
        for( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
        {
            chain = prot->ProteinChain( currentChain);
            // loop over all amino acids in the current chain
            for( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
            {
                // loop over all connections of the current amino acid
                for( currentConnection = 0;
                     currentConnection < chain.AminoAcid()[currentAminoAcid].Connectivity().Count();
                     currentConnection++ )
                {
                    first = chain.AminoAcid()[currentAminoAcid].Connectivity()[currentConnection].First();
                    first += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();
                    second = chain.AminoAcid()[currentAminoAcid].Connectivity()[currentConnection].Second();
                    second += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();

                    firstAtomPos.SetX( prot->ProteinAtomPositions()[first*3+0]);
                    firstAtomPos.SetY( prot->ProteinAtomPositions()[first*3+1]);
                    firstAtomPos.SetZ( prot->ProteinAtomPositions()[first*3+2]);
                    color1 = this->GetProteinAtomColor( first);

                    secondAtomPos.SetX( prot->ProteinAtomPositions()[second*3+0]);
                    secondAtomPos.SetY( prot->ProteinAtomPositions()[second*3+1]);
                    secondAtomPos.SetZ( prot->ProteinAtomPositions()[second*3+2]);
                    color2 = this->GetProteinAtomColor( second);

                    // compute the quaternion for the rotation of the cylinder
                    dir = secondAtomPos - firstAtomPos;
                    tmpVec.Set( 1.0f, 0.0f, 0.0f);
                    angle = - tmpVec.Angle( dir);
                    ortho = tmpVec.Cross( dir);
                    ortho.Normalise();
                    quatC.Set( angle, ortho);
                    // compute the absolute position 'position' of the cylinder (center point)
                    position = firstAtomPos + (dir/2.0f);

                    this->m_inParaCylStickRaycasting.Add( m_radiusStick/3.0f);
                    this->m_inParaCylStickRaycasting.Add( abs( (firstAtomPos-secondAtomPos).Length()));

                    this->m_quatCylinderStickRay.Add( quatC.GetX());
                    this->m_quatCylinderStickRay.Add( quatC.GetY());
                    this->m_quatCylinderStickRay.Add( quatC.GetZ());
                    this->m_quatCylinderStickRay.Add( quatC.GetW());

                    this->m_color1CylinderStickRay.Add( float(int(color1[0]))/255.0f);
                    this->m_color1CylinderStickRay.Add( float(int(color1[1]))/255.0f);
                    this->m_color1CylinderStickRay.Add( float(int(color1[2]))/255.0f);

                    this->m_color2CylinderStickRay.Add( float(int(color2[0]))/255.0f);
                    this->m_color2CylinderStickRay.Add( float(int(color2[1]))/255.0f);
                    this->m_color2CylinderStickRay.Add( float(int(color2[2]))/255.0f);

                    this->m_vertCylinderStickRay.Add( position.GetX());
                    this->m_vertCylinderStickRay.Add( position.GetY());
                    this->m_vertCylinderStickRay.Add( position.GetZ());
                    this->m_vertCylinderStickRay.Add( 1.0f);
                }
                // try to make the connection between this amino acid and its predecessor
                // --> only possible if the current amino acid is not the first in this chain
                if( currentAminoAcid > 0 )
                {
                    first = chain.AminoAcid()[currentAminoAcid-1].CCarbIndex();
                    first += chain.AminoAcid()[currentAminoAcid-1].FirstAtomIndex();
                    second = chain.AminoAcid()[currentAminoAcid].NIndex();
                    second += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();

                    firstAtomPos.SetX( prot->ProteinAtomPositions()[first*3+0]);
                    firstAtomPos.SetY( prot->ProteinAtomPositions()[first*3+1]);
                    firstAtomPos.SetZ( prot->ProteinAtomPositions()[first*3+2]);
                    color1 = this->GetProteinAtomColor( first);

                    secondAtomPos.SetX( prot->ProteinAtomPositions()[second*3+0]);
                    secondAtomPos.SetY( prot->ProteinAtomPositions()[second*3+1]);
                    secondAtomPos.SetZ( prot->ProteinAtomPositions()[second*3+2]);
                    color2 = this->GetProteinAtomColor( second);

                    // compute the quaternion for the rotation of the cylinder
                    dir = secondAtomPos - firstAtomPos;
                    tmpVec.Set( 1.0f, 0.0f, 0.0f);
                    angle = - tmpVec.Angle( dir);
                    ortho = tmpVec.Cross( dir);
                    ortho.Normalise();
                    quatC.Set( angle, ortho);
                    // compute the absolute position 'position' of the cylinder (center point)
                    position = firstAtomPos + (dir/2.0f);

                    // don't draw bonds that are too long
                    if ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) > 3.5f )
                        continue;

                    this->m_inParaCylStickRaycasting.Add( m_radiusStick/3.0f);
                    this->m_inParaCylStickRaycasting.Add( abs( (firstAtomPos-secondAtomPos).Length()));

                    this->m_quatCylinderStickRay.Add( quatC.GetX());
                    this->m_quatCylinderStickRay.Add( quatC.GetY());
                    this->m_quatCylinderStickRay.Add( quatC.GetZ());
                    this->m_quatCylinderStickRay.Add( quatC.GetW());

                    this->m_color1CylinderStickRay.Add( float(int(color1[0]))/255.0f);
                    this->m_color1CylinderStickRay.Add( float(int(color1[1]))/255.0f);
                    this->m_color1CylinderStickRay.Add( float(int(color1[2]))/255.0f);

                    this->m_color2CylinderStickRay.Add( float(int(color2[0]))/255.0f);
                    this->m_color2CylinderStickRay.Add( float(int(color2[1]))/255.0f);
                    this->m_color2CylinderStickRay.Add( float(int(color2[2]))/255.0f);

                    this->m_vertCylinderStickRay.Add( position.GetX());
                    this->m_vertCylinderStickRay.Add( position.GetY());
                    this->m_vertCylinderStickRay.Add( position.GetZ());
                    this->m_vertCylinderStickRay.Add( 1.0f);
                }
            }
        }
        this->m_prepareBallAndStick = false;
    }

    // -----------
    // -- draw  --
    // -----------
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable( GL_BLEND);

    // enable sphere shader
    this->m_sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->m_sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->m_vertSphereStickRay.PeekElements());
    glColorPointer( 3, GL_UNSIGNED_BYTE, 0, this->m_colorSphereStickRay.PeekElements());
    glDrawArrays( GL_POINTS, 0, (unsigned int)(this->m_vertSphereStickRay.Count()/4));
    // disable sphere shader
    glDisableClientState(GL_COLOR_ARRAY);
    this->m_sphereShader.Disable();

    // enable cylinder shader
    this->m_cylinderShader.Enable();
    // set shader variables
    glUniform4fvARB(this->m_cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->m_cylinderShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->m_cylinderShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->m_cylinderShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // get the attribute locations
    m_attribLocInParams = glGetAttribLocationARB( this->m_cylinderShader, "inParams");
    m_attribLocQuatC = glGetAttribLocationARB( this->m_cylinderShader, "quatC");
    m_attribLocColor1 = glGetAttribLocationARB( this->m_cylinderShader, "color1");
    m_attribLocColor2 = glGetAttribLocationARB( this->m_cylinderShader, "color2");
    // enable vertex attribute arrays for the attribute locations
    glEnableVertexAttribArrayARB( this->m_attribLocInParams);
    glEnableVertexAttribArrayARB( this->m_attribLocQuatC);
    glEnableVertexAttribArrayARB( this->m_attribLocColor1);
    glEnableVertexAttribArrayARB( this->m_attribLocColor2);
    // set vertex and attribute pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->m_vertCylinderStickRay.PeekElements());
    glVertexAttribPointerARB( this->m_attribLocInParams, 2, GL_FLOAT, 0, 0, this->m_inParaCylStickRaycasting.PeekElements());
    glVertexAttribPointerARB( this->m_attribLocQuatC, 4, GL_FLOAT, 0, 0, this->m_quatCylinderStickRay.PeekElements());
    glVertexAttribPointerARB( this->m_attribLocColor1, 3, GL_FLOAT, 0, 0, this->m_color1CylinderStickRay.PeekElements());
    glVertexAttribPointerARB( this->m_attribLocColor2, 3, GL_FLOAT, 0, 0, this->m_color2CylinderStickRay.PeekElements());
    glDrawArrays( GL_POINTS, 0, (unsigned int)(this->m_vertCylinderStickRay.Count()/4));
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArrayARB( this->m_attribLocInParams);
    glDisableVertexAttribArrayARB( this->m_attribLocQuatC);
    glDisableVertexAttribArrayARB( this->m_attribLocColor1);
    glDisableVertexAttribArrayARB( this->m_attribLocColor2);
    glDisableClientState(GL_VERTEX_ARRAY);
    // disable cylinder shader
    this->m_cylinderShader.Disable();

}


/*
 * protein::ProteinRenderer::RenderSpacefilling
 */
void protein::ProteinRenderer::RenderSpacefilling(
    const CallProteinData *prot)
{
    if( this->m_prepareSpacefilling )
    {
        unsigned int i1;
        const unsigned char *color1;

        // -----------------------------
        // -- computation for spheres --
        // -----------------------------

        // clear vertex array for spheres
        this->m_vertSphereStickRay.Clear();
        // clear color array for sphere colors
        this->m_colorSphereStickRay.Clear();

        // store the points (will be rendered as spheres by the shader)
        for( i1 = 0; i1 < prot->ProteinAtomCount(); i1++)
        {
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+0]);
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+1]);
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+2]);
            this->m_vertSphereStickRay.Add( prot->AtomTypes()[prot->ProteinAtomData()[i1].TypeIndex()].Radius() );

            color1 = this->GetProteinAtomColor( i1);
            this->m_colorSphereStickRay.Add( color1[0]);
            this->m_colorSphereStickRay.Add( color1[1]);
            this->m_colorSphereStickRay.Add( color1[2]);
        }

        this->m_prepareSpacefilling = false;
    }

    // -----------
    // -- draw  --
    // -----------
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable( GL_BLEND);

    // enable sphere shader
    this->m_sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->m_sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->m_vertSphereStickRay.PeekElements());
    glColorPointer( 3, GL_UNSIGNED_BYTE, 0, this->m_colorSphereStickRay.PeekElements());
    glDrawArrays( GL_POINTS, 0, (unsigned int)(this->m_vertSphereStickRay.Count()/4));
    // disable sphere shader
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    this->m_sphereShader.Disable();
}


/*
 * protein::ProteinRenderer::RenderSolventAccessibleSurface
 */
void protein::ProteinRenderer::RenderSolventAccessibleSurface(
    const CallProteinData *prot)
{
    if( this->m_prepareSAS )
    {
        unsigned int i1;
        const unsigned char *color1;

        // -----------------------------
        // -- computation for spheres --
        // -----------------------------

        // clear vertex array for spheres
        this->m_vertSphereStickRay.Clear();
        // clear color array for sphere colors
        this->m_colorSphereStickRay.Clear();

        // store the points (will be rendered as spheres by the shader)
        for( i1 = 0; i1 < prot->ProteinAtomCount(); i1++)
        {
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+0]);
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+1]);
            this->m_vertSphereStickRay.Add( prot->ProteinAtomPositions()[i1*3+2]);
            this->m_vertSphereStickRay.Add( prot->AtomTypes()[prot->ProteinAtomData()[i1].TypeIndex()].Radius() + this->m_probeRadius );

            color1 = this->GetProteinAtomColor( i1);
            this->m_colorSphereStickRay.Add( color1[0]);
            this->m_colorSphereStickRay.Add( color1[1]);
            this->m_colorSphereStickRay.Add( color1[2]);
        }

        this->m_prepareSAS = false;
    }

    // -----------
    // -- draw  --
    // -----------
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable( GL_BLEND);

    // enable sphere shader
    this->m_sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->m_sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->m_sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->m_vertSphereStickRay.PeekElements());
    glColorPointer( 3, GL_UNSIGNED_BYTE, 0, this->m_colorSphereStickRay.PeekElements());
    glDrawArrays( GL_POINTS, 0, (unsigned int)(this->m_vertSphereStickRay.Count()/4));
    // disable sphere shader
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    this->m_sphereShader.Disable();
}


/*
 * protein::ProteinRenderer::RenderDisulfideBondsLine
 */
void protein::ProteinRenderer::RenderDisulfideBondsLine(
    const CallProteinData *prot)
{
    // return if there are no disulfide bonds or m_drawDisulfideBonds is false
    if( prot->DisulfidBondsCount() <= 0 || !m_drawDisulfideBonds )
        return;
    // lines can not be lighted --> turn light off
    glDisable(GL_LIGHTING);
    // built the display list if it was not yet created
    if( !glIsList( this->m_disulfideBondsDisplayList ) )
    {
        // generate a new display list
        this->m_disulfideBondsDisplayList = glGenLists(1);
        // compile new display list
        glNewList( this->m_disulfideBondsDisplayList, GL_COMPILE);

        unsigned int i;
        unsigned int first, second;

        const float *protAtomPos = prot->ProteinAtomPositions();

        glPushAttrib( GL_ENABLE_BIT | GL_POINT_BIT | GL_LINE_BIT | GL_POLYGON_BIT);
        glEnable( GL_BLEND);
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable( GL_LINE_SMOOTH);
        glEnable( GL_LINE_WIDTH);
        //glEnable( GL_LINE_STIPPLE);
        //glLineStipple( 1, 0xFF00);
        glLineWidth( 3.0f);

        // set color of disulfide bonds to yellow
        glColor3f( 1.0f, 1.0f, 0.0f);
        // draw bonds
        glBegin( GL_LINES);
        for( i = 0; i < prot->DisulfidBondsCount(); i++)
        {
            first = prot->DisulfidBonds()[i].First();
            second = prot->DisulfidBonds()[i].Second();
            glVertex3f( protAtomPos[first*3+0], protAtomPos[first*3+1], protAtomPos[first*3+2]);
            glVertex3f( protAtomPos[second*3+0], protAtomPos[second*3+1], protAtomPos[second*3+2]);
        }
        glEnd(); // GL_LINES

        glPopAttrib();

        glEndList();
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO+200, "Display list for disulfide bonds built.");
    }
    else
    {
        //draw the display list
        glCallList( this->m_disulfideBondsDisplayList);
    }
    // turn light on after rendering
    glEnable(GL_LIGHTING);
    glDisable( GL_BLEND);
}

/*
 * protein::ProteinRenderer::RecomputeAll
 */
void protein::ProteinRenderer::RecomputeAll()
{
    this->m_prepareBallAndStick = true;
    this->m_prepareSpacefilling = true;
    this->m_prepareStickRaycasting = true;
    this->m_prepareSAS = true;

    glDeleteLists( this->m_disulfideBondsDisplayList, 1);
    this->m_disulfideBondsDisplayList = 0;

    glDeleteLists( this->m_proteinDisplayListLines, 1);
    this->m_proteinDisplayListLines = 0;

    this->m_protAtomColorTable.Clear();
}
