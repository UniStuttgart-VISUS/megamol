/*
 * HapticsMoleculeRenderer.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "HapticsMoleculeRenderer.h"

#ifdef WITH_OPENHAPTICS

#define _USE_MATH_DEFINES 1

#include "mmcore/CoreInstance.h"
#include "Color.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
#include "ForceDataCall.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>
#include <omp.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * protein::HapticsMoleculeRenderer::HapticsMoleculeRenderer (CTOR)
 */
HapticsMoleculeRenderer::HapticsMoleculeRenderer(void) : Renderer3DModule (),
    molDataCallerSlot( "getData", "Connects the molecule rendering with molecule data storage"),
    forceDataOutSlot( "forcedataout", "The slot providing the force data"),
    colorTableFileParam( "colorTableFilename", "The filename of the color table."),
    coloringModeParam( "coloringMode", "The coloring mode."),
    renderModeParam( "renderMode", "The rendering mode."),
    stickRadiusParam( "stickRadius", "The radius for stick rendering"),
    probeRadiusParam( "probeRadius", "The probe radius for SAS rendering"),
    minGradColorParam( "minGradColor", "The color for the minimum value for gradient coloring" ),
    midGradColorParam( "midGradColor", "The color for the middle value for gradient coloring" ),
    maxGradColorParam( "maxGradColor", "The color for the maximum value for gradient coloring" ),
    molIdxListParam( "molIdxList", "The list of molecule indices for RS computation:"),
    specialColorParam( "specialColor", "The color for the specified molecules" ),
    interpolParam( "posInterpolation", "Enable positional interpolation between frames" ),
    multiforceParam("multiforce", "Enable multiple forces to be applied simulataneously"),
    width( 0), height( 0), currentDragAtom(-1), atomPositions( 0), atomPositionCount( 0),
    forceCount(0), forceAtomIDs(NULL), forces(NULL)
{
    // Caller to get molecule data from loader
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);

    // Callback for sending force data to loader
    this->forceDataOutSlot.SetCallback( ForceDataCall::ClassName(), ForceDataCall::FunctionName(ForceDataCall::CallForGetForceData), &HapticsMoleculeRenderer::getForceData);
    this->MakeSlotAvailable( &this->forceDataOutSlot);

    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    Color::ReadColorTableFromFile( filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(new param::StringParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);
    
    // switch between immediate mode and vertex array rendering
    this->renderModeParam.SetParameter(new param::BoolParam( true));
    this->MakeSlotAvailable( &this->renderModeParam);

    // coloring mode
    //this->currentColoringMode = ELEMENT;
    this->currentColoringMode = Color::RESIDUE;
    param::EnumParam *cm = new param::EnumParam(int(this->currentColoringMode));
    cm->SetTypePair( Color::ELEMENT, "Element");
    cm->SetTypePair( Color::RESIDUE, "Residue");
    cm->SetTypePair( Color::STRUCTURE, "Structure");
    cm->SetTypePair( Color::BFACTOR, "BFactor");
    cm->SetTypePair( Color::CHARGE, "Charge");
    cm->SetTypePair( Color::OCCUPANCY, "Occupancy");
    cm->SetTypePair( Color::CHAIN, "Chain");
    cm->SetTypePair( Color::MOLECULE, "Molecule");
    cm->SetTypePair( Color::RAINBOW, "Rainbow");
    this->coloringModeParam << cm;
    this->MakeSlotAvailable( &this->coloringModeParam);

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

    // en-/disable multiple forces
    this->multiforceParam.SetParameter(new param::BoolParam( false));
    this->MakeSlotAvailable( &this->multiforceParam);

    // --- set the radius for the arrows ---
    this->radiusArrow = 0.15f;

    // make the rainbow color table
    Color::MakeRainbowColorTable( 100, this->rainbowColors);
}


/*
 * protein::HapticsMoleculeRenderer::~HapticsMoleculeRenderer (DTOR)
 */
HapticsMoleculeRenderer::~HapticsMoleculeRenderer(void)  {
    this->Release();
}


/*
 * protein::HapticsMoleculeRenderer::release
 */
void HapticsMoleculeRenderer::release(void) {
    hlDeleteShapes(this->shaderID, 1);
}


/*
 * protein::HapticsMoleculeRenderer::create
 */
bool HapticsMoleculeRenderer::create(void) {
    if (ogl_IsVersionGEQ(2,0) == 0) {
        return false;
    }
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }
    if (!vislib::graphics::gl::FramebufferObject::InitialiseExtensions()) {
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
    
    ///////////////////////////////////////////////////////////////
    // load the shader source for the sphere Id writing renderer //
    ///////////////////////////////////////////////////////////////
    if( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::writeSphereIdVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere id writing shader", this->ClassName() );
        return false;
    }
    if( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::writeSphereIdFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for sphere id writing shader", this->ClassName() );
        return false;
    }
    try {
        if( !this->writeSphereIdShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create sphere Id writing shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    // set up the absolute 3D cursor for use with the Phantom pen
    this->absoluteCursor3d.SetButtonCount(1);

    /* Setup the phantom pen */
    if (!this->phantom.Initialize()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Phantom device failed to initialize");
        return false;
    }
    this->phantom.SetCursor(&this->absoluteCursor3d);
    this->phantom.SetSpringAttributes(0.50, 0.25); // set the spring attributes
    //this->phantom.SetObjectPositionData(&this->objectData); // associate the object position data table
    this->phantom.EnableMotionInterrupts(); // enable motion interrupts for cursor
    this->phantom.SetAngularMotionTolerance(0.05); // set the tolerances to more reasonable values
    this->phantom.SetLinearMotionTolerance(0.1); // especially this one, which is normally way too high
    this->phantom.SetButtonFunction(PhantomButtonDelegate(*this, &HapticsMoleculeRenderer::getAtomID)); // set button callback function

    // Generate ids for the shaders
    // NOTE: Shapes must be rendered in the order in which their ids were generated
    // in order for event callbacks to return the correct shape id.
    this->shaderID = hlGenShapes( 1); // get shape id

    hlTouchableFace(HL_FRONT); // set the front faces as touchable

    return true;
}


/*
 * protein::HapticsMoleculeRenderer::GetExtents
 */
bool HapticsMoleculeRenderer::GetExtents(Call& call) {
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
 * protein::HapticsMoleculeRenderer::Render
 */
bool HapticsMoleculeRenderer::Render(Call& call) {
    
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL) return false;

    // set call time
    mol->SetCalltime(callTime);
    // set frame ID and call data
    mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;
    // check if atom count is zero
    if( mol->AtomCount() == 0 ) return true;
    // get positions of the first frame
    if( this->atomPositionCount != mol->AtomCount() ) {
        if( this->atomPositions )
            delete[] this->atomPositions;
        this->atomPositions = new float[mol->AtomCount() * 3];
        this->atomPositionCount = mol->AtomCount();
    }
    memcpy( this->atomPositions, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));

    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);
    
    // ---------- Query Camera View Dimensions ----------
    if( static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight()) != this->height ) {
        // store new view width and height
        this->width = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight());
        // resize the FBO
        this->sphereIdFbo.Create( this->width, this->height, GL_R32F, GL_RED, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }

    // ---------- update parameters ----------
    this->UpdateParameters( mol);

    // recompute color table, if necessary
    if( this->atomColorTable.Count()/3 < mol->AtomCount() ) {

        Color::MakeColorTable(mol,
          this->currentColoringMode,
          this->atomColorTable, this->colorLookupTable, this->rainbowColors,
          this->minGradColorParam.Param<param::StringParam>()->Value(),
          this->midGradColorParam.Param<param::StringParam>()->Value(),
          this->maxGradColorParam.Param<param::StringParam>()->Value(),
          true);

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

    // ---------- Phantom Device Updates -----------

        // Update phantom pen
    double projection[16]; // create a temp storage array for the projection matrix
    glGetDoublev(GL_PROJECTION_MATRIX, projection); // get the projection matrix
    this->phantom.UpdateWorkspace(projection); // update the workspace

    // get the gl modelview matrix for use by the phantom
    double modelview[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    this->phantom.SetModelviewMatrix(modelview);

    // Polling version of cursor motion detection.
    //this->phantom.DisableMotionInterrupts(); // disable motion interrupts
    //// poll the device for the proxy transform and set the cursor location using the transform
    //this->absoluteCursor3d.SetTransform(this->phantom.GetProxyTransform());

    // check if forces are being applied to an atom and compute these forces
    if (this->currentDragAtom != -1) {
        // currently dragging an atom - update this atom's position for the phantom
        vislib::math::Point<float, 3> objectPosition(&this->atomPositions[this->currentDragAtom * 3]);
        this->phantom.SetObjectPosition(objectPosition);

        // get the force being applied
        vislib::math::Point<float, 3> springPoint = this->phantom.GetSpringAnchorPoint();
        vislib::math::Point<float, 3> cursorPosition = this->absoluteCursor3d.GetCurrentPosition();

        // compute a vector between the spring point and the cursor
        vislib::math::Vector<float, 3> forceVector = cursorPosition - springPoint;

        // Do some scaling, perhaps?
        forceVector *= 1.0f; // What factor, here?
        
        // add force if it doesn't yet exist - otherwise replace existing force
        if (!this->forceAtomIDs.Contains(this->currentDragAtom)) {
            // array doesn't contain this force yet - add it
            // First allocate space for the new force
            this->forceCount += 1; 
            // Create space in arrays if necessary
            this->forceAtomIDs.AssertCapacity(forceCount);
            this->forces.AssertCapacity(forceCount*3); // 3 floats per atom

            // Then add the force
            this->forceAtomIDs.Add(this->currentDragAtom);
            this->forces.Add(forceVector.GetX());
            this->forces.Add(forceVector.GetY());
            this->forces.Add(forceVector.GetZ());
        } else {
            // array already contains this atom id - replace existing force
            int index = static_cast<int>(this->forceAtomIDs.IndexOf(this->currentDragAtom));
            index *= 3;
            this->forces.Erase(index, 3); // remove the previous values
            this->forces.Insert(index, forceVector.GetZ()); // write the new values
            this->forces.Insert(index, forceVector.GetY());
            this->forces.Insert(index, forceVector.GetX());
        }
    }

    // TODO: ---------- render ----------

    glClear(GL_DEPTH_BUFFER_BIT); // Have to do this in order to prevent bounding box from being haptically visible

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    hlBeginFrame();
    this->phantom.UpdateDevice(); // device callbacks

    // render data using the current rendering mode
    //if( this->renderModeParam.Param<param::BoolParam>()->Value() ) {
        this->RenderSpacefilling( mol, this->atomPositions);
    //} else {
    //    this->RenderSpacefillingImmediateMode( mol, this->atomPositions);
    //}
    hlEndFrame();
    // get the ID of the sphere
    //this->WriteAtomId( mol, this->atomPositions);

    /* TODO: This could really be moved to getAtomID, since this information is only needed on phantom button click */
    // Get the ID of atom whose surface is closest to the proxy and the distance between the proxy and this surface
    vislib::math::Point<float, 3> proxyPos = this->absoluteCursor3d.GetCurrentPosition();
    this->closestAtomDistance = cr3d->GetBoundingBoxes().ObjectSpaceBBox().LongestEdge(); // some arbitrarily large number
    this->closestAtomID = -1; // no closest atom
    // couning variable
    int cnt;
    // arrays temporary results
    this->tmpClosestAtomDistance.SetCount( omp_get_max_threads());
    this->tmpClosestAtomID.SetCount( omp_get_max_threads());
    // reset tmp arrays
    for( cnt = 0; cnt < omp_get_max_threads(); cnt += 1 ) {
        this->tmpClosestAtomDistance[cnt] = this->closestAtomDistance;
        this->tmpClosestAtomID[cnt] = -1;
    }
#pragma omp parallel for
    for (cnt = 0; cnt < static_cast<int>(mol->AtomCount()); cnt += 1) {
        /* Run through the entire list of atom positions
         * 1. Get the distance between the atom center and the proxy
         * 2. Subtract the atom radius from this distance.
         * 3. Compare this distance to the current minimum distance
         * 4. If it is less, store this distance and the associated atom id.
         */
        vislib::math::Point<float, 3> atomPos(&this->atomPositions[cnt*3]); // get the current atom location
        float distance = (atomPos - proxyPos).Length(); // get the proxy to atom distance
        distance -= mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius(); // subtract the radius for the proxy to surface distance
        if (distance < this->tmpClosestAtomDistance[omp_get_thread_num()]) {
            // new winner
            this->tmpClosestAtomDistance[omp_get_thread_num()] = distance;
            this->tmpClosestAtomID[omp_get_thread_num()] = cnt;
        }
    }
    // search for global closest distance
    for( cnt = 0; cnt < omp_get_max_threads(); cnt += 1 ) {
        if (this->closestAtomDistance > this->tmpClosestAtomDistance[cnt]) {
            // new winner
            this->closestAtomDistance = this->tmpClosestAtomDistance[cnt];
            this->closestAtomID = this->tmpClosestAtomID[cnt];
        }
    }

    this->Render3DCursor(); // render the haptic cursor
    this->RenderForceArrows(); // render the force arrows

    glPopMatrix();

    // unlock the current frame
    mol->Unlock();

    return true;
}

/*
 * HapticsMoleculeRenderer::WriteAtomId
 */
void HapticsMoleculeRenderer::WriteAtomId( const MolecularDataCall *mol, const float *atomPos) {
    // ---------- render all atoms to a FBO with color set to atom Id ----------
    // start rendering to visibility FBO
    this->sphereIdFbo.Enable();
    // set clear color & clear
    glClearColor( -1.0f, 0.0f, 0.0f, 1.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // render all atoms as spheres and write the Id to the red color channel
    this->RenderAtomIdGPU( mol, atomPos);
    // stop rendering to FBO
    this->sphereIdFbo.Disable();


    // ---------- use the histogram for generating a list of visible atoms ----------
    // read the visible atoms texture (visibility mask)
    //memset( this->visibleAtomMask, 0, sizeof(float)*protein->AtomCount());
    //glBindTexture( GL_TEXTURE_2D, this->visibleAtomsColor);
    //glGetTexImage( GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, this->visibleAtomMask);
    //glBindTexture( GL_TEXTURE_2D, 0);
}


/*
 * HapticsMoleculeRenderer::RenderAtomIdGPU
 */
void HapticsMoleculeRenderer::RenderAtomIdGPU( const MolecularDataCall *mol, const float *atomPos) {
    // resize atom parameter array
    this->atomParams.SetCount( mol->AtomCount() * 3);
    // fill the array
    for( unsigned int cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
        this->atomParams[cnt*3+0] = float( cnt);
        this->atomParams[cnt*3+1] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
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
	
    glEnable( GL_DEPTH_TEST);

	// enable sphere shader
	this->writeSphereIdShader.Enable();

	// set shader variables
    glUniform4fv( this->writeSphereIdShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv( this->writeSphereIdShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv( this->writeSphereIdShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv( this->writeSphereIdShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    glUniform3f( this->writeSphereIdShader.ParameterLocation( "zValues"), 1.0f, cameraInfo->NearClip(), cameraInfo->FarClip());

	glEnableClientState( GL_VERTEX_ARRAY);
	glEnableClientState( GL_COLOR_ARRAY);
	// set vertex and color pointers and draw them
    glColorPointer( 3, GL_FLOAT, 0, this->atomParams.PeekElements());
    glVertexPointer( 3, GL_FLOAT, 0, atomPos);
    glDrawArrays( GL_POINTS, 0, mol->AtomCount());
	// disable sphere shader
	glDisableClientState( GL_COLOR_ARRAY);
	glDisableClientState( GL_VERTEX_ARRAY);

	// disable sphere shader
	this->writeSphereIdShader.Disable();
}

/*
 * Render the molecular data in spacefilling mode.
 */
void HapticsMoleculeRenderer::RenderSpacefilling( const MolecularDataCall *mol, const float *atomPos) {
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
    
    // Set haptic material properties for the shaders
    hlMaterialf(HL_FRONT_AND_BACK, HL_STIFFNESS, 0.85f);
    hlMaterialf(HL_FRONT_AND_BACK, HL_DAMPING, 0.1f);
    hlHintb(HL_SHAPE_DYNAMIC_SURFACE_CHANGE, true); // allows for better haptic processing of moving shapes

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

    hlBeginShape(HL_SHAPE_DEPTH_BUFFER, this->shaderID);
    glDrawArrays( GL_POINTS, 0, mol->AtomCount());
    hlEndShape();

    // disable sphere shader
    this->sphereShader.Disable();
}


/*
 * Render the molecular data in spacefilling mode.
 */
void HapticsMoleculeRenderer::RenderSpacefillingImmediateMode( const MolecularDataCall *mol, const float *atomPos) {

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

    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

    // Set haptic material properties for the shaders
    hlMaterialf(HL_FRONT_AND_BACK, HL_STIFFNESS, 0.85f);
    hlMaterialf(HL_FRONT_AND_BACK, HL_DAMPING, 0.1f);
    glDisable(GL_LIGHTING);
    // set vertex and color pointers and draw them
    hlBeginShape(HL_SHAPE_DEPTH_BUFFER, this->shaderID); // start a new haptic shape using depth buffer
    glBegin( GL_POINTS);

    for( unsigned int cnt = 0; cnt < mol->AtomCount(); cnt++ ) {
        glColor3f( this->atomColorTable[cnt*3], this->atomColorTable[cnt*3+1], this->atomColorTable[cnt*3+2]); 
        glVertex4f( atomPos[cnt*3], atomPos[cnt*3+1], atomPos[cnt*3+2],
            mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius()); // render the shader
    }

    glEnd(); // GL_POINTS
    hlEndShape(); // end the haptic shape

    // disable sphere shader
    this->sphereShader.Disable();
}

/*
 * HapticsMoleculeRenderer::Render3DCursor
 */
void HapticsMoleculeRenderer::Render3DCursor (void) {
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
    glUniform1fARB ( this->arrowShader.ParameterLocation ( "radScale" ), 1.0f);
    
    vislib::math::Point<float, 3> head = this->absoluteCursor3d.GetCurrentPosition(); // get head location
    vislib::math::Vector<float, 3> orientation = this->absoluteCursor3d.GetCurrentOrientation(); // get current cursor orientation

    vislib::math::Point<float, 3> beginning = head - (1.5f * orientation); // get "beginning" of arrow (shaft end)

    glBegin( GL_POINTS);

    glColor3f(0.0f, 1.0f, 1.0f); // set cursor color to turqoise
    glTexCoord3fv(beginning.PeekCoordinates());
    glVertex4f( head.GetX(), head.GetY(), head.GetZ(), this->radiusArrow);
    glEnd();
    this->arrowShader.Disable();
}


/*
 * HapticsMoleculeRenderer::RenderForceArrows
 */
void HapticsMoleculeRenderer::RenderForceArrows (void) {
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
    glUniform1fARB ( this->arrowShader.ParameterLocation ( "radScale" ), 1.0f);

    glBegin( GL_POINTS);
    for (unsigned int cnt = 0; cnt < this->forceCount; cnt += 1) {
        vislib::math::Point<float, 3> head; // arrowhead location
        vislib::math::Point<float, 3> base(&this->atomPositions[this->forceAtomIDs[cnt] * 3]); // arrow base location
        vislib::math::Vector<float, 3> force(&this->forces[cnt * 3]); // get force vector

        force *= 1.0f; // scaling factor?

        head = base + force; // head will be in the direction of the force from the atom position

        glColor3f(0.0f, 1.0f, 0.0f); // set force arrow color to green
        glTexCoord3fv(base.PeekCoordinates()); // start the arrow at the atom center
        glVertex4f( head.GetX(), head.GetY(), head.GetZ(), this->radiusArrow); // end it in the force direction
    }
    glEnd();
    this->arrowShader.Disable();
}

/*
 * HapticsMoleculeRenderer::UpdateParameters
 */
void HapticsMoleculeRenderer::UpdateParameters( const MolecularDataCall *mol) {
    // color table param
    if( this->colorTableFileParam.IsDirty() ) {
        Color::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::StringParam>()->Value(),
            this->colorLookupTable);
        this->colorTableFileParam.ResetDirty();
    }
    // coloring mode param
    if( this->coloringModeParam.IsDirty() ) {
        
        this->currentColoringMode = static_cast<Color::ColoringMode>( int(
            this->coloringModeParam.Param<param::EnumParam>()->Value() ) );
        
        Color::MakeColorTable( mol,
          currentColoringMode,
          this->atomColorTable, this->colorLookupTable, this->rainbowColors,
          this->minGradColorParam.Param<param::StringParam>()->Value(),
          this->midGradColorParam.Param<param::StringParam>()->Value(),
          this->maxGradColorParam.Param<param::StringParam>()->Value(),
          true);
    }
    // get molecule lust
    if( this->molIdxListParam.IsDirty() ) {
        vislib::StringA tmpStr( this->molIdxListParam.Param<param::StringParam>()->Value());
        this->molIdxList = vislib::StringTokeniser<vislib::CharTraitsA>::Split( tmpStr, ';', true);
        this->molIdxListParam.ResetDirty();
    }
}

/*
 * HapticsMoleculeRenderer::getAtomID
 */
unsigned int HapticsMoleculeRenderer::getAtomID( bool click, bool touch) {
    // check if button was clicked or released and whether or not device thinks it is touching
    if (click == true && touch == true) {
        // new method - already tracking the closest atom and the distance
        if( this->closestAtomDistance > 0.1f ) {
            this->currentDragAtom = -1;
        } else {
            this->currentDragAtom = this->closestAtomID; // set the current drag atom
        }
        return this->currentDragAtom; // return the current drag atom

    } else if (click == true && touch == false) {
        // Clear the forces - this should really be handled another way (e.g. params/only retain forces if alt is held/something) but this is what we'll do for now
        this->forceCount = 0;
        this->forceAtomIDs.Clear();
        this->forces.Clear();
        // set current click object to -1 and return -1
        this->currentDragAtom = -1;
        return -1;
    } else {
        // check the multiforce param
        if (this->multiforceParam.Param<core::param::BoolParam>()->Value() == true) {
            // stop dragging, but don't clear the forces
            this->currentDragAtom = -1;
            return -1;
        } else {
            // stop dragging and clear the forces
            this->forceCount = 0;
            this->forceAtomIDs.Clear();
            this->forces.Clear();
            this->currentDragAtom = -1;
            return -1;
        }
    }
}

/*
 * HapticsMoleculeRenderer::getForceData
 */
bool HapticsMoleculeRenderer::getForceData( core::Call& call) {
    
    // Check for valid call
    ForceDataCall *dataCall = dynamic_cast<ForceDataCall*>( &call);
    if ( dataCall == NULL ) return false;

    // Set the pointers for the call
    dataCall->SetForces(this->forceCount, this->forceAtomIDs.PeekElements(), this->forces.PeekElements());

    return true;
}

#endif // WITH_OPENHAPTICS
