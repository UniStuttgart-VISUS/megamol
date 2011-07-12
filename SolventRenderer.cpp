/*
 * SolventRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "SolventRenderer.h"
#include "CoreInstance.h"
#include "param/EnumParam.h"
#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "param/StringParam.h"
#include "utility/ShaderSourceFactory.h"
#include "vislib/assert.h"
#include "vislib/File.h"
#include "vislib/Point.h"
#include "vislib/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/ShaderSource.h"
#include "vislib/AbstractOpenGLShader.h"
#include "vislib/StringTokeniser.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <glh/glh_genext.h>
#include <math.h>
#include <time.h>

using namespace megamol;
using namespace megamol::core;


/*
 * protein::SolventRenderer::SolventRenderer (CTOR)
 */
protein::SolventRenderer::SolventRenderer(void) : Renderer3DModule (), 
	protDataCallerSlot ("getData", "Connects the solvent rendering with protein data storage"),
    radiusScaleParam ( "radiusScale", "Scale factor for the atom radii"),
    elementFilterParam( "elementFilter", "Filter atoms by element type"),
    minChargeParam( "minCharge", "Minimum occupancy for filtering"),
    maxChargeParam( "maxCharge", "Maximum occupancy for filtering"),
    distanceParam( "distance", "Filter atoms by distance to protein atom"),
    minCharge( 0), maxCharge( 0), distance( 0.0f),
    atomPos( 0), atomPosSize( 0), atomColor( 0), atomColorSize( 0)
{
    this->protDataCallerSlot.SetCompatibleCall<CallProteinDataDescription>();
    this->MakeSlotAvailable(&this->protDataCallerSlot);

	// --- radius scale parameter slot  ---
	this->radiusScale = 0.3f;
    this->radiusScaleParam.SetParameter(new param::FloatParam( this->radiusScale, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->radiusScaleParam);
    
	// --- element filter parameter slot  ---
	this->elementFilter = "";
    this->elementFilterParam.SetParameter(new param::StringParam( this->elementFilter.PeekBuffer()));
    this->MakeSlotAvailable(&this->elementFilterParam);

	// --- min occupancy parameter slot  ---
	this->minCharge = -100.0f;
    this->minChargeParam.SetParameter(new param::FloatParam( this->minCharge));
    this->MakeSlotAvailable(&this->minChargeParam);
    
	// --- max occupancy parameter slot  ---
    this->maxCharge = 100.0f;
    this->maxChargeParam.SetParameter(new param::FloatParam( this->maxCharge));
    this->MakeSlotAvailable(&this->maxChargeParam);
    
	// --- distance parameter slot  ---
    this->distanceParam.SetParameter(new param::FloatParam( this->distance));
    this->MakeSlotAvailable(&this->distanceParam);
    
}


/*
 * protein::SolventRenderer::~SolventRenderer (DTOR)
 */
protein::SolventRenderer::~SolventRenderer(void)  {
    this->Release ();
}


/*
 * protein::SolventRenderer::release
 */
void protein::SolventRenderer::release(void)  {

}


/*
 * protein::SolventRenderer::create
 */
bool protein::SolventRenderer::create(void) {
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
 * protein::SolventRenderer::GetCapabilities
 */
bool protein::SolventRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING);

    return true;
}


/*
 * protein::SolventRenderer::GetExtents
 */
bool protein::SolventRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    protein::CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();
    if (protein == NULL) return false;
    if (!(*protein)()) return false;

    float scale, xoff, yoff, zoff;
    vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    scale = 2.0f / vislib::math::Max(vislib::math::Max(protein->BoundingBox().Width(),
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
    bbox.SetObjectSpaceClipBox(bbox.ObjectSpaceBBox());
    bbox.SetWorldSpaceClipBox(bbox.WorldSpaceBBox());

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::SolventRenderer::Render
 */
bool protein::SolventRenderer::Render(Call& call)
{
	// get pointer to CallProteinData
	CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();

    if( protein == NULL) 
        return false;

    if( !(*protein)() ) 
        return false;

    // get updated parameters
    this->ParameterRefresh();

	// get camera information
	this->cameraInfo = dynamic_cast<view::CallRender3D*>(&call)->GetCameraParameters();

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

   
	// -------------------------------
	// -- filter all solvent atoms  --
	// ----------- -------------------

    vislib::StringA elementName;
    bool drawAtom;
    unsigned int atomCnt;
	int filterCnt;
    // resize atom position and color arrays
    if( !this->atomPos || this->atomPosSize != protein->SolventAtomCount() ) {
        delete[] this->atomPos;
        this->atomPos = new float[protein->SolventAtomCount()*4];
        this->atomPosSize = protein->SolventAtomCount();
    }
    if( !this->atomColor || this->atomColorSize != protein->SolventAtomCount() ) {
        delete[] this->atomColor;
        this->atomColor = new unsigned char[protein->SolventAtomCount()*3];
        this->atomColorSize = protein->SolventAtomCount();
    }
    // the number of currently visible atoms
    unsigned int visibleAtomCnt = 0;

    // get all the elements names to filter out
    vislib::Array<vislib::TString> elementFilters = vislib::StringTokeniser<vislib::TCharTraits>::Split( this->elementFilter, ';', true);

    // loop over all atoms 
    for( atomCnt = 0; atomCnt < protein->SolventAtomCount(); ++atomCnt ) {
        drawAtom = true;
        // --- filter atom by element type ---
        elementName = protein->AtomTypes()[protein->SolventAtomData()[atomCnt].TypeIndex()].Name();
        // check if the name of the elements is matched by one of the filters
        for( filterCnt = 0; filterCnt < elementFilters.Count(); ++filterCnt ) {
            if( elementName.StartsWithInsensitive( elementFilters[filterCnt] ) ) {
                drawAtom = false;
                break;
            }
        }
        // --- filter atom by charge ---
        if( protein->SolventAtomData()[atomCnt].Charge() < this->minCharge ||
            protein->SolventAtomData()[atomCnt].Charge() > this->maxCharge ) {
            drawAtom = false;
        }
        // --- filter by distance to protein atom ---
        bool withinDistance = false;
        if( this->distance > vislib::math::FLOAT_EPSILON ) {
#pragma omp parallel for
            for( filterCnt = 0; filterCnt < static_cast<int>(protein->ProteinAtomCount()); ++filterCnt ) {
                if( sqrt( pow( protein->SolventAtomPositions()[3*atomCnt+0] - protein->ProteinAtomPositions()[3*filterCnt+0], 2) + 
                    pow( protein->SolventAtomPositions()[3*atomCnt+1] - protein->ProteinAtomPositions()[3*filterCnt+1], 2) + 
                    pow( protein->SolventAtomPositions()[3*atomCnt+2] - protein->ProteinAtomPositions()[3*filterCnt+2], 2) ) < this->distance ) {
                        withinDistance = true;
                }
            }
        } else {
            withinDistance = true;
        }
        // store the atom, if it was not filtered out
        if( drawAtom && withinDistance ) {
            // set color
            this->atomColor[3*visibleAtomCnt+0] = protein->AtomTypes()[protein->SolventAtomData()[atomCnt].TypeIndex()].Colour()[0];
            this->atomColor[3*visibleAtomCnt+1] = protein->AtomTypes()[protein->SolventAtomData()[atomCnt].TypeIndex()].Colour()[1];
            this->atomColor[3*visibleAtomCnt+2] = protein->AtomTypes()[protein->SolventAtomData()[atomCnt].TypeIndex()].Colour()[2];
            // set position
            this->atomPos[4*visibleAtomCnt+0] = protein->SolventAtomPositions()[3*atomCnt+0];
            this->atomPos[4*visibleAtomCnt+1] = protein->SolventAtomPositions()[3*atomCnt+1],
            this->atomPos[4*visibleAtomCnt+2] = protein->SolventAtomPositions()[3*atomCnt+2],
            this->atomPos[4*visibleAtomCnt+3] = protein->AtomTypes()[protein->SolventAtomData()[atomCnt].TypeIndex()].Radius() * this->radiusScale;
            // update number of visible atoms
            visibleAtomCnt++;
        }
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
	this->sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
	// set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
	// set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, this->atomPos);
    glColorPointer( 3, GL_UNSIGNED_BYTE, 0, this->atomColor);
    glDrawArrays( GL_POINTS, 0, visibleAtomCnt);
	// disable sphere shader
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
	this->sphereShader.Disable();



    glDisable(GL_DEPTH_TEST);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glPopMatrix();

    return true;
}

/*
 * refresh parameter values
 */
void protein::SolventRenderer::ParameterRefresh() {
    
	if ( this->radiusScaleParam.IsDirty() ) {
		this->radiusScale = this->radiusScaleParam.Param<param::FloatParam>()->Value();
		this->radiusScaleParam.ResetDirty();
	}

	if ( this->elementFilterParam.IsDirty() ) {
        this->elementFilter = this->elementFilterParam.Param<param::StringParam>()->Value();
        this->elementFilterParam.ResetDirty();
	}

    if ( this->minChargeParam.IsDirty() ) {
        this->minCharge = this->minChargeParam.Param<param::FloatParam>()->Value();
        this->minChargeParam.ResetDirty();
	}

    if ( this->maxChargeParam.IsDirty() ) {
        this->maxCharge = this->maxChargeParam.Param<param::FloatParam>()->Value();
        this->maxChargeParam.ResetDirty();
	}

    if ( this->distanceParam.IsDirty() ) {
        this->distance = this->distanceParam.Param<param::FloatParam>()->Value();
        this->distanceParam.ResetDirty();
	}
}
