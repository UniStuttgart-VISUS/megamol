/*
 * ElectrostaticsRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "ElectrostaticsRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>
#include <string>
#include <iostream>
#include <fstream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * ElectrostaticsRenderer::ElectrostaticsRenderer (CTOR)
 */
ElectrostaticsRenderer::ElectrostaticsRenderer(void) : Renderer3DModule (), 
        dataCallerSlot( "getData", "Connects the rendering with data storage."),
        cellLenghtParam( "cellLength", "The designated cell length."),
        fieldSize( 0), field( 0) {
    // the caller slot
    this->dataCallerSlot.SetCompatibleCall<ParticleDataCallDescription>();
	this->dataCallerSlot.SetCompatibleCall<megamol::protein_calls::MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->dataCallerSlot);

    // the cell length parameter
    this->cellLenghtParam.SetParameter(new param::FloatParam( 10.0f, 0.0f));
    this->MakeSlotAvailable( &this->cellLenghtParam);

}


/*
 * ElectrostaticsRenderer::~ElectrostaticsRenderer (DTOR)
 */
ElectrostaticsRenderer::~ElectrostaticsRenderer(void)  {
    this->Release ();
}


/*
 * ElectrostaticsRenderer::release
 */
void ElectrostaticsRenderer::release(void) {

}


/*
 * ElectrostaticsRenderer::create
 */
bool ElectrostaticsRenderer::create(void)
{
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

    return true;
}


/*
 * ElectrostaticsRenderer::GetExtents
 */
bool ElectrostaticsRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    ParticleDataCall *pdc = this->dataCallerSlot.CallAs<ParticleDataCall>();
	megamol::protein_calls::MolecularDataCall *mdc = this->dataCallerSlot.CallAs<megamol::protein_calls::MolecularDataCall>();
    float scale;
    if( pdc != NULL ) {
        if (!(*pdc)(ParticleDataCall::CallForGetExtent)) return false;
        if( !vislib::math::IsEqual( pdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
            scale = 2.0f / pdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        } else {
            scale = 1.0f;
        }
        cr3d->AccessBoundingBoxes() = pdc->AccessBoundingBoxes();
    } else if( mdc != NULL ) {
		if (!(*mdc)(megamol::protein_calls::MolecularDataCall::CallForGetExtent)) return false;
        if( !vislib::math::IsEqual( mdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
            scale = 2.0f / mdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        } else {
            scale = 1.0f;
        }
        cr3d->AccessBoundingBoxes() = pdc->AccessBoundingBoxes();
    } else {
        return false;
    }

    cr3d->AccessBoundingBoxes().MakeScaledWorld( scale);
    //TODO
    cr3d->SetTimeFramesCount( 1U);

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * ElectrostaticsRenderer::Render
 */
bool ElectrostaticsRenderer::Render(Call& call)
{
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to ParticleDataCall
    ParticleDataCall *particles = this->dataCallerSlot.CallAs<ParticleDataCall>();
    if( particles == NULL) return false;

    if (!(*particles)(ParticleDataCall::CallForGetData)) return false;

    // ---------- render ----------

    // -- draw particles --
    glPushMatrix();

    float scale;
    if( !vislib::math::IsEqual( particles->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        scale = 2.0f / particles->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    glScalef( scale, scale, scale);

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
    glEnable( GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    // enable sphere shader
    this->sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // draw points
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, particles->Particles());
    glColorPointer( 3, GL_FLOAT, 0, particles->Colors() );
    glDrawArrays( GL_POINTS, 0, particles->ParticleCount());
    // disable sphere shader
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    this->sphereShader.Disable();

    // ---------- compute & draw field ----------
    this->ComputeElectrostaticField( particles, this->cellLenghtParam.Param<param::FloatParam>()->Value());

    /*
    // DEBUG
    std::ifstream foutRaw( "field.raw", std::ios::in | std::ios::binary );
    float test;
    for( unsigned int i = 0; i < this->fieldSize; i++ ) {
        foutRaw.read( reinterpret_cast<char *>(&test), sizeof( float));
        this->field[i].SetX( test);
        foutRaw.read( reinterpret_cast<char *>(&test), sizeof( float));
        this->field[i].SetY( test);
        foutRaw.read( reinterpret_cast<char *>(&test), sizeof( float));
        this->field[i].SetZ( test);
    }
    foutRaw.close();
    */

    //this->sphereShader.Enable();
    //glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    //glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    //glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    //glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    //glBegin( GL_POINTS);
    glBegin( GL_LINES);
    vislib::math::Vector<float, 3> orig = particles->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack();
    float w = particles->AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float h = particles->AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float d = particles->AccessBoundingBoxes().ObjectSpaceClipBox().Depth();
    unsigned int rx = this->fieldDim[0];
    unsigned int ry = this->fieldDim[1];
    unsigned int rz = this->fieldDim[2];
    vislib::math::Vector<float, 3> pos;
    vislib::math::Vector<float, 3> gridPos;
    vislib::math::Vector<float, 3> offset( w / rx, h / ry, d / rz);
    unsigned int cntX, cntY, cntZ;
    for( cntX = 0; cntX < rx; cntX++ ) {
        for( cntY = 0; cntY < ry; cntY++ ) {
            for( cntZ = 0; cntZ < rz; cntZ++ ) {
                gridPos.Set( float( cntX) + 0.5f, float( cntY) + 0.5f, float( cntZ) + 0.5f);
                pos = orig + offset * gridPos;
                //glColor3fv( this->field[cntX + cntY * rx + cntZ * rx * ry].PeekComponents());
                //glVertex4f( pos.X(), pos.Y(), pos.Z(), 0.5f);
                glColor3f( 1.0f, 1.0f, 0.0f);
                glVertex3f( 
                    pos.X() - this->field[cntX + cntY * rx + cntZ * rx * ry].X(), 
                    pos.Y() - this->field[cntX + cntY * rx + cntZ * rx * ry].Y(),
                    pos.Z() - this->field[cntX + cntY * rx + cntZ * rx * ry].Z() );
                glColor3f( 1.0f, 0.0f, 0.0f);
                glVertex3f( 
                    pos.X() + this->field[cntX + cntY * rx + cntZ * rx * ry].X(), 
                    pos.Y() + this->field[cntX + cntY * rx + cntZ * rx * ry].Y(),
                    pos.Z() + this->field[cntX + cntY * rx + cntZ * rx * ry].Z() );
            } //cntZ
        } // cntY
    } // cntX
    glEnd();
    //this->sphereShader.Disable();

    glDisable(GL_DEPTH_TEST);
    glPopMatrix();

    return true;
}


/*
 * Compute the electrostatic field
 */
void ElectrostaticsRenderer::ComputeElectrostaticField( ParticleDataCall *particles, float stepWidth) {

    using namespace std;
    using vislib::sys::Log;

    vislib::math::Vector<float, 3> orig = particles->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack();
    float w = particles->AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float h = particles->AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float d = particles->AccessBoundingBoxes().ObjectSpaceClipBox().Depth();

    unsigned int rx = static_cast<unsigned int>(ceilf( w / stepWidth));
    unsigned int ry = static_cast<unsigned int>(ceilf( h / stepWidth));
    unsigned int rz = static_cast<unsigned int>(ceilf( d / stepWidth));
    unsigned int size = rx * ry * rz;
    this->fieldDim[0] = rx;
    this->fieldDim[1] = ry;
    this->fieldDim[2] = rz;
    vislib::math::Vector<float, 3> offset( w / rx, h / ry, d / rz);

    // (re)allocate grid, if necessary
    bool storeField = false;
    if( this->fieldSize != size ) {
        this->fieldSize = size;
        if( field )
            delete[] field;
        field = new vislib::math::Vector<float, 3>[this->fieldSize];
        storeField = true;
    }

    vislib::math::Vector<float, 3> pos;
    vislib::math::Vector<float, 3> gridPos;

    const float factor = float( 1.0 / ( 4.0 * vislib::math::PI_DOUBLE * 8.85418781762 * pow( 10.0, -12.0)));

    //time_t t = clock(); // DEBUG

    /*
    unsigned int cntX, cntY, cntZ, cntP;
    for( cntX = 0; cntX < rx; cntX++ ) {
        for( cntY = 0; cntY < ry; cntY++ ) {
            for( cntZ = 0; cntZ < rz; cntZ++ ) {
                gridPos.Set( float( cntX) + 0.5f, float( cntY) + 0.5f, float( cntZ) + 0.5f);
                pos = orig + offset * gridPos;
                vislib::math::Vector<float, 3> vec( 0.0f, 0.0f, 0.0f);
                for( cntP = 0; cntP < particles->ParticleCount(); cntP++ ) {
                    vislib::math::Vector<float, 3> pPos( 
                        particles->Particles()[cntP*4+0],
                        particles->Particles()[cntP*4+1],
                        particles->Particles()[cntP*4+2]);
                    float charge = particles->Charges()[cntP*3];
                    vec += charge * ( ( pos - pPos) / powf( ( pos - pPos).Length(), 3.0f));
                }
                this->field[cntX + cntY * rx + cntZ * rx * ry] = vec;
                //this->field[cntX + cntY * rx + cntZ * rx * ry] *= factor;
                this->field[cntX + cntY * rx + cntZ * rx * ry].Normalise();
            } //cntZ
        } // cntY
    } // cntX
    */
    int cnt, cntX, cntY, cntZ, cntP;
#pragma omp parallel for private( cntX, cntY, cntZ, cntP, gridPos, pos)
    for( cnt = 0; cnt < static_cast<int>(this->fieldSize); cnt++ ) {
        cntZ = cnt / ( rx * ry);
        cntY = ( cnt % ( rx * ry)) / rx;
        cntX = cnt % rx;
        gridPos.Set( float( cntX) + 0.5f, float( cntY) + 0.5f, float( cntZ) + 0.5f);
        pos = orig + offset * gridPos;
        for( cntP = 0; cntP < static_cast<int>(particles->ParticleCount()); cntP++ ) {
            vislib::math::Vector<float, 3> pPos( 
                particles->Particles()[cntP*4+0],
                particles->Particles()[cntP*4+1],
                particles->Particles()[cntP*4+2]);
            this->field[cnt] += particles->Charges()[cntP*3] * ( ( pos - pPos) / powf( ( pos - pPos).Length(), 3.0f));
        }
        //this->field[cntX + cntY * rx + cntZ * rx * ry] *= factor;
        this->field[cnt].Normalise();
    }

    //Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for computing the vector field: %f", ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG

    // store the field
    if( storeField ) {
        // output filenames
        string fnDat, fnRaw;
        fnDat = "field.dat";
        fnRaw = "field.raw";

        ofstream foutDat( fnDat.c_str(), ios::trunc );
        foutDat << "# Origin:" << endl << orig.X() << endl << orig.Y() << endl << orig.Z() << endl;
        foutDat << "# Bounding Box Size (W x H x D):" << endl << w << endl << h << endl << d << endl;
        foutDat << "# Number of grid cells (X,Y,Z):" << endl << rx << endl << ry << endl << rz << endl;
        foutDat << "# Number of particles:" << endl << particles->ParticleCount() << endl;
        foutDat << "# Particles (X,Y,Z,R):" << endl;
        for( unsigned int i = 0; i < particles->ParticleCount(); i++ ) {
            foutDat << particles->Particles()[i*4+0] << " " <<
                particles->Particles()[i*4+1] << " " <<
                particles->Particles()[i*4+2] << " " <<
                particles->Particles()[i*4+3] << endl;
        }
        foutDat.close();

        ofstream foutRaw( fnRaw.c_str(),  ios::out | ios::trunc | ios::binary );
        for( unsigned int i = 0; i < this->fieldSize; i++ ) {
            foutRaw.write( reinterpret_cast<char *>(this->field[i].PeekComponents()), sizeof( float)*3);
        }
        foutRaw.close();

    }
}
