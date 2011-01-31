/*
 * XYZLoader.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "XYZLoader.h"
#include "ParticleDataCall.h"
#include "param/FilePathParam.h"
#include "param/IntParam.h"
#include "param/BoolParam.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/sysfunctions.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/Quaternion.h"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <fstream>

#define USE_RANDOM_ROT_TRANS

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

/*
 * protein::XYZLoader::XYZLoader
 */
XYZLoader::XYZLoader(void) : megamol::core::Module(),
        filenameSlot( "filename", "The path to the XYZ data file to be loaded"),
        dataOutSlot( "dataout", "The slot providing the loaded data"),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), datahash(0),
        particleCount( 0), particles( 0), colors( 0), charges( 0) {

    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->filenameSlot);

    this->dataOutSlot.SetCallback( ParticleDataCall::ClassName(), ParticleDataCall::FunctionName(ParticleDataCall::CallForGetData), &XYZLoader::getData);
    this->dataOutSlot.SetCallback( ParticleDataCall::ClassName(), ParticleDataCall::FunctionName(ParticleDataCall::CallForGetExtent), &XYZLoader::getExtent);
    this->MakeSlotAvailable( &this->dataOutSlot);

}

/*
 * protein::XYZLoader::~XYZLoader
 */
XYZLoader::~XYZLoader(void) {
    this->Release ();
}

/*
 * XYZLoader::create
 */
bool XYZLoader::create(void) {
    // intentionally empty
    return true;
}

/*
 * XYZLoader::getData
 */
bool XYZLoader::getData( core::Call& call) {
    using vislib::sys::Log;

    ParticleDataCall *dc = dynamic_cast<ParticleDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    dc->SetDataHash( this->datahash);

    // set values to call
    dc->SetParticleCount( this->particleCount);
    dc->SetParticles( this->particles);
    dc->SetColors( this->colors);
    dc->SetCharges( this->charges);

    return true;
}

/*
 * XYZLoader::getExtent
 */
bool XYZLoader::getExtent( core::Call& call) {
    // get data call
    ParticleDataCall *dc = dynamic_cast<ParticleDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    dc->AccessBoundingBoxes().Clear();
    dc->AccessBoundingBoxes().SetObjectSpaceBBox( this->bbox);
    dc->AccessBoundingBoxes().SetObjectSpaceClipBox( this->bbox);

    dc->SetDataHash( this->datahash);

    return true;
}

/*
 * XYZLoader::release
 */
void XYZLoader::release(void) {

}

/*
 * XYZLoader::loadFile
 */
void XYZLoader::loadFile( const vislib::TString& filename) {
    using vislib::sys::Log;

    this->bbox.Set( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    this->datahash++;

#ifdef USE_RANDOM_ROT_TRANS
    srand( time( NULL));
    vislib::math::Vector<float, 3> rotAxis( rand(), rand(), rand());
    rotAxis.Normalise();
    vislib::math::Vector<float, 3> trans( rand(), rand(), rand());
    trans.Normalise();
    float angle = float( rand()) / float( RAND_MAX);
#endif

    vislib::StringA line, name;
    unsigned int cnt, idx, pCnt;

    time_t t = clock(); // DEBUG

    vislib::sys::ASCIIFileBuffer file;
    vislib::Array<vislib::StringA> entries;
    vislib::math::Cuboid<float> sphereBBox;

    vislib::math::Vector<float, 3> color;
    vislib::math::Vector<float, 3> charge;

    // try to load the file
    if( file.LoadFile( T2A( filename) ) ) {
        // get particle count
        pCnt = atoi( file.Line( 0).Pointer());
        this->particleCount = pCnt;
        // resize arrays
        if( this->particles )
            delete[] this->particles;
        this->particles = new float[pCnt * 4];
        if( this->colors )
            delete[] this->colors;
        this->colors = new float[pCnt * 3];
        if( this->charges )
            delete[] this->charges;
        this->charges = new float[pCnt * 3];
        for( cnt = 0; cnt < pCnt; ++cnt ) {
            // get the current line from the file
            line = file.Line( cnt + 2);
            entries = vislib::StringTokeniser<vislib::CharTraitsA>::Split( line, " ", true);
            if( entries.Count() > 3 ) {
                // position
                this->particles[cnt*4+0] = atof( entries[1]);
                this->particles[cnt*4+1] = atof( entries[2]);
                this->particles[cnt*4+2] = atof( entries[3]);
                // radius
                this->particles[cnt*4+3] = this->getElementRadius( entries[0]);
                //color
                color = this->getElementColor( entries[0]);
                this->colors[cnt*3+0] = color.X();
                this->colors[cnt*3+1] = color.Y();
                this->colors[cnt*3+2] = color.Z();
                // charges
                if( entries.Count() > 4 )
                    this->charges[cnt*3+0] = atof( entries[4]);
                if( entries.Count() > 5 )
                    this->charges[cnt*3+1] = atof( entries[5]);
                if( entries.Count() > 6 )
                    this->charges[cnt*3+2] = atof( entries[6]);
                // update the bounding box
                sphereBBox.Set( 
                    this->particles[cnt*4+0] - this->particles[cnt*4+3], 
                    this->particles[cnt*4+1] - this->particles[cnt*4+3], 
                    this->particles[cnt*4+2] - this->particles[cnt*4+3], 
                    this->particles[cnt*4+0] + this->particles[cnt*4+3], 
                    this->particles[cnt*4+1] + this->particles[cnt*4+3], 
                    this->particles[cnt*4+2] + this->particles[cnt*4+3] );
                if( cnt == 0 ) {
                    this->bbox = sphereBBox;
                } else {
                    this->bbox.Union( sphereBBox);
                }
            }
        }
        float edge = this->bbox.LongestEdge() * 5.0f;
        this->bbox.Grow( edge, edge, edge);

#ifdef USE_RANDOM_ROT_TRANS
        vislib::math::Vector<float, 3> center( 0, 0, 0);
        for( unsigned int i = 0; i < this->particleCount; i++ ) {
            center += vislib::math::Vector<float, 3>( this->particles[i*4+0], this->particles[i*4+1], this->particles[i*4+2]);
        }
        center /= float( this->particleCount);
        vislib::math::Quaternion<float> quart( angle, rotAxis);
        for( unsigned int i = 0; i < this->particleCount; i++ ) {
            vislib::math::Vector<float, 3> part( this->particles[i*4+0], this->particles[i*4+1], this->particles[i*4+2]);
            part -= center;
            part = quart * part;
            part += center;
            part += trans;
            this->particles[i*4+0] = part.X();
            this->particles[i*4+1] = part.Y();
            this->particles[i*4+2] = part.Z();
        }
#endif

        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for loading file %s: %f", static_cast<const char*>(T2A( filename)), ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG
    } else {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Could not load file %s", static_cast<const char*>(T2A( filename))); // DEBUG
    }
}

/*
 * Get the radius of the element
 */
float XYZLoader::getElementRadius( vislib::StringA name) {
    // extract the element symbol from the name
    unsigned int cnt = 0;
    vislib::StringA element;
    while( cnt < name.Length() && vislib::CharTraitsA::IsDigit( name[cnt]) ) {
        cnt++;
    }

    // --- van der Waals radii ---
    if( cnt < name.Length() ) {
        if( name[cnt] == 'H' )
            return 1.2f;
        else if( name[cnt] == 'C' ) {
            if( ( cnt + 1) < name.Length() )
                if( name[cnt+1] == 'l' )
                    return 1.75f; // chlorine
            return 1.7f; // carbon
        } else if( name[cnt] == 'N' )
            return 1.55f;
        else if( name[cnt] == 'O' )
            return 1.52f;
        else if( name[cnt] == 'S' )
            return 1.8f;
        else if( name[cnt] == 'P' )
            return 1.8f;
        else if( name[cnt] == 'C' )
            return 1.7f;
    }

    return 1.5f;
}

/*
 * Get the color of the element
 */
vislib::math::Vector<float, 3> XYZLoader::getElementColor( vislib::StringA name) {
    // extract the element symbol from the name
    unsigned int cnt = 0;
    vislib::StringA element;
    while( cnt < name.Length() && vislib::CharTraitsA::IsDigit( name[cnt]) ) {
        cnt++;
    }

    // default color
    vislib::math::Vector<unsigned char, 3> col( 191, 191, 191);
    vislib::math::Vector<float, 3> colFloat;

    // check element
    if( cnt < name.Length() ) {
        if( name[cnt] == 'H' ) // white or light grey
            col.Set( 240, 240, 240);
        else if( name[cnt] == 'C' ) { 
            // Carbon: (dark) grey or green
            col.Set( 125, 125, 125);
            //col.Set( 90, 175, 50);
            if( ( cnt + 1) < name.Length() )
                // Chlorine: yellowgreen
                col.Set( 173, 255, 47);
        } else if( name[cnt] == 'N' ) // blue
            //col.Set( 37, 136, 195);
            col.Set( 37, 136, 195);
        else if( name[cnt] == 'O' ) // red
            //col.Set( 250, 94, 82);
            col.Set( 206, 34, 34);
        else if( name[cnt] == 'S' ) // yellow
            //col.Set( 250, 230, 50);
            col.Set( 255, 215, 0);
        else if( name[cnt] == 'P' ) // orange
            col.Set( 255, 128, 64);
    }

    colFloat.Set( float( col.X()) / 255.0f, float( col.Y()) / 255.0f, float( col.Z()) / 255.0f);
    return colFloat;
}
