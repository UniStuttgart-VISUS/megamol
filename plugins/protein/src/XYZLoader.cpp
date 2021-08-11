/*
 * XYZLoader.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "XYZLoader.h"
#include "ParticleDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "vislib/ArrayAllocator.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/math/mathfunctions.h"
#include "mmcore/utility/sys/MemmappedFile.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "mmcore/utility/sys/ASCIIFileBuffer.h"
#include "vislib/math/Quaternion.h"
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

    this->dataOutSlot.SetCallback("MultiParticleDataCall", "GetData", &XYZLoader::getData);
    this->dataOutSlot.SetCallback("MultiParticleDataCall", "GetExtent", &XYZLoader::getExtent);
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
    using megamol::core::utility::log::Log;
    using namespace megamol::core::moldyn;

    ParticleDataCall *dc = dynamic_cast<ParticleDataCall*>( &call);
    MultiParticleDataCall *mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if ( dc == NULL && mpdc == NULL) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        if (mpdc != NULL) {
            this->loadFile(
                this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(), false);
        } else {
            this->loadFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        }
    }

    if (dc != NULL) {
        dc->SetDataHash( this->datahash);

        // set values to call
        dc->SetParticleCount( this->particleCount);
        dc->SetParticles( this->particles);
        dc->SetColors( this->colors);
        dc->SetCharges( this->charges);
    } else {
        mpdc->SetDataHash( this->datahash);
        mpdc->SetFrameCount(1);
        // set values to call
        mpdc->SetParticleListCount(1);
        mpdc->AccessParticles(0).SetCount(this->particleCount);
        mpdc->AccessParticles(0).SetVertexData(SimpleSphericalParticles::VERTDATA_FLOAT_XYZR, this->particles);
        mpdc->AccessParticles(0).SetColourData(SimpleSphericalParticles::COLDATA_FLOAT_RGB, this->colors);
    }
    return true;
}

/*
 * XYZLoader::getExtent
 */
bool XYZLoader::getExtent( core::Call& call) {
    // get data call
    using namespace megamol::core::moldyn;
    ParticleDataCall *dc = dynamic_cast<ParticleDataCall*>( &call);
    MultiParticleDataCall *mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if ( dc == NULL && mpdc == NULL) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        if (mpdc != NULL) {
            this->loadFile(
                this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(), false);
        } else {
            this->loadFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        }
    }

    if (dc != NULL) {
        dc->AccessBoundingBoxes().Clear();
        dc->AccessBoundingBoxes().SetObjectSpaceBBox( this->bbox);
        dc->AccessBoundingBoxes().SetObjectSpaceClipBox( this->bbox);

        dc->SetDataHash( this->datahash);
    } else {
        mpdc->SetFrameCount(1);
        mpdc->AccessBoundingBoxes().Clear();
        mpdc->AccessBoundingBoxes().SetObjectSpaceBBox( this->bbox);
        mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox( this->bbox);

        mpdc->SetDataHash( this->datahash);
    }
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
void XYZLoader::loadFile( const vislib::TString& filename, bool doElectrostatics) {
    using megamol::core::utility::log::Log;

    this->bbox.Set( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    this->datahash++;

#ifdef USE_RANDOM_ROT_TRANS
    srand( static_cast<unsigned int>(time( NULL)));
    vislib::math::Vector<float, 3> rotAxis( static_cast<float>(rand()), 
        static_cast<float>(rand()), static_cast<float>(rand()));
    rotAxis.Normalise();
    vislib::math::Vector<float, 3> trans( static_cast<float>(rand()), 
        static_cast<float>(rand()), static_cast<float>(rand()));
    trans.Normalise();
    float angle = float( rand()) / float( RAND_MAX);
#endif

    vislib::StringA line, name;
    unsigned int cnt, pCnt;

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
                this->particles[cnt*4+0] = static_cast<float>(atof( entries[1]));
                this->particles[cnt*4+1] = static_cast<float>(atof( entries[2]));
                this->particles[cnt*4+2] = static_cast<float>(atof( entries[3]));
                // radius
                this->particles[cnt*4+3] = this->getElementRadius( entries[0]);
                //color
                color = this->getElementColor( entries[0]);
                this->colors[cnt*3+0] = color.X();
                this->colors[cnt*3+1] = color.Y();
                this->colors[cnt*3+2] = color.Z();
                // charges
                if( entries.Count() > 4 )
                    this->charges[cnt*3+0] = static_cast<float>(atof( entries[4]));
                if( entries.Count() > 5 )
                    this->charges[cnt*3+1] = static_cast<float>(atof( entries[5]));
                if( entries.Count() > 6 )
                    this->charges[cnt*3+2] = static_cast<float>(atof( entries[6]));
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

        if (doElectrostatics) {
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
        }
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
    while( cnt < static_cast<unsigned int>(name.Length()) && vislib::CharTraitsA::IsDigit( name[cnt]) ) {
        cnt++;
    }

    // --- van der Waals radii ---
    if( cnt < static_cast<unsigned int>(name.Length()) ) {
        if( name[cnt] == 'H' )
            return 1.2f;
        else if( name[cnt] == 'C' ) {
            if( ( cnt + 1) < static_cast<unsigned int>(name.Length()) )
                if( name[cnt+1] == 'l' )
                    return 1.75f; // chlorine
            return 1.7f; // carbon
        } else if( name[cnt] == 'O' )
            return 1.52f;
        else if( name[cnt] == 'S' )
            return 1.8f;
        else if( name[cnt] == 'P' )
            return 1.8f;
        else if( name[cnt] == 'C' )
            return 1.7f;
        else if( name[cnt] == 'N' ) {
            if( ( cnt + 1) < static_cast<unsigned int>(name.Length()) )
                if( name[cnt+1] == 'i' )
                    return 1.63f; // nickel
            return 1.55f; // nitrogen
        }
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
    while( cnt < static_cast<unsigned int>(name.Length()) && vislib::CharTraitsA::IsDigit( name[cnt]) ) {
        cnt++;
    }

    // default color
    vislib::math::Vector<unsigned char, 3> col( 191, 191, 191);
    vislib::math::Vector<float, 3> colFloat;

    // check element
    if( cnt < static_cast<unsigned int>(name.Length()) ) {
        if( name[cnt] == 'H' ) // white or light grey
            col.Set( 240, 240, 240);
        else if( name[cnt] == 'C' ) { 
            // Carbon: (dark) grey or green
            col.Set( 125, 125, 125);
            //col.Set( 90, 175, 50);
            if( ( cnt + 1) < static_cast<unsigned int>(name.Length()) )
                // Chlorine: yellowgreen
                col.Set( 173, 255, 47);
        } else if( name[cnt] == 'O' ) // red
            //col.Set( 250, 94, 82);
            col.Set( 206, 34, 34);
        else if( name[cnt] == 'S' ) // yellow
            //col.Set( 250, 230, 50);
            col.Set( 255, 215, 0);
        else if( name[cnt] == 'P' ) // orange
            col.Set( 255, 128, 64);
        else if( name[cnt] == 'N' ) {
            if( ( cnt + 1) < static_cast<unsigned int>(name.Length()) ) {
                if( name[cnt+1] == 'i' )
                    col.Set( 136, 136, 136); // nickel
            } else {
                col.Set( 37, 136, 195); // nitrogen
            }
        }
    }

    colFloat.Set( float( col.X()) / 255.0f, float( col.Y()) / 255.0f, float( col.Z()) / 255.0f);
    return colFloat;
}
