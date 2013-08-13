#include "stdafx.h"
#include "BindingSiteDataSource.h"

#include "BindingSiteCall.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "vislib/ASCIIFileBuffer.h"
#include "param/FilePathParam.h"
#include "vislib/BufferedFile.h"
#include "vislib/sysfunctions.h"
#include "vislib/mathfunctions.h"
#include <math.h>

using namespace megamol::core;
using namespace megamol::protein;

/*
 * BindingSiteDataSource::BindingSiteDataSource (CTOR)
 */
BindingSiteDataSource::BindingSiteDataSource( void ) : megamol::core::Module(),
        dataOutSlot( "dataout", "The slot providing the binding site data"),
        pdbFilenameSlot( "pdbFilename", "The PDB file containing the binding site information") {
            
    this->pdbFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->pdbFilenameSlot);
    
    this->dataOutSlot.SetCallback( BindingSiteCall::ClassName(), BindingSiteCall::FunctionName(BindingSiteCall::CallForGetData), &BindingSiteDataSource::getData);
    this->MakeSlotAvailable( &this->dataOutSlot);

}

/*
 * BindingSiteDataSource::~BindingSiteDataSource (DTOR)
 */
BindingSiteDataSource::~BindingSiteDataSource( void ) {
    this->Release();
}

/*
 * BindingSiteDataSource::create
 */
bool BindingSiteDataSource::create() {
    
    return true;
}

/*
 * BindingSiteDataSource::release
 */
void BindingSiteDataSource::release() {
}

/*
 * BindingSiteDataSource::getData
 */
bool BindingSiteDataSource::getData( Call& call) {
    using vislib::sys::Log;

    BindingSiteCall *site = dynamic_cast<BindingSiteCall*>( &call);
    if ( !site ) return false;

    // try to load file, if necessary
    if ( this->pdbFilenameSlot.IsDirty() ) {
        this->pdbFilenameSlot.ResetDirty();
        this->loadPDBFile( this->pdbFilenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    // pass data to call, if available
    if( this->bindingSites.IsEmpty() ) {
        return false;
    } else {
        //site->SetDataHash( this->datahash);
        site->SetBindingSiteNames( &this->bindingSiteNames);
        site->SetBindingSiteResNames( &this->bindingSiteResNames);
        site->SetBindingSite( &this->bindingSites);
        return true;
    } 
}

/*
 * BindingSiteDataSource::loadPDBFile
 */
void BindingSiteDataSource::loadPDBFile( const vislib::TString& filename) {
    using vislib::sys::Log;
    
    // temp variables
    unsigned int i, j, lineCnt, bsIdx, resCnt, cnt;
    vislib::StringA line, seqNumString;
    char chainId;
    unsigned int resId;
    vislib::sys::ASCIIFileBuffer file;
    vislib::Array<vislib::StringA> bsEntries;
    SIZE_T bsEntriesCapacity = 100;
    bsEntries.AssertCapacity( bsEntriesCapacity);
    bsEntries.SetCapacityIncrement( bsEntriesCapacity);

    // reset data
    for( i = 0; i < this->bindingSites.Count(); i++ ) {
        this->bindingSites[i].Clear();
        this->bindingSiteResNames[i].Clear();
    }
    this->bindingSites.Clear();
    this->bindingSites.AssertCapacity( 20);
    this->bindingSites.SetCapacityIncrement( 10);
    this->bindingSiteResNames.Clear();
    this->bindingSiteResNames.AssertCapacity( 20);
    this->bindingSiteResNames.SetCapacityIncrement( 10);
    this->bindingSiteNames.Clear();
    this->bindingSiteNames.AssertCapacity( 20);
    this->bindingSiteNames.SetCapacityIncrement( 10);

    // try to load the file
    if( file.LoadFile( T2A( filename) ) ) {
        // file successfully loaded, read first frame
        lineCnt = 0;
        while( lineCnt < file.Count() && !line.StartsWith( "END") ) {
            // get the current line from the file
            line = file.Line( lineCnt);
            // store all site entries
            if( line.StartsWith( "SITE") ) {
                // add site entry
                bsEntries.Add( line);
            }
            // next line
            lineCnt++;
        }

        // parse site entries
        for( unsigned int i = 0; i < bsEntries.Count(); i++ ) {
            // write binding site name (check if this is the first entry)
            if( this->bindingSiteNames.IsEmpty() ) {
                this->bindingSiteNames.Add( bsEntries[i].Substring( 11, 4));
                this->bindingSites.Add( vislib::Array<vislib::Pair<char, unsigned int> >(10, 10));
                this->bindingSiteResNames.Add( vislib::Array<vislib::StringA>(10, 10));
                bsIdx = 0;
            } else {
                // check if next entry is still the same binding site
                //if( !bsEntries[i].Substring( 11, 4).Equals( bindingSiteNames[i]) ) {
                seqNumString = bsEntries[i].Substring( 7, 3);
                seqNumString.TrimSpaces();
                if( atoi( seqNumString) == 1 ) {
                    this->bindingSiteNames.Add( bsEntries[i].Substring( 11, 4));
                    this->bindingSites.Add( vislib::Array<vislib::Pair<char, unsigned int> >(10, 10));
                    this->bindingSiteResNames.Add( vislib::Array<vislib::StringA>(10, 10));
                    bsIdx++;
                }
            }

            // get number of residues
            line = bsEntries[i].Substring( 15, 2);
            line.TrimSpaces();
            resCnt = vislib::math::Clamp( 
                static_cast<unsigned int>(atoi( line) - bindingSites[bsIdx].Count()), 
                0U, 4U);

            // add residues
            cnt = 0;
            for( j = 0; j < resCnt; j++ ) {
                //resName
                line = bsEntries[i].Substring( 18 + 11 * cnt, 3);
                line.TrimSpaces();
                this->bindingSiteResNames[bsIdx].Add( line);
                // chainID
                line = bsEntries[i].Substring( 22 + 11 * cnt, 1);
                chainId = line[0];
                // seq (res seq num)
                line = bsEntries[i].Substring( 23 + 11 * cnt, 4);
                line.TrimSpaces();
                resId = static_cast<unsigned int>( atoi( line));
                // add binding site information
                this->bindingSites[bsIdx].Add( vislib::Pair<char, unsigned int>( chainId, resId));
                cnt++;
            }
        }
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Bindings Site count: %i", bindingSiteNames.Count() ); // DEBUG
    }

}
