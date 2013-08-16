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
        site->SetBindingSiteDescriptions( &this->bindingSiteDescription);
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
    vislib::Array<vislib::StringA> remarkEntries;
    SIZE_T entriesCapacity = 100;
    bsEntries.AssertCapacity( entriesCapacity);
    bsEntries.SetCapacityIncrement( entriesCapacity);
    remarkEntries.AssertCapacity( entriesCapacity);
    remarkEntries.SetCapacityIncrement( entriesCapacity);

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
            // store all remark 800 entries
            if( line.StartsWith( "REMARK 800") ) {
                line = line.Substring( 10);
                line.TrimSpaces();
                // add remark entry
                if( !line.IsEmpty() ) {
                    remarkEntries.Add( line);
                }
            }
            // next line
            lineCnt++;
        }

        // parse site entries
        for( unsigned int i = 0; i < bsEntries.Count(); i++ ) {
            // write binding site name (check if this is the first entry)
            if( this->bindingSiteNames.IsEmpty() ) {
                this->bindingSiteNames.Add( bsEntries[i].Substring( 11, 4));
                this->bindingSiteNames.Last().TrimSpaces();
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
                    this->bindingSiteNames.Last().TrimSpaces();
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
        // get binding site descriptons
        this->bindingSiteDescription.SetCount( this->bindingSiteNames.Count());
        for( unsigned int i = 0; i < this->bindingSiteNames.Count(); i++ ) {
            this->bindingSiteDescription[i] = this->ExtractBindingSiteDescripton( this->bindingSiteNames[i], remarkEntries);
        }

        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Bindings Site count: %i", bindingSiteNames.Count() ); // DEBUG
    }

}


/*
 *
 */
vislib::StringA BindingSiteDataSource::ExtractBindingSiteDescripton( vislib::StringA bsName, vislib::Array<vislib::StringA> remarkArray) {
    vislib::StringA retStr("");
    for( unsigned int i = 0; i < remarkArray.Count(); i++) {
        // search for binding site name
        if( remarkArray[i].EndsWith( bsName) ) {
            if( (i + 2) < remarkArray.Count() && remarkArray[i+2].StartsWith("SITE_DESCRIPTION:") ) {
                retStr = remarkArray[i+2].Substring( 17);
                retStr.TrimSpaces();
                remarkArray.RemoveAt(i);
                remarkArray.RemoveAt(i);
                remarkArray.RemoveAt(i);
                return retStr;  
            }
        }
    }
    return retStr;
}
