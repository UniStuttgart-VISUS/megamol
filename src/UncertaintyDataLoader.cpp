/*
* UncertaintyDataLoader.cpp
*
* Author: Matthias Braun
* Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*
* This module is based on the source code of "BindingSiteData" in megamol protein plugin (svn revision 1500).
*
*/

#include "stdafx.h"

#include "UncertaintyDataLoader.h"

#include "Python.h"

#include <cstdlib> // for: mbstowcs() - multi byte string to wide char string
#include <math.h>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"

#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/math/mathfunctions.h"

#include "UncertaintyColor.h"
#include "UncertaintyDataCall.h"


using namespace megamol::core;
using namespace megamol::protein_uncertainty;


/*
 * UncertaintyDataLoader::UncertaintyDataLoader (CTOR)
 */
UncertaintyDataLoader::UncertaintyDataLoader( void ) : megamol::core::Module(),
        dataOutSlot( "dataout", "The slot providing the uncertainty data"),
		pdbIDSlot("PDB-ID", "The PDB ID ..."),
        pdbFilenameSlot( "pdbFilename", "The PDB file containing the binding site information"),
        colorTableFileParam( "ColorTableFilename", "The filename of the color table.") {
      
	this->pdbIDSlot << new param::StringParam("");
	this->MakeSlotAvailable(&this->pdbIDSlot);

    this->pdbFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->pdbFilenameSlot);
    
	this->dataOutSlot.SetCallback(UncertaintyDataCall::ClassName(), UncertaintyDataCall::FunctionName(UncertaintyDataCall::CallForGetData), &UncertaintyDataLoader::getData);
    this->MakeSlotAvailable( &this->dataOutSlot);
    
    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    this->colorTableFileParam.SetParameter(new param::FilePathParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);
    UncertaintyColor::ReadColorTableFromFile( T2A(this->colorTableFileParam.Param<param::FilePathParam>()->Value()), this->colorLookupTable);
}

/*
 * UncertaintyDataLoader::~UncertaintyDataLoader (DTOR)
 */
UncertaintyDataLoader::~UncertaintyDataLoader( void ) {
    this->Release();
}

/*
 * UncertaintyDataLoader::create
 */
bool UncertaintyDataLoader::create() {
    
    return true;
}

/*
 * UncertaintyDataLoader::release
 */
void UncertaintyDataLoader::release() {

}

/*
 * UncertaintyDataLoader::getData
 */
bool UncertaintyDataLoader::getData( Call& call) {
    using vislib::sys::Log;

	// ...
	UncertaintyDataCall *udc = dynamic_cast<UncertaintyDataCall*>(&call);
    if ( !udc ) return false;

	// try to load run python script, if necessary
	if (this->pdbIDSlot.IsDirty()) {
		this->pdbIDSlot.ResetDirty();
		this->runPythonScript(this->pdbIDSlot.Param<core::param::StringParam>()->Value());
	}

	return true;

    // read and update the color table, if necessary
    /*if( this->colorTableFileParam.IsDirty() ) {
        UncertaintyColor::ReadColorTableFromFile( T2A(this->colorTableFileParam.Param<param::FilePathParam>()->Value()), this->colorLookupTable);
        this->colorTableFileParam.ResetDirty();
    }*/
    

    // try to load file, if necessary
    /*if ( this->pdbFilenameSlot.IsDirty() ) {
        this->pdbFilenameSlot.ResetDirty();
        this->loadPDBFile( this->pdbFilenameSlot.Param<core::param::FilePathParam>()->Value());
    }*/

    // pass data to call, if available
    /*if( this->bindingSites.IsEmpty() ) {
        return false;
    } else {
        //site->SetDataHash( this->datahash);
        //site->SetBindingSiteNames( &this->bindingSiteNames);
        //site->SetBindingSiteDescriptions( &this->bindingSiteDescription);
        //site->SetBindingSiteResNames( &this->bindingSiteResNames);
        //site->SetBindingSite( &this->bindingSites);
        //site->SetBindingSiteColors( &this->bindingSiteColors);
        return true;
    } */
}

/*
* UncertaintyDataLoader::runPythonScript
*/
void UncertaintyDataLoader::runPythonScript(const vislib::StringA &pdbid) {
	using vislib::sys::Log;


	Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Starting \"runPythonScript\" for %s", pdbid); // DEBUG


	vislib::StringA PythonArg0_Script = "UncertaintyInputData.py";
	vislib::StringA PythonArg1_PDB    = pdbid;
	vislib::StringA PythonArg2_d      = "-d";

	wchar_t *wPythonArg0 = new wchar_t[PythonArg2_d.Length()];
	wchar_t *wPythonArg1 = new wchar_t[PythonArg1_PDB.Length()];
	wchar_t *wPythonArg2 = new wchar_t[PythonArg2_d.Length()];
	mbstowcs(&wPythonArg0[0], PythonArg0_Script, sizeof(wPythonArg0));
	mbstowcs(&wPythonArg1[0], PythonArg1_PDB, sizeof(wPythonArg1));
	mbstowcs(&wPythonArg2[0], PythonArg2_d, sizeof(wPythonArg2));

	wchar_t* wPythonArgv[] = {&wPythonArg0[0], &wPythonArg1[0], &wPythonArg2[0], NULL};
	int wPythonArgc = (int)(sizeof(wPythonArgv) / sizeof(wPythonArgv[0])) - 1;

	/*
	// initialize the embedded python interpreter
	Py_SetProgramName(wPythonArgv[0]);
	Py_Initialize();
	PySys_SetArgv(wPythonArgc, wPythonArgv);

	
	// open script file
	FILE *ScriptFile;
	fopen_s(&ScriptFile, PythonArg0_Script, "r");
	if (ScriptFile != NULL) {
		// call python script with interpreter
		PyRun_SimpleFileEx(ScriptFile, PythonArg0_Script, 1); // last parameter == 1 means to close the file before returning.
		// DON'T call: fclose(ScriptFile);

	}
	else {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ">>> ERROR: Couldn't find/open file: \"%s\"", PythonArg0_Script); // DEBUG
	}
	

	// end the python interpreter
	Py_Finalize();
	*/

	delete wPythonArg0;
	delete wPythonArg1;
	delete wPythonArg2;


    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "End of \"runPythonScript\" for %s", pdbid); // DEBUG

}



/*
 * UncertaintyDataLoader::loadPDBFile
 */
void UncertaintyDataLoader::loadPDBFile( const vislib::TString& filename) {
    using vislib::sys::Log;
    
    // temp variables
    unsigned int i, j, lineCnt, bsIdx, /*resCnt,*/ cnt;
    vislib::StringA line, seqNumString, tmpBSName;
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
            if (line.StartsWith("SITE") || line.StartsWith("BSITE")) {
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
                tmpBSName = bsEntries[i].Substring(11, 4);
                tmpBSName.TrimSpaces();
                if (!tmpBSName.Equals(bindingSiteNames.Last())) {
                    seqNumString = bsEntries[i].Substring(7, 3);
                    seqNumString.TrimSpaces();
                    if (atoi(seqNumString) == 1) {
                        this->bindingSiteNames.Add(bsEntries[i].Substring(11, 4));
                        this->bindingSiteNames.Last().TrimSpaces();
                        this->bindingSites.Add(vislib::Array<vislib::Pair<char, unsigned int> >(10, 10));
                        this->bindingSiteResNames.Add(vislib::Array<vislib::StringA>(10, 10));
                        bsIdx++;
                    }
                }
            }

            // get number of residues
            //line = bsEntries[i].Substring( 15, 2);
            //line.TrimSpaces();
            //// regular PDB SITE entries can store a maximum of 4 residues per line
            //if (bsEntries[i].StartsWith("SITE")) {
            //    resCnt = vislib::math::Clamp(
            //        static_cast<unsigned int>(atoi(line) - bindingSites[bsIdx].Count()),
            //        0U, 4U);
            //}
            //else {
            //    resCnt = static_cast<unsigned int>(atoi(line) - bindingSites[bsIdx].Count());
            //}

            // add residues
            cnt = 0;
            //for( j = 0; j < resCnt; j++ ) {
            for (j = 0; j < 4; j++) {
                //resName
                line = bsEntries[i].Substring( 18 + 11 * cnt, 3);
                line.TrimSpaces();
                if (line.IsEmpty()) break;
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
        // get binding site descriptons and set colors
        this->bindingSiteDescription.SetCount( this->bindingSiteNames.Count());
        this->bindingSiteColors.SetCount( this->bindingSiteNames.Count());
        for( unsigned int i = 0; i < this->bindingSiteNames.Count(); i++ ) {
            //this->bindingSiteDescription[i] = this->ExtractBindingSiteDescripton( this->bindingSiteNames[i], remarkEntries);
            this->bindingSiteColors[i] = this->colorLookupTable[i%this->colorLookupTable.Count()];
        }

        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Bindings Site count: %i", bindingSiteNames.Count() ); // DEBUG
    }

}
