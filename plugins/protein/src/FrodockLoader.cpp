/*
 * FrodockLoader.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "FrodockLoader.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/sys/Log.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/net/SocketException.h"
#include "vislib/net/IPEndPoint.h"
#include "vislib/net/DNS.h"
#include <ctime>
#include <iostream>
#include <omp.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 * protein::FrodockLoader::FrodockLoader
 */
FrodockLoader::FrodockLoader(void) : Module(),
        filenameSlot( "filename", "The path to the PDB data file to be loaded"),
        dataOutSlot( "dataout", "The slot providing the loaded data"),
        strideFlagSlot( "strideFlag", "The flag wether STRIDE should be used or not."),
        receptorDataCallerSlot( "receptorData", "The slot providing the data of the receptor molecule."),
        ligandDataCallerSlot( "ligandData", "The slot providing the data of the ligand."),
        fileServerNameSlot( "fileServerName", "The file server name (for linux file paths)."),
        hostAddressSlot( "hostAddress", "The host address of the machine on which Frodock is running."),
        portSlot( "port", "The port over which the communication is executed."),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), datahash(0), currentSolution( -1) {
    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->filenameSlot);

	this->dataOutSlot.SetCallback(megamol::protein_calls::MolecularDataCall::ClassName(), megamol::protein_calls::MolecularDataCall::FunctionName(megamol::protein_calls::MolecularDataCall::CallForGetData), &FrodockLoader::getData);
	this->dataOutSlot.SetCallback(megamol::protein_calls::MolecularDataCall::ClassName(), megamol::protein_calls::MolecularDataCall::FunctionName(megamol::protein_calls::MolecularDataCall::CallForGetExtent), &FrodockLoader::getExtent);
    this->MakeSlotAvailable( &this->dataOutSlot);

    this->strideFlagSlot << new param::BoolParam( true);
    this->MakeSlotAvailable( &this->strideFlagSlot);

    // receptor
	this->receptorDataCallerSlot.SetCompatibleCall<megamol::protein_calls::MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->receptorDataCallerSlot);

    // ligand
	this->ligandDataCallerSlot.SetCompatibleCall<megamol::protein_calls::MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->ligandDataCallerSlot);
    
    // file server name
    this->fileServerNameSlot << new param::StringParam( "");
    this->MakeSlotAvailable( &this->fileServerNameSlot);
    
    // host
    this->hostAddressSlot << new param::StringParam( "");
    this->MakeSlotAvailable( &this->hostAddressSlot);
    // port (default, min, max)
    this->portSlot << new param::IntParam( 1234, 1024, 32767);
    this->MakeSlotAvailable( &this->portSlot);
    
    // set potential-pointer to null
    this->frodockInput.potentials = 0;
}


/*
 * protein::FrodockLoader::~FrodockLoader
 */
FrodockLoader::~FrodockLoader(void) {
    this->Release ();
}


/*
 * FrodockLoader::create
 */
bool FrodockLoader::create(void) {
    // set default values to frodockInput
    memset( frodockInput.receptor, 0, NAMESIZE * sizeof( char));
    memset( frodockInput.ligand, 0, NAMESIZE * sizeof( char));
    memset( frodockInput.vdw, 0, NAMESIZE * sizeof( char));
    frodockInput.vdw_weight = 0.0f;
    memset( frodockInput.ele, 0, NAMESIZE * sizeof( char));
    frodockInput.ele_weight = 0.0f;
    memset( frodockInput.desol_rec, 0, NAMESIZE * sizeof( char));
    memset( frodockInput.desol_lig, 0, NAMESIZE * sizeof( char));
    memset( frodockInput.asa_rec, 0, NAMESIZE * sizeof( char));
    memset( frodockInput.asa_lig, 0, NAMESIZE * sizeof( char));
    frodockInput.desol_weight = 0.0f;
    frodockInput.num_pot = 0;
    frodockInput.potentials = 0;
    frodockInput.bw = 0;
    frodockInput.lmax = 0.0f;
    frodockInput.lmin = 0.0f;
    frodockInput.th = 0.0f;
    frodockInput.lw = 0.0f;
    frodockInput.st = 0.0f;
    frodockInput.np = 0;
    frodockInput.rd = 0.0f;
    frodockInput.nt = 0;
    frodockInput.td = 0.0f;
    frodockInput.use_around = false;
    memset( frodockInput.around_point, 0, 3 * sizeof( float));
    memset( frodockInput.points, 0, NAMESIZE * sizeof( char));
    frodockInput.conv = Rosseta;

    // try to start up socket
    try {
        vislib::net::Socket::Startup();
    } catch( vislib::net::SocketException e ) {
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_ERROR, "Socket Exception during startup: %s", e.GetMsgA() );
    }
    return true;
}


/*
 * FrodockLoader::getData
 */
bool FrodockLoader::getData( core::Call& call) {
    using vislib::sys::Log;

	megamol::protein_calls::MolecularDataCall *dc = dynamic_cast<megamol::protein_calls::MolecularDataCall*>(&call);
    if ( dc == NULL ) return false;

    // try to load the input file
    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    // variables for parameter transfer
    vislib::StringA paramSlotName;
    param::ParamSlot *paramSlot;

    //////////////////////////////////////////////////////////////////
    // get pointer to receptor MolecularDataCall
    //////////////////////////////////////////////////////////////////
	megamol::protein_calls::MolecularDataCall *receptor = this->receptorDataCallerSlot.CallAs<megamol::protein_calls::MolecularDataCall>();
    vislib::StringA receptorFilename( frodockInput.receptor);
#ifdef WIN32
    // convert linux path for windows, if fileServerName is set
    if( !fileServerNameSlot.Param<param::StringParam>()->Value().IsEmpty() ) {
        if( receptorFilename.StartsWith( '/') ) {
            receptorFilename.Replace( '/', '\\');
        }
        receptorFilename.Prepend( fileServerNameSlot.Param<param::StringParam>()->Value());
        receptorFilename.Prepend( "\\\\");
    }
#endif
    // set parameter slots of the receptor
    if( receptor && frodockInput.receptor[0] != 0 ) {
        paramSlotName = "";
        paramSlot = 0;
        // get and set filename param
        paramSlotName = receptor->PeekCalleeSlot()->Parent()->FullName();
        paramSlotName += "::filename";
        paramSlot = dynamic_cast<param::ParamSlot*>(this->FindNamedObject(paramSlotName, true).get());
        if( paramSlot ) {
            paramSlot->Param<param::FilePathParam>()->SetValue( A2T( receptorFilename));
        }
        // get and set stride param
        paramSlotName = receptor->PeekCalleeSlot()->Parent()->FullName();
        paramSlotName += "::strideFlag";
        paramSlot = dynamic_cast<param::ParamSlot*>(this->FindNamedObject(paramSlotName, true).get());
        if( paramSlot ) {
            paramSlot->Param<param::BoolParam>()->SetValue( this->strideFlagSlot.Param<param::BoolParam>()->Value());
        }
        // all parameters set, execute the data call
		if (!(*receptor)(megamol::protein_calls::MolecularDataCall::CallForGetData))
            Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Could not load receptor file."); // DEBUG
        //else
        //    Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Successfully load receptor file."); // DEBUG
    }
    
    //////////////////////////////////////////////////////////////////
    // get pointer to ligand MolecularDataCall
    //////////////////////////////////////////////////////////////////
	megamol::protein_calls::MolecularDataCall *ligand = this->ligandDataCallerSlot.CallAs<megamol::protein_calls::MolecularDataCall>();
    vislib::StringA ligandFilename( frodockInput.ligand);
#ifdef WIN32
    // convert linux path for windows, if fileServerName is set
    if( !fileServerNameSlot.Param<param::StringParam>()->Value().IsEmpty() ) {
        if( ligandFilename.StartsWith( '/') ) {
            ligandFilename.Replace( '/', '\\');
        }
        ligandFilename.Prepend( fileServerNameSlot.Param<param::StringParam>()->Value());
        ligandFilename.Prepend( "\\\\");
    }
#endif
    // set parameter slots of the ligand
    if( ligand && frodockInput.ligand[0] != 0 ) {
        paramSlotName = "";
        paramSlot = 0;
        // get and set filename param
        paramSlotName = ligand->PeekCalleeSlot()->Parent()->FullName();
        paramSlotName += "::filename";
        paramSlot = dynamic_cast<param::ParamSlot*>(this->FindNamedObject(paramSlotName, true).get());
        if( paramSlot ) {
            paramSlot->Param<param::FilePathParam>()->SetValue( A2T( ligandFilename));
        }
        // get and set stride param
        paramSlotName = ligand->PeekCalleeSlot()->Parent()->FullName();
        paramSlotName += "::strideFlag";
        paramSlot = dynamic_cast<param::ParamSlot*>(this->FindNamedObject(paramSlotName, true).get());
        if( paramSlot ) {
            paramSlot->Param<param::BoolParam>()->SetValue( this->strideFlagSlot.Param<param::BoolParam>()->Value());
        }
        // all parameters set, execute the data call
		if (!(*ligand)(megamol::protein_calls::MolecularDataCall::CallForGetData))
            Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Could not load ligand file."); // DEBUG
        //else
        //    Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Successfully load ligand file."); // DEBUG
    }

    //if ( dc->FrameID() >= this->data.Count() ) return false;

    dc->SetDataHash( this->datahash);

    // TODO: assign the data from the loader to the call

    if (!(*ligand)(MolecularDataCall::CallForGetData)) return false;

    unsigned int currentSolution = 0;
    // DEBUG ...
    /*
    this->ligandCenter[0] = 30.709869f;
    this->ligandCenter[1] =-12.022730f;
    this->ligandCenter[2] =-11.278584f;
    this->numSolutions = 3;
    this->solutions.AssertSize( sizeof(float)*this->numSolutions*7);
    this->solutions.As<float>()[0] = 177.187607f;
    this->solutions.As<float>()[1] = 33.763191f;
    this->solutions.As<float>()[2] = 292.083557f;
    this->solutions.As<float>()[3] = 32.628811f;
    this->solutions.As<float>()[4] =-40.611145f;
    this->solutions.As<float>()[5] =-25.521893f;
    this->solutions.As<float>()[6] = 423.018768f;

    this->solutions.As<float>()[7] = 186.82f;
    this->solutions.As<float>()[8] = 46.25f;
    this->solutions.As<float>()[9] = 356.14f;
    this->solutions.As<float>()[10] = 32.628811f;
    this->solutions.As<float>()[11] =-40.611145f;
    this->solutions.As<float>()[12] =-25.521893f;
    this->solutions.As<float>()[13] = 403.394592f;

    this->solutions.As<float>()[14] = 318.10f;
    this->solutions.As<float>()[15] = 133.80f;
    this->solutions.As<float>()[16] = 95.74f;
    this->solutions.As<float>()[17] = 36.63f;
    this->solutions.As<float>()[18] =-22.61f;
    this->solutions.As<float>()[19] =-21.52f;
    this->solutions.As<float>()[20] = 620.939026f;
    */
    // ...DEBUG

    // apply the solution and set values to data call
    if( this->applySolution( ligand, dc->FrameID()) ) {
        dc->SetAtoms(
            ligand->AtomCount(),
            ligand->AtomTypeCount(),
            ligand->AtomTypeIndices(),
            this->atomPos.PeekElements(),
            ligand->AtomTypes(),
            ligand->AtomResidueIndices(),
            ligand->AtomBFactors(),
            ligand->AtomCharges(),
            ligand->AtomOccupancies());
    } else {
        dc->SetAtoms(
            ligand->AtomCount(),
            ligand->AtomTypeCount(),
            ligand->AtomTypeIndices(),
            ligand->AtomPositions(),
            ligand->AtomTypes(),
            ligand->AtomResidueIndices(),
            ligand->AtomBFactors(),
            ligand->AtomCharges(),
            ligand->AtomOccupancies());
    }
    dc->SetBFactorRange( 
        ligand->MaximumBFactor(),
        ligand->MinimumBFactor());
    dc->SetChargeRange( 
        ligand->MaximumCharge(),
        ligand->MinimumCharge());
    dc->SetOccupancyRange( 
        ligand->MaximumOccupancy(),
        ligand->MinimumOccupancy());
    dc->SetConnections(
        ligand->ConnectionCount(), 
        (unsigned int*)ligand->Connection());
    dc->SetResidues( 
        ligand->ResidueCount(),
        ligand->Residues());
    dc->SetResidueTypeNames(
        ligand->ResidueTypeNameCount(),
        ligand->ResidueTypeNames());
    dc->SetMolecules(
        ligand->MoleculeCount(),
        (MolecularDataCall::Molecule*)ligand->Molecules());
    dc->SetChains(
        ligand->ChainCount(),
        (MolecularDataCall::Chain*)ligand->Chains());

    /*
    if( !this->secStructAvailable && this->strideFlagSlot.Param<param::BoolParam>()->Value() ) {
        time_t t = clock(); // DEBUG
        if( this->stride ) delete this->stride;
        this->stride = new Stride( dc);
        this->stride->WriteToInterface( dc);
        this->secStructAvailable = true;
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Secondary Structure computed via STRIDE in %f seconds.", ( double( clock() - t) / double( CLOCKS_PER_SEC))); // DEBUG
    }
    */

    dc->SetUnlocker( NULL);

    return true;
}


/*
 * FrodockLoader::getExtent
 */
bool FrodockLoader::getExtent( core::Call& call) {
    MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    // get pointer to ligand MolecularDataCall
    MolecularDataCall *ligand = this->ligandDataCallerSlot.CallAs<MolecularDataCall>();

    if( ligand ) {
        this->applySolution( ligand, this->currentSolution);
        // get extends of ligand
        if (!(*ligand)(MolecularDataCall::CallForGetExtent)) return false;
        dc->AccessBoundingBoxes().Clear();
        dc->AccessBoundingBoxes() = ligand->AccessBoundingBoxes();
        dc->AccessBoundingBoxes().SetObjectSpaceClipBox( this->bbox);
    }

    // set frame count
    dc->SetFrameCount( vislib::math::Max(1U, 
        (unsigned int)(this->numSolutions)));

    dc->SetDataHash( this->datahash);

    return true;
}


/*
 * FrodockLoader::release
 */
void FrodockLoader::release(void) {
    try {
        vislib::net::Socket::Cleanup();
    } catch( vislib::net::SocketException e ) {
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_ERROR, "Socket Exception during cleanup: %s", e.GetMsgA() );
    }
}


/*
 * FrodockLoader::loadFile
 */
void FrodockLoader::loadFile( const vislib::TString& filename) {
    using vislib::sys::Log;

    // counter variable
    int cnt;

    this->datahash++;
	int around;

    FILE *f = fopen( T2A( filename), "rt");
	fscanf( f, "%*[^\n]\n"); //#FRODOCK CONFIGURATION INPUT FILE

	fscanf( f, "%*[^\n]\n");//#PDB FILES. MANDATORY
	fscanf( f, "%*[^\n]\n");//#Receptor (Mandatory)
	fscanf( f, "%[^\n]\n", frodockInput.receptor);//receptor.pdb
	fscanf( f, "%*[^\n]\n");//#Ligand (Mandatory)
	fscanf( f, "%[^\n]\n", frodockInput.ligand);//ligand.pdb

	fscanf( f, "%*[^\n]\n");//#MAIN POTENTIALS. OPTIONALS
	fscanf( f, "%*[^\n]\n");//#Van der Waals potential
	fscanf( f, "%[^\n]\n", frodockInput.vdw);//vdw.ccp4
	if(strcmp(frodockInput.vdw,"#") == 0)
		strcpy(frodockInput.vdw, "");
	fscanf( f, "%*[^\n]\n");//#Van der Waals weight (default 1.0)
	fscanf( f, "%f\n", &(frodockInput.vdw_weight));//1.0
	fscanf( f, "%*[^\n]\n");//#Electrostatic potential
	fscanf( f, "%[^\n]\n", frodockInput.ele);//ele.ccp4
	if(strcmp(frodockInput.ele,"#") == 0)
		strcpy(frodockInput.ele, "");
	fscanf( f, "%*[^\n]\n");//#Electrostatic weight (default 0.0)
	fscanf( f, "%f\n", &(frodockInput.ele_weight));//0.0;
	fscanf( f, "%*[^\n]\n");//#Receptor desolvation potential
	fscanf( f, "%[^\n]\n", frodockInput.desol_rec);//desol_rec.ccp4
	if(strcmp(frodockInput.desol_rec,"#") == 0)
		strcpy(frodockInput.desol_rec,"");
	fscanf( f, "%*[^\n]\n");//#Ligand desolvation potential
	fscanf( f, "%[^\n]\n", frodockInput.desol_lig);//desol_lig.ccp4
	if(strcmp(frodockInput.desol_lig,"#") == 0)
		strcpy(frodockInput.desol_lig,"");
	fscanf( f, "%*[^\n]\n");//#Receptor Accesibility map
	fscanf( f, "%[^\n]\n", frodockInput.asa_rec);//asa_rec.ccp4
	if(strcmp(frodockInput.asa_rec,"#") == 0)
		strcpy(frodockInput.asa_rec,"");
	fscanf( f, "%*[^\n]\n");//#Ligand Accesibility map
	fscanf( f, "%[^\n]\n", frodockInput.asa_lig);//asa_lig.ccp4
	if(strcmp(frodockInput.asa_lig,"#") == 0)
		strcpy(frodockInput.asa_lig,"");
	fscanf( f, "%*[^\n]\n");//#Desolvation weight (default 0.0)
	fscanf( f, "%f\n", &(frodockInput.desol_weight));

	fscanf( f, "%*[^\n]\n");//#EXTRA POTENTIALS. OPTIONALS
	fscanf( f, "%*[^\n]\n");//#Number of extra potentials (default 0)
	fscanf( f, "%d\n", &(frodockInput.num_pot));
	fscanf( f, "%*[^\n]\n");//#Extra potrential name, weigh and type
    // delete potentials if necessary
    if( this->frodockInput.potentials )
        delete[] this->frodockInput.potentials;
    // create new potential array
	//frodockInput.potentials = (Potential_FI*)malloc( sizeof( Potential_FI)*frodockInput.num_pot);
    this->frodockInput.potentials = new Potential_FI[frodockInput.num_pot];
	for( int i = 0; i<frodockInput.num_pot; ++i ) {
		fscanf( f, "%[^\n]\n", frodockInput.potentials[i].name);
		fscanf( f, "%f\n", &(frodockInput.potentials[i].weight));
		fscanf( f, "%d\n", &(frodockInput.potentials[i].type));
	}

	fscanf( f, "%*[^\n]\n");//#SEARCH PARAMETERS
	fscanf( f, "%*[^\n]\n");//#Bandwitdh in spherical harmonic representation. Define rotational stepsize (default: 32. Rotational stepsize ~11º)
	fscanf( f, "%d\n", &(frodockInput.bw));
	fscanf( f, "%*[^\n]\n");//#External Mask reduction ratio (default: 0.25).
	fscanf( f, "%f\n", &(frodockInput.lmax));
	fscanf( f, "%*[^\n]\n");//#Internal Mask reduction ratio (default: 0.26).
	fscanf( f, "%f\n", &(frodockInput.lmin));
	fscanf( f, "%*[^\n]\n");//#Electrostatic map threshold (default: 10.0)
	fscanf( f, "%f\n", &(frodockInput.th));
	fscanf( f, "%*[^\n]\n");//#Width between spherical layers in amstrongs (default: 1.0).
	fscanf( f, "%f\n", &(frodockInput.lw));
	fscanf( f, "%*[^\n]\n");//#Translational search stepsize in amstrongs (default: 2.0).
	fscanf( f, "%f\n", &(frodockInput.st));
	fscanf( f, "%*[^\n]\n");//#Number of solutions stored per traslational position. (default: 4)
	fscanf( f, "%d\n", &(frodockInput.np));
	fscanf( f, "%*[^\n]\n");//#Minimal rotational distance allowed between close solutions in degrees (default: 12.0)
	fscanf( f, "%f\n", &(frodockInput.rd));
	fscanf( f, "%*[^\n]\n");//#Number of solutions stored in the search (default: unlimited -1).
	fscanf( f, "%d\n", &(frodockInput.nt));
	fscanf( f, "%*[^\n]\n");//#Maximal translational distance to consider close solutions in grid units. (default: 0)
	fscanf( f, "%f\n", &(frodockInput.td));
	fscanf( f, "%*[^\n]\n");//#Limit the translational search to a region (default: false(0))
	fscanf( f, "%d\n", &(around));
	if( around == 0 )
		frodockInput.use_around = false;
	else
		frodockInput.use_around = true;
	fscanf( f, "%*[^\n]\n");//#Coordinates XYZ of the central point of the region
	fscanf( f, "%f\n", &(frodockInput.around_point[0]));//#Coordinates XYZ of the central point of the region
	fscanf( f, "%f\n", &(frodockInput.around_point[1]));//#Coordinates XYZ of the central point of the region
	fscanf( f, "%f\n", &(frodockInput.around_point[2]));//#Coordinates XYZ of the central point of the region
	strcpy(frodockInput.points,"");
	frodockInput.conv=Rosseta;
	fclose(f);

    // reset data 
    this->numSolutions = 0;
    // Communication with frodock
    try {
        // create socket
        this->socket.Create( vislib::net::Socket::FAMILY_INET, vislib::net::Socket::TYPE_STREAM, vislib::net::Socket::PROTOCOL_TCP);
    } catch( vislib::net::SocketException e) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Socket Exception during create: %s", e.GetMsgA() );
    }
    try {
        // connect to frodock
        this->socket.Connect( vislib::net::IPEndPoint::CreateIPv4(
            T2A( this->hostAddressSlot.Param<param::StringParam>()->Value()),
            this->portSlot.Param<param::IntParam>()->Value()));    
        // send input data
        this->socket.Send( &(frodockInput), sizeof(FrodockInput), vislib::net::Socket::TIMEOUT_INFINITE, 0, true);
        // send extra potentials
	    if( frodockInput.num_pot > 0 ) {
		    for( int i = 0; i < frodockInput.num_pot; ++i) {
                if( this->socket.Send( &(frodockInput.potentials[i]), sizeof(Potential_FI), vislib::net::Socket::TIMEOUT_INFINITE, 0, true) <= 0 ) {
                    Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Sending extra potential %i over the socket failed.", i);
                    return;
			    }
		    }
	    }
        // store current time
        time_t t = clock();
        // receive return value
        this->frodockResult = 0;
        this->socket.Receive( &this->frodockResult, sizeof(int), vislib::net::Socket::TIMEOUT_INFINITE, 0, true);
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Received return value after %f seconds.", ( double( clock() - t) / double( CLOCKS_PER_SEC))); // DEBUG
        if( this->frodockResult < 0 ) {
            // --- received error value (0 or -1) ---
            if( this->frodockResult == 0 ) {
                Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Execution of Frodock not successful, try to get error message...");
                char buffer[NAMESIZE];
                this->socket.Receive( buffer, sizeof(char)*NAMESIZE, vislib::net::Socket::TIMEOUT_INFINITE, 0, true);
    		    Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Input error comunicated: %s.", buffer);
    	    } else {
                Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "CATASTROPHIC ERROR: Frodock Server has collapsed. Restart the server and check input data.");
    	    }
        } else {
            // --- received success value ---
            Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Successful Execution of Frodock reported.");
            // get ligand center
            this->socket.Receive( this->ligandCenter, sizeof(float)*3, vislib::net::Socket::TIMEOUT_INFINITE, 0, true);
            Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Ligand center: %f, %f, %f.", this->ligandCenter[0], this->ligandCenter[1], this->ligandCenter[2]);
            // get the number of solutions
            this->socket.Receive( &this->numSolutions, sizeof(int), vislib::net::Socket::TIMEOUT_INFINITE, 0, true);
            Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Number of solutions: %i.", this->numSolutions);
            // get the solutions
            if( this->numSolutions > 0 ) {
                this->solutions.AssertSize( sizeof(float)*this->numSolutions*7);
                for( cnt = 0; cnt < this->numSolutions; ++cnt ) {
                    this->socket.Receive( this->solutions.AsAt<float>( sizeof(float)*cnt*7), sizeof(float)*7, vislib::net::Socket::TIMEOUT_INFINITE, 0, true);
                }
            }
        }
    } catch( vislib::net::SocketException e) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Socket Exception during connect/send: %s", e.GetMsgA() );
    }
}

/*
 * applies a solution (computes new atom positions)
 */
bool FrodockLoader::applySolution( const MolecularDataCall *ligand, unsigned int solIdx) {
    // check if the ligand pointer is null
    if( !ligand ) return false;
    // check if the requested solution is already applied
    if( this->currentSolution == solIdx ) return true;
    // check if the requested solution is out of bounds
    if( int(solIdx) < 0 || int(solIdx) >= this->numSolutions ) return false;

    // transform atom positions
    vislib::math::Vector<float, 3> pos;
    const vislib::math::Vector<float, 3> center( this->ligandCenter);
    const vislib::math::Vector<float, 3> translation( &this->solutions.As<float>()[solIdx*7+3]);
    this->atomPos.SetCount( ligand->AtomCount()*3);
    float psi = this->solutions.As<float>()[solIdx*7+0];
    float theta = this->solutions.As<float>()[solIdx*7+1];
    float phi = this->solutions.As<float>()[solIdx*7+2];
    phi *= float( vislib::math::PI_DOUBLE / 180.0);
    theta *= float( vislib::math::PI_DOUBLE / 180.0);
    psi *= float( vislib::math::PI_DOUBLE / 180.0);

    /*
    // wikipedia - Benutzer:Garufalo/Spielwiese
    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> rotMat(
        cos( phi) * cos( psi) - cos( theta) * sin( phi) * sin( psi),
        -cos( psi) * sin( phi) - cos( theta) * cos( phi) * sin( psi),
        sin( theta) * sin( psi),
        //------------------------------------------
        cos( theta) * cos( psi) * sin( phi) + cos( phi) * sin( psi),
        cos( theta) * cos( phi) * cos( psi) - sin( phi) * sin( psi),
        -cos( psi) * sin( theta),
        //------------------------------------------
        sin( theta) * sin( phi),
        cos( phi) * sin( theta),
        cos( theta)
        );
    */
    /*
    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> rotMatZ3(
        cos( psi),
        -sin( psi),
        0.0f,
        //------------------------------------------
        sin( psi),
        cos( psi),
        0.0f,
        //------------------------------------------
        0.0f,
        0.0f,
        1.0f
        );
    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> rotMatX2(
        1.0f,
        0.0f,
        0.0f,
        //------------------------------------------
        0.0f,
        cos( theta),
        -sin( theta),
        //------------------------------------------
        0.0f,
        sin( theta),
        cos( theta)
        );
    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> rotMatY2(
        cos( theta),
        0.0f,
        sin( theta),
        //------------------------------------------
        0.0f,
        1.0f,
        0.0f,
        //------------------------------------------
        -sin( theta),
        0.0f,
        cos( theta)
        );
    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> rotMatZ1(
        cos( phi),
        -sin( phi),
        0.0f,
        //------------------------------------------
        sin( phi),
        cos( phi),
        0.0f,
        //------------------------------------------
        0.0f,
        0.0f,
        1.0f
        );
    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> rotMat = rotMatZ3 * rotMatX2 * rotMatZ1;
    */
    /*
    // matheboard
    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> rotMat(
        cos( psi) * cos( phi) - cos( theta) * sin( phi) * sin( psi),
        -sin( psi) * cos( phi) - cos( theta) * sin( phi) * cos( psi),
        sin( theta) * sin( phi),
        //------------------------------------------
        cos( psi) * sin( phi) + cos( theta) * cos( phi) * sin( psi),
        -sin( psi) * sin( phi) + cos( theta) * cos( phi) * cos( psi) ,
        -sin( theta) * cos( phi),
        //------------------------------------------
        sin( psi) * sin( theta),
        cos( psi) * sin( theta),
        cos( theta)
        );
    */
    // Nacho
    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> rotMat(
        cos( psi) * cos( phi) - cos( theta) * sin( phi) * sin( psi),
        cos( psi) * sin( phi) + cos( theta) * cos( phi) * sin( psi),
        sin( psi) * sin( theta),
        -sin( psi) * cos( phi) - cos( theta) * sin( phi) * cos( psi),
        -sin( psi) * sin( phi) + cos( theta) * cos( phi) * cos( psi),
        cos( psi) * sin( theta),
        sin( theta) * sin( phi),
        -sin( theta) * cos( phi),
        cos( theta)
        );
        
#pragma omp parallel for private( pos)
    for( int cnt = 0; cnt < int( ligand->AtomCount()); ++cnt ) {
        pos.Set( ligand->AtomPositions()[3*cnt],
            ligand->AtomPositions()[3*cnt+1],
            ligand->AtomPositions()[3*cnt+2]);
        //pos -= center;
        pos = pos - center;
        pos = rotMat * pos;
        pos += translation;
        this->atomPos[3*cnt] = pos.X();
        this->atomPos[3*cnt+1] = pos.Y();
        this->atomPos[3*cnt+2] = pos.Z();
    }

    // reset bbox
    this->bbox.Set( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    for( int cnt = 0; cnt < int( ligand->AtomCount()); ++cnt ) {
        this->bbox.GrowToPoint( 
            this->atomPos[3*cnt], 
            this->atomPos[3*cnt+1], 
            this->atomPos[3*cnt+2]);
    }

    this->currentSolution = solIdx;

    // successfully applied the solution
    return true;
}
