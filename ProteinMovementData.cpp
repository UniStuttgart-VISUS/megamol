/*
 * ProteinMovementData.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "ProteinMovementData.h"
#include "param/StringParam.h"
#include "vislib/MemmappedFile.h"
#include "vislib/Log.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/sysfunctions.h"
#include <string>
#include <iostream>

using namespace megamol;
using namespace megamol::core;


/*
 * protein::ProteinMovementData::ProteinMovementData
 */
protein::ProteinMovementData::ProteinMovementData ( void ) : Module (),
        protDataCalleeSlot ( "providedata", "Connects the protein rendering with protein data storage" ),
        mainFilename ( "mainFilename", "The path to the main protein data file to load." ),
        diffFilename ( "diffFilename", "The path to the difference protein data file to load." )
{
    CallProteinMovementDataDescription cpdd;
    this->protDataCalleeSlot.SetCallback ( cpdd.ClassName(), "GetData", &ProteinMovementData::ProtDataCallback );
    this->MakeSlotAvailable ( &this->protDataCalleeSlot );

    this->mainFilename.SetParameter ( new param::StringParam ( "" ) );
    this->MakeSlotAvailable ( &this->mainFilename );
    
    this->diffFilename.SetParameter ( new param::StringParam ( "" ) );
    this->MakeSlotAvailable ( &this->diffFilename );
    
    // secondary structure
    this->secondaryStructureComputed = false;
    this->stride = 0;
}


/*
 * protein::ProteinMovementData::~ProteinMovementData
 */
protein::ProteinMovementData::~ProteinMovementData ( void )
{
    this->Release();
}


/*
 * protein::ProteinMovementData::ProtDataCallback
 */
bool protein::ProteinMovementData::ProtDataCallback ( Call& call )
{
    unsigned int counter;

    protein::CallProteinMovementData *pdi = dynamic_cast<protein::CallProteinMovementData*> ( &call );

    if( this->mainFilename.IsDirty() || this->diffFilename.IsDirty() ) {
        // load the data.
        this->tryLoadFile();
        this->mainFilename.ResetDirty();
        this->diffFilename.ResetDirty();
    }

    if ( pdi ) {
        // set the bounding box
        pdi->SetBoundingBox ( this->minX, this->minY, this->minZ, this->maxX, this->maxY, this->maxZ );
        // set scaling
        pdi->SetScaling ( 1.0f / this->maxDimension );

        // set the amino acid name table
        pdi->SetAminoAcidNameTable ( ( unsigned int ) this->aminoAcidNames.Count(),
                this->aminoAcidNames.PeekElements() );
        // set the atom type table
        pdi->SetAtomTypeTable ( ( unsigned int ) this->atomTypes.Count(), this->atomTypes.PeekElements() );

        // set the number of protein atoms and the pointers for atom data and positions
        pdi->SetProteinAtomCount ( ( unsigned int ) this->protAtomData.Count() );
        pdi->SetProteinAtomDataPointer ( ( protein::CallProteinMovementData::AtomData* ) this->protAtomData.PeekElements() );
        pdi->SetProteinAtomPositionPointer ( ( float* ) this->protAtomPos.PeekElements() );
        // allocate the chains
        pdi->AllocateChains ( ( unsigned int ) this->aminoAcidChains.Count() );
        // set amino acids and secondary structure to the chains
        for ( counter = 0; counter < pdi->ProteinChainCount(); counter++ ) {
            pdi->AccessChain ( counter ).SetAminoAcid (
                ( unsigned int ) this->aminoAcidChains[counter].Count(),
                this->aminoAcidChains[counter].PeekElements() );
        }
        // set the atom movement positions
        pdi->SetProteinAtomMovementPositionPointer( ( float* ) this->protAtomMovedPos.PeekElements() );
        pdi->SetMaxMovementDistance( this->maxMovementDist);

        // try to compute secondary structure, if necessary
        /*
        if( !this->secondaryStructureComputed ) {
            double elapsedTime = 0.0;
            time_t t = clock();
            if( stride ) delete stride;
            stride = new Stride( pdi);
            this->secondaryStructureComputed = true;
            elapsedTime = ( double( clock() - t) / double( CLOCKS_PER_SEC) );
            vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO,
                    "%s: Computed secondary structure via Stride in %.4f seconds.\n",
                    this->ClassName(), elapsedTime );
        }
        stride->WriteToInterface( pdi );
        */
    }

    return true;
}

/*
 *protein::ProteinMovementData::create
 */
bool protein::ProteinMovementData::create ( void )
{
    this->tryLoadFile();
    this->mainFilename.ResetDirty();
    this->diffFilename.ResetDirty();
    return true;
}


/*
 *protein::ProteinMovementData::tryLoadFile
 */
bool protein::ProteinMovementData::tryLoadFile ( void )
{
    using vislib::sys::MemmappedFile;
    using vislib::sys::File;
    using vislib::sys::Log;

    // temporary variables
    unsigned int counterChainId, counterResSeq, counterAtom;
    vislib::StringA tmpStr;
    // return value: false means that anything went wrong
    bool retval = false;
    // is this the first ATOM-line?
    bool firstAtom = true;

    ///////////////////////////////////////////////////////////////////////////
    // Load first PDB-file containing the general protein informations
    ///////////////////////////////////////////////////////////////////////////
    MemmappedFile file;
    // get filename
    const vislib::TString& fn = this->mainFilename.Param<param::StringParam>()->Value();

    if ( fn.IsEmpty() ) {
        // no file to load
        return false;
    }

    if ( !file.Open ( fn, File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR,
                                   "%s: Unable to open file \"%s\"", this->ClassName(),
                                            vislib::StringA ( fn ).PeekBuffer() );
        return false;
    }

    vislib::StringA str ( fn );
    // check, if file extension is 'pdb'
    str.ToLowerCase();
    if ( !str.EndsWith ( ".pdb" ) )
        return false;
    str.Clear();

    // => check for PDB file indicator: the file must contain at least one 'ATOM' ?

    // clear all containers
    this->ClearData();
    // add all elements from the periodic table to atomTypes
    this->FillAtomTypesWithPeriodicTable();
    // add all amino acid names to aminoAcidNames
    this->FillAminoAcidNames();

    // while file pointer is not at 'end of file'
    while ( !file.IsEOF() ) {
        // read next line from file
        str = vislib::sys::ReadLineFromFileA ( file );
        // skip empty lines
        if ( str.Length() <= 0 ) continue;

        if ( str.StartsWithInsensitive ( "ATOM" ) ) {
            // check if the current atom is at an alternate location
            if( !str.Substring( 16, 1 ).Equals( " ", false) &&
                !str.Substring( 16, 1 ).Equals( "A", false) )
                // only add one occurence of the ATOM (either ' ' or 'A')
                continue;
            ////////////////////////////////////////////////////////////////////////////
            // Add all ATOM entries to the 'atomEntry'-vector for further processing. //
            // Write chainIDs and resSeq to maps to be able to recover amino acids.   //
            ////////////////////////////////////////////////////////////////////////////
            if ( firstAtom )
            {
                // start the first chain
                this->tmp_atomEntries.resize ( 1 );
                // start the fist amino acid
                this->tmp_atomEntries.back().resize ( 1 );
                // add the first atom to the amino acid
                this->tmp_atomEntries.back().back().push_back ( str );

                // add the chainId to the map
                this->tmp_chainIdMap[str.Substring ( 21, 1 ) [0]] = 0;
                this->tmp_resSeqMap.SetCount ( tmp_resSeqMap.Count() + 1 );
                // add the resSeq to the map
                this->tmp_resSeqMap.Last() [str.Substring ( 22, 4 ).PeekBuffer() ] = 0;
                // the first atom entry is read --> set firstAtom to 'false'
                firstAtom = false;
            }
            else
            {
                // if a new chain starts:
                if ( !this->tmp_atomEntries.back().back().back().Substring ( 21, 1 ).Equals ( str.Substring ( 21, 1 ) ) )
                {
                    // start a new chain
                    this->tmp_atomEntries.resize ( this->tmp_atomEntries.size() + 1 );
                    // start a new amino acid
                    this->tmp_atomEntries.back().resize ( 1 );
                    // add the ATOM string to the last amino acid
                    this->tmp_atomEntries.back().back().push_back ( str );

                    // add the chainId to the map
                    this->tmp_chainIdMap[str.Substring ( 21, 1 ) [0]] = ( unsigned int ) this->tmp_atomEntries.size()-1;
                    this->tmp_resSeqMap.SetCount ( this->tmp_resSeqMap.Count() + 1 );
                    // add the resSeq to the map
                    this->tmp_resSeqMap.Last() [str.Substring ( 22, 4 ).PeekBuffer() ] = 0;
                }
                // if a new amino acid starts:
                else if ( !tmp_atomEntries.back().back().back().Substring ( 22, 4 ).Equals ( str.Substring ( 22, 4 ) ) )
                {
                    // start a new amino acid
                    this->tmp_atomEntries.back().resize ( this->tmp_atomEntries.back().size() + 1 );
                    // add the ATOM string to the last amino acid
                    this->tmp_atomEntries.back().back().push_back ( str );

                    // add the resSeq to the map
                    this->tmp_resSeqMap.Last() [str.Substring ( 22, 4 ).PeekBuffer() ] = ( unsigned int ) this->tmp_atomEntries.back().size()-1;
                }
                // if no new chain and no new amino acid starts:
                else
                {
                    // add the ATOM string to the last amino acid
                    this->tmp_atomEntries.back().back().push_back ( str );
                }

            }
        }
        else if ( str.StartsWithInsensitive ( "END" ) )
        {
            retval = true;
            break;
        }
        else if ( str.StartsWithInsensitive ( "ENDMDL" ) )
        {
            retval = true;
            break;
        }
    }

    file.Close();
    Log::DefaultLog.WriteMsg ( Log::LEVEL_INFO,
                               "%s: File \"%s\" loaded successfully\n",
                               this->ClassName(), vislib::StringA ( fn ).PeekBuffer() );

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // loop over all atom entries in the 'tmp_atomEntries'-vector to get chains, amino acids and atoms //
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // create 'aminoAcidChains'-Array
    this->aminoAcidChains.SetCount ( this->tmp_atomEntries.size() );
    // loop over all chains
    for ( counterChainId = 0;
            counterChainId < this->tmp_atomEntries.size();
            counterChainId++ ) {
        // create amino acids array in current chain with the correct size
        this->aminoAcidChains[counterChainId].SetCount ( this->tmp_atomEntries[counterChainId].size() );
        // loop over all amino acids in the current chain
        for ( counterResSeq = 0;
                counterResSeq < this->tmp_atomEntries[counterChainId].size();
                counterResSeq++ ) {
            // set the position information of the current amino acid
            this->aminoAcidChains[counterChainId][counterResSeq].SetPosition ( this->tmp_currentAtomIdx,
                    ( unsigned int ) this->tmp_atomEntries[counterChainId][counterResSeq].size() );
            // set the amino acid name for the current amino acid
            this->aminoAcidChains[counterChainId][counterResSeq].SetNameIndex ( this->GetAminoAcidNameIdx ( this->tmp_atomEntries[counterChainId][counterResSeq][0].Substring ( 17, 3 ).PeekBuffer() ) );
            // loop oder all atoms of the current amino acid
            for ( counterAtom = 0;
                    counterAtom < this->tmp_atomEntries[counterChainId][counterResSeq].size();
                    counterAtom++ ) {
                // parse string to add current atom to 'protAtomData' and the position to 'protAtomPos'
                this->ParsePDBAtom ( this->tmp_atomEntries[counterChainId][counterResSeq][counterAtom] );

                tmpStr = this->tmp_atomEntries[counterChainId][counterResSeq][counterAtom].Substring ( 12, 4 );
                tmpStr.TrimSpaces();

                // if the current atom is one of the backbone atoms --> set index to current amino acid
                if ( tmpStr.Equals ( "CA" ) )
                    this->aminoAcidChains[counterChainId][counterResSeq].SetCAlphaIndex ( counterAtom );
                else if ( tmpStr.Equals ( "C" ) )
                    this->aminoAcidChains[counterChainId][counterResSeq].SetCCarbIndex ( counterAtom );
                else if ( tmpStr.Equals ( "N" ) )
                    this->aminoAcidChains[counterChainId][counterResSeq].SetNIndex ( counterAtom );
                else if ( tmpStr.Equals ( "O" ) )
                    this->aminoAcidChains[counterChainId][counterResSeq].SetOIndex ( counterAtom );

                // increase current atom index
                this->tmp_currentAtomIdx++;
            }
            // all atoms in the current amino acid are handled --> make connections now
            this->MakeConnections ( counterChainId, counterResSeq );
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Load PDB-file containing movements
    ///////////////////////////////////////////////////////////////////////////
    MemmappedFile movementFile;
    // get filename
    const vislib::TString& fn2 = this->diffFilename.Param<param::StringParam>()->Value();

    if ( fn2.IsEmpty() ) {
        // no file to load
        return false;
    }

    if ( !movementFile.Open ( fn2, File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR,
                                   "%s: Unable to open file \"%s\"", this->ClassName(),
                                            vislib::StringA ( fn2 ).PeekBuffer() );
        return false;
    }

    str = fn2;
    // check, if file extension is 'pdb'
    str.ToLowerCase();
    if ( !str.EndsWith ( ".pdb" ) )
        return false;
    str.Clear();

    // assert capacity for moved positions
    this->protAtomMovedPos.AssertCapacity( this->protAtomPos.Count());
    float dist;
    unsigned int idx;
    // while file pointer is not at 'end of file'
    while ( !movementFile.IsEOF() ) {
        // read next line from file
        str = vislib::sys::ReadLineFromFileA( movementFile );
        // skip empty lines
        if ( str.Length() <= 0 ) continue;

        if ( str.StartsWithInsensitive ( "ATOM" ) ) {
            // check if the current atom is at an alternate location
            if( !str.Substring( 16, 1 ).Equals( " ", false) &&
                !str.Substring( 16, 1 ).Equals( "A", false) )
                // only add one occurence of the ATOM (either ' ' or 'A')
                continue;

            // get and store the position of the moved atom
            tmpStr = str.Substring ( 30, 8 );
            tmpStr.TrimSpaces();
            this->protAtomMovedPos.Add ( ( float ) atof ( tmpStr.PeekBuffer() ) );
            tmpStr = str.Substring ( 38, 8 );
            tmpStr.TrimSpaces();
            this->protAtomMovedPos.Add ( ( float ) atof ( tmpStr.PeekBuffer() ) );
            tmpStr = str.Substring ( 46, 8 );
            tmpStr.TrimSpaces();
            this->protAtomMovedPos.Add ( ( float ) atof ( tmpStr.PeekBuffer() ) );

            // check movement distance
            idx = (unsigned int)this->protAtomMovedPos.Count()-3;
            dist = sqrt( pow( this->protAtomPos[idx+0] - this->protAtomMovedPos[idx+0], 2.0f) +
                pow( this->protAtomPos[idx+1] - this->protAtomMovedPos[idx+1], 2.0f) +
                pow( this->protAtomPos[idx+2] - this->protAtomMovedPos[idx+2], 2.0f));
            if( dist > this->maxMovementDist )
                this->maxMovementDist = dist;
        }
        else if ( str.StartsWithInsensitive ( "END" ) )
        {
            retval = true;
            break;
        }
        else if ( str.StartsWithInsensitive ( "ENDMDL" ) )
        {
            retval = true;
            break;
        }
    }

    movementFile.Close();
    Log::DefaultLog.WriteMsg ( Log::LEVEL_INFO,
                               "%s: File \"%s\" loaded successfully\n",
                               this->ClassName(), vislib::StringA ( fn2 ).PeekBuffer() );

    // check if the number of atom positions are the same
    if( this->protAtomPos.Count() != this->protAtomMovedPos.Count() ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Atom counts do not comply! %i <> %i\n",
            this->ClassName(), this->protAtomPos.Count(), this->protAtomMovedPos.Count() );
        return false;
    }

    // compute the bounding box for the stored atom positions
    this->ComputeBoundingBox();

    // delete temporary variables
    for ( counterChainId = 0; counterChainId < this->tmp_atomEntries.size(); counterChainId++ ) {
        for ( counterResSeq = 0; counterResSeq < this->tmp_atomEntries[counterChainId].size(); counterResSeq++ ) {
            this->tmp_atomEntries[counterChainId][counterResSeq].clear();
        }
        this->tmp_atomEntries[counterChainId].clear();
    }
    this->tmp_atomEntries.clear();

    this->tmp_chainIdMap.clear();

    for ( counterChainId = 0; counterChainId < this->tmp_resSeqMap.Count(); counterChainId++ ) {
        this->tmp_resSeqMap[counterChainId].clear();
    }
    this->tmp_resSeqMap.Clear();

    this->tmp_atomicNumbers.clear();
    this->tmp_aminoAcidNameIdx.clear();

    // return 'true' if everything could be loaded
    return retval;
}


/*
 *protein::ProteinMovementData::ClearData
 */
void protein::ProteinMovementData::ClearData()
{
    unsigned int i, j;

    // clear temporary varibles
    this->tmp_currentAtomIdx = 0;
    for ( i = 0; i < this->tmp_atomEntries.size(); i++ )
    {
        for ( j = 0; j < this->tmp_atomEntries.at ( i ).size(); j++ )
        {
            this->tmp_atomEntries.at ( i ).at ( j ).clear();
        }
        this->tmp_atomEntries.at ( i ).clear();
    }
    this->tmp_atomEntries.clear();
    this->tmp_chainIdMap.clear();
    for ( i = 0; i < this->tmp_resSeqMap.Count(); i++ )
    {
        this->tmp_resSeqMap[i].clear();
    }
    this->tmp_resSeqMap.Clear();

    // clear data variables for DataInterface
    this->aminoAcidNames.Clear();
    this->tmp_aminoAcidNameIdx.clear();
    this->atomTypes.Clear();
    this->tmp_atomicNumbers.clear();
    for ( i = 0; i < this->aminoAcidChains.Count(); i++ )
    {
        this->aminoAcidChains[i].Clear();
    }
    this->aminoAcidChains.Clear();
    for ( i = 0; i < secStruct.Count(); i++ )
    {
        this->secStruct[i].Clear();
    }
    this->secStruct.Clear();
    this->protAtomData.Clear();
    this->protAtomPos.Clear();
    this->protAtomMovedPos.Clear();

    this->maxMovementDist = 0.0f;

    // reset bounding box
    this->minX = 0.0f;
    this->minY = 0.0f;
    this->minZ = 0.0f;
    this->maxX = 1.0f;
    this->maxY = 1.0f;
    this->maxZ = 1.0f;
    this->maxDimension = 1.0f;
}


/**********************************************************************
 * 'protein'-functions                                                *
 **********************************************************************/

/*
 *protein::ProteinMovementData::ParsePDBAtom
 */
bool protein::ProteinMovementData::ParsePDBAtom ( const vislib::StringA &line )
{
    // if input string does not start with 'ATOM' --> return false
    if ( !line.StartsWithInsensitive ( "ATOM" ) )
        return false;

    vislib::StringA str, elemName;
    unsigned int cnt;
    // temporary atom variables
    unsigned int atomTypeIdx = 0;
    float charge = 0.0f;
    float occupancy = 0.0f;
    float tempFactor = 0.0f;

    // get and store the position of the atom
    str = line.Substring ( 30, 8 );
    str.TrimSpaces();
    this->protAtomPos.Add ( ( float ) atof ( str.PeekBuffer() ) );
    str = line.Substring ( 38, 8 );
    str.TrimSpaces();
    this->protAtomPos.Add ( ( float ) atof ( str.PeekBuffer() ) );
    str = line.Substring ( 46, 8 );
    str.TrimSpaces();
    this->protAtomPos.Add ( ( float ) atof ( str.PeekBuffer() ) );

    // get the name (atom type) of the current ATOM entry
    str = line.Substring ( 12, 4 );
    str.TrimSpaces();
    if ( !str.IsEmpty() )
    {
        // search atom type entry in atom type table
        for( cnt = 0; cnt < this->atomTypes.Count(); ++cnt )
        {
            if( this->atomTypes[cnt].Name() == str )
            {
                atomTypeIdx = cnt;
            }
        }
    }
    else
    {
        atomTypeIdx = 0;
    }
    // add new atom type if necessary (i.e. if atom type was not found in table)
    if( atomTypeIdx == 0 )
    {
        unsigned int tmpTypeTableId = 0;
        // try to get element from ATOM entry (and look up periodic number)
        elemName = line.Substring ( 76, 2 );
        elemName.TrimSpaces();
        if ( !elemName.IsEmpty() )
        {
            tmpTypeTableId = this->GetAtomicNumber( elemName.PeekBuffer());
        }
        else
        {
            // try to get element from atom name
            tmpTypeTableId = this->GetAtomicNumber( str.Substring( 0, 1).PeekBuffer());
        }
        // add new atom type to atom type table
        this->atomTypes.Add( CallProteinMovementData::AtomType( str,
            this->tmp_atomTypes[tmpTypeTableId].Radius(),
            this->tmp_atomTypes[tmpTypeTableId].Colour()[0],
            this->tmp_atomTypes[tmpTypeTableId].Colour()[1],
            this->tmp_atomTypes[tmpTypeTableId].Colour()[2]) );
        // set atom type index to the index of the newly added atom type
        atomTypeIdx = this->atomTypes.Count() - 1;
    }

    // add atom data to protein atom data array
    this->protAtomData.Add ( protein::CallProteinMovementData::AtomData ( atomTypeIdx, charge, tempFactor, occupancy ) );

    return true;
}


/*
 *protein::ProteinMovementData::FillAminoAcidNames
 */
void protein::ProteinMovementData::FillAminoAcidNames()
{
    // clear atomTypes-Array if necessary
    if ( aminoAcidNames.Count() > 0 )
        aminoAcidNames.Clear();
    // clear map for atomic numbers
    tmp_aminoAcidNameIdx.clear();
    
    this->aminoAcidNames.Add ( "unknown" );
    this->aminoAcidNames.Add ( "ALA" );
    this->tmp_aminoAcidNameIdx["ALA"] = 1;
    this->aminoAcidNames.Add ( "ARG" );
    this->tmp_aminoAcidNameIdx["ARG"] = 2;
    this->aminoAcidNames.Add ( "ASN" );
    this->tmp_aminoAcidNameIdx["ASN"] = 3;
    this->aminoAcidNames.Add ( "ASP" );
    this->tmp_aminoAcidNameIdx["ASP"] = 4;
    this->aminoAcidNames.Add ( "CYS" );
    this->tmp_aminoAcidNameIdx["CYS"] = 5;
    this->aminoAcidNames.Add ( "GLN" );
    this->tmp_aminoAcidNameIdx["GLN"] = 6;
    this->aminoAcidNames.Add ( "GLU" );
    this->tmp_aminoAcidNameIdx["GLU"] = 7;
    this->aminoAcidNames.Add ( "GLY" );
    this->tmp_aminoAcidNameIdx["GLY"] = 8;
    this->aminoAcidNames.Add ( "HIS" );
    this->tmp_aminoAcidNameIdx["HIS"] = 9;
    this->aminoAcidNames.Add ( "ILE" );
    this->tmp_aminoAcidNameIdx["ILE"] = 10;
    this->aminoAcidNames.Add ( "LEU" );
    this->tmp_aminoAcidNameIdx["LEU"] = 11;
    this->aminoAcidNames.Add ( "LYS" );
    this->tmp_aminoAcidNameIdx["LYS"] = 12;
    this->aminoAcidNames.Add ( "MET" );
    this->tmp_aminoAcidNameIdx["MET"] = 13;
    this->aminoAcidNames.Add ( "PHE" );
    this->tmp_aminoAcidNameIdx["PHE"] = 14;
    this->aminoAcidNames.Add ( "PRO" );
    this->tmp_aminoAcidNameIdx["PRO"] = 15;
    this->aminoAcidNames.Add ( "SER" );
    this->tmp_aminoAcidNameIdx["SER"] = 16;
    this->aminoAcidNames.Add ( "THR" );
    this->tmp_aminoAcidNameIdx["THR"] = 17;
    this->aminoAcidNames.Add ( "TRP" );
    this->tmp_aminoAcidNameIdx["TRP"] = 18;
    this->aminoAcidNames.Add ( "TYR" );
    this->tmp_aminoAcidNameIdx["TYR"] = 19;
    this->aminoAcidNames.Add ( "VAL" );
    this->tmp_aminoAcidNameIdx["VAL"] = 20;
}


/*
 *protein::ProteinMovementData::FillAtomTypesWithPeriodicTable
 */
void protein::ProteinMovementData::FillAtomTypesWithPeriodicTable()
{
    // clear atomTypes-Array if necessary
    if ( atomTypes.Count() > 0 )
        atomTypes.Clear();
    // clear map for atomic numbers
    tmp_atomicNumbers.clear();

    // write 'unknown' to first atom type entry
    this->atomTypes.AssertCapacity( 150);
    this->atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "unknown", 1.0, 127, 127, 127 ) );
    
    // write all elements from the periodic table to tmp_atomTypes-Array
    // with their van-der-Waals-radius and natural color (white if no color or colorless)
    this->tmp_atomTypes.Clear();
    this->tmp_atomTypes.AssertCapacity( 120);
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "unknown", 1.0, 127, 127, 127 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Hydrogen", 1.2f, 240, 240, 240 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Helium", 1.4f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Lithium", 1.82f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Beryllium", 2.0f, 85, 85, 85 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Boron", 2.0f, 0, 0, 0 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Carbon", 1.7f, 0, 240, 0 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Nitrogen", 1.55f, 0, 0, 240 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Oxygen", 1.52f, 240, 0, 0 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Fluorine", 1.47f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Neon", 1.54f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Sodium", 2.27f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Magnesium", 1.73f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Aluminium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Silicon", 2.1f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Phosphorus", 1.8f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Sulphur", 1.8f, 255, 255, 0 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Chlorine", 1.75f, 255, 255, 0 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Argon", 1.88f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Potassium", 2.75f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Calcium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Scandium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Titanium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Vanadium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Chromium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Manganese", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Iron", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Cobalt", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Nickel", 1.63f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Copper", 1.4f, 255, 132, 0 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Zinc", 1.39f, 85, 85, 85 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Gallium", 1.87f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Germanium", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Arsenic", 1.85f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Selenium", 1.9f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Bromine", 1.85f, 255, 0, 0 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Krypton", 2.02f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Rubidium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Strontium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Yttrium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Zirconium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Niobium", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Molybdenum", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Technetium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Ruthenium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Rhodium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Palladium", 1.63f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Silver", 1.72f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Cadmium", 1.58f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Indium", 1.93f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Tin", 2.17f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Antimony", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Tellurium", 2.06f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Iodine", 1.98f, 85, 85, 85 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Xenon", 2.16f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Caesium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Barium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Lanthanum", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Cerium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Praseodymium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Neodymium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Promethium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Samarium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Europium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Gadolinium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Terbium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Dysprosium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Holmium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Erbium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Thulium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Ytterbium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Lutetium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Hafnium", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Tantalum", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Tungsten", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Rhenium", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Osmium", 2.0f, 85, 85, 85 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Iridium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Platinum", 1.72f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Gold", 1.66f, 238, 201, 0 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Mercury", 1.55f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Thallium", 1.96f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Lead", 2.02f, 85, 85, 85 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Bismuth", 2.0f, 170, 170, 170 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Polonium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Astatine", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Radon", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Francium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Radium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Actinium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Thorium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Protactinium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Uranium", 1.86f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Neptunium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Plutonium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Americium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Curium", 2.0f, 200, 200, 200 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Berkelium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Californium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Einsteinium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Fermium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Mendelevium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Nobelium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Lawrencium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Rutherfordium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Dubnium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Seaborgium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Bohrium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Hassium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Meitnerium", 2.0f, 255, 255, 255 ) );
    this->tmp_atomTypes.Add ( protein::CallProteinMovementData::AtomType ( "Darmstadtium", 2.0f, 255, 255, 255 ) );

    // create map that points to the element/atomic number when given the element's symbol
    this->tmp_atomicNumbers["H"] = 1;
    this->tmp_atomicNumbers["HE"] = 2;
    this->tmp_atomicNumbers["LI"] = 3;
    this->tmp_atomicNumbers["BE"] = 4;
    this->tmp_atomicNumbers["B"] = 5;
    this->tmp_atomicNumbers["C"] = 6;
    this->tmp_atomicNumbers["N"] = 7;
    this->tmp_atomicNumbers["O"] = 8;
    this->tmp_atomicNumbers["F"] = 9;
    this->tmp_atomicNumbers["NE"] = 10;
    this->tmp_atomicNumbers["NA"] = 11;
    this->tmp_atomicNumbers["MG"] = 12;
    this->tmp_atomicNumbers["AL"] = 13;
    this->tmp_atomicNumbers["SI"] = 14;
    this->tmp_atomicNumbers["P"] = 15;
    this->tmp_atomicNumbers["S"] = 16;
    this->tmp_atomicNumbers["CL"] = 17;
    this->tmp_atomicNumbers["AR"] = 18;
    this->tmp_atomicNumbers["K"] = 19;
    this->tmp_atomicNumbers["CA"] = 20;
    this->tmp_atomicNumbers["SC"] = 21;
    this->tmp_atomicNumbers["TI"] = 22;
    this->tmp_atomicNumbers["V"] = 23;
    this->tmp_atomicNumbers["CR"] = 24;
    this->tmp_atomicNumbers["MN"] = 25;
    this->tmp_atomicNumbers["FE"] = 26;
    this->tmp_atomicNumbers["CO"] = 27;
    this->tmp_atomicNumbers["NI"] = 28;
    this->tmp_atomicNumbers["CU"] = 29;
    this->tmp_atomicNumbers["ZN"] = 30;
    this->tmp_atomicNumbers["GA"] = 31;
    this->tmp_atomicNumbers["GE"] = 32;
    this->tmp_atomicNumbers["AS"] = 33;
    this->tmp_atomicNumbers["SE"] = 34;
    this->tmp_atomicNumbers["BR"] = 35;
    this->tmp_atomicNumbers["KR"] = 36;
    this->tmp_atomicNumbers["RB"] = 37;
    this->tmp_atomicNumbers["SR"] = 38;
    this->tmp_atomicNumbers["Y"] = 39;
    this->tmp_atomicNumbers["ZR"] = 40;
    this->tmp_atomicNumbers["NB"] = 41;
    this->tmp_atomicNumbers["MO"] = 42;
    this->tmp_atomicNumbers["TC"] = 43;
    this->tmp_atomicNumbers["RU"] = 44;
    this->tmp_atomicNumbers["RH"] = 45;
    this->tmp_atomicNumbers["PD"] = 46;
    this->tmp_atomicNumbers["AG"] = 47;
    this->tmp_atomicNumbers["CD"] = 48;
    this->tmp_atomicNumbers["IN"] = 49;
    this->tmp_atomicNumbers["SN"] = 50;
    this->tmp_atomicNumbers["SB"] = 51;
    this->tmp_atomicNumbers["TE"] = 52;
    this->tmp_atomicNumbers["I"] = 53;
    this->tmp_atomicNumbers["XE"] = 54;
    this->tmp_atomicNumbers["CS"] = 55;
    this->tmp_atomicNumbers["BA"] = 56;
    this->tmp_atomicNumbers["LA"] = 57;
    this->tmp_atomicNumbers["CE"] = 58;
    this->tmp_atomicNumbers["PR"] = 59;
    this->tmp_atomicNumbers["ND"] = 60;
    this->tmp_atomicNumbers["PM"] = 61;
    this->tmp_atomicNumbers["SM"] = 62;
    this->tmp_atomicNumbers["EU"] = 63;
    this->tmp_atomicNumbers["GD"] = 64;
    this->tmp_atomicNumbers["TB"] = 65;
    this->tmp_atomicNumbers["DY"] = 66;
    this->tmp_atomicNumbers["HO"] = 67;
    this->tmp_atomicNumbers["ER"] = 68;
    this->tmp_atomicNumbers["TM"] = 69;
    this->tmp_atomicNumbers["YB"] = 70;
    this->tmp_atomicNumbers["LU"] = 71;
    this->tmp_atomicNumbers["HF"] = 72;
    this->tmp_atomicNumbers["TA"] = 73;
    this->tmp_atomicNumbers["W"] = 74;
    this->tmp_atomicNumbers["RE"] = 75;
    this->tmp_atomicNumbers["OS"] = 76;
    this->tmp_atomicNumbers["IR"] = 77;
    this->tmp_atomicNumbers["PT"] = 78;
    this->tmp_atomicNumbers["AU"] = 79;
    this->tmp_atomicNumbers["HG"] = 80;
    this->tmp_atomicNumbers["TL"] = 81;
    this->tmp_atomicNumbers["PB"] = 82;
    this->tmp_atomicNumbers["BI"] = 83;
    this->tmp_atomicNumbers["PO"] = 84;
    this->tmp_atomicNumbers["AT"] = 85;
    this->tmp_atomicNumbers["RN"] = 86;
    this->tmp_atomicNumbers["FF"] = 87;
    this->tmp_atomicNumbers["RA"] = 88;
    this->tmp_atomicNumbers["AC"] = 89;
    this->tmp_atomicNumbers["TH"] = 90;
    this->tmp_atomicNumbers["PA"] = 91;
    this->tmp_atomicNumbers["U"] = 92;
    this->tmp_atomicNumbers["NP"] = 93;
    this->tmp_atomicNumbers["PU"] = 94;
    this->tmp_atomicNumbers["AM"] = 95;
    this->tmp_atomicNumbers["CM"] = 96;
    this->tmp_atomicNumbers["BK"] = 97;
    this->tmp_atomicNumbers["CF"] = 98;
    this->tmp_atomicNumbers["ES"] = 99;
    this->tmp_atomicNumbers["FM"] = 100;
    this->tmp_atomicNumbers["MD"] = 101;
    this->tmp_atomicNumbers["NO"] = 102;
    this->tmp_atomicNumbers["LR"] = 103;
    this->tmp_atomicNumbers["LW"] = 103;
    this->tmp_atomicNumbers["RF"] = 104;
    this->tmp_atomicNumbers["DB"] = 105;
    this->tmp_atomicNumbers["SG"] = 106;
    this->tmp_atomicNumbers["BH"] = 107;
    this->tmp_atomicNumbers["HS"] = 108;
    this->tmp_atomicNumbers["MT"] = 109;
    this->tmp_atomicNumbers["DS"] = 110;
}


/*
 *protein::ProteinMovementData::GetAtomicNumber
 */
unsigned int protein::ProteinMovementData::GetAtomicNumber ( const char* symbol ) const
{
    // check, if the symbol is in the tmp_atomicNumbers-map
    if ( this->tmp_atomicNumbers.find ( symbol ) != this->tmp_atomicNumbers.end() )
    {
        return this->tmp_atomicNumbers.find ( symbol )->second;
    }
    else
    {
        // return zero, if 'symbol' was not found
        return 0;
    }
}


/*
 *protein::ProteinMovementData::GetAminoAcidNameIdx
 */
unsigned int protein::ProteinMovementData::GetAminoAcidNameIdx ( const char* name ) const
{
    // check, if the name is in the tmp_aminoAcidNameIdx-map
    if ( this->tmp_aminoAcidNameIdx.find ( name ) != this->tmp_aminoAcidNameIdx.end() )
    {
        return this->tmp_aminoAcidNameIdx.find ( name )->second;
    }
    else
    {
        // return zero, if 'name' was not found
        return 0;
    }
}


/*
 *protein::ProteinMovementData::MakeConnections
 */
bool protein::ProteinMovementData::MakeConnections ( unsigned int chainIdIdx, unsigned int resSeqIdx )
{
    std::map<std::string, unsigned int> atomNamesMap;
    unsigned int counter;
    vislib::StringA name;

    unsigned int cnt1, cnt2, idx1, idx2;
    //vislib::math::Vector<float, 3> v1, v2;
    // loop over all atoms in this amino acid and fill the map with the names of the atoms
    for ( cnt1 = 0; cnt1 < this->tmp_atomEntries[chainIdIdx][resSeqIdx].size() - 1; ++cnt1 ){
        for ( cnt2 = cnt1 + 1; cnt2 < this->tmp_atomEntries[chainIdIdx][resSeqIdx].size(); ++cnt2 ){
            idx1 = aminoAcidChains[chainIdIdx][resSeqIdx].FirstAtomIndex() + cnt1;
            idx2 = aminoAcidChains[chainIdIdx][resSeqIdx].FirstAtomIndex() + cnt2;
            vislib::math::Vector<float, 3> v1( &this->protAtomPos[idx1 * 3]);
            vislib::math::Vector<float, 3> v2( &this->protAtomPos[idx2 * 3]);
            if( ( v1 - v2).Length() <
                0.6f * ( this->atomTypes[this->protAtomData[idx1].TypeIndex()].Radius() +
                this->atomTypes[this->protAtomData[idx2].TypeIndex()].Radius() ) ) {
                aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                    protein::CallProteinMovementData::IndexPair ( cnt1, cnt2 ) );
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    return true;
    ///////////////////////////////////////////////////////////////////////////

    // loop over all atoms in this amino acid and fill the map with the names of the atoms
    for ( counter = 0; counter < this->tmp_atomEntries[chainIdIdx][resSeqIdx].size(); counter++ )
    {
        name = tmp_atomEntries[chainIdIdx][resSeqIdx][counter].Substring ( 12, 4 );
        name.TrimSpaces();
        atomNamesMap[name.PeekBuffer() ] = counter;
    }

    //////////////////////////////////////
    // try to make backbone connections //
    //////////////////////////////////////

    // check for C-alpha atom
    if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() )
    {
        // try to make C-alpha -- C connection
        if ( atomNamesMap.find ( "C" ) != atomNamesMap.end() )
        {
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["C"] ) );
        }
        // try to make C-alpha -- N connection
        if ( atomNamesMap.find ( "N" ) != atomNamesMap.end() )
        {
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["N"] ) );
        }
    }
    // check for C atom
    if ( atomNamesMap.find ( "C" ) != atomNamesMap.end() )
    {
        // try to make C -- O connection
        if ( atomNamesMap.find ( "O" ) != atomNamesMap.end() )
        {
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["C"], atomNamesMap["O"] ) );
        }
        // try to make C -- OXT connection (only available for the last amino acid in the chain)
        if ( atomNamesMap.find ( "OXT" ) != atomNamesMap.end() )
        {
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["C"], atomNamesMap["OXT"] ) );
        }
        // try to make C -- O' connection (only available for the last amino acid in the chain)
        if ( atomNamesMap.find ( "O'" ) != atomNamesMap.end() )
        {
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["C"], atomNamesMap["O'"] ) );
        }
        // try to make C -- O'' connection (only available for the last amino acid in the chain)
        if ( atomNamesMap.find ( "O''" ) != atomNamesMap.end() )
        {
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["C"], atomNamesMap["O''"] ) );
        }
    }

    ////////////////////////////////////////
    // try to make side chain connections //
    ////////////////////////////////////////

    // ALA
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ALA"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
    }

    // ARG
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ARG"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD"] ) );
        if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "NE" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["NE"] ) );
        if ( atomNamesMap.find ( "NE" ) != atomNamesMap.end() && atomNamesMap.find ( "CZ" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NE"], atomNamesMap["CZ"] ) );
        if ( atomNamesMap.find ( "CZ" ) != atomNamesMap.end() && atomNamesMap.find ( "NH1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CZ"], atomNamesMap["NH1"] ) );
        if ( atomNamesMap.find ( "CZ" ) != atomNamesMap.end() && atomNamesMap.find ( "NH2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CZ"], atomNamesMap["NH2"] ) );
    }

    // ASN
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ASN"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "OD1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["OD1"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "ND2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["ND2"] ) );
    }

    // ASP
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ASP"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "OD1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["OD1"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "OD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["OD2"] ) );
    }

    // CYS
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["CYS"] ||
         this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["CYX"] ||
         this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["CYM"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "SG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["SG"] ) );
    }

    // GLU
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["GLU"] ||
         this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["GLH"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD"] ) );
        if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "OE1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["OE1"] ) );
        if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "OE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["OE2"] ) );
    }

    // GLN
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["GLN"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD"] ) );
        if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "OE1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["OE1"] ) );
        if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "NE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["NE2"] ) );
    }

    // GLY --> has no side chain, consists only of backbone

    // HIS
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["HIS"] ||
         this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["HID"] ||
         this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["HIE"] ||
         this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["HIP"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "ND1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["ND1"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD2"] ) );
        if ( atomNamesMap.find ( "ND1" ) != atomNamesMap.end() && atomNamesMap.find ( "CE1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["ND1"], atomNamesMap["CE1"] ) );
        if ( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() && atomNamesMap.find ( "NE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["NE2"] ) );
        if ( atomNamesMap.find ( "CE1" ) != atomNamesMap.end() && atomNamesMap.find ( "NE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE1"], atomNamesMap["NE2"] ) );
    }

    // ILE
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ILE"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG1"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG2"] ) );
        if ( atomNamesMap.find ( "CG1" ) != atomNamesMap.end() && atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG1"], atomNamesMap["CD1"] ) );
    }

    // LEU
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["LEU"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD1"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD2"] ) );
    }

    // LYS
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["LYS"] ||
         this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["LYN"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD"] ) );
        if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "CE" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["CE"] ) );
        if ( atomNamesMap.find ( "CE" ) != atomNamesMap.end() && atomNamesMap.find ( "NZ" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE"], atomNamesMap["NZ"] ) );
    }

    // MET
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["MET"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "SD" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["SD"] ) );
        if ( atomNamesMap.find ( "SD" ) != atomNamesMap.end() && atomNamesMap.find ( "CE" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["SD"], atomNamesMap["CE"] ) );
    }

    // PHE
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["PHE"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD1"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD2"] ) );
        if ( atomNamesMap.find ( "CD1" ) != atomNamesMap.end() && atomNamesMap.find ( "CE1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["CE1"] ) );
        if ( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() && atomNamesMap.find ( "CE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["CE2"] ) );
        if ( atomNamesMap.find ( "CE1" ) != atomNamesMap.end() && atomNamesMap.find ( "CZ" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE1"], atomNamesMap["CZ"] ) );
        if ( atomNamesMap.find ( "CE2" ) != atomNamesMap.end() && atomNamesMap.find ( "CZ" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE2"], atomNamesMap["CZ"] ) );
    }

    // PRO
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["PRO"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD"] ) );
        if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "N" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["N"] ) );
    }

    // SER
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["SER"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "OG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["OG"] ) );
    }

    // THR
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["THR"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "OG1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["OG1"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG2"] ) );
    }

    // TRP
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["TRP"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD1"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD2"] ) );
        if ( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() && atomNamesMap.find ( "CE3" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["CE3"] ) );
        if ( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() && atomNamesMap.find ( "CE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["CE2"] ) );
        if ( atomNamesMap.find ( "CD1" ) != atomNamesMap.end() && atomNamesMap.find ( "NE1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["NE1"] ) );
        if ( atomNamesMap.find ( "NE1" ) != atomNamesMap.end() && atomNamesMap.find ( "CE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NE1"], atomNamesMap["CE2"] ) );
        if ( atomNamesMap.find ( "CE3" ) != atomNamesMap.end() && atomNamesMap.find ( "CZ3" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE3"], atomNamesMap["CZ3"] ) );
        if ( atomNamesMap.find ( "CE2" ) != atomNamesMap.end() && atomNamesMap.find ( "CZ2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE2"], atomNamesMap["CZ2"] ) );
        if ( atomNamesMap.find ( "CZ2" ) != atomNamesMap.end() && atomNamesMap.find ( "CH2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CZ2"], atomNamesMap["CH2"] ) );
        if ( atomNamesMap.find ( "CZ3" ) != atomNamesMap.end() && atomNamesMap.find ( "CH2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CZ3"], atomNamesMap["CH2"] ) );
    }

    // TYR
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["TYR"] ||
         this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["TYM"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD1"] ) );
        if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["CD2"] ) );
        if ( atomNamesMap.find ( "CD1" ) != atomNamesMap.end() && atomNamesMap.find ( "CE1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["CE1"] ) );
        if ( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() && atomNamesMap.find ( "CE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["CE2"] ) );
        if ( atomNamesMap.find ( "CE1" ) != atomNamesMap.end() && atomNamesMap.find ( "CZ" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE1"], atomNamesMap["CZ"] ) );
        if ( atomNamesMap.find ( "CE2" ) != atomNamesMap.end() && atomNamesMap.find ( "CZ" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE2"], atomNamesMap["CZ"] ) );
        if ( atomNamesMap.find ( "CZ" ) != atomNamesMap.end() && atomNamesMap.find ( "OH" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CZ"], atomNamesMap["OH"] ) );
    }

    // VAL
    if ( this->aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["VAL"] )
    {
        if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() && atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["CB"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG1"] ) );
        if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() && atomNamesMap.find ( "CG2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["CG2"] ) );
    }

    //////////////////////////////////////////////////////////////////////
    // Hydrogen atoms
    //////////////////////////////////////////////////////////////////////
    
    // amino group
    if ( atomNamesMap.find ( "N" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "H" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["N"], atomNamesMap["H"] ) );
        if( atomNamesMap.find ( "H1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["N"], atomNamesMap["H1"] ) );
        if( atomNamesMap.find ( "H2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["N"], atomNamesMap["H2"] ) );
        if( atomNamesMap.find ( "H3" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["N"], atomNamesMap["H3"] ) );
    }
    
    // amino group
    if ( atomNamesMap.find ( "N" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "H" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["N"], atomNamesMap["H"] ) );
        if( atomNamesMap.find ( "H1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["N"], atomNamesMap["H1"] ) );
        if( atomNamesMap.find ( "H2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["N"], atomNamesMap["H2"] ) );
        if( atomNamesMap.find ( "H3" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["N"], atomNamesMap["H3"] ) );
    }
    
    // A alpha
    if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "HA" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["HA"] ) );
        if( atomNamesMap.find ( "HA2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["HA2"] ) );
        if( atomNamesMap.find ( "HA3" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CA"], atomNamesMap["HA3"] ) );
    }
    
    // B beta
    if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "HB" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["HB"] ) );
        if( atomNamesMap.find ( "HB1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["HB1"] ) );
        if( atomNamesMap.find ( "HB2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["HB2"] ) );
        if( atomNamesMap.find ( "HB3" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CB"], atomNamesMap["HB3"] ) );
    }

    // G gamma
    if ( atomNamesMap.find ( "HG" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["HG"] ) );
        if( atomNamesMap.find ( "OG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["OG"], atomNamesMap["HG"] ) );
        if( atomNamesMap.find ( "SG" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["SG"], atomNamesMap["HG"] ) );
    }
    
    if ( atomNamesMap.find ( "OG1" ) != atomNamesMap.end() && atomNamesMap.find ( "HG1" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["OG1"], atomNamesMap["HG1"] ) );
    
    if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "HG2" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["HG2"] ) );
    
    if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "HG3" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG"], atomNamesMap["HG3"] ) );
    
    if ( atomNamesMap.find ( "CG1" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "HG11" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG1"], atomNamesMap["HG11"] ) );
        if( atomNamesMap.find ( "HG12" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG1"], atomNamesMap["HG12"] ) );
        if( atomNamesMap.find ( "HG13" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG1"], atomNamesMap["HG13"] ) );
    }
    
    if ( atomNamesMap.find ( "CG2" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "HG21" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG2"], atomNamesMap["HG21"] ) );
        if( atomNamesMap.find ( "HG22" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG2"], atomNamesMap["HG22"] ) );
        if( atomNamesMap.find ( "HG23" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CG2"], atomNamesMap["HG23"] ) );
    }
    
    // D delta
    if ( atomNamesMap.find ( "HD1" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["HD1"] ) );
        if( atomNamesMap.find ( "ND1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["ND1"], atomNamesMap["HD1"] ) );
    }
    
    if ( atomNamesMap.find ( "HD2" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "CD" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["HD2"] ) );
        if( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["HD2"] ) );
        if( atomNamesMap.find ( "OD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["OD2"], atomNamesMap["HD2"] ) );
    }
    
    if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "HD3" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD"], atomNamesMap["HD3"] ) );
    
    if ( atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "HD11" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["HD11"] ) );
        if( atomNamesMap.find ( "HD12" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["HD12"] ) );
        if( atomNamesMap.find ( "HD13" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["HD13"] ) );
    }
    
    if ( atomNamesMap.find ( "HD21" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["HD21"] ) );
        if( atomNamesMap.find ( "ND2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["ND2"], atomNamesMap["HD21"] ) );
    }
    
    if ( atomNamesMap.find ( "HD22" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["HD22"] ) );
        if( atomNamesMap.find ( "ND2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["ND2"], atomNamesMap["HD22"] ) );
    }
    
    if ( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() && atomNamesMap.find ( "HD23" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["HD23"] ) );
    
    // E epsilon
    if ( atomNamesMap.find ( "NE" ) != atomNamesMap.end() && atomNamesMap.find ( "HE" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["NE"], atomNamesMap["HE"] ) );
    
    if ( atomNamesMap.find ( "HE1" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "CE" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE"], atomNamesMap["HE1"] ) );
        if( atomNamesMap.find ( "CE1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE1"], atomNamesMap["HE1"] ) );
        if( atomNamesMap.find ( "NE1" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NE1"], atomNamesMap["HE1"] ) );
    }
    
    if ( atomNamesMap.find ( "HE2" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "CE" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE"], atomNamesMap["HE2"] ) );
        if( atomNamesMap.find ( "CE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE2"], atomNamesMap["HE2"] ) );
        if( atomNamesMap.find ( "OE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["OE2"], atomNamesMap["HE2"] ) );
        if( atomNamesMap.find ( "NE2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NE2"], atomNamesMap["HE2"] ) );
    }
    
    if ( atomNamesMap.find ( "HE3" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "CE" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE"], atomNamesMap["HE3"] ) );
        if( atomNamesMap.find ( "CE3" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CE3"], atomNamesMap["HE3"] ) );
    }
    
    if ( atomNamesMap.find ( "NE2" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "HE21" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NE2"], atomNamesMap["HE21"] ) );
        if( atomNamesMap.find ( "HE22" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NE2"], atomNamesMap["HE22"] ) );
    }
    
    // Z zeta
    if ( atomNamesMap.find ( "CZ" ) != atomNamesMap.end() && atomNamesMap.find ( "HZ" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["CZ"], atomNamesMap["HZ"] ) );
    
    if ( atomNamesMap.find ( "NZ" ) != atomNamesMap.end() && atomNamesMap.find ( "HZ1" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["NZ"], atomNamesMap["HZ1"] ) );
    
    if ( atomNamesMap.find ( "HZ2" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "NZ" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NZ"], atomNamesMap["HZ2"] ) );
        if( atomNamesMap.find ( "CZ2" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CZ2"], atomNamesMap["HZ2"] ) );
    }
    
    if ( atomNamesMap.find ( "HZ3" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "NZ" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NZ"], atomNamesMap["HZ3"] ) );
        if( atomNamesMap.find ( "CZ3" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["CZ3"], atomNamesMap["HZ3"] ) );
    }
    
    // H eta
    if ( atomNamesMap.find ( "OH" ) != atomNamesMap.end() && atomNamesMap.find ( "HH" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["OH"], atomNamesMap["HH"] ) );
    
    if ( atomNamesMap.find ( "CH2" ) != atomNamesMap.end() && atomNamesMap.find ( "HH2" ) != atomNamesMap.end() )
        aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
            protein::CallProteinMovementData::IndexPair ( atomNamesMap["CH2"], atomNamesMap["HH2"] ) );
    
    if ( atomNamesMap.find ( "NH1" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "HH11" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NH1"], atomNamesMap["HH11"] ) );
        if( atomNamesMap.find ( "HH12" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NH1"], atomNamesMap["HH12"] ) );
    }
    
    if ( atomNamesMap.find ( "NH2" ) != atomNamesMap.end() )
    {
        if( atomNamesMap.find ( "HH21" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NH2"], atomNamesMap["HH21"] ) );
        if( atomNamesMap.find ( "HH22" ) != atomNamesMap.end() )
            aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                protein::CallProteinMovementData::IndexPair ( atomNamesMap["NH2"], atomNamesMap["HH22"] ) );
    }
    
    // -------------- END Hydrogen atoms --------------
        
    // all possible connections created, return true
    return true;
}


/*
 *protein::ProteinMovementData::ComputeBoundingBox
 */
void protein::ProteinMovementData::ComputeBoundingBox()
{
    unsigned int counter;
    // if no positions are stored --> return
    if ( this->protAtomData.Count() <= 0 )
        return;
    // get first position
    this->minX = this->maxX = this->protAtomPos[0];
    this->minY = this->maxY = this->protAtomPos[1];
    this->minZ = this->maxZ = this->protAtomPos[2];
    // check all values in the protAtomPos-List
    for ( counter = 3; counter < this->protAtomPos.Count(); counter+=3 )
    {
        // min/max X
        if ( this->minX > this->protAtomPos[counter + 0] )
            this->minX = this->protAtomPos[counter + 0];
        else if ( this->maxX < this->protAtomPos[counter + 0] )
            this->maxX = this->protAtomPos[counter + 0];
        // min/max Y
        if ( this->minY > this->protAtomPos[counter + 1] )
            this->minY = this->protAtomPos[counter + 1];
        else if ( this->maxY < this->protAtomPos[counter + 1] )
            this->maxY = this->protAtomPos[counter + 1];
        // min/max Z
        if ( this->minZ > this->protAtomPos[counter + 2] )
            this->minZ = this->protAtomPos[counter + 2];
        else if ( this->maxZ < this->protAtomPos[counter + 2] )
            this->maxZ = this->protAtomPos[counter + 2];
    }
    // add maximum atom radius to min/max-values to prevent atoms sticking out
    this->minX +=-3.0f;
    this->maxX += 3.0f;
    this->minY +=-3.0f;
    this->maxY += 3.0f;
    this->minZ +=-3.0f;
    this->maxZ += 3.0f;
    // compute maximum dimension
    this->maxDimension = this->maxX - this->minX;
    if ( this->maxDimension < ( this->maxY - this->minY ) )
        this->maxDimension = this->maxY - this->minY;
    if ( this->maxDimension < ( this->maxZ - this->minZ ) )
        this->maxDimension = this->maxZ - this->minZ;
}
