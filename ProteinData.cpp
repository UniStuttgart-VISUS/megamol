/*
 * ProteinData.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "ProteinData.h"
#include "param/StringParam.h"
#include "vislib/MemmappedFile.h"
#include "vislib/Log.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/sysfunctions.h"
#include "vislib/StringTokeniser.h"
#include <string>
#include <iostream>

using namespace megamol;
using namespace megamol::core;


/*
 * protein::ProteinData::ProteinData
 */
protein::ProteinData::ProteinData(void) : Module (),
        m_protDataCalleeSlot("providedata", "Connects the protein rendering with protein data storage"),
		m_filename("filename", "The path to the protein data file to load.")
{
    CallProteinDataDescription cpdd;
    this->m_protDataCalleeSlot.SetCallback(cpdd.ClassName(), "GetData", &ProteinData::ProtDataCallback);
    this->MakeSlotAvailable(&this->m_protDataCalleeSlot);

    this->m_filename.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->m_filename);
	
	// secondary structure
	this->m_secondaryStructureComputed = false;
	this->m_stride = 0;
}


/*
 * protein::ProteinData::~ProteinData
 */
protein::ProteinData::~ProteinData(void)
{
    this->Release ();
}


/*
 * protein::ProteinData::ProtDataCallback
 */
bool protein::ProteinData::ProtDataCallback(Call& call) {
    unsigned int counter;

    protein::CallProteinData *pdi = dynamic_cast<protein::CallProteinData*>(&call);

    if (this->m_filename.IsDirty()) 
	{
		// load the data.
		this->tryLoadFile();
		this->m_filename.ResetDirty();
    }

	if( pdi )
	{
		// set the bounding box
		pdi->SetBoundingBox( this->m_minX, this->m_minY, this->m_minZ, this->m_maxX, this->m_maxY, this->m_maxZ);
		// set scaling
		pdi->SetScaling( 1.0f / this->m_maxDimension);

		// set the amino acid name table
		pdi->SetAminoAcidNameTable( (unsigned int)this->m_aminoAcidNames.Count(), 
			this->m_aminoAcidNames.PeekElements());
		// set the atom type table
		pdi->SetAtomTypeTable( (unsigned int)this->m_atomTypes.Count(), this->m_atomTypes.PeekElements());

		// set the number of protein atoms and the pointers for atom data and positions
		pdi->SetProteinAtomCount( (unsigned int)this->m_protAtomData.Count());
		pdi->SetProteinAtomDataPointer( (protein::CallProteinData::AtomData*)this->m_protAtomData.PeekElements());
		pdi->SetProteinAtomPositionPointer( (float*)this->m_protAtomPos.PeekElements());
		// allocate the chains
		pdi->AllocateChains( (unsigned int)this->m_aminoAcidChains.Count());
		// set amino acids and secondary structure to the chains
		for( counter = 0; counter < pdi->ProteinChainCount(); counter++ )
		{
			pdi->AccessChain( counter).SetAminoAcid(
				(unsigned int)this->m_aminoAcidChains[counter].Count(),
				this->m_aminoAcidChains[counter].PeekElements());
			}

		// try to compute secondary structure, if necessary
		if( !this->m_secondaryStructureComputed )
		{
			double elapsedTime = 0.0;
			time_t t = clock();
			if( m_stride ) delete m_stride;
			m_stride = new Stride( pdi);
			this->m_secondaryStructureComputed = true;
			elapsedTime = ( double( clock() - t) / double( CLOCKS_PER_SEC) );
			vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO,
					"%s: Computed secondary structure via Stride in %.4f seconds.\n",
					this->ClassName(), elapsedTime );
		}
		m_stride->WriteToInterface( pdi );

		// set the disulfide bonds
		pdi->SetDisulfidBondsPointer( (unsigned int)this->m_dsBonds.Count(), 
			(protein::CallProteinData::IndexPair*)this->m_dsBonds.PeekElements());

		// set the min and max temperature factor
		pdi->SetMinimumTemperatureFactor( m_minTempFactor);
		pdi->SetMaximumTemperatureFactor( m_maxTempFactor);
		// set the min and max occupancy
		pdi->SetMinimumOccupancy( m_minOccupancy);
		pdi->SetMaximumOccupancy( m_maxOccupancy);
		// set the min and max charge
		pdi->SetMinimumCharge( m_minCharge );
		pdi->SetMaximumCharge( m_maxCharge );
	}

    return true;
}

/*
 *protein::ProteinData::create
 */
bool protein::ProteinData::create(void)
{
    this->tryLoadFile();
    this->m_filename.ResetDirty();
    return true;
}


/*
 *protein::ProteinData::tryLoadFile
 */
bool protein::ProteinData::tryLoadFile(void)
{
	using vislib::sys::MemmappedFile;
    using vislib::sys::File;
    using vislib::sys::Log;

	// clear all containers
	this->ClearData();
	// add all elements from the periodic table to m_atomTypes
	this->FillAtomTypesWithPeriodicTable();
	// add all amino acid names to m_aminoAcidNames
	this->FillAminoAcidNames();

	// temporary variables
	unsigned int counterChainId, counterResSeq, counterAtom;
	vislib::StringA tmpStr;
	// return value: false means that anything went wrong
	bool retval = false;
	// is this the first ATOM-line?
	bool firstAtom = true;
	// next file
	bool newFile = true;
	// file 
    MemmappedFile file;
	// get filename 
    const vislib::TString& fn = this->m_filename.Param<param::StringParam>()->Value();

	if ( fn.IsEmpty() )
	{
        // no file to load
        return false;
    }

    //vislib::StringTokeniser<vislib::TCharTraits> filenames( fn, ' ');
    vislib::Array<vislib::TString> filenames = vislib::StringTokeniser<vislib::TCharTraits>::Split( fn, ';', true);

    for( unsigned int cnt = 0; cnt < filenames.Count(); ++ cnt ) {

	    //if( !file.Open( fn, File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY ) ) {
        if( !file.Open( filenames[cnt], File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY ) ) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
		                               "%s: Unable to open file \"%s\"", this->ClassName(),
											    vislib::StringA ( fn ).PeekBuffer() );
            return false;
        }
        newFile = true;

	    vislib::StringA str( filenames[cnt]);
	    // check, if file extension is 'pdb'
	    str.ToLowerCase();
	    if( !str.EndsWith( ".pdb") )
		    continue;
	    str.Clear();

	    // => check for PDB file indicator: the file must contain at least one 'ATOM' ?

	    // while file pointer is not at 'end of file'
	    while( !file.IsEOF() ) {
		    vislib::StringA str;
		    // read next line from file
		    str = vislib::sys::ReadLineFromFileA( file );
		    // skip empty lines
		    if( str.Length() <= 0 ) continue;
    	    
		    if( str.StartsWithInsensitive( "ATOM") ) {
			    // check if the current atom is at an alternate location
			    if( !str.Substring( 16, 1 ).Equals( " ", false) &&
			        !str.Substring( 16, 1 ).Equals( "A", false) )
				    // only add one occurence of the ATOM (either ' ' or 'A')
				    continue;
			    ////////////////////////////////////////////////////////////////////////////
			    // Add all ATOM entries to the 'atomEntry'-vector for further processing. //
			    // Write chainIDs and resSeq to maps to be able to recover amino acids.   //
			    ////////////////////////////////////////////////////////////////////////////
			    if( firstAtom ) {
				    // start the first chain
				    this->tmp_atomEntries.resize(1);
				    // start the fist amino acid
				    this->tmp_atomEntries.back().resize(1);
				    // add the first atom to the amino acid
				    this->tmp_atomEntries.back().back().push_back( str);

				    // add the chainId to the map
				    this->tmp_chainIdMap[str.Substring( 21, 1)[0]] = 0;
				    this->tmp_resSeqMap.SetCount( tmp_resSeqMap.Count() + 1);
				    // add the resSeq to the map
				    this->tmp_resSeqMap.Last()[str.Substring( 22, 4).PeekBuffer()] = 0;
				    // the first atom entry is read --> set firstAtom to 'false'
				    firstAtom = false;
                    newFile = false;
			    } else {
				    // if a new chain starts:
				    if( !this->tmp_atomEntries.back().back().back().Substring( 21, 1).Equals( str.Substring( 21, 1)) || newFile ) {
					    // start a new chain
					    this->tmp_atomEntries.resize( this->tmp_atomEntries.size() + 1);
					    // start a new amino acid
					    this->tmp_atomEntries.back().resize(1);
					    // add the ATOM string to the last amino acid
					    this->tmp_atomEntries.back().back().push_back( str);
    					
					    // add the chainId to the map
					    this->tmp_chainIdMap[str.Substring( 21, 1)[0]] = (unsigned int)this->tmp_atomEntries.size()-1;
					    this->tmp_resSeqMap.SetCount( this->tmp_resSeqMap.Count() + 1);
					    // add the resSeq to the map
					    this->tmp_resSeqMap.Last()[str.Substring( 22, 4).PeekBuffer()] = 0;
				    }
				    // if a new amino acid starts:
				    else if( !tmp_atomEntries.back().back().back().Substring( 22, 4).Equals( str.Substring( 22, 4)) ) {
					    // start a new amino acid
					    this->tmp_atomEntries.back().resize( this->tmp_atomEntries.back().size() + 1);
					    // add the ATOM string to the last amino acid
					    this->tmp_atomEntries.back().back().push_back( str);
    					
					    // add the resSeq to the map
					    this->tmp_resSeqMap.Last()[str.Substring( 22, 4).PeekBuffer()] = (unsigned int)this->tmp_atomEntries.back().size()-1;
				    }
				    // if no new chain and no new amino acid starts:
				    else {
					    // add the ATOM string to the last amino acid
					    this->tmp_atomEntries.back().back().push_back( str);
				    }
    				newFile = false;	
			    }
		    }
            else if( str.StartsWithInsensitive( "HETATM") ) {
			    // TODO: handle HETATM entries --> atoms of solvent molecules
		    }
		    else if( str.StartsWithInsensitive( "CONECT") ) {
			    // TODO: handle CONECT entries to get the connections of the solvent molecules atoms
		    }
		    else if( str.StartsWithInsensitive( "END") ) {
			    retval = true;
			    break;
		    }
		    else if( str.StartsWithInsensitive( "ENDMDL") ) {
			    retval = true;
			    break;
		    }
	    }

        file.Close();
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
	                               "%s: File \"%s\" loaded successfully\n",
	                               this->ClassName(), vislib::StringA ( fn ).PeekBuffer() );
        
    }

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// loop over all atom entries in the 'tmp_atomEntries'-vector to get chains, amino acids and atoms //
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	// create 'm_aminoAcidChains'-Array
	this->m_aminoAcidChains.SetCount( this->tmp_atomEntries.size());
	// loop over all chains
	for(counterChainId = 0; 
		counterChainId < this->tmp_atomEntries.size(); 
		counterChainId++)
	{
		// create amino acids array in current chain with the correct size
		this->m_aminoAcidChains[counterChainId].SetCount( this->tmp_atomEntries[counterChainId].size() );
		// loop over all amino acids in the current chain
		for(counterResSeq = 0; 
			counterResSeq < this->tmp_atomEntries[counterChainId].size(); 
			counterResSeq++)
		{
			// set the position information of the current amino acid
			this->m_aminoAcidChains[counterChainId][counterResSeq].SetPosition( this->tmp_currentAtomIdx, 
				(unsigned int)this->tmp_atomEntries[counterChainId][counterResSeq].size() );
			// set the amino acid name for the current amino acid
			this->m_aminoAcidChains[counterChainId][counterResSeq].SetNameIndex ( this->GetAminoAcidNameIdx ( this->tmp_atomEntries[counterChainId][counterResSeq][0].Substring ( 17, 3 ).PeekBuffer() ) );
			// loop oder all atoms of the current amino acid
			for(counterAtom = 0;
				counterAtom < this->tmp_atomEntries[counterChainId][counterResSeq].size();
				counterAtom++)
			{
				// parse string to add current atom to 'm_protAtomData' and the position to 'm_protAtomPos'
				this->ParsePDBAtom( this->tmp_atomEntries[counterChainId][counterResSeq][counterAtom]);
				
				tmpStr = this->tmp_atomEntries[counterChainId][counterResSeq][counterAtom].Substring( 12, 4);
				tmpStr.TrimSpaces();

				// if the current atom is one of the backbone atoms --> set index to current amino acid
				if( tmpStr.Equals( "CA") )
					this->m_aminoAcidChains[counterChainId][counterResSeq].SetCAlphaIndex( counterAtom );
				else if( tmpStr.Equals( "C" ) )
					this->m_aminoAcidChains[counterChainId][counterResSeq].SetCCarbIndex( counterAtom );
				else if( tmpStr.Equals( "N" ) )
					this->m_aminoAcidChains[counterChainId][counterResSeq].SetNIndex( counterAtom );
				else if( tmpStr.Equals( "O" ) )
					this->m_aminoAcidChains[counterChainId][counterResSeq].SetOIndex( counterAtom );

				// increase current atom index
				this->tmp_currentAtomIdx++;
			}
			// all atoms in the current amino acid are handled --> make connections now
			this->MakeConnections( counterChainId, counterResSeq);
		}
	}

	// try to load secondary structure information from STRIDE file
	//this->ReadSecondaryStructure();

	// check for potential disulfide bonds
	this->EstimateDisulfideBonds();

	// compute the bounding box for the stored atom positions
	this->ComputeBoundingBox();

	// delete temporary variables
	for( counterChainId = 0; counterChainId < this->tmp_atomEntries.size(); counterChainId++ )
	{
		for( counterResSeq = 0; counterResSeq < this->tmp_atomEntries[counterChainId].size(); counterResSeq++ )
		{
			this->tmp_atomEntries[counterChainId][counterResSeq].clear();
		}
		this->tmp_atomEntries[counterChainId].clear();
	}
	this->tmp_atomEntries.clear();

	this->tmp_chainIdMap.clear();

	for( counterChainId = 0; counterChainId < this->tmp_resSeqMap.Count(); counterChainId++ )
	{
		this->tmp_resSeqMap[counterChainId].clear();
	}
	this->tmp_resSeqMap.Clear();

	this->tmp_atomicNumbers.clear();
	this->tmp_aminoAcidNameIdx.clear();

	this->tmp_cysteineSulfurAtoms.Clear();

	// return 'true' if everything could be loaded
	return retval;
}


/*
 *protein::ProteinData::ClearData
 */
void protein::ProteinData::ClearData()
{
	unsigned int i, j;

	// clear temporary varibles
	this->tmp_currentAtomIdx = 0;
	for( i = 0; i < this->tmp_atomEntries.size(); i++ )
	{
		for( j = 0; j < this->tmp_atomEntries.at(i).size(); j++ )
		{
			this->tmp_atomEntries.at(i).at(j).clear();
		}
		this->tmp_atomEntries.at(i).clear();
	}
	this->tmp_atomEntries.clear();
	this->tmp_chainIdMap.clear();
	for( i = 0; i < this->tmp_resSeqMap.Count(); i++ )
	{
		this->tmp_resSeqMap[i].clear();
	}
	this->tmp_resSeqMap.Clear();

	// clear data variables for DataInterface
	this->m_aminoAcidNames.Clear();
	this->tmp_aminoAcidNameIdx.clear();
	this->m_atomTypes.Clear();
	this->tmp_atomicNumbers.clear();
	for( i = 0; i < this->m_aminoAcidChains.Count(); i++ )
	{
		this->m_aminoAcidChains[i].Clear();
	}
	this->m_aminoAcidChains.Clear();
	for( i = 0; i < m_secStruct.Count(); i++ )
	{
		this->m_secStruct[i].Clear();
	}
	this->m_secStruct.Clear();
	this->m_dsBonds.Clear();
	this->m_protAtomData.Clear();
	this->m_protAtomPos.Clear();
	
	// reset bounding box
	this->m_minX = 0.0f;
	this->m_minY = 0.0f;
	this->m_minZ = 0.0f;
	this->m_maxX = 1.0f;
	this->m_maxY = 1.0f;
	this->m_maxZ = 1.0f;
	this->m_maxDimension = 1.0f;

	this->tmp_cysteineSulfurAtoms.Clear();
}


/**********************************************************************
 * 'protein'-functions                                                *
 **********************************************************************/

/*
 *protein::ProteinData::ReadSecondaryStructure
 */
bool protein::ProteinData::ReadSecondaryStructure()
{
	using vislib::sys::MemmappedFile;
    using vislib::sys::File;
    using vislib::sys::Log;

	unsigned int counterChainId;
	unsigned int counterResSeq;
	vislib::StringA line;
	vislib::StringA currentChainId;
	vislib::StringA currentResSeq;
	vislib::StringA currentType;
	unsigned int firstAminoAcidIdx;
	unsigned int aminoAcidCnt;
	unsigned int firstAtomIdx;
	unsigned int atomCnt;
	// temp array that stores all secondary structure information for each chain
	vislib::Array<vislib::Array<vislib::StringA> > asgEntries;

    MemmappedFile file;
	// get filename 
    const vislib::TString& fn = this->m_filename.Param<param::StringParam>()->Value();

	if ( fn.IsEmpty() )
	{
        // no file to load
        return false;
    }

	// check, if file extension is 'pdb'
	vislib::StringA str( fn );
	str.ToLowerCase();
	if( !str.EndsWith( ".pdb") )
	{
		return false;
	}
	else
	{
		// substitute the extension '.pdb' with '.stride'
		str = fn;
		str.Truncate( fn.Length()-4);
		str.Append( ".stride");
	}

    // try reading stride file
	if ( !file.Open ( str, File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY ) )
	{
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
		                           "%s: Unable to open file \"%s\" for reading secondary structure", this->ClassName(), vislib::StringA ( str ).PeekBuffer() );
        return false;
    }
	
	// resize 'asgEntries' to the number of chains
	asgEntries.SetCount( this->tmp_atomEntries.size());

	// read all ASG entries from the stride-file
	while( !file.IsEOF() )
	{
		line = vislib::sys::ReadLineFromFileA( file);
		// add ASG entries to the correct chains
		if( line.StartsWithInsensitive( "ASG") )
		{
			asgEntries[this->tmp_chainIdMap[line.Substring( 9, 1)[0]]].Add( line);
		}
	}

	file.Close();
	Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
	                           "%s: File \"%s\" loaded successfully\n",
	                           this->ClassName(), vislib::StringA ( str ).PeekBuffer() );

	// resize secondary structure string to the number of chains
	this->m_secStruct.SetCount( this->tmp_atomEntries.size());
	// loop over all chains
	for( counterChainId = 0; counterChainId < asgEntries.Count(); counterChainId++ )
	{
		this->m_secStruct[counterChainId].Clear();
		// read and set first amino acid's sec struct info
		this->m_secStruct[counterChainId].Add( protein::CallProteinData::SecStructure());
		// read the type
		currentType = asgEntries[counterChainId][0].Substring( 24, 1);
		// set the type
		if( currentType.Equals( "H") ||	currentType.Equals( "G") ||	currentType.Equals( "I") )
			this->m_secStruct[counterChainId].Last().SetType( protein::CallProteinData::SecStructure::TYPE_HELIX );
		else if( currentType.Equals( "E") )
			this->m_secStruct[counterChainId].Last().SetType( protein::CallProteinData::SecStructure::TYPE_SHEET );
		else if( currentType.Equals( "T") )
			this->m_secStruct[counterChainId].Last().SetType( protein::CallProteinData::SecStructure::TYPE_TURN );
		else
			this->m_secStruct[counterChainId].Last().SetType( protein::CallProteinData::SecStructure::TYPE_COIL );
		// store first amino acid index
		firstAminoAcidIdx = 0;
		aminoAcidCnt = 1;
		firstAtomIdx = this->m_aminoAcidChains[counterChainId][0].FirstAtomIndex();
		atomCnt = this->m_aminoAcidChains[counterChainId][0].AtomCount();

		// loop over all following ASG entries in this chain
		for( counterResSeq = 1; counterResSeq < asgEntries[counterChainId].Count(); counterResSeq++ )
		{
			// if another secondary structure element starts:
			if( currentType != asgEntries[counterChainId][counterResSeq].Substring( 24, 1) )
			{
				// add the position information to the last secondary structure element
				this->m_secStruct[counterChainId].Last().SetPosition( firstAtomIdx, atomCnt, firstAminoAcidIdx, aminoAcidCnt);
				// add a new secondary stucture element
				this->m_secStruct[counterChainId].Add( protein::CallProteinData::SecStructure());
				// read the type
				currentType = asgEntries[counterChainId][counterResSeq].Substring( 24, 1);
				// set the type
				if( currentType.Equals( "H") ||	currentType.Equals( "G") ||	currentType.Equals( "I") )
					this->m_secStruct[counterChainId].Last().SetType( protein::CallProteinData::SecStructure::TYPE_HELIX );
				else if( currentType.Equals( "E") )
					this->m_secStruct[counterChainId].Last().SetType( protein::CallProteinData::SecStructure::TYPE_SHEET );
				else if( currentType.Equals( "T") )
					this->m_secStruct[counterChainId].Last().SetType( protein::CallProteinData::SecStructure::TYPE_TURN );
				else
					this->m_secStruct[counterChainId].Last().SetType( protein::CallProteinData::SecStructure::TYPE_COIL );
				// store first amino acid index
				firstAminoAcidIdx = counterResSeq;
				aminoAcidCnt = 1;
				firstAtomIdx = this->m_aminoAcidChains[counterChainId][firstAminoAcidIdx].FirstAtomIndex();
				atomCnt = this->m_aminoAcidChains[counterChainId][firstAminoAcidIdx].AtomCount();
			}
			else
			{
				// --> add the next amino acid to the current sec struct
				aminoAcidCnt++;
				atomCnt += this->m_aminoAcidChains[counterChainId][counterResSeq].AtomCount();
			}
		}
		// after all ASG entries are read: add the position information to the last secondary structure element
		this->m_secStruct[counterChainId].Last().SetPosition( firstAtomIdx, atomCnt, firstAminoAcidIdx, aminoAcidCnt);
	}

	return true;
}


/*
 *protein::ProteinData::ParsePDBAtom
 */
bool protein::ProteinData::ParsePDBAtom( const vislib::StringA &line )
{
	// if input string does not start with 'ATOM' --> return false
	if( !line.StartsWithInsensitive( "ATOM") )
		return false;

	vislib::StringA str, elemName;
	unsigned int cnt;
	// temporary atom variables
	unsigned int atomTypeIdx = 0;
	float charge = 0.0f;
	float occupancy = 0.0f;
	float tempFactor = 0.0f;

	// get and store the position of the atom
	str = line.Substring( 30, 8);
	str.TrimSpaces();
	this->m_protAtomPos.Add( (float)atof(str.PeekBuffer()));
	str = line.Substring( 38, 8);
	str.TrimSpaces();
	this->m_protAtomPos.Add( (float)atof(str.PeekBuffer()));
	str = line.Substring( 46, 8);
	str.TrimSpaces();
	this->m_protAtomPos.Add( (float)atof(str.PeekBuffer()));

	// get the name (atom type) of the current ATOM entry
	str = line.Substring ( 12, 4 );
	str.TrimSpaces();
	if ( !str.IsEmpty() )
	{
		// search atom type entry in atom type table
		for( cnt = 0; cnt < this->m_atomTypes.Count(); ++cnt )
		{
			if( this->m_atomTypes[cnt].Name() == str )
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
		this->m_atomTypes.Add( CallProteinData::AtomType( str,
			this->tmp_atomTypes[tmpTypeTableId].Radius(),
			this->tmp_atomTypes[tmpTypeTableId].Colour()[0],
			this->tmp_atomTypes[tmpTypeTableId].Colour()[1],
			this->tmp_atomTypes[tmpTypeTableId].Colour()[2]) );
		// set atom type index to the index of the newly added atom type
		atomTypeIdx = this->m_atomTypes.Count() - 1;
	}
	// check if the atom is the sulfur atom of a cysteine
	if( str.Equals( "SG" ) )
	{
		// add the index of the atom to the cysteine sulfur atom list
		this->tmp_cysteineSulfurAtoms.Add ( this->m_protAtomData.Count() );
	}

	// get the temperature factor (b-factor)
	str = line.Substring( 60, 6);
	str.TrimSpaces();
	tempFactor = (float)atof(str.PeekBuffer());
	if( this->m_protAtomData.Count() == 0 )
	{
		this->m_minTempFactor = tempFactor;
		this->m_maxTempFactor = tempFactor;
	}
	else
	{
		if( this->m_minTempFactor > tempFactor )
			this->m_minTempFactor = tempFactor;
		else if( this->m_maxTempFactor < tempFactor )
			this->m_maxTempFactor = tempFactor;
	}
	
	// get the occupancy
	str = line.Substring( 54, 6);
	str.TrimSpaces();
	occupancy = (float)atof(str.PeekBuffer());
	if( this->m_protAtomData.Count() == 0 )
	{
		this->m_minOccupancy = occupancy;
		this->m_maxOccupancy = occupancy;
	}
	else
	{
		if( this->m_minOccupancy > occupancy )
			this->m_minOccupancy = occupancy;
		else if( this->m_maxOccupancy < occupancy )
			this->m_maxOccupancy = occupancy;
	}
	
	// get the charge
	str = line.Substring( 78, 2);
	str.TrimSpaces();
	charge = (float)atof(str.PeekBuffer());
	if( this->m_protAtomData.Count() == 0 )
	{
		this->m_minCharge = charge;
		this->m_maxCharge = charge;
	}
	else
	{
		if( this->m_minCharge > charge )
			this->m_minCharge = charge;
		else if( this->m_maxCharge < charge )
			this->m_maxCharge = charge;
	}

	// add atom data to protein atom data array
    this->m_protAtomData.Add( protein::CallProteinData::AtomData( atomTypeIdx, charge, tempFactor, occupancy));

	// check if the atom is the sulfur atom of a cysteine
	/*
	if( atomTypeIdx == 16 && line.Substring( 17, 3).Equals( "CYS") )
	{
		// add the index of the atom to the cysteine sulfur atom list
		this->tmp_cysteineSulfurAtoms.Add( (unsigned int)this->m_protAtomData.Count()-1);
	}
	*/

	return true;
}


/*
 *protein::ProteinData::ParsePDBHetatm
 */
bool protein::ProteinData::ParsePDBHetatm( const vislib::StringA &line )
{
	return true;
}


/*
 *protein::ProteinData::ParsePDBConect
 */
bool protein::ProteinData::ParsePDBConect( const vislib::StringA &line )
{
	return true;
}


/*
 *protein::ProteinData::FillAminoAcidNames
 */
void protein::ProteinData::FillAminoAcidNames()
{
	// clear m_atomTypes-Array if necessary
	if( m_aminoAcidNames.Count() > 0 )
		m_aminoAcidNames.Clear();
	// clear map for atomic numbers
	tmp_aminoAcidNameIdx.clear();

	this->m_aminoAcidNames.Add( "unknown");
	this->m_aminoAcidNames.Add ( "ALA" );
	this->tmp_aminoAcidNameIdx["ALA"] = 1;
	this->m_aminoAcidNames.Add ( "ARG" );
	this->tmp_aminoAcidNameIdx["ARG"] = 2;
	this->m_aminoAcidNames.Add ( "ASN" );
	this->tmp_aminoAcidNameIdx["ASN"] = 3;
	this->m_aminoAcidNames.Add ( "ASP" );
	this->tmp_aminoAcidNameIdx["ASP"] = 4;
	this->m_aminoAcidNames.Add ( "CYS" );
	this->tmp_aminoAcidNameIdx["CYS"] = 5;
	this->m_aminoAcidNames.Add ( "GLN" );
	this->tmp_aminoAcidNameIdx["GLN"] = 6;
	this->m_aminoAcidNames.Add ( "GLU" );
	this->tmp_aminoAcidNameIdx["GLU"] = 7;
	this->m_aminoAcidNames.Add ( "GLY" );
	this->tmp_aminoAcidNameIdx["GLY"] = 8;
	this->m_aminoAcidNames.Add ( "HIS" );
	this->tmp_aminoAcidNameIdx["HIS"] = 9;
	this->m_aminoAcidNames.Add ( "ILE" );
	this->tmp_aminoAcidNameIdx["ILE"] = 10;
	this->m_aminoAcidNames.Add ( "LEU" );
	this->tmp_aminoAcidNameIdx["LEU"] = 11;
	this->m_aminoAcidNames.Add ( "LYS" );
	this->tmp_aminoAcidNameIdx["LYS"] = 12;
	this->m_aminoAcidNames.Add ( "MET" );
	this->tmp_aminoAcidNameIdx["MET"] = 13;
	this->m_aminoAcidNames.Add ( "PHE" );
	this->tmp_aminoAcidNameIdx["PHE"] = 14;
	this->m_aminoAcidNames.Add ( "PRO" );
	this->tmp_aminoAcidNameIdx["PRO"] = 15;
	this->m_aminoAcidNames.Add ( "SER" );
	this->tmp_aminoAcidNameIdx["SER"] = 16;
	this->m_aminoAcidNames.Add ( "THR" );
	this->tmp_aminoAcidNameIdx["THR"] = 17;
	this->m_aminoAcidNames.Add ( "TRP" );
	this->tmp_aminoAcidNameIdx["TRP"] = 18;
	this->m_aminoAcidNames.Add ( "TYR" );
	this->tmp_aminoAcidNameIdx["TYR"] = 19;
	this->m_aminoAcidNames.Add ( "VAL" );
	this->tmp_aminoAcidNameIdx["VAL"] = 20;
	
	/*
	this->m_aminoAcidNames.Add ( "unknown" );
	this->m_aminoAcidNames.Add( "Alanin");
	this->tmp_aminoAcidNameIdx["ALA"] = 1;
	this->m_aminoAcidNames.Add( "Arginin");
	this->tmp_aminoAcidNameIdx["ARG"] = 2;
	this->m_aminoAcidNames.Add( "Asparagin");
	this->tmp_aminoAcidNameIdx["ASN"] = 3;
	this->m_aminoAcidNames.Add ( "Asparaginsaeure" );
	this->tmp_aminoAcidNameIdx["ASP"] = 4;
	this->m_aminoAcidNames.Add( "Cystein");
	this->tmp_aminoAcidNameIdx["CYS"] = 5;
	this->m_aminoAcidNames.Add( "Glutamin");
	this->tmp_aminoAcidNameIdx["GLN"] = 6;
	this->m_aminoAcidNames.Add ( "Glutaminsaeure" );
	this->tmp_aminoAcidNameIdx["GLU"] = 7;
	this->m_aminoAcidNames.Add( "Glycin");
	this->tmp_aminoAcidNameIdx["GLY"] = 8;
	this->m_aminoAcidNames.Add( "Histidin");
	this->tmp_aminoAcidNameIdx["HIS"] = 9;
	this->m_aminoAcidNames.Add( "Isoleucin");
	this->tmp_aminoAcidNameIdx["ILE"] = 10;
	this->m_aminoAcidNames.Add( "Leucin");
	this->tmp_aminoAcidNameIdx["LEU"] = 11;
	this->m_aminoAcidNames.Add( "Lysin");
	this->tmp_aminoAcidNameIdx["LYS"] = 12;
	this->m_aminoAcidNames.Add( "Methionin");
	this->tmp_aminoAcidNameIdx["MET"] = 13;
	this->m_aminoAcidNames.Add( "Phenylalanin");
	this->tmp_aminoAcidNameIdx["PHE"] = 14;
	this->m_aminoAcidNames.Add( "Prolin");
	this->tmp_aminoAcidNameIdx["PRO"] = 15;
	this->m_aminoAcidNames.Add( "Serin");
	this->tmp_aminoAcidNameIdx["SER"] = 16;
	this->m_aminoAcidNames.Add( "Threonin");
	this->tmp_aminoAcidNameIdx["THR"] = 17;
	this->m_aminoAcidNames.Add( "Tryptophan");
	this->tmp_aminoAcidNameIdx["TRP"] = 18;
	this->m_aminoAcidNames.Add( "Tyrosin");
	this->tmp_aminoAcidNameIdx["TYR"] = 19;
	this->m_aminoAcidNames.Add( "Valin");
	this->tmp_aminoAcidNameIdx["VAL"] = 20;
	*/
}


/*
 *protein::ProteinData::FillAtomTypesWithPeriodicTable
 */
void protein::ProteinData::FillAtomTypesWithPeriodicTable()
{
    bool VIS09colors = false;
	// clear m_atomTypes-Array if necessary
	if( m_atomTypes.Count() > 0 )
		m_atomTypes.Clear();
	// clear map for atomic numbers
	tmp_atomicNumbers.clear();

	// write 'unknown' to first atom type entry
	this->m_atomTypes.AssertCapacity( 150);
	this->m_atomTypes.Add( protein::CallProteinData::AtomType( "unknown", 1.0, 127, 127, 127));
	
	// write all elements from the periodic table to tmp_atomTypes-Array
	// with their van-der-Waals-radius and natural color (white if no color or colorless)
	this->tmp_atomTypes.Clear();
	this->tmp_atomTypes.AssertCapacity( 120);
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "unknown", 1.0, 127, 127, 127 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Hydrogen", 1.2f, 240, 240, 240 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Helium", 1.4f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Lithium", 1.82f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Beryllium", 2.0f, 85, 85, 85 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Boron", 2.0f, 0, 0, 0 ) );
	if( VIS09colors )
        this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Carbon", 1.7f, 130, 130, 130 ) );
    else
        this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Carbon", 1.7f, 0, 240, 0 ) );
    if( VIS09colors )
	    this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Nitrogen", 1.55f, 0, 153, 153 ) );
    else
        this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Nitrogen", 1.55f, 0, 0, 240 ) );
    if( VIS09colors )
        this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Oxygen", 1.52f, 250, 60, 0 ) );
    else
	    this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Oxygen", 1.52f, 240, 0, 0 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Fluorine", 1.47f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Neon", 1.54f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Sodium", 2.27f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Magnesium", 1.73f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Aluminium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Silicon", 2.1f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Phosphorus", 1.8f, 255, 255, 255 ) );
    if( VIS09colors )
	    this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Sulphur", 1.8f, 255, 210, 0 ) );
    else
        this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Sulphur", 1.8f, 255, 255, 0 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Chlorine", 1.75f, 255, 255, 0 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Argon", 1.88f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Potassium", 2.75f, 200, 200, 200 ) );
    if( VIS09colors )
	    this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Calcium", 2.0f, 100, 100, 100 ) );
    else
        this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Calcium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Scandium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Titanium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Vanadium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Chromium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Manganese", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Iron", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Cobalt", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Nickel", 1.63f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Copper", 1.4f, 255, 132, 0 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Zinc", 1.39f, 85, 85, 85 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Gallium", 1.87f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Germanium", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Arsenic", 1.85f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Selenium", 1.9f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Bromine", 1.85f, 255, 0, 0 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Krypton", 2.02f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Rubidium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Strontium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Yttrium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Zirconium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Niobium", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Molybdenum", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Technetium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Ruthenium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Rhodium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Palladium", 1.63f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Silver", 1.72f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Cadmium", 1.58f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Indium", 1.93f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Tin", 2.17f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Antimony", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Tellurium", 2.06f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Iodine", 1.98f, 85, 85, 85 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Xenon", 2.16f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Caesium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Barium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Lanthanum", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Cerium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Praseodymium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Neodymium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Promethium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Samarium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Europium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Gadolinium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Terbium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Dysprosium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Holmium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Erbium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Thulium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Ytterbium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Lutetium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Hafnium", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Tantalum", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Tungsten", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Rhenium", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Osmium", 2.0f, 85, 85, 85 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Iridium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Platinum", 1.72f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Gold", 1.66f, 238, 201, 0 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Mercury", 1.55f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Thallium", 1.96f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Lead", 2.02f, 85, 85, 85 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Bismuth", 2.0f, 170, 170, 170 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Polonium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Astatine", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Radon", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Francium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Radium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Actinium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Thorium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Protactinium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Uranium", 1.86f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Neptunium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Plutonium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Americium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Curium", 2.0f, 200, 200, 200 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Berkelium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Californium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Einsteinium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Fermium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Mendelevium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Nobelium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Lawrencium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Rutherfordium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Dubnium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Seaborgium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Bohrium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Hassium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Meitnerium", 2.0f, 255, 255, 255 ) );
	this->tmp_atomTypes.Add ( protein::CallProteinData::AtomType ( "Darmstadtium", 2.0f, 255, 255, 255 ) );

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
 *protein::ProteinData::GetAtomicNumber
 */
unsigned int protein::ProteinData::GetAtomicNumber( const char* symbol) const
{
	// check, if the symbol is in the tmp_atomicNumbers-map
	if( this->tmp_atomicNumbers.find( symbol) != this->tmp_atomicNumbers.end() )
	{
		return this->tmp_atomicNumbers.find( symbol)->second;
	}
	else
	{
		// return zero, if 'symbol' was not found
		return 0;
	}
}


/*
 *protein::ProteinData::GetAminoAcidNameIdx
 */
unsigned int protein::ProteinData::GetAminoAcidNameIdx( const char* name) const
{
	// check, if the name is in the tmp_aminoAcidNameIdx-map
	if( this->tmp_aminoAcidNameIdx.find( name) != this->tmp_aminoAcidNameIdx.end() )
	{
		return this->tmp_aminoAcidNameIdx.find( name)->second;
	}
	else
	{
		// return zero, if 'name' was not found
		return 0;
	}
}


/*
 *protein::ProteinData::MakeConnections
 */
bool protein::ProteinData::MakeConnections( unsigned int chainIdIdx, unsigned int resSeqIdx )
{
	std::map<std::string, unsigned int> atomNamesMap;
	unsigned int counter;
	vislib::StringA name;

    // TEMP!!!! 

    unsigned int cnt1, cnt2, idx1, idx2;
    //vislib::math::Vector<float, 3> v1, v2;
    // loop over all atoms in this amino acid and fill the map with the names of the atoms
    for ( cnt1 = 0; cnt1 < this->tmp_atomEntries[chainIdIdx][resSeqIdx].size() - 1; ++cnt1 ){
        for ( cnt2 = cnt1 + 1; cnt2 < this->tmp_atomEntries[chainIdIdx][resSeqIdx].size(); ++cnt2 ){
            idx1 = m_aminoAcidChains[chainIdIdx][resSeqIdx].FirstAtomIndex() + cnt1;
            idx2 = m_aminoAcidChains[chainIdIdx][resSeqIdx].FirstAtomIndex() + cnt2;
            vislib::math::Vector<float, 3> v1( &this->m_protAtomPos[idx1 * 3]);
            vislib::math::Vector<float, 3> v2( &this->m_protAtomPos[idx2 * 3]);
            if( ( v1 - v2).Length() <
                0.6f * ( this->m_atomTypes[this->m_protAtomData[idx1].TypeIndex()].Radius() +
                this->m_atomTypes[this->m_protAtomData[idx2].TypeIndex()].Radius() ) ) {
                m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
                    protein::CallProteinMovementData::IndexPair ( cnt1, cnt2 ) );
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    return true;
    ///////////////////////////////////////////////////////////////////////////

	// loop over all atoms in this amino acid and fill the map with the names of the atoms
	for( counter = 0; counter < this->tmp_atomEntries[chainIdIdx][resSeqIdx].size(); counter++ )
	{
		name = tmp_atomEntries[chainIdIdx][resSeqIdx][counter].Substring( 12, 4);
		name.TrimSpaces();
		atomNamesMap[name.PeekBuffer()] = counter;
	}

	//////////////////////////////////////
	// try to make backbone connections //
	//////////////////////////////////////

	// check for C-alpha atom
	if( atomNamesMap.find( "CA") != atomNamesMap.end() )
	{
		// try to make C-alpha -- C connection
		if( atomNamesMap.find( "C") != atomNamesMap.end() )
		{
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["C"]));
		}
		// try to make C-alpha -- N connection
		if( atomNamesMap.find( "N") != atomNamesMap.end() )
		{
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["N"]));
		}
	}
	// check for C atom
	if( atomNamesMap.find( "C") != atomNamesMap.end() )
	{
		// try to make C -- O connection
		if( atomNamesMap.find( "O") != atomNamesMap.end() )
		{
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["C"], atomNamesMap["O"]));
		}
		// try to make C -- OXT connection (only available for the last amino acid in the chain)
		if( atomNamesMap.find( "OXT") != atomNamesMap.end() )
		{
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["C"], atomNamesMap["OXT"]));
		}
		// try to make C -- O' connection (only available for the last amino acid in the chain)
		if ( atomNamesMap.find ( "O'" ) != atomNamesMap.end() )
		{
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			    protein::CallProteinData::IndexPair ( atomNamesMap["C"], atomNamesMap["O'"] ) );
		}
		// try to make C -- O'' connection (only available for the last amino acid in the chain)
		if ( atomNamesMap.find ( "O''" ) != atomNamesMap.end() )
		{
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			    protein::CallProteinData::IndexPair ( atomNamesMap["C"], atomNamesMap["O''"] ) );
		}
	}

	////////////////////////////////////////
	// try to make side chain connections //
	////////////////////////////////////////

	// ALA
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ALA"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
	}

	// ARG
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ARG"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD"]));
		if( atomNamesMap.find( "CD") != atomNamesMap.end() && atomNamesMap.find( "NE") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD"], atomNamesMap["NE"]));
		if( atomNamesMap.find( "NE") != atomNamesMap.end() && atomNamesMap.find( "CZ") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["NE"], atomNamesMap["CZ"]));
		if( atomNamesMap.find( "CZ") != atomNamesMap.end() && atomNamesMap.find( "NH1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CZ"], atomNamesMap["NH1"]));
		if( atomNamesMap.find( "CZ") != atomNamesMap.end() && atomNamesMap.find( "NH2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CZ"], atomNamesMap["NH2"]));
	}

	// ASN
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ASN"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "OD1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["OD1"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "ND2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["ND2"]));
	}

	// ASP
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ASP"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "OD1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["OD1"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "OD2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["OD2"]));
	}

	// CYS
	if ( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["CYS"] ||
	     this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["CYX"] ||
	     this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["CYM"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "SG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["SG"]));
	}

	// GLU
	if ( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["GLU"] ||
	     this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["GLH"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD"]));
		if( atomNamesMap.find( "CD") != atomNamesMap.end() && atomNamesMap.find( "OE1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD"], atomNamesMap["OE1"]));
		if( atomNamesMap.find( "CD") != atomNamesMap.end() && atomNamesMap.find( "OE2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD"], atomNamesMap["OE2"]));
	}

	// GLN
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["GLN"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD"]));
		if( atomNamesMap.find( "CD") != atomNamesMap.end() && atomNamesMap.find( "OE1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD"], atomNamesMap["OE1"]));
		if( atomNamesMap.find( "CD") != atomNamesMap.end() && atomNamesMap.find( "NE2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD"], atomNamesMap["NE2"]));
	}

	// GLY --> has no side chain, consists only of backbone

	// HIS
	if ( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["HIS"] ||
	     this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["HID"] ||
	     this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["HIE"] ||
	     this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["HIP"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "ND1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["ND1"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD2"]));
		if( atomNamesMap.find( "ND1") != atomNamesMap.end() && atomNamesMap.find( "CE1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["ND1"], atomNamesMap["CE1"]));
		if( atomNamesMap.find( "CD2") != atomNamesMap.end() && atomNamesMap.find( "NE2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD2"], atomNamesMap["NE2"]));
		if( atomNamesMap.find( "CE1") != atomNamesMap.end() && atomNamesMap.find( "NE2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CE1"], atomNamesMap["NE2"]));
	}

	// ILE
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["ILE"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG1"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG2"]));
		if( atomNamesMap.find( "CG1") != atomNamesMap.end() && atomNamesMap.find( "CD1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG1"], atomNamesMap["CD1"]));
	}

	// LEU
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["LEU"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD1"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD2"]));
	}

	// LYS
	if ( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["LYS"] ||
	     this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["LYN"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD"]));
		if( atomNamesMap.find( "CD") != atomNamesMap.end() && atomNamesMap.find( "CE") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD"], atomNamesMap["CE"]));
		if( atomNamesMap.find( "CE") != atomNamesMap.end() && atomNamesMap.find( "NZ") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CE"], atomNamesMap["NZ"]));
	}

	// MET
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["MET"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "SD") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["SD"]));
		if( atomNamesMap.find( "SD") != atomNamesMap.end() && atomNamesMap.find( "CE") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["SD"], atomNamesMap["CE"]));
	}

	// PHE
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["PHE"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD1"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD2"]));
		if( atomNamesMap.find( "CD1") != atomNamesMap.end() && atomNamesMap.find( "CE1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD1"], atomNamesMap["CE1"]));
		if( atomNamesMap.find( "CD2") != atomNamesMap.end() && atomNamesMap.find( "CE2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD2"], atomNamesMap["CE2"]));
		if( atomNamesMap.find( "CE1") != atomNamesMap.end() && atomNamesMap.find( "CZ") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CE1"], atomNamesMap["CZ"]));
		if( atomNamesMap.find( "CE2") != atomNamesMap.end() && atomNamesMap.find( "CZ") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CE2"], atomNamesMap["CZ"]));
	}

	// PRO
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["PRO"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD"]));
		if( atomNamesMap.find( "CD") != atomNamesMap.end() && atomNamesMap.find( "N") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD"], atomNamesMap["N"]));
	}

	// SER
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["SER"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "OG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["OG"]));
	}

	// THR
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["THR"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "OG1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["OG1"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG2"]));
	}

	// TRP
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["TRP"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD1"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD2"]));
		if( atomNamesMap.find( "CD2") != atomNamesMap.end() && atomNamesMap.find( "CE3") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD2"], atomNamesMap["CE3"]));
		if( atomNamesMap.find( "CD2") != atomNamesMap.end() && atomNamesMap.find( "CE2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD2"], atomNamesMap["CE2"]));
		if( atomNamesMap.find( "CD1") != atomNamesMap.end() && atomNamesMap.find( "NE1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD1"], atomNamesMap["NE1"]));
		if( atomNamesMap.find( "NE1") != atomNamesMap.end() && atomNamesMap.find( "CE2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["NE1"], atomNamesMap["CE2"]));
		if( atomNamesMap.find( "CE3") != atomNamesMap.end() && atomNamesMap.find( "CZ3") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CE3"], atomNamesMap["CZ3"]));
		if( atomNamesMap.find( "CE2") != atomNamesMap.end() && atomNamesMap.find( "CZ2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CE2"], atomNamesMap["CZ2"]));
		if( atomNamesMap.find( "CZ2") != atomNamesMap.end() && atomNamesMap.find( "CH2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CZ2"], atomNamesMap["CH2"]));
		if( atomNamesMap.find( "CZ3") != atomNamesMap.end() && atomNamesMap.find( "CH2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CZ3"], atomNamesMap["CH2"]));
	}

	// TYR
	if ( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["TYR"] ||
	     this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["TYM"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD1"]));
		if( atomNamesMap.find( "CG") != atomNamesMap.end() && atomNamesMap.find( "CD2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CG"], atomNamesMap["CD2"]));
		if( atomNamesMap.find( "CD1") != atomNamesMap.end() && atomNamesMap.find( "CE1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD1"], atomNamesMap["CE1"]));
		if( atomNamesMap.find( "CD2") != atomNamesMap.end() && atomNamesMap.find( "CE2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CD2"], atomNamesMap["CE2"]));
		if( atomNamesMap.find( "CE1") != atomNamesMap.end() && atomNamesMap.find( "CZ") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CE1"], atomNamesMap["CZ"]));
		if( atomNamesMap.find( "CE2") != atomNamesMap.end() && atomNamesMap.find( "CZ") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CE2"], atomNamesMap["CZ"]));
		if( atomNamesMap.find( "CZ") != atomNamesMap.end() && atomNamesMap.find( "OH") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CZ"], atomNamesMap["OH"]));
	}

	// VAL
	if( this->m_aminoAcidChains[chainIdIdx][resSeqIdx].NameIndex() == this->tmp_aminoAcidNameIdx["VAL"] )
	{
		if( atomNamesMap.find( "CA") != atomNamesMap.end() && atomNamesMap.find( "CB") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CA"], atomNamesMap["CB"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG1") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG1"]));
		if( atomNamesMap.find( "CB") != atomNamesMap.end() && atomNamesMap.find( "CG2") != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add( 
				protein::CallProteinData::IndexPair( atomNamesMap["CB"], atomNamesMap["CG2"]));
	}

	//////////////////////////////////////////////////////////////////////
	// Hydrogen atoms
	//////////////////////////////////////////////////////////////////////
	
	// amino group
	if ( atomNamesMap.find ( "N" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "H" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["N"], atomNamesMap["H"] ) );
		if( atomNamesMap.find ( "H1" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["N"], atomNamesMap["H1"] ) );
		if( atomNamesMap.find ( "H2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["N"], atomNamesMap["H2"] ) );
		if( atomNamesMap.find ( "H3" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["N"], atomNamesMap["H3"] ) );
	}
	
	// amino group
	if ( atomNamesMap.find ( "N" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "H" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["N"], atomNamesMap["H"] ) );
		if( atomNamesMap.find ( "H1" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["N"], atomNamesMap["H1"] ) );
		if( atomNamesMap.find ( "H2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["N"], atomNamesMap["H2"] ) );
		if( atomNamesMap.find ( "H3" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["N"], atomNamesMap["H3"] ) );
	}
	
	// A alpha
	if ( atomNamesMap.find ( "CA" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "HA" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CA"], atomNamesMap["HA"] ) );
		if( atomNamesMap.find ( "HA2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CA"], atomNamesMap["HA2"] ) );
		if( atomNamesMap.find ( "HA3" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CA"], atomNamesMap["HA3"] ) );
	}
	
	// B beta
	if ( atomNamesMap.find ( "CB" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "HB" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CB"], atomNamesMap["HB"] ) );
		if( atomNamesMap.find ( "HB1" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CB"], atomNamesMap["HB1"] ) );
		if( atomNamesMap.find ( "HB2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CB"], atomNamesMap["HB2"] ) );
		if( atomNamesMap.find ( "HB3" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CB"], atomNamesMap["HB3"] ) );
	}

	// G gamma
	if ( atomNamesMap.find ( "HG" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "CG" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CG"], atomNamesMap["HG"] ) );
		if( atomNamesMap.find ( "OG" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["OG"], atomNamesMap["HG"] ) );
		if( atomNamesMap.find ( "SG" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["SG"], atomNamesMap["HG"] ) );
	}
	
	if ( atomNamesMap.find ( "OG1" ) != atomNamesMap.end() && atomNamesMap.find ( "HG1" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["OG1"], atomNamesMap["HG1"] ) );
	
	if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "HG2" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["CG"], atomNamesMap["HG2"] ) );
	
	if ( atomNamesMap.find ( "CG" ) != atomNamesMap.end() && atomNamesMap.find ( "HG3" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["CG"], atomNamesMap["HG3"] ) );
	
	if ( atomNamesMap.find ( "CG1" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "HG11" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CG1"], atomNamesMap["HG11"] ) );
		if( atomNamesMap.find ( "HG12" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CG1"], atomNamesMap["HG12"] ) );
		if( atomNamesMap.find ( "HG13" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CG1"], atomNamesMap["HG13"] ) );
	}
	
	if ( atomNamesMap.find ( "CG2" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "HG21" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CG2"], atomNamesMap["HG21"] ) );
		if( atomNamesMap.find ( "HG22" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CG2"], atomNamesMap["HG22"] ) );
		if( atomNamesMap.find ( "HG23" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CG2"], atomNamesMap["HG23"] ) );
	}
	
	// D delta
	if ( atomNamesMap.find ( "HD1" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["HD1"] ) );
		if( atomNamesMap.find ( "ND1" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["ND1"], atomNamesMap["HD1"] ) );
	}
	
	if ( atomNamesMap.find ( "HD2" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "CD" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CD"], atomNamesMap["HD2"] ) );
		if( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["HD2"] ) );
		if( atomNamesMap.find ( "OD2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["OD2"], atomNamesMap["HD2"] ) );
	}
	
	if ( atomNamesMap.find ( "CD" ) != atomNamesMap.end() && atomNamesMap.find ( "HD3" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["CD"], atomNamesMap["HD3"] ) );
	
	if ( atomNamesMap.find ( "CD1" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "HD11" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["HD11"] ) );
		if( atomNamesMap.find ( "HD12" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["HD12"] ) );
		if( atomNamesMap.find ( "HD13" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CD1"], atomNamesMap["HD13"] ) );
	}
	
	if ( atomNamesMap.find ( "HD21" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["HD21"] ) );
		if( atomNamesMap.find ( "ND2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["ND2"], atomNamesMap["HD21"] ) );
	}
	
	if ( atomNamesMap.find ( "HD22" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["HD22"] ) );
		if( atomNamesMap.find ( "ND2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["ND2"], atomNamesMap["HD22"] ) );
	}
	
	if ( atomNamesMap.find ( "CD2" ) != atomNamesMap.end() && atomNamesMap.find ( "HD23" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["CD2"], atomNamesMap["HD23"] ) );
	
	// E epsilon
	if ( atomNamesMap.find ( "NE" ) != atomNamesMap.end() && atomNamesMap.find ( "HE" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["NE"], atomNamesMap["HE"] ) );
	
	if ( atomNamesMap.find ( "HE1" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "CE" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CE"], atomNamesMap["HE1"] ) );
		if( atomNamesMap.find ( "CE1" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CE1"], atomNamesMap["HE1"] ) );
		if( atomNamesMap.find ( "NE1" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NE1"], atomNamesMap["HE1"] ) );
	}
	
	if ( atomNamesMap.find ( "HE2" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "CE" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CE"], atomNamesMap["HE2"] ) );
		if( atomNamesMap.find ( "CE2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CE2"], atomNamesMap["HE2"] ) );
		if( atomNamesMap.find ( "OE2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["OE2"], atomNamesMap["HE2"] ) );
		if( atomNamesMap.find ( "NE2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NE2"], atomNamesMap["HE2"] ) );
	}
	
	if ( atomNamesMap.find ( "HE3" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "CE" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CE"], atomNamesMap["HE3"] ) );
		if( atomNamesMap.find ( "CE3" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CE3"], atomNamesMap["HE3"] ) );
	}
	
	if ( atomNamesMap.find ( "NE2" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "HE21" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NE2"], atomNamesMap["HE21"] ) );
		if( atomNamesMap.find ( "HE22" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NE2"], atomNamesMap["HE22"] ) );
	}
	
	// Z zeta
	if ( atomNamesMap.find ( "CZ" ) != atomNamesMap.end() && atomNamesMap.find ( "HZ" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["CZ"], atomNamesMap["HZ"] ) );
	
	if ( atomNamesMap.find ( "NZ" ) != atomNamesMap.end() && atomNamesMap.find ( "HZ1" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["NZ"], atomNamesMap["HZ1"] ) );
	
	if ( atomNamesMap.find ( "HZ2" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "NZ" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NZ"], atomNamesMap["HZ2"] ) );
		if( atomNamesMap.find ( "CZ2" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CZ2"], atomNamesMap["HZ2"] ) );
	}
	
	if ( atomNamesMap.find ( "HZ3" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "NZ" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NZ"], atomNamesMap["HZ3"] ) );
		if( atomNamesMap.find ( "CZ3" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["CZ3"], atomNamesMap["HZ3"] ) );
	}
	
	// H eta
	if ( atomNamesMap.find ( "OH" ) != atomNamesMap.end() && atomNamesMap.find ( "HH" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["OH"], atomNamesMap["HH"] ) );
	
	if ( atomNamesMap.find ( "CH2" ) != atomNamesMap.end() && atomNamesMap.find ( "HH2" ) != atomNamesMap.end() )
		m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
			protein::CallProteinData::IndexPair ( atomNamesMap["CH2"], atomNamesMap["HH2"] ) );
	
	if ( atomNamesMap.find ( "NH1" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "HH11" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NH1"], atomNamesMap["HH11"] ) );
		if( atomNamesMap.find ( "HH12" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NH1"], atomNamesMap["HH12"] ) );
	}
	
	if ( atomNamesMap.find ( "NH2" ) != atomNamesMap.end() )
	{
		if( atomNamesMap.find ( "HH21" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NH2"], atomNamesMap["HH21"] ) );
		if( atomNamesMap.find ( "HH22" ) != atomNamesMap.end() )
			m_aminoAcidChains[chainIdIdx][resSeqIdx].AccessConnectivity().Add (
				protein::CallProteinData::IndexPair ( atomNamesMap["NH2"], atomNamesMap["HH22"] ) );
	}
	
	// -------------- END Hydrogen atoms --------------
		
	// all possible connections created, return true
	return true;
}


/*
 *protein::ProteinData::ComputeBoundingBox
 */
void protein::ProteinData::ComputeBoundingBox()
{
	unsigned int counter;
	// if no positions are stored --> return
	if( this->m_protAtomData.Count() <= 0 )
		return;
	// get first position
	this->m_minX = this->m_maxX = this->m_protAtomPos[0];
	this->m_minY = this->m_maxY = this->m_protAtomPos[1];
	this->m_minZ = this->m_maxZ = this->m_protAtomPos[2];
	// check all values in the m_protAtomPos-List
	for( counter = 3; counter < this->m_protAtomPos.Count(); counter+=3 )
	{
		// min/max X
		if (this->m_minX > this->m_protAtomPos[counter + 0] )
			this->m_minX = this->m_protAtomPos[counter + 0];
		else if( this->m_maxX < this->m_protAtomPos[counter + 0] )
			this->m_maxX = this->m_protAtomPos[counter + 0];
		// min/max Y
		if (this->m_minY > this->m_protAtomPos[counter + 1] )
			this->m_minY = this->m_protAtomPos[counter + 1];
		else if( this->m_maxY < this->m_protAtomPos[counter + 1] )
			this->m_maxY = this->m_protAtomPos[counter + 1];
		// min/max Z
		if (this->m_minZ > this->m_protAtomPos[counter + 2] )
			this->m_minZ = this->m_protAtomPos[counter + 2];
		else if( this->m_maxZ < this->m_protAtomPos[counter + 2] )
			this->m_maxZ = this->m_protAtomPos[counter + 2];
	}
	// add maximum atom radius to min/max-values to prevent atoms sticking out
	this->m_minX +=-3.0f;
	this->m_maxX += 3.0f;
	this->m_minY +=-3.0f;
	this->m_maxY += 3.0f;
	this->m_minZ +=-3.0f;
	this->m_maxZ += 3.0f;
	// compute maximum dimension
	this->m_maxDimension = this->m_maxX - this->m_minX;
	if( this->m_maxDimension < (this->m_maxY - this->m_minY) )
		this->m_maxDimension = this->m_maxY - this->m_minY;
	if( this->m_maxDimension < (this->m_maxZ - this->m_minZ) )
		this->m_maxDimension = this->m_maxZ - this->m_minZ;
}


/*
 *protein::ProteinData::EstimateDisulfideBonds
 */
void protein::ProteinData::EstimateDisulfideBonds()
{
	unsigned int c1, c2;
	unsigned int idx1, idx2;
	vislib::math::Vector<float, 3> pos1, pos2;
	// loop over all sulfur atoms
	for( c1 = 0; c1 < this->tmp_cysteineSulfurAtoms.Count(); c1++ )
	{
		// get the index of the first sulfur
		idx1 = this->tmp_cysteineSulfurAtoms[c1];
		// get the position of the first sulfur
		pos1.Set( this->m_protAtomPos[idx1*3+0], this->m_protAtomPos[idx1*3+1], this->m_protAtomPos[idx1*3+2]);
		// check the current sulfur against all others
		for( c2 = c1+1; c2 < this->tmp_cysteineSulfurAtoms.Count(); c2++ )
		{
			// get the index of the second sulfur
			idx2 = this->tmp_cysteineSulfurAtoms[c2];
			// get the position of the second sulfur
			pos2.Set( this->m_protAtomPos[idx2*3+0], this->m_protAtomPos[idx2*3+1], this->m_protAtomPos[idx2*3+2]);
			// check if the distance is smaller than the minimum bond length
			if( (pos1 - pos2).Length() < 3.0f )
			{
				// add the index pair to the disulfide bond list
				this->m_dsBonds.Add( protein::CallProteinData::IndexPair( idx1, idx2));
			}
		}
	}

	if( m_dsBonds.Count() > 0 )
		vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO+300, "%s: %d disulfide bonds detected.", this->ClassName(), ( int ) m_dsBonds.Count() );
	else
		vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO+300, "%s: No disulfide bonds detected.", this->ClassName() );
}

