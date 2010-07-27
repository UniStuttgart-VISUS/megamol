/*
 * PDBLoader.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "PDBLoader.h"
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
#include "vislib/stringconverter.h"
#include "vislib/stringtokeniser.h"
#include "vislib/ASCIIFileBuffer.h"
#include <ctime>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

/*
 * PDBLoader::Frame::Frame
 */
PDBLoader::Frame::Frame(void) : atomCount( 0),
    maxBFactor(0), minBFactor( 0), 
    maxCharge( 0), minCharge( 0),
    maxOccupancy( 0), minOccupancy( 0) {
    // Intentionally empty
}


/*
 * PDBLoader::Frame::~Frame
 */
PDBLoader::Frame::~Frame(void) {
}

/*
 * PDBLoader::Frame::operator==
 */
bool PDBLoader::Frame::operator==(const PDBLoader::Frame& rhs) {
    // TODO: extend this accordingly
    return true;
}


/*
 * Assign a position to the array of positions.
 */
bool PDBLoader::Frame::SetAtomPosition( unsigned int idx, float x, float y, float z) {
    if( idx >= this->atomCount ) return false;
    this->atomPosition[idx*3+0] = x;
    this->atomPosition[idx*3+1] = y;
    this->atomPosition[idx*3+2] = z;
    return true;
}

/*
 * Assign a position to the array of positions.
 */
bool PDBLoader::Frame::SetAtomBFactor( unsigned int idx, float val) {
    if( idx >= this->atomCount ) return false;
    this->bfactor[idx] = val;
    return true;
}

/*
 * Assign a charge to the array of charges.
 */
bool PDBLoader::Frame::SetAtomCharge( unsigned int idx, float val) {
    if( idx >= this->atomCount ) return false;
    this->charge[idx] = val;
    return true;
}

/*
 * Assign a occupancy to the array of occupancies.
 */
bool PDBLoader::Frame::SetAtomOccupancy( unsigned int idx, float val) {
    if( idx >= this->atomCount ) return false;
    this->occupancy[idx] = val;
    return true;
}

// ======================================================================

/*
 * protein::PDBLoader::PDBLoader
 */
PDBLoader::PDBLoader(void) : Module(),
        filenameSlot( "filename", "The path to the PDB data file to be loaded"),
        dataOutSlot( "dataout", "The slot providing the loaded data"),
        maxFramesSlot( "maxFrames", "The maximum number of frames to be loaded"),
        strideFlagSlot( "strideFlag", "The flag wether STRIDE should be used or not."),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), datahash(0), 
        stride( 0), secStructAvailable( false) {
    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->filenameSlot);

    this->dataOutSlot.SetCallback( CallProteinData::ClassName(), CallProteinData::FunctionName(CallProteinData::CallForGetData), &PDBLoader::getData);
    this->dataOutSlot.SetCallback( CallProteinData::ClassName(), CallProteinData::FunctionName(CallProteinData::CallForGetExtent), &PDBLoader::getExtent);
    this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &PDBLoader::getData);
    this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &PDBLoader::getExtent);
    this->MakeSlotAvailable( &this->dataOutSlot);

    this->maxFramesSlot << new param::IntParam( 500);
    this->MakeSlotAvailable( &this->maxFramesSlot);
    
    this->strideFlagSlot << new param::BoolParam( true);
    this->MakeSlotAvailable( &this->strideFlagSlot);
}


/*
 * protein::PDBLoader::~PDBLoader
 */
PDBLoader::~PDBLoader(void) {
    this->Release ();
}


/*
 * PDBLoader::create
 */
bool PDBLoader::create(void) {
    // intentionally empty
    return true;
}


/*
 * PDBLoader::getData
 */
bool PDBLoader::getData( core::Call& call) {
    using vislib::sys::Log;

    MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    if ( dc->FrameID() >= this->data.Count() ) return false;

    dc->SetDataHash( this->datahash);

    // TODO: assign the data from the loader to the call

    dc->SetAtoms( this->data[dc->FrameID()].AtomCount(), this->atomType.Count(), 
        (unsigned int*)this->atomTypeIdx.PeekElements(), 
        (float*)this->data[dc->FrameID()].AtomPositions(), 
        (MolecularDataCall::AtomType*)this->atomType.PeekElements(),
        (float*)this->data[dc->FrameID()].AtomBFactor(),
        (float*)this->data[dc->FrameID()].AtomCharge(),
        (float*)this->data[dc->FrameID()].AtomOccupancy());
    dc->SetBFactorRange( this->data[dc->FrameID()].MinBFactor(), 
        this->data[dc->FrameID()].MaxBFactor());
    dc->SetChargeRange( this->data[dc->FrameID()].MinCharge(), 
        this->data[dc->FrameID()].MaxCharge());
    dc->SetOccupancyRange( this->data[dc->FrameID()].MinOccupancy(), 
        this->data[dc->FrameID()].MaxOccupancy());
    dc->SetConnections( this->connectivity.Count() / 2, 
        (unsigned int*)this->connectivity.PeekElements());
    dc->SetResidues( this->residue.Count(),
        (MolecularDataCall::Residue**)this->residue.PeekElements());
    dc->SetResidueTypeNames( this->residueTypeName.Count(),
        (vislib::StringA*)this->residueTypeName.PeekElements());
    dc->SetMolecules( this->molecule.Count(), 
        (MolecularDataCall::Molecule*)this->molecule.PeekElements());
    dc->SetChains( this->chain.Count(),
        (MolecularDataCall::Chain*)this->chain.PeekElements());

    if( !this->secStructAvailable && this->strideFlagSlot.Param<param::BoolParam>()->Value() ) {
        time_t t = clock(); // DEBUG
        if( this->stride ) delete this->stride;
        this->stride = new Stride( dc);
        this->stride->WriteToInterface( dc);
        this->secStructAvailable = true;
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Secondary Structure computed via STRIDE in %f seconds.", ( double( clock() - t) / double( CLOCKS_PER_SEC))); // DEBUG
    }

    dc->SetUnlocker( NULL);

    return true;
}


/*
 * PDBLoader::getExtent
 */
bool PDBLoader::getExtent( core::Call& call) {
    MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    dc->AccessBoundingBoxes().Clear();
    dc->AccessBoundingBoxes().SetObjectSpaceBBox( this->bbox);
    dc->AccessBoundingBoxes().SetObjectSpaceClipBox( this->bbox);

    dc->SetFrameCount( vislib::math::Max(1U, 
        static_cast<unsigned int>( this->data.Count())));

    dc->SetDataHash( this->datahash);

    return true;
}


/*
 * PDBLoader::release
 */
void PDBLoader::release(void) {
    this->data.Clear();
}


/*
 * PDBLoader::loadFile
 */
void PDBLoader::loadFile( const vislib::TString& filename) {
    using vislib::sys::Log;

    this->bbox.Set( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    this->data.Clear();
    this->datahash++;

    time_t t = clock(); // DEBUG

    vislib::StringA line;
    unsigned int idx, cnt, atomCnt, lineCnt, frameCnt, resCnt, chainCnt;
    
    t = clock(); // DEBUG

    vislib::sys::ASCIIFileBuffer file;
    vislib::Array<vislib::StringA> atomEntries;
    SIZE_T atomEntriesCapacity = 10000;
    SIZE_T frameCapacity = 10000;
    atomEntries.AssertCapacity( atomEntriesCapacity);

    // try to load the file
    if( file.LoadFile( T2A( filename) ) ) {
        // file successfully loaded, read first frame
        lineCnt = 0;
        while( lineCnt < file.Count() && !line.StartsWith( "END") ) {
            // get the current line from the file
            line = file.Line( lineCnt);
            // store all atom entries
            if( line.StartsWith( "ATOM") ) {
                // ignore alternate locations
                if( line.Substring( 16, 1 ).Equals( " ", false) || 
                    line.Substring( 16, 1 ).Equals( "A", false) ) {
                    // resize atom entry array, if necessary
                    if( atomEntries.Count() == atomEntriesCapacity ) {
                        atomEntriesCapacity += 10000;
                        atomEntries.AssertCapacity( atomEntriesCapacity);
                    }
                    // add atom entry
                    atomEntries.Add( line);
                }
            }
            // next line
            lineCnt++;
        }
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Atom count: %i", atomEntries.Count() ); // DEBUG

        // set the atom count for the first frame
        frameCnt = 0;
        this->data.AssertCapacity( frameCapacity);
        this->data.SetCount( 1);
        this->data[0].SetAtomCount( atomEntries.Count());
        // resize atom type index array
        this->atomTypeIdx.SetCount( atomEntries.Count());
        // set the capacity of the atom type array
        this->atomType.AssertCapacity( atomEntries.Count());
        // set the capacity of the residue array
        this->residue.AssertCapacity( atomEntries.Count());
        
        // parse all atoms of the first frame
        for( atomCnt = 0; atomCnt < atomEntries.Count(); ++atomCnt ) {
            this->parseAtomEntry( atomEntries[atomCnt], atomCnt, frameCnt);
        }
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for parsing first frame: %f", ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG

        // parsed first frame - load all other frames now
        atomCnt = 0;
        while( lineCnt < file.Count() ) {
            // get the current line from the file
            line = file.Line( lineCnt);
            // store all atom entries
            if( line.StartsWith( "ATOM") ) {
                // found new frame, resize data array
                if( atomCnt == 0 ) {
                    frameCnt++;
                    // check if max frame count is reached
                    if( frameCnt > this->maxFramesSlot.Param<param::IntParam>()->Value() ) {
                        break;
                    }
                    if( this->data.Count() == frameCapacity ) {
                        frameCapacity += 10000;
                        this->data.AssertCapacity( frameCapacity);
                    }
                    this->data.SetCount( frameCnt + 1);
                    this->data[frameCnt].SetAtomCount( atomEntries.Count());
                }
                // ignore alternate locations
                if( line.Substring( 16, 1 ).Equals( " ", false) || 
                    line.Substring( 16, 1 ).Equals( "A", false) ) {
                    // add atom position to the current frame
                    this->setAtomPositionToFrame( line, atomCnt, frameCnt);
                    atomCnt++;
                }
            } else if( line.StartsWith( "END") ) {
                atomCnt = 0;
            }
            // next line
            lineCnt++;
        }
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for parsing %i frames: %f", this->data.Count(), ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG

        // all information loaded, delete file
        file.Clear();
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for clearing the file: %f", ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG

        this->molecule.AssertCapacity( this->residue.Count());
        this->chain.AssertCapacity( this->residue.Count());

        unsigned int first, cnt;

        unsigned int firstConIdx;
        // loop over all chains
        for( chainCnt = 0; chainCnt < this->chainFirstRes.Count(); ++chainCnt ) {
            // add new molecule
            if( chainCnt == 0 ) {
                this->molecule.Add( MolecularDataCall::Molecule(  0, 1));
                firstConIdx = 0;
            } else {
                this->molecule.Add( MolecularDataCall::Molecule(
                    this->molecule.Last().FirstResidueIndex()
                    + this->molecule.Last().ResidueCount(), 1));
                firstConIdx = this->connectivity.Count();
            }
            // add new chain
            this->chain.Add( MolecularDataCall::Chain( this->molecule.Count()-1, 1));
            // get the residue range of the current chain
            first = this->chainFirstRes[chainCnt];
            cnt = first + this->chainResCount[chainCnt];
            // loop over all residues in the current chain
            for( resCnt = first; resCnt < cnt; ++resCnt ) {
                // search for connections inside the current residue
                this->MakeResidueConnections( resCnt, 0);
                // search for connections between consecutive residues
                if( ( resCnt + 1) < cnt ) {
                    if( this->MakeResidueConnections( resCnt, resCnt+1, 0) ) {
                        this->molecule.Last().SetPosition(
                            this->molecule.Last().FirstResidueIndex(),
                            this->molecule.Last().ResidueCount() + 1);
                    } else {
                        this->molecule.Last().SetConnectionRange( firstConIdx, ( this->connectivity.Count() - firstConIdx) / 2);
                        firstConIdx = this->connectivity.Count();
                        this->molecule.Add( MolecularDataCall::Molecule( resCnt+1, 1));
                        this->chain.Last().SetPosition(
                            this->chain.Last().FirstMoleculeIndex(),
                            this->chain.Last().MoleculeCount() + 1 );
                    }
                }
            }
            this->molecule.Last().SetConnectionRange( firstConIdx, ( this->connectivity.Count() - firstConIdx) / 2);
        }
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for finding all bonds: %f", ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG

        // search for CA, C, O and N in amino acids
        MolecularDataCall::AminoAcid *aminoacid;
        for( resCnt = 0; resCnt < this->residue.Count(); ++resCnt ) {
            // check if the current residue is an amino acid
            if( this->residue[resCnt]->Identifier() == MolecularDataCall::Residue::AMINOACID ) {
                aminoacid = (MolecularDataCall::AminoAcid*)this->residue[resCnt];
                idx = aminoacid->FirstAtomIndex();
                cnt = idx + aminoacid->AtomCount();
                // loop over all atom of the current amino acid
                for( atomCnt = idx; atomCnt < cnt; ++atomCnt ) {
                    if( this->atomType[this->atomTypeIdx[atomCnt]].Name().Equals( "CA") ) {
                        aminoacid->SetCAlphaIndex( atomCnt);
                    } else if( this->atomType[this->atomTypeIdx[atomCnt]].Name().Equals( "N") ) {
                        aminoacid->SetNIndex( atomCnt);
                    } else if( this->atomType[this->atomTypeIdx[atomCnt]].Name().Equals( "C") ) {
                        aminoacid->SetCCarbIndex( atomCnt);
                    } else if( this->atomType[this->atomTypeIdx[atomCnt]].Name().Equals( "O") ) {
                        aminoacid->SetOIndex( atomCnt);
                    }
                }
            }
        }

        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for loading file %s: %f", T2A( filename), ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG
    } else {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Could not load file %s", T2A( filename)); // DEBUG
    }

}

/*
 * parse one atom entry
 */
void PDBLoader::parseAtomEntry( vislib::StringA &atomEntry, unsigned int atom, 
        unsigned int frame) {
    // temp variables
    vislib::StringA tmpStr;
    vislib::math::Vector<float, 3> pos;
    // set atom position
    pos.Set( float( atof( atomEntry.Substring( 30, 8))),
        float( atof( atomEntry.Substring( 38, 8))),
        float( atof( atomEntry.Substring( 46, 8))));
    this->data[frame].SetAtomPosition( atom, pos.X(), pos.Y(), pos.Z());
    
    // get the name (atom type) of the current ATOM entry
    tmpStr = atomEntry.Substring( 12, 4);
    tmpStr.TrimSpaces();
    // get the radius of the element
    float radius = getElementRadius( tmpStr);
    // get the color of the element
    vislib::math::Vector<unsigned char, 3> color = getElementColor( tmpStr);
    // set the new atom type
    MolecularDataCall::AtomType type( tmpStr, radius, color.X(), color.Y(), 
        color.Z());
    // search for current atom type in atom type array
    INT_PTR atomTypeIdx = atomType.IndexOf( type);
    if( atomTypeIdx == 
            vislib::Array<MolecularDataCall::AtomType>::INVALID_POS ) {
        this->atomTypeIdx[atom] = this->atomType.Count();
        this->atomType.Add( type);
    } else {
        this->atomTypeIdx[atom] = atomTypeIdx;
    }

    // update the bounding box
    vislib::math::Cuboid<float> atomBBox( 
        pos.X() - this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.Y() - this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.Z() - this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.X() + this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.Y() + this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.Z() + this->atomType[this->atomTypeIdx[atom]].Radius());
    if( atom == 0 ) {
        this->bbox = atomBBox;
    } else {
        this->bbox.Union( atomBBox);
    }

    // get chain id
    char tmpChainId = atomEntry.Substring( 21, 1)[0];
    // get the name of the residue
    tmpStr = atomEntry.Substring( 17, 4);
    tmpStr.TrimSpaces();
    vislib::StringA resName = tmpStr;
    unsigned int resTypeIdx;
    // search for current residue type name in the array
    INT_PTR resTypeNameIdx = this->residueTypeName.IndexOf( resName);
    if( resTypeNameIdx ==  vislib::Array<vislib::StringA>::INVALID_POS ) {
        resTypeIdx = this->residueTypeName.Count();
        this->residueTypeName.Add( resName);
    } else {
        resTypeIdx = resTypeNameIdx;
    }
    // get the sequence number of the residue
    tmpStr = atomEntry.Substring( 22, 4);
    tmpStr.TrimSpaces();
    unsigned int newResSeq = float( atof(tmpStr));
    // handle residue
    if( this->residue.Count() == 0 ) {
        // create first residue
        this->resSeq = newResSeq;
        if( this->IsAminoAcid( resName) ) {
            MolecularDataCall::AminoAcid *res = 
                new MolecularDataCall::AminoAcid( atom, 1, 0, 0, 0, 0, atomBBox, resTypeIdx);
            this->residue.Add( (MolecularDataCall::Residue*)res);
        } else {
            MolecularDataCall::Residue *res = 
                new MolecularDataCall::Residue( atom, 1, atomBBox, resTypeIdx);
            this->residue.Add( res);
        }
        // first chain
        this->chainId = tmpChainId;
        this->chainFirstRes.Add( 0);
        this->chainResCount.Add( 1);
    } else if( newResSeq == this->resSeq ) {
        // still the same residue - add one atom
        this->residue.Last()->SetPosition( 
            this->residue.Last()->FirstAtomIndex(), 
            this->residue.Last()->AtomCount() + 1);
        // compute and set the bounding box
        vislib::math::Cuboid<float> resBBox( 
            this->residue.Last()->BoundingBox());
        resBBox.Union( atomBBox);
        this->residue.Last()->SetBoundingBox( resBBox);
    } else if( newResSeq != this->resSeq ) {
        // starting new residue
        this->resSeq = newResSeq;
        if( this->IsAminoAcid( resName) ) {
            MolecularDataCall::AminoAcid *res = 
                new MolecularDataCall::AminoAcid( atom, 1, 0, 0, 0, 0, atomBBox, resTypeIdx);
            this->residue.Add( (MolecularDataCall::Residue*)res);
        } else {
            MolecularDataCall::Residue *res = 
                new MolecularDataCall::Residue( atom, 1, atomBBox, resTypeIdx);
            this->residue.Add( res);
        }
        // elongate existing chain or create new chain
        if( tmpChainId == this->chainId ) {
            this->chainResCount.Last()++;
        } else {
            this->chainId = tmpChainId;
            this->chainFirstRes.Add( this->residue.Count()-1);
            this->chainResCount.Add( 1);
        }
    }

    // get the temperature factor (b-factor)
    tmpStr = atomEntry.Substring( 60, 6);
    tmpStr.TrimSpaces();
    float tempFactor = float( atof( tmpStr));
    if( atom == 0 ) {
        this->data[frame].SetBFactorRange( tempFactor, tempFactor);
    } else {
        if( this->data[frame].MinBFactor() > tempFactor )
            this->data[frame].SetMinBFactor( tempFactor);
        else if( this->data[frame].MaxBFactor() < tempFactor )
            this->data[frame].SetMaxBFactor( tempFactor);
    }
    this->data[frame].SetAtomBFactor( atom, tempFactor);
    
    // get the occupancy
    tmpStr = atomEntry.Substring( 54, 6);
    tmpStr.TrimSpaces();
    float occupancy = float( atof( tmpStr));
    if( atom == 0 ) {
        this->data[frame].SetOccupancyRange( occupancy, occupancy);
    } else {
        if( this->data[frame].MinOccupancy() > occupancy )
            this->data[frame].SetMinOccupancy( occupancy);
        else if( this->data[frame].MaxOccupancy() < occupancy )
            this->data[frame].SetMaxOccupancy( occupancy);
    }
    this->data[frame].SetAtomOccupancy( atom, occupancy);
    
    // get the charge
    tmpStr = atomEntry.Substring( 78, 2);
    tmpStr.TrimSpaces();
    float charge = float( atof( tmpStr));
    if( atom == 0 ) {
        this->data[frame].SetChargeRange( charge, charge);
    } else {
        if( this->data[frame].MinCharge() > charge )
            this->data[frame].SetMinCharge( charge);
        else if( this->data[frame].MaxCharge() < charge )
            this->data[frame].SetMaxCharge( charge);
    }
    this->data[frame].SetAtomCharge( atom, charge);

}

/*
 * Get the radius of the element
 */
float PDBLoader::getElementRadius( vislib::StringA name) {
    // extract the element symbol from the name
    unsigned int cnt = 0;
    vislib::StringA element;
    while( vislib::CharTraitsA::IsDigit( name[cnt]) ) {
        cnt++;
    }

    // --- van der Waals radii ---
    if( name[cnt] == 'H' )
        return 1.2f;
    if( name[cnt] == 'C' )
        return 1.7f;
    if( name[cnt] == 'N' )
        return 1.55f;
    if( name[cnt] == 'O' )
        return 1.52f;
    if( name[cnt] == 'S' )
        return 1.8f;
    if( name[cnt] == 'P' )
        return 1.8f;
    if( name[cnt] == 'C' )
        return 1.7f;

    return 1.5f;
}

/*
 * Get the color of the element
 */
vislib::math::Vector<unsigned char, 3> PDBLoader::getElementColor( vislib::StringA name) {
    // extract the element symbol from the name
    unsigned int cnt = 0;
    vislib::StringA element;
    while( vislib::CharTraitsA::IsDigit( name[cnt]) ) {
        cnt++;
    }

    if( name[cnt] == 'H' ) // white or light grey
        return vislib::math::Vector<unsigned char, 3>( 240, 240, 240);
    if( name[cnt] == 'C' ) // (dark) grey or green
        return vislib::math::Vector<unsigned char, 3>( 125, 125, 125);
        //return vislib::math::Vector<unsigned char, 3>( 90, 175, 50);
    if( name[cnt] == 'N' ) // blue
        //return vislib::math::Vector<unsigned char, 3>( 37, 136, 195);
        return vislib::math::Vector<unsigned char, 3>( 37, 136, 195);
    if( name[cnt] == 'O' ) // red
        //return vislib::math::Vector<unsigned char, 3>( 250, 94, 82);
        return vislib::math::Vector<unsigned char, 3>( 206, 34, 34);
    if( name[cnt] == 'S' ) // yellow
        //return vislib::math::Vector<unsigned char, 3>( 250, 230, 50);
        return vislib::math::Vector<unsigned char, 3>( 255, 215, 0);
    if( name[cnt] == 'P' ) // orange
        return vislib::math::Vector<unsigned char, 3>( 255, 128, 64);

    return vislib::math::Vector<unsigned char, 3>( 191, 191, 191);
}

/*
 * set the position of the current atom entry to the frame
 */
void PDBLoader::setAtomPositionToFrame( vislib::StringA &atomEntry, unsigned int atom, 
        unsigned int frame) {
    // temp variables
    vislib::StringA tmpStr;
    vislib::math::Vector<float, 3> pos;
    // set atom position
    pos.Set( float( atof( atomEntry.Substring( 30, 8))),
        float( atof( atomEntry.Substring( 38, 8))),
        float( atof( atomEntry.Substring( 46, 8))));
    this->data[frame].SetAtomPosition( atom, pos.X(), pos.Y(), pos.Z());
    
    // update bounding box
    vislib::math::Cuboid<float> atomBBox( 
        pos.X() - this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.Y() - this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.Z() - this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.X() + this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.Y() + this->atomType[this->atomTypeIdx[atom]].Radius(), 
        pos.Z() + this->atomType[this->atomTypeIdx[atom]].Radius());
    this->bbox.Union( atomBBox);

    // get the temperature factor (b-factor)
    tmpStr = atomEntry.Substring( 60, 6);
    tmpStr.TrimSpaces();
    float tempFactor = float( atof( tmpStr));
    if( atom == 0 ) {
        this->data[frame].SetBFactorRange( tempFactor, tempFactor);
    } else {
        if( this->data[frame].MinBFactor() > tempFactor )
            this->data[frame].SetMinBFactor( tempFactor);
        else if( this->data[frame].MaxBFactor() < tempFactor )
            this->data[frame].SetMaxBFactor( tempFactor);
    }
    
    // get the occupancy
    tmpStr = atomEntry.Substring( 54, 6);
    tmpStr.TrimSpaces();
    float occupancy = float( atof( tmpStr));
    if( atom == 0 ) {
        this->data[frame].SetOccupancyRange( occupancy, occupancy);
    } else {
        if( this->data[frame].MinOccupancy() > occupancy )
            this->data[frame].SetMinOccupancy( occupancy);
        else if( this->data[frame].MaxOccupancy() < occupancy )
            this->data[frame].SetMaxOccupancy( occupancy);
    }
    
    // get the charge
    tmpStr = atomEntry.Substring( 78, 2);
    tmpStr.TrimSpaces();
    float charge = float( atof( tmpStr));
    if( atom == 0 ) {
        this->data[frame].SetChargeRange( charge, charge);
    } else {
        if( this->data[frame].MinCharge() > charge )
            this->data[frame].SetMinCharge( charge);
        else if( this->data[frame].MaxCharge() < charge )
            this->data[frame].SetMaxCharge( charge);
    }
    
}

/*
 * Search for connections in the given residue and add them to the
 * global connection array.
 */
void PDBLoader::MakeResidueConnections( unsigned int resIdx, unsigned int frame) {
    // check bounds
    if( resIdx >= this->residue.Count() ) return;
    if( frame >= this->data.Count() ) return;
    // get capacity of connectivity array
    SIZE_T connectionCapacity = this->connectivity.Capacity();
    // increase capacity of connectivity array, of necessary
    if( this->connectivity.Count() == this->connectivity.Capacity() ) {
        connectionCapacity += 10000;
        this->connectivity.AssertCapacity( connectionCapacity);
    }
    // loop over all atoms in the residue
    unsigned int cnt0, cnt1, atomIdx0, atomIdx1;
    vislib::math::Vector<float, 3> atomPos0, atomPos1;
    for( cnt0 = 0; cnt0 < this->residue[resIdx]->AtomCount() - 1; ++cnt0 ) {
        for( cnt1 = cnt0 + 1; cnt1 < this->residue[resIdx]->AtomCount(); ++cnt1 ) {
            // get atom indices
            atomIdx0 = this->residue[resIdx]->FirstAtomIndex() + cnt0;
            atomIdx1 = this->residue[resIdx]->FirstAtomIndex() + cnt1;
            // get atom positions
            atomPos0.Set( this->data[frame].AtomPositions()[3*atomIdx0+0],
                this->data[frame].AtomPositions()[3*atomIdx0+1],
                this->data[frame].AtomPositions()[3*atomIdx0+2]);
            atomPos1.Set( this->data[frame].AtomPositions()[3*atomIdx1+0],
                this->data[frame].AtomPositions()[3*atomIdx1+1],
                this->data[frame].AtomPositions()[3*atomIdx1+2]);
            // check distance
            if( ( atomPos0 - atomPos1).Length() < 
                0.58f * ( this->atomType[this->atomTypeIdx[atomIdx0]].Radius() +
                this->atomType[this->atomTypeIdx[atomIdx1]].Radius() ) ) {
                // add connection
                this->connectivity.Add( atomIdx0);
                this->connectivity.Add( atomIdx1);
            }
        }
    }
}

/*
 * Search for connections between two residues.
 */
bool PDBLoader::MakeResidueConnections( unsigned int resIdx0, unsigned int resIdx1, unsigned int frame) {
    // flag wether the two residues are connected
    bool connected = false;
    // check bounds
    if( resIdx0 >= this->residue.Count() ) return connected;
    if( resIdx1 >= this->residue.Count() ) return connected;
    if( frame >= this->data.Count() ) return connected;

    // get capacity of connectivity array
    SIZE_T connectionCapacity = this->connectivity.Capacity();
    // increase capacity of connectivity array, of necessary
    if( this->connectivity.Count() == this->connectivity.Capacity() ) {
        connectionCapacity += 10000;
        this->connectivity.AssertCapacity( connectionCapacity);
    }

    // loop over all atoms in the residue
    unsigned int cnt0, cnt1, atomIdx0, atomIdx1;
    vislib::math::Vector<float, 3> atomPos0, atomPos1;
    for( cnt0 = 0; cnt0 < this->residue[resIdx0]->AtomCount(); ++cnt0 ) {
        for( cnt1 = 0; cnt1 < this->residue[resIdx1]->AtomCount(); ++cnt1 ) {
            // get atom indices
            atomIdx0 = this->residue[resIdx0]->FirstAtomIndex() + cnt0;
            atomIdx1 = this->residue[resIdx1]->FirstAtomIndex() + cnt1;
            // get atom positions
            atomPos0.Set( this->data[frame].AtomPositions()[3*atomIdx0+0],
                this->data[frame].AtomPositions()[3*atomIdx0+1],
                this->data[frame].AtomPositions()[3*atomIdx0+2]);
            atomPos1.Set( this->data[frame].AtomPositions()[3*atomIdx1+0],
                this->data[frame].AtomPositions()[3*atomIdx1+1],
                this->data[frame].AtomPositions()[3*atomIdx1+2]);
            // check distance
            if( ( atomPos0 - atomPos1).Length() < 
                0.58f * ( this->atomType[this->atomTypeIdx[atomIdx0]].Radius() +
                this->atomType[this->atomTypeIdx[atomIdx1]].Radius() ) ) {
                // add connection
                this->connectivity.Add( atomIdx0);
                this->connectivity.Add( atomIdx1);
                connected = true;
            }
        }
    }
    return connected;
}

/*
 * Check if the residue is an amino acid.
 */
bool PDBLoader::IsAminoAcid( vislib::StringA resName ) {
    if( resName.Equals( "ALA" ) |
        resName.Equals( "ARG" ) |
        resName.Equals( "ASN" ) |
        resName.Equals( "ASP" ) |
        resName.Equals( "CYS" ) |
        resName.Equals( "GLN" ) |
        resName.Equals( "GLU" ) |
        resName.Equals( "GLY" ) |
        resName.Equals( "HIS" ) |
        resName.Equals( "ILE" ) |
        resName.Equals( "LEU" ) |
        resName.Equals( "LYS" ) |
        resName.Equals( "MET" ) |
        resName.Equals( "PHE" ) |
        resName.Equals( "PRO" ) |
        resName.Equals( "SER" ) |
        resName.Equals( "THR" ) |
        resName.Equals( "TRP" ) |
        resName.Equals( "TYR" ) |
        resName.Equals( "VAL" ) | 
        resName.Equals( "ASH" ) |
        resName.Equals( "CYX" ) |
        resName.Equals( "CYM" ) |
        resName.Equals( "GLH" ) |
        resName.Equals( "HID" ) |
        resName.Equals( "HIE" ) |
        resName.Equals( "HIP" ) |
        resName.Equals( "LYN" ) |
        resName.Equals( "TYM" ) )
        return true;
    return false;
}