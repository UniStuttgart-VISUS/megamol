/*
 * CartoonDataSource.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "CartoonDataSource.h"
#include "param/FilePathParam.h"
#include "param/IntParam.h"
#include "param/BoolParam.h"
#include "param/StringParam.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include <omp.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::misc;
using namespace megamol::protein;

/*
 * protein::CartoonDataSource::CartoonDataSource
 */
CartoonDataSource::CartoonDataSource(void) : Module(),
        filenameSlot( "filename", "The path to the PDB data file to be loaded"),
        dataOutSlot( "dataout", "The slot providing the loaded data"),
        strideFlagSlot( "strideFlag", "The flag wether STRIDE should be used or not."),
        molDataCallerSlot( "molData", "The slot providing the data of the molecule."),
        datahash(0), ellipCurves(), rectCurves(), tubeCurves() {
    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->filenameSlot);
    
    // ExtBezierDataCall
    this->dataOutSlot.SetCallback(ExtBezierDataCall::ClassName(), "GetData",
        &CartoonDataSource::getData);
    this->dataOutSlot.SetCallback(ExtBezierDataCall::ClassName(), "GetExtent",
        &CartoonDataSource::getExtent);
    // BezierDataCall
    this->dataOutSlot.SetCallback(BezierDataCall::ClassName(), "GetData",
        &CartoonDataSource::getData);
    this->dataOutSlot.SetCallback(BezierDataCall::ClassName(), "GetExtent",
        &CartoonDataSource::getExtent);
    // make available
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->strideFlagSlot << new param::BoolParam( true);
    this->MakeSlotAvailable( &this->strideFlagSlot);

    // molecule
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);
    
}


/*
 * protein::CartoonDataSource::~CartoonDataSource
 */
CartoonDataSource::~CartoonDataSource(void) {
    this->Release ();
}


/*
 * CartoonDataSource::create
 */
bool CartoonDataSource::create(void) {
    return true;
}


/*
 * CartoonDataSource::getData
 */
bool CartoonDataSource::getData( core::Call& call) {

    ExtBezierDataCall *dc = dynamic_cast<ExtBezierDataCall*>( &call);
    BezierDataCall *bdc = dynamic_cast<BezierDataCall*>( &call);
    if ( dc == NULL && bdc == NULL ) return false;

    // try to load the input file
    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        if( !this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value()) )
            return false;
    }

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    
    // compute bezier points
    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;

    // set values to data call
    if( dc ) {
        // using ExtBezierDataCall
        this->ComputeBezierPoints( mol);

        dc->SetData(static_cast<unsigned int>(this->ellipCurves.Count()),
            static_cast<unsigned int>(this->rectCurves.Count()),
            this->ellipCurves.PeekElements(),
            this->rectCurves.PeekElements());

        if( dc->FrameID() >= mol->FrameCount() ) return false;

        dc->SetDataHash( mol->DataHash());
        
        if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;
        dc->AccessBoundingBoxes().Clear();
        dc->AccessBoundingBoxes() = mol->AccessBoundingBoxes();

        dc->SetUnlocker( NULL);
    } else {
        // using BezierDataCall
        this->ComputeBezierPointsTubes( mol);

        bdc->SetData(static_cast<unsigned int>(this->tubeCurves.Count()),
            this->tubeCurves.PeekElements());

        if( bdc->FrameID() >= mol->FrameCount() ) return false;

        bdc->SetDataHash( mol->DataHash());
        
        if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;
        bdc->AccessBoundingBoxes().Clear();
        bdc->AccessBoundingBoxes() = mol->AccessBoundingBoxes();

        bdc->SetUnlocker( NULL);
    }

    return true;
}


/*
 * CartoonDataSource::getExtent
 */
bool CartoonDataSource::getExtent( core::Call& call) {
    ExtBezierDataCall *dc = dynamic_cast<ExtBezierDataCall*>( &call);
    BezierDataCall *bdc = dynamic_cast<BezierDataCall*>( &call);
    if( dc == NULL && bdc == NULL )
        return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        if( !this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value()) )
            return false;
    }

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol ) {
        // get extends of molecule
        if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;
        if( dc ) {
            dc->AccessBoundingBoxes().Clear();
            dc->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
        } else {
            bdc->AccessBoundingBoxes().Clear();
            bdc->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
        }
    }

    // set frame count
    if( dc ) {
        dc->SetFrameCount( vislib::math::Max(1U, mol->FrameCount()));
        dc->SetDataHash( this->datahash);
    } else {
        bdc->SetFrameCount( vislib::math::Max(1U, mol->FrameCount()));
        bdc->SetDataHash( this->datahash);
    }

    return true;
}


/*
 * CartoonDataSource::release
 */
void CartoonDataSource::release(void) {
}


/*
 * CartoonDataSource::loadFile
 */
bool CartoonDataSource::loadFile( const vislib::TString& filename) {
    // use namespace Log
    using vislib::sys::Log;

    // variables for parameter transfer
    vislib::StringA paramSlotName;
    param::ParamSlot *paramSlot;

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    // set parameter slots of the molecule
    if( mol ) {
        paramSlotName = "";
        paramSlot = 0;
        // get and set filename param
        paramSlotName = mol->PeekCalleeSlot()->Parent()->FullName();
        paramSlotName += "::filename";
        paramSlot = dynamic_cast<param::ParamSlot*>( this->FindNamedObject( paramSlotName, true));
        if( paramSlot ) {
            paramSlot->Param<param::FilePathParam>()->SetValue( filename);
        }
        // get and set stride param
        paramSlotName = mol->PeekCalleeSlot()->Parent()->FullName();
        paramSlotName += "::strideFlag";
        paramSlot = dynamic_cast<param::ParamSlot*>( this->FindNamedObject( paramSlotName, true));
        if( paramSlot ) {
            paramSlot->Param<param::BoolParam>()->SetValue( this->strideFlagSlot.Param<param::BoolParam>()->Value());
        }
        // all parameters set, execute the data call
        if( !(*mol)(MolecularDataCall::CallForGetData)) {
            Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Could not load molecule file."); // DEBUG
            return false;
        }
    } else {
        // could not get MolecularDataCall
        return false;
    }

    return true;
}

/*
 * Compute the bezier points
 */
void CartoonDataSource::ComputeBezierPoints( const MolecularDataCall *mol) {
    this->ellipCurves.Clear();
    this->rectCurves.Clear();

    // reserve memory
    unsigned int numEllipCurves = 0;
    this->ellipCurves.SetCount( mol->ResidueCount());
    unsigned int numRectCurves = 0;
    this->rectCurves.SetCount( mol->ResidueCount());

    MolecularDataCall::AminoAcid *aa0;
    MolecularDataCall::AminoAcid *aa1;
    MolecularDataCall::AminoAcid *aa2;
    MolecularDataCall::AminoAcid *aa3;
    vislib::math::Vector<float, 3> pos0, pos1, pos2, pos3, cp1, cp2, dir0, dir1, dir2, dir3;
    bool flip = false;

    unsigned int firstRes, resCount, resIdx;
    unsigned int currentSecS, secSFirstRes, secSResCnt;
    for( unsigned int cntMol = 0; cntMol < mol->MoleculeCount(); ++cntMol ) {
        // get bounds of the current molecule
        firstRes = mol->Molecules()[cntMol].FirstResidueIndex();
        resCount = mol->Molecules()[cntMol].ResidueCount();
        currentSecS = mol->Molecules()[cntMol].FirstSecStructIndex();
        secSFirstRes = mol->SecondaryStructures()[currentSecS].FirstAminoAcidIndex();
        secSResCnt = mol->SecondaryStructures()[currentSecS].AminoAcidCount();
        // check bounds
        if( resCount < 4 ) continue;
        // loop over all residues of the current molecule
        for( unsigned int cntRes = 1; cntRes < resCount - 3; ++cntRes ) {
            resIdx = firstRes + cntRes;

            while( resIdx > ( secSFirstRes + secSResCnt) ) {
                currentSecS++;
                secSFirstRes = mol->SecondaryStructures()[currentSecS].FirstAminoAcidIndex();
                secSResCnt = mol->SecondaryStructures()[currentSecS].AminoAcidCount();
            }

            aa0 = aa1 = aa2 = aa3 = 0;
            aa0 = dynamic_cast<MolecularDataCall::AminoAcid*>( mol->Residues()[resIdx-1]);
            aa1 = dynamic_cast<MolecularDataCall::AminoAcid*>( mol->Residues()[resIdx]);
            aa2 = dynamic_cast<MolecularDataCall::AminoAcid*>( mol->Residues()[resIdx+1]);
            aa3 = dynamic_cast<MolecularDataCall::AminoAcid*>( mol->Residues()[resIdx+2]);
            if( !aa0 || !aa1 || !aa2 || !aa3 ) continue;

            pos0.Set( mol->AtomPositions()[3*aa0->CAlphaIndex()],
                mol->AtomPositions()[3*aa0->CAlphaIndex()+1],
                mol->AtomPositions()[3*aa0->CAlphaIndex()+2]);
            pos1.Set( mol->AtomPositions()[3*aa1->CAlphaIndex()],
                mol->AtomPositions()[3*aa1->CAlphaIndex()+1],
                mol->AtomPositions()[3*aa1->CAlphaIndex()+2]);
            pos2.Set( mol->AtomPositions()[3*aa2->CAlphaIndex()],
                mol->AtomPositions()[3*aa2->CAlphaIndex()+1],
                mol->AtomPositions()[3*aa2->CAlphaIndex()+2]);
            pos3.Set( mol->AtomPositions()[3*aa3->CAlphaIndex()],
                mol->AtomPositions()[3*aa3->CAlphaIndex()+1],
                mol->AtomPositions()[3*aa3->CAlphaIndex()+2]);
            cp1 = ( pos2 - pos0) * 0.3f + pos1;
            cp2 = ( pos1 - pos3) * 0.3f + pos2;
            dir0.Set( mol->AtomPositions()[3*aa1->OIndex()] - mol->AtomPositions()[3*aa1->CCarbIndex()],
                mol->AtomPositions()[3*aa1->OIndex()+1] - mol->AtomPositions()[3*aa1->CCarbIndex()+1],
                mol->AtomPositions()[3*aa1->OIndex()+2] - mol->AtomPositions()[3*aa1->CCarbIndex()+2] );
            dir0.Normalise();
            if( cntRes > 1 && dir0.Angle( dir3) > 0.5 * vislib::math::PI_DOUBLE ) {
                dir0 *= -1.0f;
            }
            dir3.Set( mol->AtomPositions()[3*aa2->OIndex()] - mol->AtomPositions()[3*aa2->CCarbIndex()],
                mol->AtomPositions()[3*aa2->OIndex()+1] - mol->AtomPositions()[3*aa2->CCarbIndex()+1],
                mol->AtomPositions()[3*aa2->OIndex()+2] - mol->AtomPositions()[3*aa2->CCarbIndex()+2] );
            dir3.Normalise();
            if( dir0.Angle( dir3) > 0.5 * vislib::math::PI_DOUBLE ) {
                dir3 *= -1.0f;
            }
            dir1 = 2.0f * dir0 + dir3;
            dir1.Normalise();
            dir2 = dir0 + 2.0f * dir3;
            dir2.Normalise();

            float widthY;
            if( mol->SecondaryStructures()[currentSecS].Type() == MolecularDataCall::SecStructure::TYPE_SHEET ) {
                widthY = 0.9f;
                if( ( resIdx + 1) > ( secSFirstRes + secSResCnt) ) {
                    widthY = 0.3f;
                }
                this->rectCurves[numRectCurves][0].Set( pos1.X(), pos1.Y(), pos1.Z(), 
                    dir0.X(), dir0.Y(), dir0.Z(), 
                    0.9f, 0.3f, 
                    255, 0, 0);
                this->rectCurves[numRectCurves][1].Set( cp1.X(), cp1.Y(), cp1.Z(), 
                    dir1.X(), dir1.Y(), dir1.Z(), 
                    0.9f, 0.3f, 
                    255, 255, 0);
                this->rectCurves[numRectCurves][2].Set( cp2.X(), cp2.Y(), cp2.Z(), 
                    dir2.X(), dir2.Y(), dir2.Z(), 
                    0.9f, 0.3f, 
                    0, 255, 255);
                this->rectCurves[numRectCurves][3].Set( pos2.X(), pos2.Y(), pos2.Z(), 
                    dir3.X(), dir3.Y(), dir3.Z(), 
                    widthY, 0.3f, 
                    0, 0, 255);
                numRectCurves++;
            } else if( mol->SecondaryStructures()[currentSecS].Type() == MolecularDataCall::SecStructure::TYPE_HELIX ) {
                widthY = 0.9f;
                if( ( resIdx + 1) > ( secSFirstRes + secSResCnt) ) {
                    widthY = 0.3f;
                }
                this->ellipCurves[numEllipCurves][0].Set( pos1.X(), pos1.Y(), pos1.Z(), 
                    dir0.X(), dir0.Y(), dir0.Z(), 
                    0.9f, 0.3f, 
                    255, 0, 0);
                this->ellipCurves[numEllipCurves][1].Set( cp1.X(), cp1.Y(), cp1.Z(), 
                    dir1.X(), dir1.Y(), dir1.Z(), 
                    0.9f, 0.3f, 
                    255, 255, 0);
                this->ellipCurves[numEllipCurves][2].Set( cp2.X(), cp2.Y(), cp2.Z(), 
                    dir2.X(), dir2.Y(), dir2.Z(), 
                    0.9f, 0.3f, 
                    0, 255, 255);
                this->ellipCurves[numEllipCurves][3].Set( pos2.X(), pos2.Y(), pos2.Z(), 
                    dir3.X(), dir3.Y(), dir3.Z(), 
                    widthY, 0.3f, 
                    0, 0, 255);
                numEllipCurves++;
            } else {
                widthY = 0.3f;
                if( ( resIdx + 1) > ( secSFirstRes + secSResCnt) && 
                    mol->SecondaryStructures()[currentSecS + 1].Type() == MolecularDataCall::SecStructure::TYPE_HELIX) {
                    widthY = 0.9f;
                }
                this->ellipCurves[numEllipCurves][0].Set( pos1.X(), pos1.Y(), pos1.Z(), 
                    dir0.X(), dir0.Y(), dir0.Z(), 
                    0.3f, 0.3f, 
                    255, 0, 0);
                this->ellipCurves[numEllipCurves][1].Set( cp1.X(), cp1.Y(), cp1.Z(), 
                    dir1.X(), dir1.Y(), dir1.Z(), 
                    0.3f, 0.3f, 
                    255, 255, 0);
                this->ellipCurves[numEllipCurves][2].Set( cp2.X(), cp2.Y(), cp2.Z(), 
                    dir2.X(), dir2.Y(), dir2.Z(), 
                    0.3f, 0.3f, 
                    0, 255, 255);
                this->ellipCurves[numEllipCurves][3].Set( pos2.X(), pos2.Y(), pos2.Z(), 
                    dir3.X(), dir3.Y(), dir3.Z(), 
                    widthY, 0.3f, 
                    0, 0, 255);
                numEllipCurves++;
            }
        }
    }

}

/*
 * Compute the bezier points for tubes
 */
void CartoonDataSource::ComputeBezierPointsTubes( const MolecularDataCall *mol) {
    this->tubeCurves.Clear();

    // reserve memory
    unsigned int numTubeCurves = 0;
    this->tubeCurves.SetCount( mol->ResidueCount());

    MolecularDataCall::AminoAcid *aa0;
    MolecularDataCall::AminoAcid *aa1;
    MolecularDataCall::AminoAcid *aa2;
    MolecularDataCall::AminoAcid *aa3;
    vislib::math::Vector<float, 3> pos0, pos1, pos2, pos3, cp1, cp2;
    bool flip = false;

    unsigned int firstRes, resCount, resIdx;
    unsigned int currentSecS, secSFirstRes, secSResCnt;
    for( unsigned int cntMol = 0; cntMol < mol->MoleculeCount(); ++cntMol ) {
        // get bounds of the current molecule
        firstRes = mol->Molecules()[cntMol].FirstResidueIndex();
        resCount = mol->Molecules()[cntMol].ResidueCount();
        currentSecS = mol->Molecules()[cntMol].FirstSecStructIndex();
        secSFirstRes = mol->SecondaryStructures()[currentSecS].FirstAminoAcidIndex();
        secSResCnt = mol->SecondaryStructures()[currentSecS].AminoAcidCount();
        // check bounds
        if( resCount < 4 ) continue;
        // loop over all residues of the current molecule
        for( unsigned int cntRes = 1; cntRes < resCount - 3; ++cntRes ) {
            resIdx = firstRes + cntRes;

            while( resIdx > ( secSFirstRes + secSResCnt) ) {
                currentSecS++;
                secSFirstRes = mol->SecondaryStructures()[currentSecS].FirstAminoAcidIndex();
                secSResCnt = mol->SecondaryStructures()[currentSecS].AminoAcidCount();
            }

            aa0 = aa1 = aa2 = aa3 = 0;
            aa0 = dynamic_cast<MolecularDataCall::AminoAcid*>( mol->Residues()[resIdx-1]);
            aa1 = dynamic_cast<MolecularDataCall::AminoAcid*>( mol->Residues()[resIdx]);
            aa2 = dynamic_cast<MolecularDataCall::AminoAcid*>( mol->Residues()[resIdx+1]);
            aa3 = dynamic_cast<MolecularDataCall::AminoAcid*>( mol->Residues()[resIdx+2]);
            if( !aa0 || !aa1 || !aa2 || !aa3 ) continue;

            pos0.Set( mol->AtomPositions()[3*aa0->CAlphaIndex()],
                mol->AtomPositions()[3*aa0->CAlphaIndex()+1],
                mol->AtomPositions()[3*aa0->CAlphaIndex()+2]);
            pos1.Set( mol->AtomPositions()[3*aa1->CAlphaIndex()],
                mol->AtomPositions()[3*aa1->CAlphaIndex()+1],
                mol->AtomPositions()[3*aa1->CAlphaIndex()+2]);
            pos2.Set( mol->AtomPositions()[3*aa2->CAlphaIndex()],
                mol->AtomPositions()[3*aa2->CAlphaIndex()+1],
                mol->AtomPositions()[3*aa2->CAlphaIndex()+2]);
            pos3.Set( mol->AtomPositions()[3*aa3->CAlphaIndex()],
                mol->AtomPositions()[3*aa3->CAlphaIndex()+1],
                mol->AtomPositions()[3*aa3->CAlphaIndex()+2]);
            cp1 = ( pos2 - pos0) * 0.3f + pos1;
            cp2 = ( pos1 - pos3) * 0.3f + pos2;

            float widthY, arrowpoint0, arrowpoint1;
            if( mol->SecondaryStructures()[currentSecS].Type() == MolecularDataCall::SecStructure::TYPE_SHEET ) {
                arrowpoint0 = arrowpoint1 = 0.9f;
                widthY = 0.9f;
                if( ( resIdx + 1) > ( secSFirstRes + secSResCnt) ) {
                    widthY = 0.3f;
                    arrowpoint0 = 1.2f;
                    arrowpoint1 = 0.6f;
                }
                this->tubeCurves[numTubeCurves][0].Set( pos1.X(), pos1.Y(), pos1.Z(), 
                    arrowpoint0,
                    0, 0, 255);
                this->tubeCurves[numTubeCurves][1].Set( cp1.X(), cp1.Y(), cp1.Z(), 
                    0.9f,
                    0, 0, 255);
                this->tubeCurves[numTubeCurves][2].Set( cp2.X(), cp2.Y(), cp2.Z(), 
                    arrowpoint1,
                    0, 0, 255);
                this->tubeCurves[numTubeCurves][3].Set( pos2.X(), pos2.Y(), pos2.Z(), 
                    widthY,
                    0, 0, 255);
                numTubeCurves++;
            } else if( mol->SecondaryStructures()[currentSecS].Type() == MolecularDataCall::SecStructure::TYPE_HELIX ) {
                widthY = 0.9f;
                if( ( resIdx + 1) > ( secSFirstRes + secSResCnt) ) {
                    widthY = 0.3f;
                }
                this->tubeCurves[numTubeCurves][0].Set( pos1.X(), pos1.Y(), pos1.Z(),  
                    0.9f,
                    255, 0, 0);
                this->tubeCurves[numTubeCurves][1].Set( cp1.X(), cp1.Y(), cp1.Z(), 
                    0.9f,
                    255, 0, 0);
                this->tubeCurves[numTubeCurves][2].Set( cp2.X(), cp2.Y(), cp2.Z(), 
                    0.9f,
                    255, 0, 0);
                this->tubeCurves[numTubeCurves][3].Set( pos2.X(), pos2.Y(), pos2.Z(), 
                    widthY,
                    255, 0, 0);
                numTubeCurves++;
            } else {
                widthY = 0.3f;
                if( ( resIdx + 1) > ( secSFirstRes + secSResCnt) && 
                    mol->SecondaryStructures()[currentSecS + 1].Type() == MolecularDataCall::SecStructure::TYPE_HELIX) {
                    widthY = 0.9f;
                }
                this->tubeCurves[numTubeCurves][0].Set( pos1.X(), pos1.Y(), pos1.Z(), 
                    0.3f,
                    200, 200, 200);
                this->tubeCurves[numTubeCurves][1].Set( cp1.X(), cp1.Y(), cp1.Z(), 
                    0.3f,
                    200, 200, 200);
                this->tubeCurves[numTubeCurves][2].Set( cp2.X(), cp2.Y(), cp2.Z(), 
                    0.3f,
                    200, 200, 200);
                this->tubeCurves[numTubeCurves][3].Set( pos2.X(), pos2.Y(), pos2.Z(), 
                    widthY,
                    200, 200, 200);
                numTubeCurves++;
            }
        }
    }

}
