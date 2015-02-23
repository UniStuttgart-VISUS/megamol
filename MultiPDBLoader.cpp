/*
 * PDBLoader.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "MultiPDBLoader.h"
#include "PDBLoader.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/sys/Log.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/types.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/ASCIIFileBuffer.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::core::moldyn;



MultiPDBLoader::MultiPDBLoader(void) : Module(),
        filenameSlot("filename", "The file name"),
        molecularDataOutSlot( "dataout", "The slot providing the loaded data"),
        dataHash(0) {
    // filename slot
    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);
    
    // data out slot for molecular data
    this->molecularDataOutSlot.SetCallback( MolecularDataCall::ClassName(),
            MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData),
            &MultiPDBLoader::getData);
    this->molecularDataOutSlot.SetCallback( MolecularDataCall::ClassName(),
            MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent),
            &MultiPDBLoader::getExtent);
    this->MakeSlotAvailable( &this->molecularDataOutSlot);

}


MultiPDBLoader::~MultiPDBLoader(void) {
    this->Release ();
}

bool MultiPDBLoader::create(void) {
    // nothing to do
    return true;
}

void MultiPDBLoader::release(void) { 
    // clear data arrays
    this->datacall.Clear();
    this->pdb.Clear();
}

bool MultiPDBLoader::getExtent(core::Call& call) {
    MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>( &call );
    if ( dc == NULL ) return false;
    // load aquaria query
    this->assertData();

    if (!this->pdb.Count()) return false;
    
    unsigned int frame = dc->FrameID();
    unsigned int i = frame % this->pdb.Count();
    
    dc->SetFrameID(frame);
    dc->SetFrameCount(this->pdb.Count());
    dc->AccessBoundingBoxes() = this->datacall[i]->AccessBoundingBoxes();
	dc->SetDataHash(this->dataHash);

    return true;

}

bool MultiPDBLoader::getData(core::Call& caller) {
    MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>(&caller);
    if ( dc == NULL ) return false;
    // load all pdb files
    this->assertData();

    if (this->pdb.Count() == 0) return false;

    unsigned int frame = dc->FrameID();
    unsigned int i = frame % this->pdb.Count();
    
    // copy call
    *dc = *this->datacall[i];
    dc->SetFrameID(frame);
    dc->SetFrameCount(this->pdb.Count());
    dc->SetDataHash(this->dataHash);

    return true;
}

void MultiPDBLoader::assertData(void) {
    using vislib::sys::Log;

    if (!this->filenameSlot.IsDirty()) return;  // nothing to do
    this->filenameSlot.ResetDirty();    
    
    // clear data arrays
    this->datacall.Clear();
    this->pdb.Clear();
    
    // temp variables
    vislib::StringA line;

    // try to load the file
    vislib::sys::ASCIIFileBuffer file;
    if (file.LoadFile(this->filenameSlot.Param<param::FilePathParam>()->Value())) {
        // TODO test if line is really a valid file!
        this->pdb.SetCount(file.Count());
        this->datacall.SetCount(file.Count());
        // for each line in the file, try to load the pdb file
        for (unsigned int i = 0; i < file.Count(); i++) {
            this->pdb[i] = new PDBLoader();
            this->datacall[i] = new MolecularDataCall();
            this->pdb[i]->pdbFilenameSlot.Param<core::param::FilePathParam>()->SetValue(file.Line(i), true);
            this->datacall[i]->SetFrameID(0);
            if (this->pdb[i]->getExtent(*this->datacall[i])) {
                this->pdb[i]->getData(*this->datacall[i]);
            }
        }
    }

    this->dataHash++;
}
