#include "stdafx.h"
#include "BindingSiteDataSource.h"

#include "BindingSiteCall.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "param/FilePathParam.h"
#include "vislib/BufferedFile.h"
#include "vislib/sysfunctions.h"
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

    //MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>( &call);
    //if ( dc == NULL ) return false;

    if ( this->pdbFilenameSlot.IsDirty() ) {
        this->pdbFilenameSlot.ResetDirty();
        //this->loadFile( this->pdbFilenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    //dc->SetDataHash( this->datahash);

    return true;
}
