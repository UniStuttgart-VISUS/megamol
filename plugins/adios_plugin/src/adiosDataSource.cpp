#include "stdafx.h"
#include "adiosDataSource.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/sys/Log.h"
#include "mmcore/param/FilePathParam.h"

#include <adios2.h>


namepspace megamol {
namespace adios {

adiosDataSource::adiosDataSource : 
  view::AnimDataModule(),
  filename("filename", "The path to the ADIOS-based file to load."),
  getData("getdata", "Slot to request data from this data source."),
  bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f),
  data_hash(0)
 {

    this->filename.SetParameter(new param::FilePathParam(""));
    this->filename.SetUpdateCallback(&MMPLDDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);


    this->getData.SetCallback("MultiParticleDataCall", "GetData", &adiosDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &adiosDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);

}

  adiosDataSource::~adiosDataSource(void) {
    this->Release();
  }

/*
 * adiosDataSource::create
 */
bool adiosDataSource::create(void) {
    return true;
}


/*
 * adiosDDataSource::release
 */
void moldyn::MMPLDDataSource::release(void) {
    this->resetFrameCache();
}


/*
 * adiosDataSource::getDataCallback
 */
bool adiosDataSource::getDataCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (c2 == NULL) return false;

    Frame *f = NULL;
    if (c2 != NULL) {
        f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID(), c2->IsFrameForced()));
        if (f == NULL) return false;
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetFrameID(f->FrameNumber());
        c2->SetDataHash(this->data_hash);
        f->SetData(*c2);
    }

    return true;
}

/*
 * adiosDataSource::getExtentCallback
 */
bool adiosDataSource::getExtentCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);

    if (c2 != NULL) {
        c2->SetFrameCount(this->FrameCount());
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
        c2->SetDataHash(this->data_hash);
        return true;
    }

    return false;
}




} /* end namespace megamol */
} /* end namespace adios */
