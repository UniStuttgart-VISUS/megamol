#include "stdafx.h"
#include "MainDirAnalysis.h"

#include "geometry_calls/LinesDataCall.h"


megamol::stdplugin::datatools::MainDirAnalysis::MainDirAnalysis(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "Data output slot")
    , dataInSlot("dataIn", "Data input slot") {
    this->dataInSlot.SetCompatibleCall<megamol::geocalls::LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);
}


megamol::stdplugin::datatools::MainDirAnalysis::~MainDirAnalysis(void) {
    this->Release();
}


bool megamol::stdplugin::datatools::MainDirAnalysis::create(void) {
    return true;
}


void megamol::stdplugin::datatools::MainDirAnalysis::release(void) {

}


bool megamol::stdplugin::datatools::MainDirAnalysis::getDataCallback(megamol::core::Call& c) {
    return true;
}


bool megamol::stdplugin::datatools::MainDirAnalysis::getExtentCallback(megamol::core::Call& c) {
    return true;
}