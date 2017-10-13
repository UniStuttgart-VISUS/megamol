#include "stdafx.h"
#include "PBSDataSource.h"

using namespace megamol;
using namespace megamol::pbs;


PBSDataSource::PBSDataSource(void) : core::Module() {

}


PBSDataSource::~PBSDataSource(void) {
    this->Release();
}


bool PBSDataSource::create(void) {
    return true;
}


void PBSDataSource::release(void) {

}
