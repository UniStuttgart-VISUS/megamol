#include "stdafx.h"
#include "TrajectoryDataSource.h"


megamol::stdplugin::datatools::io::TrajectoryDataSource::TrajectoryDataSource()
    : megamol::core::Module()
    , trajOutSlot("trajOut", "Trajectory output")
    , trajFilepath("filename", "Trajectory file to read") {

}


megamol::stdplugin::datatools::io::TrajectoryDataSource::~TrajectoryDataSource() {
    this->Release();
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::create() {
    return true;
}


void megamol::stdplugin::datatools::io::TrajectoryDataSource::release() {

}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::getDataCallback(megamol::core::Call& c) {
    return true;
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::getExtentCallback(megamol::core::Call& c) {
    return true;
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::assertData() {
    return true;
}
