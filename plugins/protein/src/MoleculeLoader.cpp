#include "MoleculeLoader.h"

#include "mmcore/param/FilePathParam.h"

using namespace megamol::protein;

MoleculeLoader::MoleculeLoader(void)
        : core::Module()
        , filenameSlot_("filename", "Path to the file containing the structure of the molecule(s) to visualize.")
        , trajectoryFilenameSlot_("trajectoryFilename", "Path to the file containing the trajectory of the molecule.")
        , trajectory_(nullptr) {
    // TODO
}

MoleculeLoader::~MoleculeLoader(void) {
    this->Release();
}

bool MoleculeLoader::create(void) {
    // TODO

    return true;
}

void MoleculeLoader::release(void) {
    // TODO
}

bool MoleculeLoader::getData(core::Call& call) {
    // TODO
    return true;
}

bool MoleculeLoader::getExtent(core::Call& call) {
    // TODO
    return true;
}
