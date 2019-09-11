#include "ExtractMesh.h"

namespace megamol {
namespace probe {


ExtractMesh::ExtractMesh()
    : Module()
    , _getDataCall("", "")
    , _getMeshCall("", "")
    , _algorithmSlot("", "")
    , _xSlot("", "")
    , _ySlot("", "")
    , _zSlot("", "")
    , _xyzSlot("", "")
    , _formatSlot("", "") {
    



}

ExtractMesh::~ExtractMesh() {
}

bool ExtractMesh::InterfaceIsDirty() {
}

void ExtractMesh::calculateAlphaShape() {





}


} // namespace probe
} // namespace megamol