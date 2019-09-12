#include "ExtractMesh.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/EnumParam.h"
#include "adios_plugin/CallADIOSData.h"

namespace megamol {
namespace probe {


ExtractMesh::ExtractMesh()
    : Module()
    , _getDataCall("", "")
    , _deployMeshCall("", "")
    , _algorithmSlot("", "")
    , _xSlot("", "")
    , _ySlot("", "")
    , _zSlot("", "")
    , _xyzSlot("", "")
    , _formatSlot("", "") {
    


    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "alpha_shape");
    this->_algorithmSlot << ep;
    this->MakeSlotAvailable(&this->_algorithmSlot);

    core::param::FlexEnumParam* xEp = new core::param::FlexEnumParam("undef");
    this->_xSlot << xEp;
    this->MakeSlotAvailable(&this->_xSlot);

    core::param::FlexEnumParam* yEp = new core::param::FlexEnumParam("undef");
    this->_ySlot << yEp;
    this->MakeSlotAvailable(&this->_ySlot);

    core::param::FlexEnumParam* zEp = new core::param::FlexEnumParam("undef");
    this->_zSlot << zEp;
    this->MakeSlotAvailable(&this->_zSlot);

    //this->_deployMeshCall.SetCallback(geocalls::LinesDataCall::ClassName(), "GetData", &ExtractMesh::getData);
    //this->_deployMeshCall.SetCallback(
    //    geocalls::LinesDataCall::ClassName(), "GetExtent", &ExtractMesh::getMetaData);
    this->MakeSlotAvailable(&this->_deployMeshCall);

    this->_getDataCall.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_getDataCall);

}

ExtractMesh::~ExtractMesh() {
}

bool ExtractMesh::create() {

    return true;
}

void ExtractMesh::release() {
}

bool ExtractMesh::InterfaceIsDirty() {

    return true;
}

void ExtractMesh::calculateAlphaShape() {





}

bool ExtractMesh::getData(core::Call& call) {

    return true;
}

bool ExtractMesh::getMetaData(core::Call& call) {

    return true;
}


} // namespace probe
} // namespace megamol