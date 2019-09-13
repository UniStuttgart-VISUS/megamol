#include "ExtractMesh.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/EnumParam.h"
#include "adios_plugin/CallADIOSData.h"
#include "mesh/MeshCalls.h"


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
    , _formatSlot("", "")
    , _alphaSlot("","") {
    
    
    this->_alphaSlot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->_alphaSlot);

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

    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ExtractMesh::getData);
    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ExtractMesh::getMetaData);
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

    ::pcl::ConcaveHull<pcl::PointXYZ> hull;

    


    hull.setAlpha(this->_alphaSlot.Param<core::param::FloatParam>()->Value());
    hull.setDimension(3);
    hull.reconstruct(cloud, polygons);




}

bool ExtractMesh::getData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);

    // get data from adios


    // put data in mesh 


    return true;
}

bool ExtractMesh::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    // get metadata from adios
    
    
    
    // put metadata in mesh call
    auto meta_data = cm->getMetaData();
    //meta_data.


    return true;
}


} // namespace probe
} // namespace megamol