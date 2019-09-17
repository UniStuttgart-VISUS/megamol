#include "ExtractMesh.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/EnumParam.h"
#include "adios_plugin/CallADIOSData.h"



namespace megamol {
namespace probe {


ExtractMesh::ExtractMesh()
    : Module()
    , _getDataCall("getData", "")
    , _deployMeshCall("deployMesh", "")
    , _algorithmSlot("algorithm", "")
    , _xSlot("x", "")
    , _ySlot("y", "")
    , _zSlot("z", "")
    , _xyzSlot("xyz", "")
    , _formatSlot("format", "")
    , _alphaSlot("alpha","") {
    
    this->_alphaSlot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->_alphaSlot);

    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "alpha_shape");
    this->_algorithmSlot << ep;
    this->MakeSlotAvailable(&this->_algorithmSlot);

    core::param::EnumParam* fp = new core::param::EnumParam(0);
    fp->SetTypePair(0, "separated");
    fp->SetTypePair(0, "interleaved");
    this->_formatSlot << fp;
    this->_formatSlot.SetUpdateCallback(&ExtractMesh::toggleFormat);
    this->MakeSlotAvailable(&this->_formatSlot);

    core::param::FlexEnumParam* xEp = new core::param::FlexEnumParam("undef");
    this->_xSlot << xEp;
    this->MakeSlotAvailable(&this->_xSlot);

    core::param::FlexEnumParam* yEp = new core::param::FlexEnumParam("undef");
    this->_ySlot << yEp;
    this->MakeSlotAvailable(&this->_ySlot);

    core::param::FlexEnumParam* zEp = new core::param::FlexEnumParam("undef");
    this->_zSlot << zEp;
    this->MakeSlotAvailable(&this->_zSlot);

    core::param::FlexEnumParam* xyzEp = new core::param::FlexEnumParam("undef");
    this->_xyzSlot << xyzEp;
    xyzEp->SetGUIVisible(false);
    this->MakeSlotAvailable(&this->_xyzSlot);

    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ExtractMesh::getData);
    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ExtractMesh::getMetaData);
    this->MakeSlotAvailable(&this->_deployMeshCall);

    this->_getDataCall.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_getDataCall);

}

ExtractMesh::~ExtractMesh() {
    this->Release();
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
    hull.reconstruct(_cloud, _polygons);

}

void ExtractMesh::createPointCloud(std::vector<std::string>& vars) {

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return;

    auto count = cd->getData(vars[0])->size();

    _cloud.points.resize(count);

    for (auto var : vars) {
        if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
            auto x =
                cd->getData(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            auto y =
                cd->getData(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            auto z =
                cd->getData(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();

            auto xminmax = std::minmax_element(x.begin(),x.end());
            auto yminmax = std::minmax_element(y.begin(), y.end());
            auto zminmax = std::minmax_element(z.begin(), z.end());
            _bbox.SetObjectSpaceBBox(*xminmax.first,*yminmax.first,*zminmax.second,*xminmax.second,*yminmax.second,*zminmax.first);

            for (unsigned long long i = 0; i < count; i++) {
                _cloud.points[i].x = x[i];
                _cloud.points[i].y = y[i];
                _cloud.points[i].z = z[i];
            }

        } else {
            auto xyz = cd->getData(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()))
                           ->GetAsFloat();
            float xmin = 0;
            float xmax = 0;
            float ymin = 0;
            float ymax = 0;
            float zmin = 0;
            float zmax = 0;
            for (unsigned long long i = 0; i < count; i++) {
                _cloud.points[i].x = xyz[3 * i + 0];
                _cloud.points[i].y = xyz[3 * i + 1];
                _cloud.points[i].z = xyz[3 * i + 2];

                xmin = std::min(xmin, _cloud.points[i].x);
                xmax = std::max(xmax, _cloud.points[i].x);
                ymin = std::min(ymin, _cloud.points[i].y);
                ymax = std::max(ymax, _cloud.points[i].y);
                zmin = std::min(zmin, _cloud.points[i].z);
                zmax = std::max(zmax, _cloud.points[i].z);
            }
            _bbox.SetObjectSpaceBBox(xmin, ymin, zmax, xmax, ymax, zmin);
        }
    }
}

void ExtractMesh::convertToMesh() {
    
    _mesh_attribs.resize(1);
    _mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
    _mesh_attribs[0].byte_size = _cloud.points.size() * sizeof(pcl::PointXYZ);
    _mesh_attribs[0].component_cnt = 3;
    _mesh_attribs[0].stride = sizeof(pcl::PointXYZ);
    _mesh_attribs[0].data = reinterpret_cast<uint8_t*>(&_cloud.points);

    _mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
    _mesh_indices.data = reinterpret_cast<uint8_t*>(_polygons.data());

    }

bool ExtractMesh::getData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;


    if (cd->getDataHash() == _old_datahash) return true;

    std::vector<std::string> toInq;
    toInq.clear();
    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        toInq.emplace_back(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    } else {
        toInq.emplace_back(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    }

    // get data from adios
    for (auto var : toInq) {
        cd->inquire(var);
    }

    this->createPointCloud(toInq);

    auto meta_data = cm->getMetaData();
    meta_data.m_bboxs = _bbox;
    cm->setMetaData(meta_data);

    this->calculateAlphaShape();

    this->convertToMesh();

    // put data in mesh 
    mesh::MeshDataAccessCollection mesh;

    mesh.addMesh(_mesh_attribs, _mesh_indices);
    cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)));

    return true;
}

bool ExtractMesh::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    // get metadata from adios
    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->_xSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_ySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_zSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }
    
    // put metadata in mesh call
    auto meta_data = cm->getMetaData();
    meta_data.m_frame_cnt = cd->getFrameCount();
    cd->setFrameIDtoLoad(meta_data.m_frame_ID);
    meta_data.m_data_hash = cd->getDataHash();
    cm->setMetaData(meta_data);

    return true;
}

bool ExtractMesh::toggleFormat(core::param::ParamSlot& p) {

    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        this->_xSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(true);
        this->_ySlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(true);
        this->_zSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(true);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(false);
    } else {
        this->_xSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(false);
        this->_ySlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(false);
        this->_zSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(false);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(true);
    }

    return true;
}


} // namespace probe
} // namespace megamol