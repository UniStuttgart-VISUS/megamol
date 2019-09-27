#include "ExtractMesh.h"
#include <limits>
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "normal_3d_omp.h"


namespace megamol {
namespace probe {


ExtractMesh::ExtractMesh()
    : Module()
    , _getDataCall("getData", "")
    , _deployMeshCall("deployMesh", "")
    , _deploySpheresCall("deploySpheres", "")
    , _algorithmSlot("algorithm", "")
    , _xSlot("x", "")
    , _ySlot("y", "")
    , _zSlot("z", "")
    , _xyzSlot("xyz", "")
    , _formatSlot("format", "")
    , _alphaSlot("alpha", "") {

    this->_alphaSlot << new core::param::FloatParam(1.0f);
    this->_alphaSlot.SetUpdateCallback(&ExtractMesh::alphaChanged);
    this->MakeSlotAvailable(&this->_alphaSlot);

    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "alpha_shape");
    this->_algorithmSlot << ep;
    this->MakeSlotAvailable(&this->_algorithmSlot);

    core::param::EnumParam* fp = new core::param::EnumParam(0);
    fp->SetTypePair(0, "separated");
    fp->SetTypePair(1, "interleaved");
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

    this->_deploySpheresCall.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ExtractMesh::getParticleData);
    this->_deploySpheresCall.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ExtractMesh::getParticleMetaData);
    this->MakeSlotAvailable(&this->_deploySpheresCall);
}

ExtractMesh::~ExtractMesh() { this->Release(); }

bool ExtractMesh::create() { return true; }

void ExtractMesh::release() {}

bool ExtractMesh::InterfaceIsDirty() { return true; }

void ExtractMesh::calculateAlphaShape() {

    bool useOrigData = true;


    pcl::ConcaveHull<pcl::PointXYZ> hull;

    hull.setAlpha(this->_alphaSlot.Param<core::param::FloatParam>()->Value());
    hull.setDimension(3);
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(_cloud);
    hull.setInputCloud(inputCloud);
    // hull.setDoFiltering(true);
    if (!useOrigData) {
        hull.reconstruct(_resultCloud, _polygons);
    }
    if (usePoisson) {


        pcl::Poisson<pcl::PointNormal> surface;

        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> normal_est;

        std::shared_ptr<const pcl::PointCloud<pcl::PointXYZ>> result_shared;

        if (useOrigData) {
            result_shared = std::make_shared<const pcl::PointCloud<pcl::PointXYZ>>(_cloud);
        } else {
            result_shared = std::make_shared<const pcl::PointCloud<pcl::PointXYZ>>(_resultCloud);
        }

        normal_est.setInputCloud(result_shared);
        auto center = _bbox.ObjectSpaceBBox().CalcCenter();
        normal_est.setViewPoint(center.GetX(), center.GetY(), center.GetZ());

        pcl::PointCloud<pcl::PointNormal> normal_cloud;

        normal_est.setRadiusSearch(0.1f);

        normal_est.compute(normal_cloud);


        std::shared_ptr<const pcl::PointCloud<pcl::PointNormal>> normal_shared =
            std::make_shared<const pcl::PointCloud<pcl::PointNormal>>(normal_cloud);

        _polygons.clear();
        surface.setInputCloud(normal_shared);
        surface.setDepth(9);
        surface.setSamplesPerNode(15.0f);
        //surface.setConfidence(true);

        surface.reconstruct(_resultSurface, _polygons);

    }
}

bool ExtractMesh::createPointCloud(std::vector<std::string>& vars) {

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    if (vars.empty()) return false;

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

            auto xminmax = std::minmax_element(x.begin(), x.end());
            auto yminmax = std::minmax_element(y.begin(), y.end());
            auto zminmax = std::minmax_element(z.begin(), z.end());
            _bbox.SetObjectSpaceBBox(
                *xminmax.first, *yminmax.first, *zminmax.second, *xminmax.second, *yminmax.second, *zminmax.first);

            for (unsigned long long i = 0; i < count; i++) {
                _cloud.points[i].x = x[i];
                _cloud.points[i].y = y[i];
                _cloud.points[i].z = z[i];
            }

        } else {
            auto xyz = cd->getData(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()))
                           ->GetAsFloat();
            float xmin = std::numeric_limits<float>::max();
            float xmax = std::numeric_limits<float>::min();
            float ymin = std::numeric_limits<float>::max();
            float ymax = std::numeric_limits<float>::min();
            float zmin = std::numeric_limits<float>::max();
            float zmax = std::numeric_limits<float>::min();
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
    return true;
}

void ExtractMesh::convertToMesh() {

    if (!usePoisson) {
        _mesh_attribs.resize(1);
        _mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh_attribs[0].byte_size = _resultCloud.points.size() * sizeof(pcl::PointXYZ);
        _mesh_attribs[0].component_cnt = 3;
        _mesh_attribs[0].stride = sizeof(pcl::PointXYZ);
        _mesh_attribs[0].data = reinterpret_cast<uint8_t*>(_resultCloud.points.data());
    } else {
        _mesh_attribs.resize(1);
        _mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        vertex_data.resize(3*_resultSurface.points.size());
        for (auto i = 0; i < _resultSurface.points.size(); i++) {
            vertex_data[3 * i + 0] = _resultSurface.points[i].x;
            vertex_data[3 * i + 1] = _resultSurface.points[i].y;
            vertex_data[3 * i + 2] = _resultSurface.points[i].z;
        }


        _mesh_attribs[0].byte_size = sizeof(float) * vertex_data.size();
        _mesh_attribs[0].component_cnt = 3;
        _mesh_attribs[0].stride = 3 * sizeof(float);
        _mesh_attribs[0].data = reinterpret_cast<uint8_t*>(vertex_data.data());
    }

    _mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
    _mesh_indices.byte_size = 3 * _polygons.size() * sizeof(uint32_t);
    _mesh_indices.data = reinterpret_cast<uint8_t*>(_polygons.data());
}

bool ExtractMesh::getData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto meta_data = cm->getMetaData();
    // if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() &&
    //    meta_data.m_data_hash == _recalc_hash)
    //    return true;

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
        if (!cd->inquire(var)) return false;
    }
    if (!(*cd)(0)) return false;


    if (!this->createPointCloud(toInq)) return false;


    meta_data.m_bboxs = _bbox;
    cm->setMetaData(meta_data);

    this->calculateAlphaShape();

    // this->filterResult();
    // this->filterByIndex();

    this->convertToMesh();

    // put data in mesh
    mesh::MeshDataAccessCollection mesh;

    mesh.addMesh(_mesh_attribs, _mesh_indices);
    cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)));
    _old_datahash = cd->getDataHash();


    return true;
}

bool ExtractMesh::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto meta_data = cm->getMetaData();
    if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() &&
        meta_data.m_data_hash == _recalc_hash)
        return true;

    // get metadata from adios
    if (!(*cd)(1)) return false;
    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->_xSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_ySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_zSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }

    // put metadata in mesh call

    meta_data.m_frame_cnt = cd->getFrameCount();
    cd->setFrameIDtoLoad(meta_data.m_frame_ID);
    meta_data.m_data_hash = _recalc_hash;
    cm->setMetaData(meta_data);

    return true;
}

bool ExtractMesh::getParticleData(core::Call& call) {
    auto cm = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

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
        if (!cd->inquire(var)) return false;
    }
    if (!(*cd)(0)) return false;


    if (!this->createPointCloud(toInq)) return false;


    cm->AccessBoundingBoxes().SetObjectSpaceBBox(_bbox.ObjectSpaceBBox());

    this->calculateAlphaShape();

    // this->filterResult();
    // this->filterByIndex();

    cm->SetParticleListCount(1);
    cm->AccessParticles(0).SetGlobalRadius(0.02f);
    cm->AccessParticles(0).SetCount(_resultCloud.points.size());
    cm->AccessParticles(0).SetVertexData(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
        reinterpret_cast<uint8_t*>(_resultCloud.points.data()), 4 * sizeof(float));

    _old_datahash = cd->getDataHash();

    return true;
}

bool ExtractMesh::getParticleMetaData(core::Call& call) {
    auto cm = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    if (cd->getDataHash() == _old_datahash && cm->FrameID() == cd->getFrameIDtoLoad() && cm->DataHash() == _recalc_hash)
        return true;

    // get metadata from adios
    if (!(*cd)(1)) return false;
    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->_xSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_ySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_zSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }


    cm->SetFrameCount(cd->getFrameCount());
    cd->setFrameIDtoLoad(cm->FrameID());
    cd->setDataHash(_recalc_hash);

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

bool ExtractMesh::alphaChanged(core::param::ParamSlot& p) {

    _recalc_hash++;

    return true;
}

bool ExtractMesh::filterResult() {

    const int factor = 3;

    std::vector<pcl::Vertices> new_polygon;
    new_polygon.reserve(_polygons.size());

    for (auto& polygon : _polygons) {
        auto v1 = _resultCloud.points[polygon.vertices[0]];
        auto v2 = _resultCloud.points[polygon.vertices[1]];
        auto v3 = _resultCloud.points[polygon.vertices[2]];

        auto a = v1 - v2;
        auto length_a = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);

        auto b = v1 - v3;
        auto length_b = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);

        auto c = v2 - v3;
        auto length_c = std::sqrt(c.x * c.x + c.y * c.y + c.z * c.z);

        auto l1 = length_b > length_a ? length_b / length_a : length_a / length_b;
        auto l2 = length_c > length_a ? length_c / length_a : length_a / length_c;
        auto l3 = length_b > length_c ? length_b / length_c : length_c / length_b;

        if (l1 >= factor || l2 >= factor || l3 >= factor) {
            vislib::sys::Log::DefaultLog.WriteInfo("[ExtractMesh] Found bad polygon.");
        } else {
            new_polygon.emplace_back(polygon);
        }
    }
    new_polygon.shrink_to_fit();

    _polygons = std::move(new_polygon);

    return true;
}

bool ExtractMesh::filterByIndex() {

    size_t max_distance = 10;

    std::vector<pcl::Vertices> new_polygon;
    new_polygon.reserve(_polygons.size());

    for (auto polygon : _polygons) {
        auto dist_a = std::abs(static_cast<int>(polygon.vertices[0]) - static_cast<int>(polygon.vertices[1]));
        auto dist_b = std::abs(static_cast<int>(polygon.vertices[2]) - static_cast<int>(polygon.vertices[1]));
        auto dist_c = std::abs(static_cast<int>(polygon.vertices[0]) - static_cast<int>(polygon.vertices[2]));

        if (!(dist_a >= max_distance || dist_b >= max_distance || dist_c >= max_distance)) {
            new_polygon.emplace_back(polygon);
        }
    }
    new_polygon.shrink_to_fit();
    _polygons = std::move(new_polygon);

    return true;
}

} // namespace probe
} // namespace megamol
