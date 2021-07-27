#include "PointInterfaceClassification.h"

#include "mesh/MeshCalls.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FloatParam.h"


megamol::thermodyn::PointInterfaceClassification::PointInterfaceClassification()
        : point_out_slot_("pointOut", "")
        , point_in_slot_("pointIn", "")
        , mesh_in_slot_("meshIn", "")
        , mesh_points_in_slot_("meshPointsIn", "")
        , critical_temp_slot_("Temp Critical", "") {
    point_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &PointInterfaceClassification::get_data_cb);
    point_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &PointInterfaceClassification::get_extent_cb);
    MakeSlotAvailable(&point_out_slot_);

    point_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&point_in_slot_);

    mesh_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&mesh_in_slot_);

    mesh_points_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&mesh_points_in_slot_);

    critical_temp_slot_ << new core::param::FloatParam(0.8f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&critical_temp_slot_);
}


megamol::thermodyn::PointInterfaceClassification::~PointInterfaceClassification() {
    this->Release();
}


bool megamol::thermodyn::PointInterfaceClassification::create() {
    return true;
}


void megamol::thermodyn::PointInterfaceClassification::release() {}


bool megamol::thermodyn::PointInterfaceClassification::get_data_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_mesh = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto in_data = point_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;
    auto in_mesh_points = mesh_points_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_mesh_points == nullptr)
        return false;

    /*static bool not_init = true;
    if (not_init) {
        init();
        not_init = false;
    }*/

    if (!(*in_mesh)(1))
        return false;
    if (!(*in_data)(1))
        return false;
    if (!(*in_mesh_points)(1))
        return false;

    auto meta = in_mesh->getMetaData();
    meta.m_frame_ID = out_data->FrameID();
    in_mesh->setMetaData(meta);
    if (!(*in_mesh)(1))
        return false;
    if (!(*in_mesh)(0))
        return false;
    meta = in_mesh->getMetaData();
    in_data->SetFrameID(out_data->FrameID());
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;
    in_mesh_points->SetFrameID(out_data->FrameID());
    if (!(*in_mesh_points)(1))
        return false;
    if (!(*in_mesh_points)(0))
        return false;

    // if (/*in_data->hasUpdate()*/ meta.m_frame_ID != frame_id_ /*|| meta.m_data_hash != _data_hash*/) {
    if (in_data->FrameID() != frame_id_ /*|| meta.m_data_hash != _data_hash*/) {
        if (!assert_data(*in_data, *in_mesh, *in_mesh_points))
            return false;
        frame_id_ = in_data->FrameID();
        //_data_hash = meta.m_data_hash;
        ++out_data_hash_;
    }

    out_data->SetParticleListCount(distances_.size());
    out_data->SetDataHash(out_data_hash_);
    out_data->SetFrameID(frame_id_);

    for (unsigned int plIdx = 0; plIdx < distances_.size(); ++plIdx) {
        auto const& in_parts = in_data->AccessParticles(plIdx);
        auto& out_parts = out_data->AccessParticles(plIdx);
        auto const& pos = positions_[plIdx];
        // auto const& col = distances_[plIdx];
        auto const& col = in_interface_[plIdx];
        auto const& minmax_el = dist_minmax_[plIdx];

        out_parts.SetCount(pos.size());
        out_parts.SetGlobalRadius(in_parts.GetGlobalRadius());
        out_parts.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, pos.data());
        out_parts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, col.data());
        out_parts.SetColourMapIndexValues(minmax_el.first, minmax_el.second);
    }

    return true;
}


bool megamol::thermodyn::PointInterfaceClassification::get_extent_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_mesh = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto in_data = point_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;
    auto in_mesh_points = mesh_points_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_mesh_points == nullptr)
        return false;

    auto in_meta = in_mesh->getMetaData();
    in_meta.m_frame_ID = out_data->FrameID();
    if (!(*in_mesh)(1))
        return false;
    in_meta = in_mesh->getMetaData(), in_data->SetFrameCount(in_meta.m_frame_cnt);
    in_data->SetFrameID(in_meta.m_frame_ID);
    if (!(*in_data)(1))
        return false;
    in_mesh_points->SetFrameID(in_meta.m_frame_ID);
    if (!(*in_mesh_points)(1))
        return false;

    out_data->SetFrameCount(in_data->FrameCount());
    // out_data->SetFrameID(frame_id_);
    out_data->AccessBoundingBoxes() = in_data->AccessBoundingBoxes();

    return true;
}

bool megamol::thermodyn::PointInterfaceClassification::assert_data(core::moldyn::MultiParticleDataCall& points,
    mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& mesh_points) {
    auto const thickness = [](float T, float T_c) -> float {
        return -1.720f * std::powf((T_c - T) / T_c, 1.89f) + 1.103f * std::powf((T_c - T) / T_c, -0.62f);
    };

    auto const& mesh_collection = mesh.getData();
    auto const& meshes = mesh_collection->accessMeshes();

    auto& points_in_mesh = mesh_points.AccessParticles(0);
    auto tempAcc = points_in_mesh.GetParticleStore().GetCRAcc();

    std::vector<float> data;
    std::array<float, 3> weights = {1.f, 1.f, 1.f};

    std::vector<float> temps;

    for (auto const& mesh : meshes) {
        auto const& mesh_data = mesh.second;

        auto const num_triangles = mesh_data.indices.byte_size / sizeof(glm::uvec3);

        data.reserve(data.size() + num_triangles * 3 * 3);
        temps.reserve(temps.size() + num_triangles * 3);

        auto indices = reinterpret_cast<glm::uvec3 const*>(mesh_data.indices.data);

        auto const& attributes = mesh_data.attributes;

        std::remove_cv_t<std::remove_reference_t<decltype(attributes)>>::value_type const* attr_ptr = nullptr;

        for (auto const& attr : attributes) {
            if (attr.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION) {
                attr_ptr = &attr;
            }
        }

        auto positions = reinterpret_cast<glm::vec3 const*>(attr_ptr->data);

        for (std::remove_cv_t<decltype(num_triangles)> t_idx = 0; t_idx < num_triangles; ++t_idx) {
            auto a = positions[indices[t_idx].x];
            auto b = positions[indices[t_idx].y];
            auto c = positions[indices[t_idx].z];

            data.push_back(a.x);
            data.push_back(a.y);
            data.push_back(a.z);
            data.push_back(b.x);
            data.push_back(b.y);
            data.push_back(b.z);
            data.push_back(c.x);
            data.push_back(c.y);
            data.push_back(c.z);

            temps.push_back(tempAcc->Get_f(indices[t_idx].x));
            temps.push_back(tempAcc->Get_f(indices[t_idx].y));
            temps.push_back(tempAcc->Get_f(indices[t_idx].z));
        }
    }

    auto const meta = mesh.getMetaData();

    auto const bbox_mesh = meta.m_bboxs.BoundingBox();

    std::array<float, 6> bbox = {
        bbox_mesh.Left(), bbox_mesh.Right(), bbox_mesh.Bottom(), bbox_mesh.Top(), bbox_mesh.Back(), bbox_mesh.Front()};

    point_cloud_ = std::make_shared<stdplugin::datatools::genericPointcloud<float, 3>>(data, bbox, weights);
    nanoflann::KDTreeSingleIndexAdaptorParams params;
    params.leaf_max_size = 100;
    kd_tree_ = std::make_shared<kd_tree_t<float, 3>>(3, *point_cloud_, params);
    kd_tree_->buildIndex();

    auto const pl_count = points.GetParticleListCount();

    distances_.resize(pl_count);
    positions_.resize(pl_count);
    dist_minmax_.resize(pl_count);
    in_interface_.resize(pl_count);

    for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = points.AccessParticles(pl_idx);
        auto const p_count = parts.GetCount();

        auto& distances = distances_[pl_idx];
        distances.resize(p_count);

        auto& positions = positions_[pl_idx];
        positions.resize(p_count);

        auto& in_interface = in_interface_[pl_idx];
        in_interface.resize(p_count);

        auto const x_acc = parts.GetParticleStore().GetXAcc();
        auto const y_acc = parts.GetParticleStore().GetYAcc();
        auto const z_acc = parts.GetParticleStore().GetZAcc();

        std::array<size_t, 100> index;
        std::array<float, 100> dist;

        for (std::remove_cv_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            auto const query = kd_tree_->dataset.get_position(p_idx);

            kd_tree_->knnSearch(query, 100, index.data(), dist.data());

            auto dis_it = std::find_if(dist.begin(), dist.end(), [](auto const f) { return f != 0.0f; });

            // if (dis[0] == 0.f) {

            //    /*auto const query = Point(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx));
            //    auto const dis = tree.squared_distance(query);
            //    auto const closest_P = tree.closest_point_and_primitive(query);
            //    auto tri_idx = std::distance(triangles.begin(), closest_P.second);*/

            //    // auto const closest_P = tree.closest_point(query);

            //    distances[p_idx] = std::sqrtf(dis[1]);
            //} else {
            //    distances[p_idx] = std::sqrtf(dis[0]);
            //}
            if (dis_it == dist.end()) {
                distances[p_idx] = 0.f;
                in_interface[p_idx] = -1.f;
            } else {
                distances[p_idx] = std::sqrtf(*dis_it);
                auto const idx = std::distance(dist.begin(), dis_it);
                auto const id = index[idx];
                auto const temp = temps[idx];
                auto const thick = thickness(temp, 1.7f);
                in_interface[p_idx] = distances[p_idx] < 0.1f;
            }

            positions[p_idx] = glm::vec3(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx));
        }

        auto const minmax = std::minmax_element(distances.begin(), distances.end());
        dist_minmax_[pl_idx] = std::make_pair(*minmax.first, *minmax.second);

        auto const minmax_inter = std::minmax_element(in_interface.begin(), in_interface.end());
        dist_minmax_[pl_idx] = std::make_pair(*minmax_inter.first, *minmax_inter.second);
    }

    return true;
}
