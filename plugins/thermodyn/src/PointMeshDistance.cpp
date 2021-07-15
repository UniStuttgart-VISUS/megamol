#include "PointMeshDistance.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"


megamol::thermodyn::PointMeshDistance::PointMeshDistance()
        : in_points_slot_("inPoints", "")
        , in_mesh_slot_("inMesh", "")
        , out_distances_slot_("outDist", "")
        , toggle_threshold_slot_("toggle threshold", "")
        , threshold_slot_("threshold", "")
        , toggle_invert_slot_("toggle inversion", "") {
    in_points_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_points_slot_);

    in_mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&in_mesh_slot_);

    out_distances_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &PointMeshDistance::get_data_cb);
    out_distances_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &PointMeshDistance::get_extent_cb);
    MakeSlotAvailable(&out_distances_slot_);

    toggle_threshold_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&toggle_threshold_slot_);

    threshold_slot_ << new core::param::FloatParam(0.1f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&threshold_slot_);

    toggle_invert_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&toggle_invert_slot_);
}


megamol::thermodyn::PointMeshDistance::~PointMeshDistance() {
    this->Release();
}


bool megamol::thermodyn::PointMeshDistance::create() {
    return true;
}


void megamol::thermodyn::PointMeshDistance::release() {}


bool megamol::thermodyn::PointMeshDistance::get_data_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_mesh = in_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto in_data = in_points_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
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

    // if (/*in_data->hasUpdate()*/ meta.m_frame_ID != frame_id_ /*|| meta.m_data_hash != _data_hash*/) {
    if (in_data->FrameID() != frame_id_ || is_dirty() /*|| meta.m_data_hash != _data_hash*/) {
        if (!assert_data(*in_data, *in_mesh))
            return false;
        frame_id_ = in_data->FrameID();
        //_data_hash = meta.m_data_hash;
        ++out_data_hash_;
        reset_dirty();
    }

    out_data->SetParticleListCount(distances_.size());
    out_data->SetDataHash(out_data_hash_);
    out_data->SetFrameID(frame_id_);

    for (unsigned int plIdx = 0; plIdx < distances_.size(); ++plIdx) {
        auto const& in_parts = in_data->AccessParticles(plIdx);
        auto& out_parts = out_data->AccessParticles(plIdx);
        auto const& pos = positions_[plIdx];
        auto const& col = distances_[plIdx];
        auto const& minmax_el = dist_minmax_[plIdx];

        out_parts.SetCount(pos.size());
        out_parts.SetGlobalRadius(in_parts.GetGlobalRadius());
        out_parts.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, pos.data());
        out_parts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, col.data());
        out_parts.SetColourMapIndexValues(minmax_el.first, minmax_el.second);
    }

    return true;
}


bool megamol::thermodyn::PointMeshDistance::get_extent_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_mesh = in_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto in_data = in_points_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    auto in_meta = in_mesh->getMetaData();
    in_meta.m_frame_ID = out_data->FrameID();
    if (!(*in_mesh)(1))
        return false;
    in_meta = in_mesh->getMetaData(), in_data->SetFrameCount(in_meta.m_frame_cnt);
    in_data->SetFrameID(in_meta.m_frame_ID);
    if (!(*in_data)(1))
        return false;

    out_data->SetFrameCount(in_data->FrameCount());
    // out_data->SetFrameID(frame_id_);
    out_data->AccessBoundingBoxes() = in_data->AccessBoundingBoxes();

    return true;
}


bool megamol::thermodyn::PointMeshDistance::assert_data(
    core::moldyn::MultiParticleDataCall& points, mesh::CallMesh& mesh) {
    auto const& mesh_collection = mesh.getData();
    auto const& meshes = mesh_collection->accessMeshes();

    std::list<Triangle> triangles;

    for (auto const& mesh : meshes) {
        auto const& mesh_data = mesh.second;

        auto const num_triangles = mesh_data.indices.byte_size / sizeof(glm::uvec3);

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
            auto a = Point(positions[indices[t_idx].x].x, positions[indices[t_idx].x].y, positions[indices[t_idx].x].z);
            auto b = Point(positions[indices[t_idx].y].x, positions[indices[t_idx].y].y, positions[indices[t_idx].y].z);
            auto c = Point(positions[indices[t_idx].z].x, positions[indices[t_idx].z].y, positions[indices[t_idx].z].z);

            triangles.push_back(Triangle(a, b, c));
        }
    }

    Tree tree(triangles.begin(), triangles.end());

    auto const pl_count = points.GetParticleListCount();

    distances_.resize(pl_count);
    positions_.resize(pl_count);
    dist_minmax_.resize(pl_count);

    auto const toggle_threshold = toggle_threshold_slot_.Param<core::param::BoolParam>()->Value();
    auto const threshold = threshold_slot_.Param<core::param::FloatParam>()->Value();
    auto const toggle_invert = toggle_invert_slot_.Param<core::param::BoolParam>()->Value();

    for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = points.AccessParticles(pl_idx);
        auto const p_count = parts.GetCount();

        auto& distances = distances_[pl_idx];
        distances.resize(p_count);

        auto& positions = positions_[pl_idx];
        positions.resize(p_count);

        auto const x_acc = parts.GetParticleStore().GetXAcc();
        auto const y_acc = parts.GetParticleStore().GetYAcc();
        auto const z_acc = parts.GetParticleStore().GetZAcc();

        for (std::remove_cv_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            auto const query = Point(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx));
            auto const dis = tree.squared_distance(query);
            auto const closest_P = tree.closest_point_and_primitive(query);
            // auto tri_idx = std::distance(triangles.begin(), closest_P.second);
            auto const& a = closest_P.second->vertex(0);
            auto const& b = closest_P.second->vertex(1);
            auto const& c = closest_P.second->vertex(2);
            auto normal = CGAL::normal(a, b, c);

            auto const dis_vec = query - closest_P.first;
            auto const angle = std::acosf(CGAL::scalar_product(dis_vec, normal) /
                                          (std::sqrtf(normal.squared_length()) * std::sqrtf(dis_vec.squared_length())));

            // auto const closest_P = tree.closest_point(query);

            if (toggle_threshold) {
                distances[p_idx] = std::sqrtf(dis) < threshold ? 0.f : 1.f;
            } else {
                if (toggle_invert) {
                    distances[p_idx] = std::sqrtf(dis) * (angle < 3.14f * 0.5f ? -1.0f : 1.0f);
                } else {
                    distances[p_idx] = std::sqrtf(dis) * (angle < 3.14f * 0.5f ? 1.0f : -1.0f);
                }
            }

            positions[p_idx] = glm::vec3(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx));
        }

        auto const minmax = std::minmax_element(distances.begin(), distances.end());
        dist_minmax_[pl_idx] = std::make_pair(*minmax.first, *minmax.second);
    }

    return true;
}
