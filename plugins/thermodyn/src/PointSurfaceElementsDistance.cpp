#include "PointSurfaceElementsDistance.h"

#include <array>

#include "mesh/MeshDataAccessor.h"


megamol::thermodyn::PointSurfaceElementsDistance::PointSurfaceElementsDistance()
        : data_out_slot_("dataOut", "")
        , part_mesh_in_slot_("partMeshIn", "")
        , inter_mesh_in_slot_("interMeshIn", "")
        , parts_in_slot_("partsIn", "") {
    data_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &PointSurfaceElementsDistance::get_data_cb);
    data_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &PointSurfaceElementsDistance::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    part_mesh_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&part_mesh_in_slot_);

    inter_mesh_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&inter_mesh_in_slot_);

    parts_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&parts_in_slot_);
}


megamol::thermodyn::PointSurfaceElementsDistance::~PointSurfaceElementsDistance() {
    this->Release();
}


bool megamol::thermodyn::PointSurfaceElementsDistance::create() {
    return true;
}


void megamol::thermodyn::PointSurfaceElementsDistance::release() {}


bool megamol::thermodyn::PointSurfaceElementsDistance::get_data_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto part_mesh_data = part_mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (part_mesh_data == nullptr)
        return false;
    auto inter_mesh_data = inter_mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (inter_mesh_data == nullptr)
        return false;
    auto parts_in_data = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in_data == nullptr)
        return false;

    auto req_frame_id = out_data->FrameID();

    auto part_mesh_meta = part_mesh_data->getMetaData();
    part_mesh_meta.m_frame_ID = req_frame_id;
    part_mesh_data->setMetaData(part_mesh_meta);

    auto inter_mesh_meta = inter_mesh_data->getMetaData();
    inter_mesh_meta.m_frame_ID = req_frame_id;
    inter_mesh_data->setMetaData(inter_mesh_meta);

    parts_in_data->SetFrameID(req_frame_id);

    if (!(*part_mesh_data)(1))
        return false;
    if (!(*part_mesh_data)(0))
        return false;

    if (!(*inter_mesh_data)(1))
        return false;
    if (!(*inter_mesh_data)(0))
        return false;

    if (!(*parts_in_data)(1))
        return false;
    if (!(*parts_in_data)(0))
        return false;

    if (part_mesh_data->hasUpdate() /* || inter_mesh_data->hasUpdate()*/ ||
        parts_in_data->DataHash() != parts_data_hash_ ||
        parts_in_data->FrameID() != frame_id_) {
        assert_data(*part_mesh_data, *inter_mesh_data, *parts_in_data);

        parts_data_hash_ = parts_in_data->DataHash();
        frame_id_ = parts_in_data->FrameID();
        ++out_data_hash_;
    }

    out_data->SetDataHash(out_data_hash_);
    out_data->SetFrameID(frame_id_);

    auto const pl_count = inter_pos_classes_.size();

    out_data->SetParticleListCount(pl_count);

    for (uint64_t pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& in_parts = parts_in_data->AccessParticles(pl_idx);
        auto& out_parts = out_data->AccessParticles(pl_idx);

        out_parts = in_parts;

        out_parts.SetColourData(
            core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, inter_pos_classes_[pl_idx].data());
        out_parts.SetColourMapIndexValues(
            inter_pos_classes_minmax_[pl_idx].first, inter_pos_classes_minmax_[pl_idx].second);
    }

    return true;
}


bool megamol::thermodyn::PointSurfaceElementsDistance::get_extent_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto part_mesh_data = part_mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (part_mesh_data == nullptr)
        return false;
    auto inter_mesh_data = inter_mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (inter_mesh_data == nullptr)
        return false;
    auto parts_in_data = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in_data == nullptr)
        return false;

    if (!(*part_mesh_data)(1))
        return false;

    if (!(*inter_mesh_data)(1))
        return false;

    if (!(*parts_in_data)(1))
        return false;

    auto const frame_count = parts_in_data->FrameCount();
    auto const bboxes = parts_in_data->AccessBoundingBoxes();

    /*auto part_mesh_meta = part_mesh_data->getMetaData();
    part_mesh_meta.m_frame_cnt = frame_count;
    part_mesh_meta.m_bboxs.SetBoundingBox(bboxes.ObjectSpaceBBox());
    part_mesh_meta.m_bboxs.SetClipBox(bboxes.ObjectSpaceClipBox());

    auto inter_mesh_meta = inter_mesh_data->getMetaData();
    inter_mesh_meta.m_frame_cnt = frame_count;
    inter_mesh_meta.m_bboxs.SetBoundingBox(bboxes.ObjectSpaceBBox());
    inter_mesh_meta.m_bboxs.SetClipBox(bboxes.ObjectSpaceClipBox());*/

    out_data->SetFrameCount(frame_count);
    out_data->AccessBoundingBoxes() = bboxes;

    return true;
}


void compute_bbox(glm::vec3 const& pos, glm::vec3& lower, glm::vec3& upper) {
    lower.x = std::fmin(pos.x, lower.x);
    lower.y = std::fmin(pos.y, lower.y);
    lower.z = std::fmin(pos.z, lower.z);
    upper.x = std::fmax(pos.x, upper.x);
    upper.y = std::fmax(pos.y, upper.y);
    upper.z = std::fmax(pos.z, upper.z);
}


bool megamol::thermodyn::PointSurfaceElementsDistance::assert_data(
    mesh::CallMesh& in_part_mesh, mesh::CallMesh& in_inter_mesh, core::moldyn::MultiParticleDataCall& in_parts) {
    nanoflann::KDTreeSingleIndexAdaptorParams params;

    inter_pos_classes_.clear();
    inter_pos_classes_minmax_.clear();

    inner_point_clouds_.clear();
    inner_kd_trees_.clear();

    auto part_mesh = in_part_mesh.getData();
    auto part_mesh_meta = in_part_mesh.getMetaData();
    auto pmm_bbox = part_mesh_meta.m_bboxs.BoundingBox();
    /*std::array<glm::vec3, 2> part_mesh_bbox = {glm::vec3(pmm_bbox.Left(), pmm_bbox.Bottom(), pmm_bbox.Back()),
        glm::vec3(pmm_bbox.Right(), pmm_bbox.Top(), pmm_bbox.Front())};*/
    std::array<glm::vec3, 2> part_mesh_bbox = {glm::vec3(std::numeric_limits<float>::max()),
        glm::vec3(std::numeric_limits<float>::min())};

    auto part_mesh_coll = part_mesh->accessMeshes();

    std::vector<std::vector<glm::vec3>> part_ref_points;
    std::vector<std::vector<glm::vec3>> part_ref_normals;

    for (auto& [name, mesh] : part_mesh_coll) {
        auto part_mesh_acc = mesh::MeshDataTriangleAccessor(mesh);

        auto const tri_count = part_mesh_acc.GetCount();

        std::vector<glm::vec3> ref_points;
        ref_points.reserve(tri_count);
        std::vector<glm::vec3> ref_normals;
        ref_normals.reserve(tri_count);

        for (std::decay_t<decltype(tri_count)> t_idx = 0; t_idx < tri_count; ++t_idx) {
            auto const tri_pos = part_mesh_acc.GetPosition(t_idx);
            auto const ref_pos = (tri_pos[0] + tri_pos[1] + tri_pos[2]) / 3.0f;
            ref_points.push_back(ref_pos);
            compute_bbox(ref_pos, part_mesh_bbox[0], part_mesh_bbox[1]);
            auto const tri_norm = part_mesh_acc.GetNormal(t_idx);
            if (tri_norm.has_value()) {
                auto const tri_norm_v = tri_norm.value();
                auto const ref_norm = (tri_norm_v[0] + tri_norm_v[1] + tri_norm_v[2]) / 3.0f;
                ref_normals.push_back(ref_norm);
            } else {
                ref_normals.push_back(glm::vec3(0));
            }
        }

        part_ref_points.push_back(ref_points);
        part_ref_normals.push_back(ref_normals);

        inner_point_clouds_.push_back(std::make_shared<stdplugin::datatools::glmPointcloud>(
            part_ref_points.back(), part_mesh_bbox, glm::vec3(1)));
        inner_kd_trees_.push_back(std::make_shared<kd_tree_t>(3, *inner_point_clouds_.back(), params));
        inner_kd_trees_.back()->buildIndex();
    }


    outer_point_clouds_.clear();
    outer_kd_trees_.clear();

    auto inter_mesh = in_inter_mesh.getData();
    auto inter_mesh_meta = in_inter_mesh.getMetaData();
    auto imm_bbox = inter_mesh_meta.m_bboxs.BoundingBox();
    /*std::array<glm::vec3, 2> inter_mesh_bbox = {glm::vec3(imm_bbox.Left(), imm_bbox.Bottom(), imm_bbox.Back()),
        glm::vec3(imm_bbox.Right(), imm_bbox.Top(), imm_bbox.Front())};*/
    std::array<glm::vec3, 2> inter_mesh_bbox = {glm::vec3(std::numeric_limits<float>::max()),
        glm::vec3(std::numeric_limits<float>::min())};

    auto inter_mesh_coll = inter_mesh->accessMeshes();

    std::vector<std::vector<glm::vec3>> inter_ref_points;
    std::vector<std::vector<glm::vec3>> inter_ref_normals;

    for (auto& [name, mesh] : inter_mesh_coll) {
        auto inter_mesh_acc = mesh::MeshDataTriangleAccessor(mesh);

        auto const tri_count = inter_mesh_acc.GetCount();

        std::vector<glm::vec3> ref_points;
        ref_points.reserve(tri_count);
        std::vector<glm::vec3> ref_normals;
        ref_normals.reserve(tri_count);

        for (std::decay_t<decltype(tri_count)> t_idx = 0; t_idx < tri_count; ++t_idx) {
            auto const tri_pos = inter_mesh_acc.GetPosition(t_idx);
            auto const ref_pos = (tri_pos[0] + tri_pos[1] + tri_pos[2]) / 3.0f;
            ref_points.push_back(ref_pos);
            compute_bbox(ref_pos, inter_mesh_bbox[0], inter_mesh_bbox[1]);
            auto const tri_norm = inter_mesh_acc.GetNormal(t_idx);
            if (tri_norm.has_value()) {
                auto const tri_norm_v = tri_norm.value();
                auto const ref_norm = (tri_norm_v[0] + tri_norm_v[1] + tri_norm_v[2]) / 3.0f;
                ref_normals.push_back(ref_norm);
            } else {
                ref_normals.push_back(glm::vec3(0));
            }
        }

        inter_ref_points.push_back(ref_points);
        inter_ref_normals.push_back(ref_normals);

        outer_point_clouds_.push_back(std::make_shared<stdplugin::datatools::glmPointcloud>(
            inter_ref_points.back(), inter_mesh_bbox, glm::vec3(1)));
        outer_kd_trees_.push_back(std::make_shared<kd_tree_t>(3, *outer_point_clouds_.back(), params));
        outer_kd_trees_.back()->buildIndex();
    }


    auto const pl_count = in_parts.GetParticleListCount();
    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = in_parts.AccessParticles(pl_idx);

        auto const p_count = parts.GetCount();

        auto const xAcc = parts.GetParticleStore().GetXAcc();
        auto const yAcc = parts.GetParticleStore().GetYAcc();
        auto const zAcc = parts.GetParticleStore().GetZAcc();

        std::vector<float> tmp_inner_distances(p_count, std::numeric_limits<float>::max());
        std::vector<float> tmp_outer_distances(p_count, std::numeric_limits<float>::max());

        for (uint64_t t_idx = 0; t_idx < inner_kd_trees_.size(); ++t_idx) {
            auto& tree = inner_kd_trees_[t_idx];
            auto const& points = part_ref_points[t_idx];
            auto const& normals = part_ref_normals[t_idx];
            for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                auto const query = glm::vec3(xAcc->Get_f(p_idx), yAcc->Get_f(p_idx), zAcc->Get_f(p_idx));
                size_t idx = 0;
                float dis = 0;
                tree->knnSearch(glm::value_ptr(query), 1, &idx, &dis);
                auto const to_q_dir = query - points[idx];
                auto const normal = normals[idx];
                auto const sign_fac =
                    std::signbit(glm::dot(to_q_dir, normal) / (glm::length(to_q_dir) * glm::length(normal))) ? -1.f
                                                                                                             : 1.f;
                tmp_inner_distances[p_idx] =
                    sign_fac * (std::fabs(tmp_inner_distances[p_idx]) > dis ? dis : tmp_inner_distances[p_idx]);
            }
        }

        for (uint64_t t_idx = 0; t_idx < outer_kd_trees_.size(); ++t_idx) {
            auto& tree = outer_kd_trees_[t_idx];
            auto const& points = inter_ref_points[t_idx];
            auto const& normals = inter_ref_normals[t_idx];
            for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                auto const query = glm::vec3(xAcc->Get_f(p_idx), yAcc->Get_f(p_idx), zAcc->Get_f(p_idx));
                size_t idx = 0;
                float dis = 0;
                tree->knnSearch(glm::value_ptr(query), 1, &idx, &dis);
                auto const to_q_dir = query - points[idx];
                auto const normal = normals[idx];
                auto const sign_fac =
                    std::signbit(glm::dot(to_q_dir, normal) / (glm::length(to_q_dir) * glm::length(normal))) ? -1.f
                                                                                                             : 1.f;
                tmp_outer_distances[p_idx] =
                    sign_fac * (std::fabs(tmp_outer_distances[p_idx]) > dis ? dis : tmp_outer_distances[p_idx]);
            }
        }

        std::vector<float> inter_pos_class(p_count);
        std::transform(tmp_inner_distances.begin(), tmp_inner_distances.end(), tmp_outer_distances.begin(),
            inter_pos_class.begin(), [](auto inner, auto outer) {
                auto const inner_bit = std::signbit(inner);
                auto const outer_bit = std::signbit(outer);
                if (!inner_bit && outer_bit) {
                    return 1.f;
                } else if (inner_bit && outer_bit) {
                    return 2.f;
                } else if (!inner_bit && !outer_bit) {
                    return 3.f;
                }
                return 0.f;
            });

        auto const minmax_el = std::minmax_element(inter_pos_class.begin(), inter_pos_class.end());
        inter_pos_classes_minmax_.emplace_back(std::make_pair(*minmax_el.first, *minmax_el.second));
        inter_pos_classes_.push_back(inter_pos_class);
    }

    return true;
}
