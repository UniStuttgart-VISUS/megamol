#include "PrepareSurfaceEvents.h"

#include "mmcore/param/BoolParam.h"


megamol::thermodyn::PrepareSurfaceEvents::PrepareSurfaceEvents()
        : data_out_slot_("dataOut", "")
        , parts_out_slot_("partsOut", "")
        , parts_in_slot_("partsIn", "")
        , mesh_in_slot_("meshIn", "")
        , table_in_slot_("tableIn", "")
        , show_all_parts_slot_("show all", "") {
    data_out_slot_.SetCallback(
        CallEvents::ClassName(), CallEvents::FunctionName(0), &PrepareSurfaceEvents::get_data_cb);
    data_out_slot_.SetCallback(
        CallEvents::ClassName(), CallEvents::FunctionName(1), &PrepareSurfaceEvents::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    parts_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &PrepareSurfaceEvents::get_parts_data_cb);
    parts_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &PrepareSurfaceEvents::get_parts_extent_cb);
    MakeSlotAvailable(&parts_out_slot_);

    parts_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&parts_in_slot_);

    mesh_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&mesh_in_slot_);

    table_in_slot_.SetCompatibleCall<stdplugin::datatools::table::TableDataCallDescription>();
    MakeSlotAvailable(&table_in_slot_);

    show_all_parts_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&show_all_parts_slot_);
}


megamol::thermodyn::PrepareSurfaceEvents::~PrepareSurfaceEvents() {
    this->Release();
}


bool megamol::thermodyn::PrepareSurfaceEvents::create() {
    return true;
}


void megamol::thermodyn::PrepareSurfaceEvents::release() {}


bool megamol::thermodyn::PrepareSurfaceEvents::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<CallEvents*>(&c);
    if (data_out == nullptr)
        return false;

    auto mesh_in = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto table_in = table_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (table_in == nullptr)
        return false;

    auto const out_meta = data_out->getMetaData();

    auto mesh_meta = mesh_in->getMetaData();
    mesh_meta.m_frame_ID = out_meta.m_frame_ID;
    mesh_in->setMetaData(mesh_meta);
    if (!(*mesh_in)(1))
        return false;
    if (!(*mesh_in)(0))
        return false;

    parts_in->SetFrameID(out_meta.m_frame_ID);
    if (!(*parts_in)(1))
        return false;
    if (!(*parts_in)(0))
        return false;

    table_in->SetFrameID(out_meta.m_frame_ID);
    if (!(*table_in)(1))
        return false;
    if (!(*table_in)(0))
        return false;

    if (mesh_in->hasUpdate() || parts_in->DataHash() != parts_in_data_hash_ || parts_in->FrameID() != frame_id_ ||
        table_in->DataHash() != table_in_data_hash_ /*|| table_in->GetFrameID() != frame_id_*/ || is_dirty()) {
        prepare_maps(*table_in);
        assert_data(*parts_in, *mesh_in, *table_in, parts_in->FrameID());

        parts_in_data_hash_ = parts_in->DataHash();
        table_in_data_hash_ = table_in->DataHash();
        frame_id_ = parts_in->FrameID();
        ++data_out_hash_;
        reset_dirty();
    }

    auto meta = out_meta;
    meta.m_frame_ID = frame_id_;
    meta.m_frame_cnt = parts_in->FrameCount();
    data_out->setMetaData(meta);
    data_out->setData(events_, data_out_hash_);

    return true;
}


bool megamol::thermodyn::PrepareSurfaceEvents::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<CallEvents*>(&c);
    if (data_out == nullptr)
        return false;

    auto mesh_in = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto table_in = table_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (table_in == nullptr)
        return false;

    auto const out_meta = data_out->getMetaData();

    auto mesh_meta = mesh_in->getMetaData();
    mesh_meta.m_frame_ID = out_meta.m_frame_ID;
    mesh_in->setMetaData(mesh_meta);
    if (!(*mesh_in)(1))
        return false;

    parts_in->SetFrameID(out_meta.m_frame_ID);
    if (!(*parts_in)(1))
        return false;

    table_in->SetFrameID(out_meta.m_frame_ID);
    if (!(*table_in)(1))
        return false;

    auto meta = out_meta;
    meta.m_frame_ID = parts_in->FrameID();
    meta.m_frame_cnt = parts_in->FrameCount();
    data_out->setMetaData(meta);

    return true;
}


bool megamol::thermodyn::PrepareSurfaceEvents::get_parts_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto mesh_in = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto table_in = table_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (table_in == nullptr)
        return false;

    auto const out_frame_id = data_out->FrameID();

    auto mesh_meta = mesh_in->getMetaData();
    mesh_meta.m_frame_ID = out_frame_id;
    mesh_in->setMetaData(mesh_meta);
    if (!(*mesh_in)(1))
        return false;
    if (!(*mesh_in)(0))
        return false;

    parts_in->SetFrameID(out_frame_id);
    if (!(*parts_in)(1))
        return false;
    if (!(*parts_in)(0))
        return false;

    table_in->SetFrameID(out_frame_id);
    if (!(*table_in)(1))
        return false;
    if (!(*table_in)(0))
        return false;

    if (mesh_in->hasUpdate() || parts_in->DataHash() != parts_in_data_hash_ || parts_in->FrameID() != frame_id_ ||
        table_in->DataHash() != table_in_data_hash_ /*|| table_in->GetFrameID() != frame_id_*/ || is_dirty()) {
        prepare_maps(*table_in);
        assert_data(*parts_in, *mesh_in, *table_in, parts_in->FrameID());

        parts_in_data_hash_ = parts_in->DataHash();
        table_in_data_hash_ = table_in->DataHash();
        frame_id_ = parts_in->FrameID();
        ++parts_out_hash_;
        reset_dirty();
    }

    data_out->SetFrameCount(parts_in->FrameCount());
    data_out->SetFrameID(frame_id_);
    data_out->SetParticleListCount(part_data_.size());
    for (decltype(part_data_)::size_type pl_idx = 0; pl_idx < part_data_.size(); ++pl_idx) {
        auto& out_parts = data_out->AccessParticles(pl_idx);
        auto const& part_data = part_data_[pl_idx];

        auto const p_count = part_data.size() / 8;

        out_parts.SetCount(p_count);

        out_parts.SetIDData(core::moldyn::SimpleSphericalParticles::IDDATA_UINT32, part_data.data(), 8 * sizeof(float));
        out_parts.SetVertexData(
            core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, part_data.data() + 1, 8 * sizeof(float));
        out_parts.SetColourData(
            core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA, part_data.data() + 4, 8 * sizeof(float));
        out_parts.SetGlobalRadius(parts_in->AccessParticles(pl_idx).GetGlobalRadius());
    }

    return true;
}


bool megamol::thermodyn::PrepareSurfaceEvents::get_parts_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto mesh_in = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto table_in = table_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (table_in == nullptr)
        return false;

    auto const out_frame_id = data_out->FrameID();

    auto mesh_meta = mesh_in->getMetaData();
    mesh_meta.m_frame_ID = out_frame_id;
    mesh_in->setMetaData(mesh_meta);
    if (!(*mesh_in)(1))
        return false;

    parts_in->SetFrameID(out_frame_id);
    if (!(*parts_in)(1))
        return false;

    table_in->SetFrameID(out_frame_id);
    if (!(*table_in)(1))
        return false;

    data_out->SetFrameCount(parts_in->FrameCount());
    data_out->SetFrameID(frame_id_);
    data_out->AccessBoundingBoxes() = parts_in->AccessBoundingBoxes();

    return true;
}


bool megamol::thermodyn::PrepareSurfaceEvents::assert_data(core::moldyn::MultiParticleDataCall& in_parts,
    mesh::CallMesh& in_mesh, stdplugin::datatools::table::TableDataCall& in_table, int frame_id) {
    auto const& mesh_collection = in_mesh.getData();
    auto const& meshes = mesh_collection->accessMeshes();

    auto const show_all = show_all_parts_slot_.Param<core::param::BoolParam>()->Value();

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

    events_ = std::make_shared<std::vector<glm::vec4>>();

    try {
        auto const event_list = frame_id_map_.at(frame_id);
        events_->reserve(event_list.size());
        for (auto const& event_el : event_list) {
            auto const& [s_fid, e_fid, s_pos, e_pos] = event_map_.at(event_el);
            auto const query = Point(s_pos.x, s_pos.y, s_pos.z);
            // auto const dis = tree.squared_distance(query);
            auto const closest_P = tree.closest_point_and_primitive(query);
            auto const P = closest_P.first;
            events_->push_back(glm::vec4(P.x(), P.y(), P.z(), frame_id));
        }

        part_data_.clear();

        auto const pl_count = in_parts.GetParticleListCount();

        part_data_.resize(pl_count);

        for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = in_parts.AccessParticles(pl_idx);
            auto& part_data = part_data_[pl_idx];

            auto const p_count = parts.GetCount();

            part_data.reserve(p_count);

            auto const idAcc = parts.GetParticleStore().GetIDAcc();
            auto const xAcc = parts.GetParticleStore().GetXAcc();
            auto const yAcc = parts.GetParticleStore().GetYAcc();
            auto const zAcc = parts.GetParticleStore().GetZAcc();
            auto const crAcc = parts.GetParticleStore().GetCRAcc();
            auto const cgAcc = parts.GetParticleStore().GetCGAcc();
            auto const cbAcc = parts.GetParticleStore().GetCBAcc();
            auto const caAcc = parts.GetParticleStore().GetCAAcc();

            for (std::remove_cv_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                auto const id = idAcc->Get_u32(p_idx);
                auto const fit = std::find(event_list.begin(), event_list.end(), id);
                if (show_all) {
                    part_data.push_back(*reinterpret_cast<float const*>(&id));
                    part_data.push_back(xAcc->Get_f(p_idx));
                    part_data.push_back(yAcc->Get_f(p_idx));
                    part_data.push_back(zAcc->Get_f(p_idx));
                    if (fit == event_list.end()) {
                        part_data.push_back(crAcc->Get_f(p_idx));
                        part_data.push_back(cgAcc->Get_f(p_idx));
                        part_data.push_back(cbAcc->Get_f(p_idx));
                        part_data.push_back(caAcc->Get_f(p_idx));
                    } else {
                        part_data.push_back(0.0f);
                        part_data.push_back(1.0f);
                        part_data.push_back(0.0f);
                        part_data.push_back(1.0f);
                    }
                } else {
                    if (fit != event_list.end()) {
                        part_data.push_back(*reinterpret_cast<float const*>(&id));
                        part_data.push_back(xAcc->Get_f(p_idx));
                        part_data.push_back(yAcc->Get_f(p_idx));
                        part_data.push_back(zAcc->Get_f(p_idx));
                        part_data.push_back(0.0f);
                        part_data.push_back(1.0f);
                        part_data.push_back(0.0f);
                        part_data.push_back(1.0f);
                    }
                }
            }
        }
    } catch (...) { return false; }

    return true;
}


void megamol::thermodyn::PrepareSurfaceEvents::prepare_maps(
    stdplugin::datatools::table::TableDataCall const& in_table) {
    auto const col_count = in_table.GetColumnsCount();
    auto const row_count = in_table.GetRowsCount();
    auto const table_data = in_table.GetData();

    frame_id_map_.clear();
    // frame_id_map_.reserve(row_count);

    event_map_.clear();
    event_map_.reserve(row_count);

    for (std::remove_cv_t<decltype(row_count)> row = 0; row < row_count; ++row) {
        event_map_[in_table.GetData(0, row)] = {static_cast<int>(in_table.GetData(1, row)),
            static_cast<int>(in_table.GetData(2, row)),
            {in_table.GetData(3, row), in_table.GetData(4, row), in_table.GetData(5, row)},
            {in_table.GetData(6, row), in_table.GetData(7, row), in_table.GetData(8, row)}};

        frame_id_map_[in_table.GetData(1, row)].push_back(in_table.GetData(0, row));
    }
}
