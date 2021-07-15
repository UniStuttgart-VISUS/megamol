#include "stdafx.h"
#include "ParticleSurface.h"

#include "mmcore/UniFlagCalls.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/FilenameHelper.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "Eigen/Dense"

#include "mmstd_datatools/TFUtils.h"


megamol::thermodyn::ParticleSurface::ParticleSurface()
        : _out_mesh_slot("outMesh", "")
        , _out_part_slot("outPart", "")
        , _in_data_slot("inData", "")
        , _tf_slot("inTF", "")
        , _flags_read_slot("readFlags", "")
        , _alpha_slot("alpha", "")
        , _type_slot("type", "")
        , _vert_type_slot("vert type", "")
        , toggle_interface_slot_("toggle interface", "")
        , info_filename_slot_("info filename", "")
        , write_info_slot_("write info", "") {
    _out_mesh_slot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ParticleSurface::get_data_cb);
    _out_mesh_slot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ParticleSurface::get_extent_cb);
    MakeSlotAvailable(&_out_mesh_slot);

    _out_part_slot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ParticleSurface::get_part_data_cb);
    _out_part_slot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ParticleSurface::get_part_extent_cb);
    MakeSlotAvailable(&_out_part_slot);

    _in_data_slot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&_in_data_slot);

    _tf_slot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    MakeSlotAvailable(&_tf_slot);

    _flags_read_slot.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&_flags_read_slot);

    _alpha_slot << new core::param::FloatParam(1.25f, 0.0f);
    MakeSlotAvailable(&_alpha_slot);

    using surface_type_ut = std::underlying_type_t<surface_type>;
    auto ep = new core::param::EnumParam(static_cast<surface_type_ut>(surface_type::alpha_shape));
    ep->SetTypePair(static_cast<surface_type_ut>(surface_type::alpha_shape), "alpha shape");
    ep->SetTypePair(static_cast<surface_type_ut>(surface_type::gtim), "gtim");
    _type_slot << ep;
    MakeSlotAvailable(&_type_slot);

    using vert_type_ut = std::underlying_type_t<Alpha_shape_3::Classification_type>;
    ep = new core::param::EnumParam(static_cast<vert_type_ut>(Alpha_shape_3::REGULAR));
    ep->SetTypePair(static_cast<vert_type_ut>(Alpha_shape_3::REGULAR), "regular");
    ep->SetTypePair(static_cast<vert_type_ut>(Alpha_shape_3::EXTERIOR), "exterior");
    ep->SetTypePair(static_cast<vert_type_ut>(Alpha_shape_3::INTERIOR), "interior");
    ep->SetTypePair(static_cast<vert_type_ut>(Alpha_shape_3::SINGULAR), "singular");
    _vert_type_slot << ep;
    MakeSlotAvailable(&_vert_type_slot);

    toggle_interface_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&toggle_interface_slot_);

    info_filename_slot_ << new core::param::FilePathParam("");
    MakeSlotAvailable(&info_filename_slot_);

    write_info_slot_ << new core::param::BoolParam(false);
    write_info_slot_.SetUpdateCallback(&ParticleSurface::write_info_cb);
    MakeSlotAvailable(&write_info_slot_);
}


megamol::thermodyn::ParticleSurface::~ParticleSurface() {
    this->Release();
}


bool megamol::thermodyn::ParticleSurface::create() {
    return true;
}


void megamol::thermodyn::ParticleSurface::release() {}


bool compute_touchingsphere_radius(megamol::thermodyn::Point_3 const& a_pos, float a_r,
    megamol::thermodyn::Point_3 const& b_pos, float b_r, megamol::thermodyn::Point_3 const& c_pos, float c_r,
    megamol::thermodyn::Point_3 const& d_pos, float d_r, float& res) {
    Eigen::Vector3f a_vec;
    a_vec << a_pos.x(), a_pos.y(), a_pos.z();
    Eigen::Vector3f b_vec;
    b_vec << b_pos.x(), b_pos.y(), b_pos.z();
    Eigen::Vector3f c_vec;
    c_vec << c_pos.x(), c_pos.y(), c_pos.z();
    Eigen::Vector3f d_vec;
    d_vec << d_pos.x(), d_pos.y(), d_pos.z();

    auto row_0_tmp = a_vec - b_vec;
    auto row_1_tmp = a_vec - c_vec;
    auto row_2_tmp = a_vec - d_vec;

    Eigen::Matrix3f M;
    M << row_0_tmp[0], row_0_tmp[1], row_0_tmp[2], row_1_tmp[0], row_1_tmp[1], row_1_tmp[2], row_2_tmp[0], row_2_tmp[1],
        row_2_tmp[2];
    Eigen::Vector3f d;
    d << a_r - b_r, a_r - c_r, a_r - d_r;
    Eigen::Vector3f s;
    s << a_vec.dot(a_vec) - b_vec.dot(b_vec) - a_r * a_r + b_r * b_r,
        a_vec.dot(a_vec) - c_vec.dot(c_vec) - a_r * a_r + c_r * c_r,
        a_vec.dot(a_vec) - d_vec.dot(d_vec) - a_r * a_r + d_r * d_r;
    s = 0.5f * s;
    Eigen::Matrix3f invM;
    bool invertible;
    M.computeInverseWithCheck(invM, invertible);
    if (!invertible)
        return false;
    auto r_0 = invM * s;
    auto u = invM * d;
    auto v = a_vec - r_0;
    auto u_norm = u.stableNorm();
    if (u_norm == 1.0f)
        return false;
    auto res_0 = (-(a_r - u.dot(v)) + (a_r * u + v).stableNorm()) / (1.0f - u_norm * u_norm);
    auto res_1 = (-(a_r - u.dot(v)) - (a_r * u + v).stableNorm()) / (1.0f - u_norm * u_norm);
    auto tmp = std::max(res_0, res_1);
    if (tmp < 0.0f)
        return false;
    res = tmp;
    return true;
}


bool megamol::thermodyn::ParticleSurface::assert_data(core::moldyn::MultiParticleDataCall& call) {
    core::view::CallGetTransferFunction* cgtf = _tf_slot.CallAs<core::view::CallGetTransferFunction>();
    if (cgtf != nullptr && !(*cgtf)())
        return false;

    auto fcr = _flags_read_slot.CallAs<core::FlagCallRead_CPU>();

    bool tf_changed = false;

    auto const pl_count = call.GetParticleListCount();
    if (call.DataHash() != _in_data_hash || call.FrameID() != _frame_id || is_dirty()) {

        _vertices.resize(pl_count);
        _normals.resize(pl_count);
        _indices.resize(pl_count);
        _part_data.resize(pl_count);

        auto const alpha = _alpha_slot.Param<core::param::FloatParam>()->Value();
        auto const squared_alpha = alpha * alpha;

        auto const type = static_cast<surface_type>(_type_slot.Param<core::param::EnumParam>()->Value());

        auto const vert_type =
            static_cast<Alpha_shape_3::Classification_type>(_vert_type_slot.Param<core::param::EnumParam>()->Value());

        auto const toggle_interface = toggle_interface_slot_.Param<core::param::BoolParam>()->Value();


        for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = call.AccessParticles(pl_idx);

            auto const& [vertices, normals, indices, part_data] = compute_alpha_shape(alpha, parts);

            _vertices[pl_idx] = vertices;
            _normals[pl_idx] = normals;
            _indices[pl_idx] = indices;
            _part_data[pl_idx] = part_data;

            auto const thickness = [](float T, float T_c) -> float {
                return -1.720f * std::powf((T_c - T) / T_c, 1.89f) + 1.103f * std::powf((T_c - T) / T_c, -0.62f);
            };

            std::list<std::pair<Point_3, std::size_t>> interface_points;

            for (decltype(part_data)::size_type idx = 0; idx < part_data.size(); ++idx) {
                auto const D = thickness(part_data[idx].i, 1.7f);
                auto const nx = normals[idx * 3];
                auto const ny = normals[idx * 3 + 1];
                auto const nz = normals[idx * 3 + 2];

                interface_points.push_back(std::make_pair(
                    Point_3(part_data[idx].x + nx * D, part_data[idx].y + ny * D, part_data[idx].z + nz * D),
                    part_data[idx].idx));
            }

            if (toggle_interface) {
                auto const& [inter_vertices, inter_normals, inter_indices, inter_part_data] =
                    compute_alpha_shape(2.f * alpha, interface_points, parts);

                _vertices[pl_idx] = inter_vertices;
                _normals[pl_idx] = inter_normals;
                _indices[pl_idx] = inter_indices;
            }
        }
    }

    if (call.DataHash() != _in_data_hash || call.FrameID() != _frame_id || is_dirty() ||
        (cgtf != nullptr && (tf_changed = cgtf->IsDirty()))) {
        _colors.resize(pl_count);

        float const* color_tf = nullptr;
        auto color_tf_size = 0;
        std::array<float, 2> range = {0.f, 0.f};

        if (call.DataHash() != _in_data_hash || call.FrameID() != _frame_id) {
            for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
                auto const& parts = call.AccessParticles(pl_idx);

                range[0] = std::min(parts.GetMinColourIndexValue(), range[0]);
                range[1] = std::max(parts.GetMaxColourIndexValue(), range[1]);
            }
            if (cgtf != nullptr) {
                cgtf->SetRange(range);
                (*cgtf)(0);
                color_tf = cgtf->GetTextureData();
                color_tf_size = cgtf->TextureSize();
            }
        } else {
            if (cgtf != nullptr) {
                (*cgtf)(0);
                color_tf = cgtf->GetTextureData();
                color_tf_size = cgtf->TextureSize();
                range = cgtf->Range();
            }
        }

        auto const def_color = glm::vec4(1.0f);

        for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const min_i = range[0];
            auto const max_i = range[1];
            auto const fac_i = 1.0f / (max_i - min_i + 1e-8f);

            auto& colors = _colors[pl_idx];
            auto const& part_data = _part_data[pl_idx];

            colors.clear();
            colors.reserve(part_data.size() * 4);

            for (auto const& el : part_data) {
                auto col_a = def_color;
                auto col_b = def_color;
                auto col_c = def_color;

                if (color_tf != nullptr) {
                    col_a = stdplugin::datatools::get_sample_from_tf(color_tf, color_tf_size, el.i, min_i, fac_i);
                }

                colors.push_back(col_a.r);
                colors.push_back(col_a.g);
                colors.push_back(col_a.b);
                colors.push_back(col_a.a);
            }
        }

        _unsel_colors = _colors;
    }

    /*bool flags_changed = false;
    if (fcr != nullptr) {
        (*fcr)(0);
        flags_changed = fcr->version() != _fcr_version;
        _fcr_version = fcr->version();
    }*/

    if (call.DataHash() != _in_data_hash || call.FrameID() != _frame_id || is_dirty() ||
        (cgtf != nullptr && (tf_changed = cgtf->IsDirty())) || (fcr != nullptr && fcr->hasUpdate())) {
        _colors = _unsel_colors;

        std::vector<size_t> sel_jobs;
        // std::vector<std::tuple<size_t, size_t, glm::vec4, glm::vec4, glm::vec4>> unsel;
        bool data_flag_change = false;
        if ((fcr != nullptr && fcr->hasUpdate())) {
            auto const data = fcr->getData();
            auto fit = data->flags->begin();
            do {
                fit = std::find_if(
                    fit, data->flags->end(), [](auto const& el) { return el == core::FlagStorage::SELECTED; });
                if (fit != data->flags->end()) {
                    sel_jobs.push_back(std::distance(data->flags->begin(), fit));
                    // core::utility::log::Log::DefaultLog.WriteInfo("[ParticleSurface] Selected %d", sel_jobs.back());
                    // break;
                    fit = std::next(fit);
                }
            } while (fit != data->flags->end());

            std::vector<size_t> base_sizes;
            base_sizes.reserve(pl_count);
            for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
                auto& indices = _indices[pl_idx];
                auto const num_el = indices.size() / 3;
                base_sizes.push_back(num_el);
            }
            std::vector<std::tuple<size_t, size_t>> tmp_sel;
            for (auto sel : sel_jobs) {
                size_t pl_idx = 0;
                for (size_t i = 0; i < base_sizes.size(); ++i) {
                    if (sel > base_sizes[i]) {
                        sel -= base_sizes[i];
                        ++pl_idx;
                    }
                }
                tmp_sel.push_back(std::make_tuple(pl_idx, sel));
            }

            _sel = tmp_sel;

            for (auto const& el : _sel) {
                auto const pl_i = std::get<0>(el);
                // auto const p_i = std::get<1>(el) * 4;
                auto const i0 = _indices[pl_i][std::get<1>(el) * 3];
                auto const i1 = _indices[pl_i][std::get<1>(el) * 3 + 1];
                auto const i2 = _indices[pl_i][std::get<1>(el) * 3 + 2];
                _colors[pl_i][i0 * 4 + 0] = 1.f;
                _colors[pl_i][i0 * 4 + 1] = 0.f;
                _colors[pl_i][i0 * 4 + 2] = 0.f;
                _colors[pl_i][i0 * 4 + 3] = 1.0f;
                _colors[pl_i][i1 * 4 + 0] = 1.f;
                _colors[pl_i][i1 * 4 + 1] = 0.f;
                _colors[pl_i][i1 * 4 + 2] = 0.f;
                _colors[pl_i][i1 * 4 + 3] = 1.0f;
                _colors[pl_i][i2 * 4 + 0] = 1.f;
                _colors[pl_i][i2 * 4 + 1] = 0.f;
                _colors[pl_i][i2 * 4 + 2] = 0.f;
                _colors[pl_i][i2 * 4 + 3] = 1.0f;
                // core::utility::log::Log::DefaultLog.WriteInfo("[ParticleSurface] Selecting %d", std::get<1>(el));
                data_flag_change = true;
            }
        }
        // flags_changed = false;

        if (call.DataHash() != _in_data_hash || call.FrameID() != _frame_id || is_dirty() ||
            (cgtf != nullptr && (tf_changed = cgtf->IsDirty())) || data_flag_change) {
            total_tri_count_ = 0;
            for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
                auto& vertices = _vertices[pl_idx];
                auto& normals = _normals[pl_idx];
                auto& colors = _colors[pl_idx];
                auto& indices = _indices[pl_idx];

                std::vector<mesh::MeshDataAccessCollection::VertexAttribute> mesh_attributes;
                mesh::MeshDataAccessCollection::IndexData mesh_indices;

                total_tri_count_ += indices.size() / 3;

                mesh_indices.byte_size = indices.size() * sizeof(uint32_t);
                mesh_indices.data = reinterpret_cast<uint8_t*>(indices.data());
                mesh_indices.type = mesh::MeshDataAccessCollection::UNSIGNED_INT;

                mesh_attributes.emplace_back(
                    mesh::MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(normals.data()),
                        normals.size() * sizeof(float), 3, mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0,
                        mesh::MeshDataAccessCollection::NORMAL});
                mesh_attributes.emplace_back(
                    mesh::MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(colors.data()),
                        colors.size() * sizeof(float), 4, mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 4, 0,
                        mesh::MeshDataAccessCollection::COLOR});
                mesh_attributes.emplace_back(
                    mesh::MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(vertices.data()),
                        vertices.size() * sizeof(float), 3, mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0,
                        mesh::MeshDataAccessCollection::POSITION});

                auto identifier = std::string("particle_surface_") + std::to_string(pl_idx);
                _mesh_access_collection = std::make_shared<mesh::MeshDataAccessCollection>();
                _mesh_access_collection->addMesh(identifier, mesh_attributes, mesh_indices);
            }
            ++_out_data_hash;
        }
        if (fcr != nullptr) {
            if ((*fcr)(0)) {
                auto data = fcr->getData();
                data->validateFlagCount(total_tri_count_);
            }
            // flags_changed = fcr->version() != _fcr_version;
        }
    }

    return true;
}


std::tuple<std::vector<float>, std::vector<float>, std::vector<unsigned int>,
    std::vector<megamol::thermodyn::ParticleSurface::particle_data>>
megamol::thermodyn::ParticleSurface::compute_alpha_shape(
    float alpha, core::moldyn::SimpleSphericalParticles const& parts) {
    auto const xAcc = parts.GetParticleStore().GetXAcc();
    auto const yAcc = parts.GetParticleStore().GetYAcc();
    auto const zAcc = parts.GetParticleStore().GetZAcc();

    std::list<std::pair<Point_3, std::size_t>> points;

    auto const p_count = parts.GetCount();

    for (std::decay_t<decltype(p_count)> pidx = 0; pidx < p_count; ++pidx) {
        points.emplace_back(std::make_pair(Point_3(xAcc->Get_f(pidx), yAcc->Get_f(pidx), zAcc->Get_f(pidx)), pidx));
    }

    return compute_alpha_shape(alpha, points, parts);
}


std::tuple<std::vector<float>, std::vector<float>, std::vector<unsigned int>,
    std::vector<megamol::thermodyn::ParticleSurface::particle_data>>
megamol::thermodyn::ParticleSurface::compute_alpha_shape(float alpha,
    std::list<std::pair<Point_3, std::size_t>> const& points, core::moldyn::SimpleSphericalParticles const& parts) {
    auto const iAcc = parts.GetParticleStore().GetCRAcc();

    auto const idAcc = parts.GetParticleStore().GetIDAcc();

    auto const dxAcc = parts.GetParticleStore().GetDXAcc();
    auto const dyAcc = parts.GetParticleStore().GetDYAcc();
    auto const dzAcc = parts.GetParticleStore().GetDZAcc();

    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<unsigned int> indices;
    std::list<Facet> facets;
    std::vector<particle_data> part_data;

    Triangulation_3 tri = Triangulation_3(points.begin(), points.end());

    auto as = std::make_shared<Alpha_shape_3>(tri, alpha);

    facets.clear();
    as->get_alpha_shape_facets(std::back_inserter(facets), Alpha_shape_3::REGULAR);

    vertices.clear();
    vertices.reserve(facets.size() * 9);
    normals.clear();
    normals.reserve(facets.size() * 9);
    indices.clear();
    indices.reserve(facets.size() * 3);

    part_data.clear();
    part_data.reserve(facets.size() * 3);

    /*auto const thickness = [](float T, float T_c) -> float {
        return -1.720f * std::powf((T_c - T) / T_c, 1.89f) + 1.103f * std::powf((T_c - T) / T_c, -0.62f);
    };*/

    // https://stackoverflow.com/questions/15905833/saving-cgal-alpha-shape-surface-mesh
    std::size_t ih = 0;
    for (auto& face : facets) {
        if (as->classify(face.first) != Alpha_shape_3::EXTERIOR)
            face = as->mirror_facet(face);
        // CGAL_assertion(as.classify(facets[i].first) == Alpha_shape_3::EXTERIOR);

        int idx[3] = {
            (face.second + 1) % 4,
            (face.second + 2) % 4,
            (face.second + 3) % 4,
        };

        if (face.second % 2 == 0)
            std::swap(idx[0], idx[1]);


        auto const& a = (face.first->vertex(idx[0])->point());
        auto const& b = (face.first->vertex(idx[1])->point());
        auto const& c = (face.first->vertex(idx[2])->point());

        auto normal = CGAL::normal(a, b, c);
        auto const length = std::sqrtf(normal.squared_length());
        normal /= length;


        auto const id_a = idAcc->Get_u32((face.first->vertex(idx[0])->info()));
        part_data.emplace_back(particle_data{(face.first->vertex(idx[0])->info()), id_a, static_cast<float>(a.x()),
            static_cast<float>(a.y()), static_cast<float>(a.z()), iAcc->Get_f((face.first->vertex(idx[0])->info())),
            dxAcc->Get_f((face.first->vertex(idx[0])->info())), dyAcc->Get_f((face.first->vertex(idx[0])->info())),
            dzAcc->Get_f((face.first->vertex(idx[0])->info()))});


        auto const id_b = idAcc->Get_u32((face.first->vertex(idx[1])->info()));
        part_data.emplace_back(particle_data{(face.first->vertex(idx[1])->info()), id_b, static_cast<float>(b.x()),
            static_cast<float>(b.y()), static_cast<float>(b.z()), iAcc->Get_f((face.first->vertex(idx[1])->info())),
            dxAcc->Get_f((face.first->vertex(idx[1])->info())), dyAcc->Get_f((face.first->vertex(idx[1])->info())),
            dzAcc->Get_f((face.first->vertex(idx[1])->info()))});


        auto const id_c = idAcc->Get_u32((face.first->vertex(idx[2])->info()));
        part_data.emplace_back(particle_data{(face.first->vertex(idx[2])->info()), id_c, static_cast<float>(c.x()),
            static_cast<float>(c.y()), static_cast<float>(c.z()), iAcc->Get_f((face.first->vertex(idx[2])->info())),
            dxAcc->Get_f((face.first->vertex(idx[2])->info())), dyAcc->Get_f((face.first->vertex(idx[2])->info())),
            dzAcc->Get_f((face.first->vertex(idx[2])->info()))});


        vertices.push_back(a.x());
        vertices.push_back(a.y());
        vertices.push_back(a.z());

        vertices.push_back(b.x());
        vertices.push_back(b.y());
        vertices.push_back(b.z());

        vertices.push_back(c.x());
        vertices.push_back(c.y());
        vertices.push_back(c.z());

        normals.push_back(normal.x());
        normals.push_back(normal.y());
        normals.push_back(normal.z());
        normals.push_back(normal.x());
        normals.push_back(normal.y());
        normals.push_back(normal.z());
        normals.push_back(normal.x());
        normals.push_back(normal.y());
        normals.push_back(normal.z());

        indices.push_back(3 * ih);
        indices.push_back(3 * ih + 1);
        indices.push_back(3 * ih + 2);
        ++ih;
    }

    return std::make_tuple(vertices, normals, indices, part_data);
}


bool megamol::thermodyn::ParticleSurface::get_data_cb(core::Call& c) {
    auto out_mesh = dynamic_cast<mesh::CallMesh*>(&c);
    if (out_mesh == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;
    core::view::CallGetTransferFunction* cgtf = _tf_slot.CallAs<core::view::CallGetTransferFunction>();
    if (cgtf != nullptr && !(*cgtf)())
        return false;

    auto meta_data = out_mesh->getMetaData();

    in_data->SetFrameID(meta_data.m_frame_ID);
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    bool tf_changed = false;

    auto fcr = _flags_read_slot.CallAs<core::FlagCallRead_CPU>();
    // bool flags_changed = false;
    if (fcr != nullptr) {
        (*fcr)(0);
        /*if ((*fcr)(0)) {
            auto data = fcr->getData();
            data->validateFlagCount(total_tri_count_);
        }*/
        // flags_changed = fcr->version() != _fcr_version;
    }

    if (in_data->DataHash() != _in_data_hash || in_data->FrameID() != _frame_id || is_dirty() ||
        (cgtf != nullptr && (tf_changed = cgtf->IsDirty())) || (fcr != nullptr && fcr->hasUpdate())) {
        auto const res = assert_data(*in_data);

        auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
        auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

        _frame_id = in_data->FrameID();
        _in_data_hash = in_data->DataHash();
        reset_dirty();

        meta_data.m_bboxs.SetBoundingBox(bbox);
        meta_data.m_bboxs.SetClipBox(cbox);

        if (cgtf != nullptr) {
            cgtf->ResetDirty();
        }
    }

    meta_data.m_frame_cnt = in_data->FrameCount();
    meta_data.m_frame_ID = _frame_id;
    // meta_data.m_data_hash = ++_out_data_hash;
    out_mesh->setMetaData(meta_data);
    out_mesh->setData(_mesh_access_collection, _out_data_hash);

    return true;
}


bool megamol::thermodyn::ParticleSurface::get_extent_cb(core::Call& c) {
    auto out_mesh = dynamic_cast<mesh::CallMesh*>(&c);
    if (out_mesh == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;
    auto meta_data = out_mesh->getMetaData();
    in_data->SetFrameID(meta_data.m_frame_ID);
    if (!(*in_data)(1))
        return false;

    auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

    meta_data.m_bboxs.SetBoundingBox(bbox);
    meta_data.m_bboxs.SetClipBox(cbox);
    meta_data.m_frame_cnt = in_data->FrameCount();
    meta_data.m_frame_ID = in_data->FrameID();
    out_mesh->setMetaData(meta_data);


    return true;
}


bool megamol::thermodyn::ParticleSurface::get_part_data_cb(core::Call& c) {
    auto out_part = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_part == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;
    core::view::CallGetTransferFunction* cgtf = _tf_slot.CallAs<core::view::CallGetTransferFunction>();
    if (cgtf != nullptr && !(*cgtf)())
        return false;

    in_data->SetFrameID(out_part->FrameID());
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    bool tf_changed = false;

    if (in_data->DataHash() != _in_data_hash || in_data->FrameID() != _frame_id || is_dirty() ||
        (cgtf != nullptr && (tf_changed = cgtf->IsDirty()))) {
        auto const res = assert_data(*in_data);


        _frame_id = in_data->FrameID();
        _in_data_hash = in_data->DataHash();
        reset_dirty();

        if (cgtf != nullptr) {
            cgtf->ResetDirty();
        }
        ++_out_data_hash;
    }

    auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

    out_part->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    out_part->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);

    out_part->SetParticleListCount(in_data->GetParticleListCount());

    for (std::remove_const_t<decltype(in_data->GetParticleListCount())> pl_idx = 0;
         pl_idx < in_data->GetParticleListCount(); ++pl_idx) {

        auto const& part_data = _part_data[pl_idx];
        auto& parts = out_part->AccessParticles(pl_idx);

        parts.SetCount(part_data.size());
        parts.SetIDData(
            core::moldyn::SimpleSphericalParticles::IDDATA_UINT32, &(part_data[0].id), sizeof(particle_data));
        parts.SetVertexData(
            core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, &(part_data[0].x), sizeof(particle_data));
        parts.SetColourData(
            core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, &(part_data[0].i), sizeof(particle_data));
        parts.SetDirData(
            core::moldyn::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, &(part_data[0].dx), 7 * sizeof(float));
        parts.SetGlobalRadius(in_data->AccessParticles(pl_idx).GetGlobalRadius());
        parts.SetColourMapIndexValues(in_data->AccessParticles(pl_idx).GetMinColourIndexValue(),
            in_data->AccessParticles(pl_idx).GetMaxColourIndexValue());
    }

    out_part->SetFrameCount(in_data->FrameCount());
    out_part->SetFrameID(in_data->FrameID());

    out_part->SetDataHash(_out_data_hash);

    return true;
}


bool megamol::thermodyn::ParticleSurface::get_part_extent_cb(core::Call& c) {
    auto out_part = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_part == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(out_part->FrameID());
    if (!(*in_data)(1))
        return false;

    auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

    out_part->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    out_part->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);

    out_part->SetFrameCount(in_data->FrameCount());
    out_part->SetFrameID(in_data->FrameID());

    return true;
}


bool megamol::thermodyn::ParticleSurface::write_info_cb(core::param::ParamSlot& p) {
    if (write_info_slot_.Param<core::param::BoolParam>()->Value()) {
        auto const part_count = _part_data[0].size();
        std::vector<float> infos;
        infos.reserve(part_count);
        for (uint64_t i = 0; i < part_count; ++i) {
            auto const info = _part_data[0][i].i;
            infos.push_back(info);
        }
        // auto const filename = info_filename_slot_.Param<core::param::FilePathParam>()->Value();
        auto const filename = core::utility::get_extended_filename(
            info_filename_slot_, _frame_id, increment, core::utility::increment_type::INCREMENT_SAFE);
        auto ofs = std::ofstream(filename, std::ios::binary);
        uint64_t data_size = infos.size() * sizeof(float);
        ofs.write(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
        ofs.write(reinterpret_cast<char*>(infos.data()), data_size);
    }

    return true;
}
