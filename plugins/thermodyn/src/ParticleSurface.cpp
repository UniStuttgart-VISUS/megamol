#include "stdafx.h"
#include "ParticleSurface.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "Eigen/Dense"

#include "TFUtils.h"


megamol::thermodyn::ParticleSurface::ParticleSurface()
        : _out_mesh_slot("outMesh", "")
        , _in_data_slot("inData", "")
        , _tf_slot("inTF", "")
        , _alpha_slot("alpha", "")
        , _type_slot("type", "") {
    _out_mesh_slot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ParticleSurface::get_data_cb);
    _out_mesh_slot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ParticleSurface::get_data_cb);
    MakeSlotAvailable(&_out_mesh_slot);

    _in_data_slot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&_in_data_slot);

    _tf_slot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    MakeSlotAvailable(&_tf_slot);

    _alpha_slot << new core::param::FloatParam(1.25f, 0.0f);
    MakeSlotAvailable(&_alpha_slot);

    using surface_type_ut = std::underlying_type_t<surface_type>;
    auto ep = new core::param::EnumParam(static_cast<surface_type_ut>(surface_type::alpha_shape));
    ep->SetTypePair(static_cast<surface_type_ut>(surface_type::alpha_shape), "alpha shape");
    ep->SetTypePair(static_cast<surface_type_ut>(surface_type::gtim), "gtim");
    _type_slot << ep;
    MakeSlotAvailable(&_type_slot);
}


megamol::thermodyn::ParticleSurface::~ParticleSurface() {
    this->Release();
}


bool megamol::thermodyn::ParticleSurface::create() {
    return true;
}


void megamol::thermodyn::ParticleSurface::release() {}


bool compute_touchingsphere_radius(megamol::thermodyn::ParticleSurface::Point_3 const& a_pos, float a_r,
    megamol::thermodyn::ParticleSurface::Point_3 const& b_pos, float b_r,
    megamol::thermodyn::ParticleSurface::Point_3 const& c_pos, float c_r,
    megamol::thermodyn::ParticleSurface::Point_3 const& d_pos, float d_r, float& res) {
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

    if (in_data->DataHash() != _in_data_hash || in_data->FrameID() != _frame_id || is_dirty() || (cgtf != nullptr && cgtf->IsDirty())) {

        auto const pl_count = in_data->GetParticleListCount();
        _vertices.resize(pl_count);
        _normals.resize(pl_count);
        _colors.resize(pl_count);
        _indices.resize(pl_count);

        auto const alpha = _alpha_slot.Param<core::param::FloatParam>()->Value();
        auto const squared_alpha = alpha * alpha;

        auto const type = static_cast<surface_type>(_type_slot.Param<core::param::EnumParam>()->Value());

        _mesh_access_collection = std::make_shared<mesh::MeshDataAccessCollection>();

        float const* color_tf = nullptr;
        auto color_tf_size = 0;
        std::array<float, 2> range = {0.f, 0.f};

        if (in_data->DataHash() != _in_data_hash || in_data->FrameID() != _frame_id) {
            for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
                auto const& parts = in_data->AccessParticles(pl_idx);

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

        /*if (cgtf != nullptr) {
            cgtf->SetRange(range);
            (*cgtf)(0);
            color_tf = cgtf->GetTextureData();
            color_tf_size = cgtf->TextureSize();
            range = cgtf->Range();
        }*/

        for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = in_data->AccessParticles(pl_idx);

            auto const min_i = range[0];
            auto const max_i = range[1];
            auto const fac_i = 1.0f / (max_i - min_i + 1e-8f);

            auto& vertices = _vertices[pl_idx];
            auto& normals = _normals[pl_idx];
            auto& colors = _colors[pl_idx];
            auto& indices = _indices[pl_idx];

            auto const p_count = parts.GetCount();

            auto const xAcc = parts.GetParticleStore().GetXAcc();
            auto const yAcc = parts.GetParticleStore().GetYAcc();
            auto const zAcc = parts.GetParticleStore().GetZAcc();

            auto const iAcc = parts.GetParticleStore().GetCRAcc();

            std::vector<std::pair<Point_3, float>> points;
            points.reserve(p_count);

            for (std::remove_const_t<decltype(p_count)> pidx = 0; pidx < p_count; ++pidx) {
                points.emplace_back(std::make_pair(
                    Point_3(xAcc->Get_f(pidx), yAcc->Get_f(pidx), zAcc->Get_f(pidx)), iAcc->Get_f(pidx)));
            }

            if (type == surface_type::alpha_shape) {
                Triangulation_3 tri = Triangulation_3(points.begin(), points.end());

                Alpha_shape_3 as(tri, alpha);

                std::list<Facet> facets;
                as.get_alpha_shape_facets(std::back_inserter(facets), Alpha_shape_3::REGULAR);

                std::list<Triangle> alpha_mesh;

                for (auto& facet : facets) {
                    /*mesh_points.push_back(facet.first->vertex(Triangulation_3::vertex_triple_index(facet.second,
                    0))->point());
                    mesh_points.push_back(facet.first->vertex(Triangulation_3::vertex_triple_index(facet.second,
                    1))->point());
                    mesh_points.push_back(facet.first->vertex(Triangulation_3::vertex_triple_index(facet.second,
                    2))->point());*/
                    alpha_mesh.push_back(as.triangle(facet));
                }

                vertices.clear();
                vertices.reserve(facets.size() * 9);
                normals.clear();
                normals.reserve(facets.size() * 9);
                colors.clear();
                colors.reserve(facets.size() * 12);
                indices.resize(facets.size() * 3);

                // for (auto& triangle : alpha_mesh) {
                for (auto& face : facets) {
                    /*auto const& a = triangle.vertex(0);
                    auto const& b = triangle.vertex(1);
                    auto const& c = triangle.vertex(2);*/

                    auto const& vert_a = face.first->vertex(as.vertex_triple_index(face.second, 0));
                    auto const& vert_b = face.first->vertex(as.vertex_triple_index(face.second, 1));
                    auto const& vert_c = face.first->vertex(as.vertex_triple_index(face.second, 2));

                    auto const& a = vert_a->point();
                    auto const& b = vert_b->point();
                    auto const& c = vert_c->point();

                    vertices.push_back(a.x());
                    vertices.push_back(a.y());
                    vertices.push_back(a.z());

                    vertices.push_back(b.x());
                    vertices.push_back(b.y());
                    vertices.push_back(b.z());

                    vertices.push_back(c.x());
                    vertices.push_back(c.y());
                    vertices.push_back(c.z());

                    auto normal = CGAL::normal(a, b, c);
                    auto const length = std::sqrtf(normal.squared_length());
                    normal /= length;
                    normals.push_back(normal.x());
                    normals.push_back(normal.y());
                    normals.push_back(normal.z());
                    normals.push_back(normal.x());
                    normals.push_back(normal.y());
                    normals.push_back(normal.z());
                    normals.push_back(normal.x());
                    normals.push_back(normal.y());
                    normals.push_back(normal.z());

                    auto col_a = def_color;
                    auto col_b = def_color;
                    auto col_c = def_color;

                    if (color_tf != nullptr) {
                        auto const val_a = (vert_a->info() - min_i) * fac_i * static_cast<float>(color_tf_size);
                        auto const val_b = (vert_b->info() - min_i) * fac_i * static_cast<float>(color_tf_size);
                        auto const val_c = (vert_c->info() - min_i) * fac_i * static_cast<float>(color_tf_size);
                        std::remove_const_t<decltype(val_a)> main_a = 0;
                        auto rest_a = std::modf(val_a, &main_a);
                        rest_a =
                            static_cast<int>(main_a) >= 0 && static_cast<int>(main_a) < color_tf_size ? rest_a : 0.0f;
                        std::remove_const_t<decltype(val_b)> main_b = 0;
                        auto rest_b = std::modf(val_b, &main_b);
                        rest_b =
                            static_cast<int>(main_b) >= 0 && static_cast<int>(main_b) < color_tf_size ? rest_b : 0.0f;
                        std::remove_const_t<decltype(val_c)> main_c = 0;
                        auto rest_c = std::modf(val_c, &main_c);
                        rest_c =
                            static_cast<int>(main_c) >= 0 && static_cast<int>(main_c) < color_tf_size ? rest_c : 0.0f;
                        main_a = std::clamp(static_cast<int>(main_a), 0, color_tf_size - 1);
                        main_b = std::clamp(static_cast<int>(main_b), 0, color_tf_size - 1);
                        main_c = std::clamp(static_cast<int>(main_c), 0, color_tf_size - 1);
                        col_a =
                            stdplugin::datatools::sample_tf(color_tf, color_tf_size, static_cast<int>(main_a), rest_a);
                        col_b =
                            stdplugin::datatools::sample_tf(color_tf, color_tf_size, static_cast<int>(main_b), rest_b);
                        col_c =
                            stdplugin::datatools::sample_tf(color_tf, color_tf_size, static_cast<int>(main_c), rest_c);
                    }

                    colors.push_back(col_a.r);
                    colors.push_back(col_a.g);
                    colors.push_back(col_a.b);
                    colors.push_back(col_a.a);
                    colors.push_back(col_b.r);
                    colors.push_back(col_b.g);
                    colors.push_back(col_b.b);
                    colors.push_back(col_b.a);
                    colors.push_back(col_c.r);
                    colors.push_back(col_c.g);
                    colors.push_back(col_c.b);
                    colors.push_back(col_c.a);
                }

                std::iota(indices.begin(), indices.end(), 0);
            }
            //} else {
            //    Delaunay tri = Delaunay(points.cbegin(), points.cend());

            //    std::list<Delaunay::Cell> cells;

            //    auto first_cell = tri.cells_begin();
            //    auto center = first_cell->circumcenter();
            //    auto first_point = first_cell->vertex(0)->point();
            //    auto first_length = (center - first_point).squared_length();

            //    DFacet test_facet = DFacet(first_cell, 0);


            //    /*std::copy_if(tri.cells_begin(), tri.cells_end(), std::back_inserter(cells),
            //        [squared_alpha](Delaunay::Cell const& cell) {
            //            auto center = cell.circumcenter();
            //            auto squared_radius = (cell.vertex(0)->point() - center).squared_length();
            //            return squared_radius <= squared_alpha;
            //        });*/
            //    std::copy_if(
            //        tri.cells_begin(), tri.cells_end(), std::back_inserter(cells), [alpha](Delaunay::Cell const& cell)
            //        {
            //            auto const& a = cell.vertex(0)->point();
            //            auto const& b = cell.vertex(1)->point();
            //            auto const& c = cell.vertex(2)->point();
            //            auto const& d = cell.vertex(3)->point();

            //            float radius = 0.0f;
            //            auto res = compute_touchingsphere_radius(a, 0.5f, b, 0.5f, c, 0.5f, d, 0.5f, radius);
            //            return res && (radius <= alpha);
            //        });

            //    vertices.clear();
            //    vertices.reserve(cells.size() * 12);
            //    normals.clear();
            //    normals.reserve(cells.size() * 12);
            //    indices.reserve(cells.size() * 12);

            //    std::size_t idx = 0;
            //    for (auto const& cell : cells) {
            //        auto const base_idx = idx * 4;

            //        auto const& a = cell.vertex(0)->point();
            //        auto const& b = cell.vertex(1)->point();
            //        auto const& c = cell.vertex(2)->point();
            //        auto const& d = cell.vertex(3)->point();

            //        vertices.push_back(a.x());
            //        vertices.push_back(a.y());
            //        vertices.push_back(a.z());

            //        vertices.push_back(b.x());
            //        vertices.push_back(b.y());
            //        vertices.push_back(b.z());

            //        vertices.push_back(c.x());
            //        vertices.push_back(c.y());
            //        vertices.push_back(c.z());

            //        vertices.push_back(d.x());
            //        vertices.push_back(d.y());
            //        vertices.push_back(d.z());

            //        indices.push_back(base_idx + 0);
            //        indices.push_back(base_idx + 1);
            //        indices.push_back(base_idx + 2);

            //        indices.push_back(base_idx + 0);
            //        indices.push_back(base_idx + 1);
            //        indices.push_back(base_idx + 3);

            //        indices.push_back(base_idx + 1);
            //        indices.push_back(base_idx + 2);
            //        indices.push_back(base_idx + 3);

            //        indices.push_back(base_idx + 2);
            //        indices.push_back(base_idx + 0);
            //        indices.push_back(base_idx + 3);

            //        ++idx;
            //    }
            //}


            std::vector<mesh::MeshDataAccessCollection::VertexAttribute> mesh_attributes;
            mesh::MeshDataAccessCollection::IndexData mesh_indices;

            mesh_indices.byte_size = indices.size() * sizeof(uint32_t);
            mesh_indices.data = reinterpret_cast<uint8_t*>(indices.data());
            mesh_indices.type = mesh::MeshDataAccessCollection::UNSIGNED_INT;

            mesh_attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                reinterpret_cast<uint8_t*>(normals.data()), normals.size() * sizeof(float), 3,
                mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0, mesh::MeshDataAccessCollection::NORMAL});

            mesh_attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                reinterpret_cast<uint8_t*>(colors.data()), colors.size() * sizeof(float), 4,
                mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 4, 0, mesh::MeshDataAccessCollection::COLOR});

            mesh_attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                reinterpret_cast<uint8_t*>(vertices.data()), vertices.size() * sizeof(float), 3,
                mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0, mesh::MeshDataAccessCollection::POSITION});

            auto identifier = std::string("particle_surface_") + std::to_string(pl_idx);
            _mesh_access_collection->addMesh(identifier, mesh_attributes, mesh_indices);
        }

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
    out_mesh->setData(_mesh_access_collection, ++_out_data_hash);

    return true;
}


bool megamol::thermodyn::ParticleSurface::get_extent_cb(core::Call& c) {
    auto out_mesh = dynamic_cast<mesh::CallMesh*>(&c);
    if (out_mesh == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    if (!(*in_data)(1))
        return false;

    auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

    auto meta_data = out_mesh->getMetaData();
    meta_data.m_bboxs.SetBoundingBox(bbox);
    meta_data.m_bboxs.SetClipBox(cbox);
    meta_data.m_frame_cnt = in_data->FrameCount();
    meta_data.m_frame_ID = in_data->FrameID();
    out_mesh->setMetaData(meta_data);


    return true;
}
