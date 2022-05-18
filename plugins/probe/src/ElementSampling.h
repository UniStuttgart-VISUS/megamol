/*
 * ElementSampling.h
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#pragma once

#include <CGAL/convex_hull_3_to_face_graph.h>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "probe/ProbeCollection.h"
#include <CGAL/Polygon_mesh_processing/shape_predicates.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>


namespace megamol {
namespace probe {
// default triangulation for Surface_mesher
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
typedef Tr::Geom_traits GT;
typedef GT::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Surface_mesh;

class ElementSampling : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ElementSampling";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    ElementSampling();
    virtual ~ElementSampling();

protected:
    virtual bool create();
    virtual void release();

    uint32_t _version;

    core::CalleeSlot _probe_lhs_slot;

    core::CallerSlot _elements_rhs_slot;
    size_t _elements_cached_hash;

    core::CallerSlot _adios_rhs_slot;
    size_t _adios_cached_hash;

    core::param::ParamSlot _parameter_to_sample_slot;
    core::param::ParamSlot _xSlot;
    core::param::ParamSlot _ySlot;
    core::param::ParamSlot _zSlot;
    core::param::ParamSlot _xyzSlot;
    core::param::ParamSlot _formatSlot;

private:
    bool getData(core::Call& call);
    bool getMetaData(core::Call& call);

    bool readElements();

    template<typename T>
    void doScalarSampling(const std::vector<std::vector<Surface_mesh>>& elements, const std::vector<T>& data,
        const std::vector<T>& data_positions);
    void do_triangulation(Surface_mesh& mesh_);
    void placeProbes(const std::vector<std::vector<Surface_mesh>>& elements);

    std::shared_ptr<ProbeCollection> _probes;

    size_t _old_datahash;
    bool _trigger_recalc;
    bool paramChanged(core::param::ParamSlot& p);

    megamol::core::BoundingBoxes_2 _bbox;

    std::vector<std::vector<Surface_mesh>> _elements;
};


template<typename T>
void ElementSampling::doScalarSampling(const std::vector<std::vector<Surface_mesh>>& elements,
    const std::vector<T>& data, const std::vector<T>& data_positions) {

    float global_min = std::numeric_limits<T>::max();
    float global_max = -std::numeric_limits<T>::max();

    // select Element
    for (int j = 0; j < elements[0].size(); ++j) {
        FloatDistributionProbe probe;


        auto visitor = [&probe, j, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::Vec4Probe> ||
                          std::is_same_v<T, probe::FloatProbe>) {

                probe.m_timestamp = arg.m_timestamp;
                probe.m_value_name = arg.m_value_name;
                probe.m_position = arg.m_position;
                probe.m_direction = arg.m_direction;
                probe.m_begin = arg.m_begin;
                probe.m_end = arg.m_end;
                probe.m_cluster_id = arg.m_cluster_id;
                probe.m_geo_ids = arg.m_geo_ids;

                probe.m_sample_radius = 0;

                _probes->setProbe(j, probe);

            } else if constexpr (std::is_same_v<T, probe::FloatDistributionProbe>) {
                probe = arg;

            } else {
                // unknown/incompatible probe type, throw error? do nothing?
            }
        };

        auto generic_probe = _probes->getGenericProbe(j);
        std::visit(visitor, generic_probe);

        std::shared_ptr<FloatDistributionProbe::SamplingResult> samples = probe.getSamplingResult();
        float min_value = std::numeric_limits<T>::max();
        float max_value = -std::numeric_limits<T>::max();
        float avg_value = 0.0f;
        samples->samples.resize(elements.size());

        // go through shells
        for (int i = 0; i < elements.size(); ++i) {

            glm::vec3 element_max_coord = glm::vec3(-std::numeric_limits<float>::infinity());
            glm::vec3 element_min_coord = glm::vec3(std::numeric_limits<float>::infinity());
            for (auto p : elements[i][j].points()) {
                element_max_coord.x = std::max<float>(element_max_coord.x, p.x());
                element_max_coord.y = std::max<float>(element_max_coord.y, p.y());
                element_max_coord.z = std::max<float>(element_max_coord.z, p.z());
                element_min_coord.x = std::min<float>(element_min_coord.x, p.x());
                element_min_coord.y = std::min<float>(element_min_coord.y, p.y());
                element_min_coord.z = std::min<float>(element_min_coord.z, p.z());
            }

            typedef boost::graph_traits<Surface_mesh>::edge_descriptor edge_descriptor;
            std::vector<edge_descriptor> edges;
            Surface_mesh current_mesh = elements[i][j];

            CGAL::Polygon_mesh_processing::degenerate_edges(current_mesh, std::back_inserter(edges));
            if (!edges.empty()) {
                core::utility::log::Log::DefaultLog.WriteWarn("[ElementSampling] degenerated mesh detected.");
                //this->do_triangulation(current_mesh);
            }
            std::shared_ptr<CGAL::Side_of_triangle_mesh<Surface_mesh, K>> inside;
            try {
                inside = std::make_shared<CGAL::Side_of_triangle_mesh<Surface_mesh, K>>(current_mesh);
            } catch (std::exception& e) {
                std::string message = "[ElementSampling] Could not create inside check: " + std::string(e.what());
                core::utility::log::Log::DefaultLog.WriteError(message.c_str());
                return;
            }
            float value = 0;
            int nb_inside = 0;
            float min_data = std::numeric_limits<T>::max();
            float max_data = -std::numeric_limits<T>::max();
#pragma parallel for
            for (int n = 0; n < (data_positions.size() / 3); ++n) {

                const glm::vec3 pos =
                    glm::vec3(data_positions[3 * n + 0], data_positions[3 * n + 1], data_positions[3 * n + 2]);

                if (pos.x <= element_max_coord.x && pos.y <= element_max_coord.y && pos.z <= element_max_coord.z &&
                    pos.x >= element_min_coord.x && pos.y >= element_min_coord.y && pos.z >= element_min_coord.z) {

                    const Point p =
                        Point(data_positions[3 * n + 0], data_positions[3 * n + 1], data_positions[3 * n + 2]);

                    CGAL::Bounded_side res;
                    try {
                        res = inside->operator()(p);
                    } catch (std::exception& e) {
                        std::string message = "[ElementSampling] Inside check threw error:" + std::string(e.what());
                        core::utility::log::Log::DefaultLog.WriteError(message.c_str());
                        return;
                    }
                    if (res == CGAL::ON_BOUNDED_SIDE) {
                        value += data[n];
                        min_data = std::min(min_data, data[n]);
                        max_data = std::max(max_data, data[n]);
                        ++nb_inside;
                    }
                    //if (res == CGAL::ON_BOUNDARY) {
                    //    ++nb_boundary;
                    //}
                }
            }

            value = nb_inside > 0 ? value / nb_inside : value;
            min_data = nb_inside > 0 ? min_data : value;
            max_data = nb_inside > 0 ? max_data : value;
            FloatDistributionProbe::SampleValue s_result;
            s_result.mean = value;
            s_result.lower_bound = min_data;
            s_result.upper_bound = max_data;

            samples->samples[i] = s_result;

            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);


            avg_value += value;
        } // end for shells
        avg_value /= elements.size();

        samples->average_value = avg_value;
        samples->max_value = max_value;
        samples->min_value = min_value;

        global_min = std::min(global_min, samples->min_value);
        global_max = std::max(global_max, samples->max_value);
    } // end for elements
    _probes->setGlobalMinMax(global_min, global_max);
}

inline void ElementSampling::do_triangulation(Surface_mesh& mesh_) {

    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Point_3 Point_3;
    std::list<Point_3> points_for_triangulation;

    for (auto point : mesh_.points()) {
        points_for_triangulation.push_back(Point(point.x(), point.y(), point.z()));
    }

    typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
    typedef Delaunay::Vertex_handle Vertex_handle;
    typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
    // void CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, TriangleMesh & graph)

    Delaunay T(points_for_triangulation.begin(), points_for_triangulation.end());
    mesh_.clear();
    CGAL::convex_hull_3_to_face_graph(T, mesh_);
}

} // namespace probe
} // namespace megamol
