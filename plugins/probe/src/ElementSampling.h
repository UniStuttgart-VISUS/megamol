/*
 * ElementSampling.h
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "ProbeCollection.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include "mmcore/BoundingBoxes_2.h"


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
    static const char* ClassName() { return "ElementSampling"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "..."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

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

    template <typename T>
    void doScalarSampling(const std::vector<std::vector<Surface_mesh>>& elements, const std::vector<T>& data,
        const std::vector<T>& data_positions);
    void placeProbes(const std::vector<std::vector<Surface_mesh>>& elements);

    std::shared_ptr<ProbeCollection> _probes;

    size_t _old_datahash;
    bool _trigger_recalc;
    bool paramChanged(core::param::ParamSlot& p);

    megamol::core::BoundingBoxes_2 _bbox;

     std::vector<std::vector<Surface_mesh>> _elements;
};


template <typename T>
void ElementSampling::doScalarSampling(const std::vector<std::vector<Surface_mesh>>& elements, const std::vector<T>& data, const std::vector<T>& data_positions) {

    float global_min = std::numeric_limits<T>::max();
    float global_max = -std::numeric_limits<T>::max();

    // select Element
    for (int j = 0; j < elements[0].size(); ++j) {
        FloatProbe probe;

        auto visitor = [&probe, j, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::Vec4Probe>) {

                probe.m_timestamp = arg.m_timestamp;
                probe.m_value_name = arg.m_value_name;
                probe.m_position = arg.m_position;
                probe.m_direction = arg.m_direction;
                probe.m_begin = arg.m_begin;
                probe.m_end = arg.m_end;
                probe.m_cluster_id = arg.m_cluster_id;

                probe.m_sample_radius = 0;

                _probes->setProbe(j, probe);

            } else if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                probe = arg;

            } else {
                // unknown/incompatible probe type, throw error? do nothing?
            }
        };

        auto generic_probe = _probes->getGenericProbe(j);
        std::visit(visitor, generic_probe);

        std::shared_ptr<FloatProbe::SamplingResult> samples = probe.getSamplingResult();
        float min_value = std::numeric_limits<T>::max();
        float max_value = -std::numeric_limits<T>::max();
        float avg_value = 0.0f;
        samples->samples.resize(elements.size());


        // go through shells
        for (int i = 0; i < elements.size(); ++i) {

            glm::vec3 max_coord = glm::vec3(-std::numeric_limits<float>::infinity());
            glm::vec3 min_coord = glm::vec3(std::numeric_limits<float>::infinity());
            for (auto p : elements[i][j].points()) {
                max_coord.x = std::max<float>(max_coord.x, p.x());
                max_coord.y = std::max<float>(max_coord.y, p.y());
                max_coord.z = std::max<float>(max_coord.z, p.z());
                min_coord.x = std::min<float>(min_coord.x, p.x());
                min_coord.y = std::min<float>(min_coord.y, p.y());
                min_coord.z = std::min<float>(min_coord.z, p.z());
            }

            CGAL::Side_of_triangle_mesh<Surface_mesh, K> inside(elements[i][j]);
            float value = 0;
            int nb_inside = 0;
            float min_data = std::numeric_limits<T>::max();
            float max_data = -std::numeric_limits<T>::max();
            for (int n = 0; n < (data_positions.size()/3); ++n) {

                const glm::vec3 pos =
                    glm::vec3(data_positions[3 * n + 0], data_positions[3 * n + 1], data_positions[3 * n + 2]);

                if (pos.x <= max_coord.x && pos.y <= max_coord.y && pos.z <= max_coord.z && pos.x >= min_coord.x &&
                    pos.y >= min_coord.y && pos.z >= min_coord.z) {

                    const Point p = Point(data_positions[3 * n + 0], data_positions[3 * n + 1], data_positions[3 * n + 2]);

                    CGAL::Bounded_side res = inside(p);
                    if (res == CGAL::ON_BOUNDED_SIDE) {
                        value += data[n];
                        min_data = std::min(min_data, value);
                        max_data = std::max(max_data, value);
                        ++nb_inside;
                    }
                    //if (res == CGAL::ON_BOUNDARY) {
                    //    ++nb_boundary;
                    //}
                }
            }

            value /= nb_inside;
            samples->samples[i] = value;

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

} // namespace probe
} // namespace megamol
