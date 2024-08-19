#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <vector>

#include <owl/common/math/vec.h>

#include "RDF.h"

namespace megamol::optix_owl {
using namespace owl::common;
void dump_analysis_data(std::filesystem::path const& output_path, std::shared_ptr<std::vector<vec3f>> op_s,
    std::shared_ptr<std::vector<vec3f>> sp_s, std::shared_ptr<std::vector<vec3f>> diffs, unsigned int pl_idx,
    float radius, bool do_rdf) {
    if (do_rdf) {
        auto rdf = RDF(op_s, sp_s);
        auto const [org_rdf, new_rdf] = rdf.BuildHistogram(4.0f * radius, 100);

        {
            auto f = std::ofstream(output_path / ("org_rdf_" + std::to_string(pl_idx) + ".blobb"), std::ios::binary);
            f.write(
                reinterpret_cast<char const*>(org_rdf.data()), org_rdf.size() * sizeof(decltype(org_rdf)::value_type));
            f.close();
        }

        {
            auto f = std::ofstream(output_path / ("new_rdf_" + std::to_string(pl_idx) + ".blobb"), std::ios::binary);
            f.write(
                reinterpret_cast<char const*>(new_rdf.data()), new_rdf.size() * sizeof(decltype(new_rdf)::value_type));
            f.close();
        }
    }

    {
        auto const dx_minmax = std::minmax_element(
            diffs->begin(), diffs->end(), [](auto const& lhs, auto const& rhs) { return lhs.x < rhs.x; });
        auto const dy_minmax = std::minmax_element(
            diffs->begin(), diffs->end(), [](auto const& lhs, auto const& rhs) { return lhs.y < rhs.y; });
        auto const dz_minmax = std::minmax_element(
            diffs->begin(), diffs->end(), [](auto const& lhs, auto const& rhs) { return lhs.z < rhs.z; });
        auto const d_acc = std::accumulate(diffs->begin(), diffs->end(), vec3f(0),
            [](auto const& lhs, auto const& rhs) { return abs(lhs) + abs(rhs); });
        auto const csv_file_path = output_path / "comp_stats.csv";
        if (std::filesystem::exists(csv_file_path)) {
            // already exists ... append stats
            auto f = std::ofstream(csv_file_path, std::ios::app);
            f << dx_minmax.first->x << "," << dx_minmax.second->x << "," << dy_minmax.first->y << ","
              << dy_minmax.second->y << "," << dz_minmax.first->z << "," << dz_minmax.second->z << ","
              << d_acc.x / static_cast<float>(diffs->size()) << "," << d_acc.y / static_cast<float>(diffs->size())
              << "," << d_acc.z / static_cast<float>(diffs->size()) << "\n";
            f.close();
        } else {
            // create file
            auto f = std::ofstream(csv_file_path);
            f << "dx_min,dx_max,dy_min,dy_max,dz_min,dz_max,dx_mean,dy_mean,dz_mean\n";
            f << dx_minmax.first->x << "," << dx_minmax.second->x << "," << dy_minmax.first->y << ","
              << dy_minmax.second->y << "," << dz_minmax.first->z << "," << dz_minmax.second->z << ","
              << d_acc.x / static_cast<float>(diffs->size()) << "," << d_acc.y / static_cast<float>(diffs->size())
              << "," << d_acc.z / static_cast<float>(diffs->size()) << "\n";
            f.close();
        }
    }
}
} // namespace megamol::optix_owl
