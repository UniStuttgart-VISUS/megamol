#pragma once

#include <array>
#include <unordered_map>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "thermodyn/PathLineDataCall.h"
#include "mmstd_datatools/DBSCAN.h"

namespace megamol {
namespace thermodyn {

class PathClustering : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PathClustering"; }

    /** Return module class description */
    static const char* Description(void) { return "Computes a clustering of particle pathlines"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PathClustering();

    virtual ~PathClustering();

protected:
    bool create() override;

    void release() override;

private:
    using DB_3DIM = stdplugin::datatools::DBSCAN<float, true, 3, false>;
    using DB_4DIM = stdplugin::datatools::DBSCAN<float, true, 4, true>;
    using DB_7DIM = stdplugin::datatools::DBSCAN<float, true, 7, true>;

    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    using vertex_t = std::array<float, 4>;

    // using edge_t = std::pair<int64_t, int64_t>;

    using cluster_assoc_t = std::unordered_map<size_t, std::vector<int64_t>>;

    using vertex_set_t = std::vector<vertex_t>;

    // using edge_set_t = std::vector<edge_t>;

    static void prepareFrameData(std::vector<float>& data, thermodyn::PathLineDataCall::pathline_store_t const& input,
        size_t const fidx, size_t const entrySize, size_t const tempOffset) {
        for (auto const& el : input) {
            auto const& path = el.second;
            data.push_back(path[fidx * entrySize + 0]);
            data.push_back(path[fidx * entrySize + 1]);
            data.push_back(path[fidx * entrySize + 2]);
            // temperature
            data.push_back(path[fidx * entrySize + tempOffset]);
            // id
            data.push_back(static_cast<float>(el.first));
        }
    }

    static void prepareFrameData(std::vector<float>& data, thermodyn::PathLineDataCall::pathline_store_t const& input,
        size_t const fidx, size_t const entrySize, size_t const tempOffset, bool const colsPresent, bool const dirsPresent, bool const useTemp) {
        size_t dirOffset = 3;
        if (colsPresent) dirOffset += 4;
        for (auto const& el : input) {
            auto const& path = el.second;
            data.push_back(path[fidx * entrySize + 0]);
            data.push_back(path[fidx * entrySize + 1]);
            data.push_back(path[fidx * entrySize + 2]);
            if (dirsPresent) {
                data.push_back(path[fidx * entrySize + dirOffset + 0]);
                data.push_back(path[fidx * entrySize + dirOffset + 1]);
                data.push_back(path[fidx * entrySize + dirOffset + 2]);
            }
            // temperature
            if (useTemp) {
                data.push_back(path[fidx * entrySize + tempOffset]);
            }
            // id
            data.push_back(static_cast<float>(el.first));
        }
    }

    static void createClusterAssoc(size_t const idxOffset, std::vector<std::vector<float>> const& clusters,
        cluster_assoc_t& clusterAssoc) {
        for (size_t idx = 0; idx < clusters.size(); ++idx) {
            auto const& cluster = clusters[idx];
            for (size_t pidx = 0; pidx < cluster.size() / 5; ++pidx) {
                auto const id = static_cast<size_t>(cluster[pidx * 5 + 4]);
                clusterAssoc[id].push_back(idxOffset + idx);
            }
        }
    }

    static void createClusterAssoc(size_t const idxOffset, std::vector<std::vector<float>> const& clusters,
        cluster_assoc_t& clusterAssoc, bool const dirsPresent, bool const useTemp) {
        size_t off = 5;
        if (dirsPresent) off = 8;
        if (!useTemp) off -= 1;
        for (size_t idx = 0; idx < clusters.size(); ++idx) {
            auto const& cluster = clusters[idx];
            for (size_t pidx = 0; pidx < cluster.size() / off; ++pidx) {
                auto const id = static_cast<size_t>(cluster[pidx * off + off - 1]);
                clusterAssoc[id].push_back(idxOffset + idx);
            }
        }
    }

    /*static void progressClusterAssoc(cluster_assoc_t& clusterAssoc, std::vector<size_t> const& assigned) {
        for (auto it = clusterAssoc.begin(); it != clusterAssoc.end();) {
            auto fit = std::find(assigned.begin(), assigned.end(), it->first);
            if (fit == assigned.end()) {
                clusterAssoc.erase(it);
            } else {
                ++it;
            }
        }
        for (auto& el : clusterAssoc) {
            auto& p = el.second;
            p.first = p.second;
            p.second = -1;
        }
    }*/

    static vertex_t getClusterCenter(std::vector<float> const& cluster, bool const dirsPresent, bool const useTemp) {
        vertex_t p{0.0f, 0.0f, 0.0f, 0.0f};
        size_t off = 5;
        if (dirsPresent) off = 8;
        if (!useTemp) off -= 1;
        for (size_t pidx = 0; pidx < cluster.size() / off; ++pidx) {
            p[0] += cluster[pidx * off + 0];
            p[1] += cluster[pidx * off + 1];
            p[2] += cluster[pidx * off + 2];
            p[3] += cluster[pidx * off + 3];
        }
        p[0] /= static_cast<float>(cluster.size() / off);
        p[1] /= static_cast<float>(cluster.size() / off);
        p[2] /= static_cast<float>(cluster.size() / off);
        p[3] /= static_cast<float>(cluster.size() / off);
        return p;
    }

    static vertex_set_t getVertexList(std::vector<std::vector<float>> const& clusters, bool const dirsPresent, bool const useTemp) {
        vertex_set_t ret;
        ret.reserve(clusters.size());
        for (auto const& el : clusters) {
            ret.emplace_back(getClusterCenter(el, dirsPresent, useTemp));
        }
        return ret;
    }

    /*static edge_set_t getEdgeList(cluster_assoc_t const& clusterAssoc) {
        edge_set_t ret;
        ret.reserve(clusterAssoc.size());
        for (auto const& el : clusterAssoc) {
            if (el.second.first > -1 && el.second.second > -1) {
                ret.emplace_back(el.second.first, el.second.second);
            }
        }
        ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
        return ret;
    }*/

    static std::vector<float> createPathline(std::vector<int64_t> const& clusterIdxs, vertex_set_t const& vertexSet) {
        std::vector<float> ret;
        ret.reserve(clusterIdxs.size()*3);
        for (auto const& idx : clusterIdxs) {
            ret.push_back(vertexSet[idx][0]);
            ret.push_back(vertexSet[idx][1]);
            ret.push_back(vertexSet[idx][2]);
        }
        return ret;
    }

    bool isDirty() const {
        return minPtsSlot_.IsDirty() || sigmaSlot_.IsDirty() || conDirsSlot_.IsDirty() || conTempSlot_.IsDirty();
    }

    void resetDirty() {
        minPtsSlot_.ResetDirty();
        sigmaSlot_.ResetDirty();
        conDirsSlot_.ResetDirty();
        conTempSlot_.ResetDirty();
    }

    /** input of pathlines */
    core::CallerSlot dataInSlot_;

    /** output of clustered pathlines */
    core::CalleeSlot dataOutSlot_;

    core::param::ParamSlot minPtsSlot_;

    core::param::ParamSlot sigmaSlot_;

    core::param::ParamSlot conDirsSlot_;

    core::param::ParamSlot conTempSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    PathLineDataCall::pathline_store_set_t outPathStore_;

    PathLineDataCall::entrysizes_t outEntrySizes_;

    PathLineDataCall::color_flags_t outColsPresent_;

    PathLineDataCall::dir_flags_t outDirsPresent_;

}; // end class PathClustering

} // end namespace thermodyn
} // end namespace megamol
