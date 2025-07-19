#include "RDF.h"

#include <fstream>
#include <numbers>
#include <thread>

#include <glm/gtc/type_ptr.hpp>
#include <omp.h>

namespace megamol::optix_owl {
RDF::RDF(std::shared_ptr<std::vector<vec3f>> org_data, std::shared_ptr<std::vector<vec3f>> new_data)
        : org_data_{org_data}
        , new_data_{new_data} {
    auto t1 = std::thread([this, org_data_adr = &org_data]() {
        this->org_Pts_ = std::make_shared<OWLPointcloud>(org_data_adr->get());
        this->org_particleTree_ =
            std::make_shared<kd_tree_t>(3, *this->org_Pts_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        this->org_particleTree_->buildIndex();
    });
    auto t2 = std::thread([this, new_data_adr = &new_data]() {
        this->new_Pts_ = std::make_shared<OWLPointcloud>(new_data_adr->get());
        this->new_particleTree_ =
            std::make_shared<kd_tree_t>(3, *this->new_Pts_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        this->new_particleTree_->buildIndex();
    });
    t1.join();
    t2.join();
}

std::vector<float> compute_histo(float cut_off, unsigned int num_bins, std::shared_ptr<std::vector<vec3f>> data,
    std::shared_ptr<RDF::kd_tree_t> tree) {
    std::vector<float> org_histo(num_bins);

    auto const num_threads = omp_get_max_threads();

    std::vector<std::vector<uint64_t>> org_histo_t(num_threads);
    for (auto& v : org_histo_t) {
        v.resize(num_bins);
    }

    std::vector<std::vector<nanoflann::ResultItem<size_t, float>>> results(num_threads);
    for (auto& r : results) {
        r.reserve(20);
    }

    nanoflann::SearchParameters params = {};
    params.sorted = true;

    auto const diff = (cut_off * cut_off) / static_cast<double>(num_bins);

    uint64_t num_samples = 0;
#pragma omp parallel for reduction(+ : num_samples)
    for (int64_t i = 0; i < data->size(); ++i) {
        auto const& od = (*data)[i];
        auto const threadID = omp_get_thread_num();
        auto const tmp_od = glm::vec3(od.x, od.y, od.z);
        auto const N = tree->radiusSearch(glm::value_ptr(tmp_od), cut_off * cut_off, results[threadID], params);
        if (N > 1) {
            num_samples += N - 1;
            std::vector<uint64_t> tmp_histo(num_bins);
            for (auto it = results[threadID].begin() + 1; it != results[threadID].end(); ++it) {
                auto const idx = it->second / diff;
                ++tmp_histo[idx];
            }
            std::transform(org_histo_t[threadID].begin(), org_histo_t[threadID].end(), tmp_histo.begin(),
                org_histo_t[threadID].begin(), std::plus<uint64_t>());
        }
    }

    for (int i = 1; i < num_threads; ++i) {
        std::transform(org_histo_t[0].begin(), org_histo_t[0].end(), org_histo_t[i].begin(), org_histo_t[0].begin(),
            std::plus<uint64_t>());
    }

    std::transform(org_histo_t[0].begin(), org_histo_t[0].end(), org_histo.begin(),
        [&num_samples](auto const& val) { return static_cast<float>(val) / static_cast<float>(num_samples); });

    return org_histo;
}

std::vector<float> compute_volumes(float cut_off, unsigned int num_bins) {
    std::vector<float> volumes(num_bins);

    auto const diff = cut_off / static_cast<double>(num_bins);

    for (unsigned int i = 0; i < num_bins; ++i) {
        auto const rad = static_cast<float>(i + 1) * diff;
        auto const vol = 4.0f * rad * rad * rad * std::numbers::pi / 3.0f;
        volumes[i] = vol;
    }

    for (unsigned int i = 1; i < num_bins; ++i) {
        volumes[i] -= volumes[i - 1];
    }

    return volumes;
}

std::tuple<std::vector<float>, std::vector<float>> RDF::BuildHistogram(
    float cut_off, unsigned int num_bins, box3f bbox, uint64_t num_particles) {
#if 0
    std::vector<float> org_histo(num_bins);
    std::vector<float> new_histo(num_bins);

    auto const num_threads = omp_get_max_threads();

    std::vector<std::vector<uint64_t>> org_histo_t(num_threads);
    for (auto& v : org_histo_t) {
        v.resize(num_bins);
    }
    std::vector<std::vector<uint64_t>> new_histo_t(num_threads);
    for (auto& v : new_histo_t) {
        v.resize(num_bins);
    }

    std::vector<std::vector<nanoflann::ResultItem<size_t, float>>> results(num_threads);
    for (auto& r : results) {
        r.reserve(20);
    }

    nanoflann::SearchParameters params = {};
    params.sorted = true;

    auto const diff = (cut_off * cut_off) / static_cast<double>(num_bins);

    uint64_t num_samples = 0;
#pragma omp parallel for reduction(+ : num_samples)
    for (int64_t i = 0; i < org_data_->size(); ++i) {
        auto const& od = (*org_data_)[i];
        auto const threadID = omp_get_thread_num();

        auto const N =
            org_particleTree_->radiusSearch(glm::value_ptr(od), cut_off * cut_off, results[threadID], params);
        if (N > 1) {
            num_samples += N - 1;
            std::vector<uint64_t> tmp_histo(num_bins);
            for (auto it = results[threadID].begin() + 1; it != results[threadID].end(); ++it) {
                auto const idx = it->second / diff;
                ++tmp_histo[idx];
            }
            std::transform(org_histo_t[threadID].begin(), org_histo_t[threadID].end(), tmp_histo.begin(),
                org_histo_t[threadID].begin(), std::plus<uint64_t>());
        }
    }

    for (int i = 1; i < num_threads; ++i) {
        std::transform(org_histo_t[0].begin(), org_histo_t[0].end(), org_histo_t[i].begin(), org_histo_t[0].begin(),
            std::plus<uint64_t>());
    }

    std::transform(org_histo_t[0].begin(), org_histo_t[0].end(), org_histo.begin(),
        [&num_samples](auto const& val) { return static_cast<float>(val) / static_cast<float>(num_samples); });
#endif

    auto const ideal_dense = num_particles / bbox.volume();

    auto const org_histo = compute_histo(cut_off, num_bins, org_data_, org_particleTree_);
    auto const new_histo = compute_histo(cut_off, num_bins, new_data_, new_particleTree_);

    auto const volumes = compute_volumes(cut_off, num_bins);

    std::vector<float> org_rdf(num_bins);
    std::vector<float> new_rdf(num_bins);

    std::transform(org_histo.begin(), org_histo.end(), volumes.begin(), org_rdf.begin(), std::divides<float>());
    std::transform(new_histo.begin(), new_histo.end(), volumes.begin(), new_rdf.begin(), std::divides<float>());

    std::transform(
        org_rdf.begin(), org_rdf.end(), org_rdf.begin(), [&ideal_dense](auto const& val) { return val / ideal_dense; });
    std::transform(
        new_rdf.begin(), new_rdf.end(), new_rdf.begin(), [&ideal_dense](auto const& val) { return val / ideal_dense; });

#if 0
    auto f = std::ofstream("org_rdf.blobb");
    f.write(reinterpret_cast<char const*>(org_rdf.data()), org_rdf.size() * sizeof(float));
    f.close();

    f = std::ofstream("new_rdf.blobb");
    f.write(reinterpret_cast<char const*>(new_rdf.data()), new_rdf.size() * sizeof(float));
    f.close();
#endif

    return std::make_tuple(org_rdf, new_rdf);
}
} // namespace megamol::optix_owl
