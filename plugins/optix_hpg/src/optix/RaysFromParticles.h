#pragma once

#include <unordered_map>
#include <vector>

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "glm/glm.hpp"

#include "optix/utils_host.h"

namespace megamol::optix_hpg {
inline std::vector<std::vector<RayH>> rays_from_particles(
    core::moldyn::MultiParticleDataCall& call, unsigned int base_frame_id, unsigned int frame_skip) {
    std::vector<std::vector<RayH>> rays;
    std::vector<std::vector<std::pair<glm::vec3, glm::vec3>>> origins;
    std::vector<std::unordered_map<std::uint64_t, std::uint64_t>> idx_map;

    bool got_0 = false;
    do {
        call.SetFrameID(base_frame_id, true);
        got_0 = call(1);
        got_0 = got_0 && call(0);
    } while (call.FrameID() != base_frame_id && !got_0 && (call.Unlock(), true));

    auto const a_pl_count = call.GetParticleListCount();

    rays.resize(a_pl_count);
    origins.resize(a_pl_count);
    idx_map.resize(a_pl_count);

    for (unsigned int plIdx = 0; plIdx < a_pl_count; ++plIdx) {
        auto const& part_a = call.AccessParticles(plIdx);
        auto const a_pCount = part_a.GetCount();

        auto& orgs = origins[plIdx];
        auto& idc = idx_map[plIdx];

        orgs.resize(a_pCount);
        idc.reserve(a_pCount);
        std::fill(orgs.begin(), orgs.end(),
            std::make_pair<glm::vec3, glm::vec3>(
                glm::vec3(std::numeric_limits<float>::lowest()), glm::vec3(std::numeric_limits<float>::lowest())));

        auto a_xAcc = part_a.GetParticleStore().GetXAcc();
        auto a_yAcc = part_a.GetParticleStore().GetYAcc();
        auto a_zAcc = part_a.GetParticleStore().GetZAcc();
        auto a_idAcc = part_a.GetParticleStore().GetIDAcc();

        for (std::size_t pIdx = 0; pIdx < a_pCount; ++pIdx) {
            orgs[pIdx] = std::make_pair(glm::vec3(a_xAcc->Get_f(pIdx), a_yAcc->Get_f(pIdx), a_zAcc->Get_f(pIdx)),
                glm::vec3(std::numeric_limits<float>::lowest()));
            idc[a_idAcc->Get_u64(pIdx)] = pIdx;
        }
    }

    bool got_1 = false;
    do {
        call.SetFrameID(base_frame_id + 1, true);
        got_1 = call(1);
        got_1 = got_1 && call(0);
    } while (call.FrameID() != base_frame_id + 1 && !got_1 && (call.Unlock(), true));

    auto const b_pl_count = call.GetParticleListCount();
    if (b_pl_count != a_pl_count) {
        throw std::runtime_error("[RaysFromParticles] Particle list count does not match");
    }

    for (unsigned int plIdx = 0; plIdx < a_pl_count; ++plIdx) {
        auto const& part_b = call.AccessParticles(plIdx);
        auto const b_pCount = part_b.GetCount();

        auto& orgs = origins[plIdx];
        auto& idc = idx_map[plIdx];

        auto b_xAcc = part_b.GetParticleStore().GetXAcc();
        auto b_yAcc = part_b.GetParticleStore().GetYAcc();
        auto b_zAcc = part_b.GetParticleStore().GetZAcc();
        auto b_idAcc = part_b.GetParticleStore().GetIDAcc();

        for (std::size_t pIdx = 0; pIdx < b_pCount; ++pIdx) {
            auto const id = b_idAcc->Get_u64(pIdx);
            auto fit = idc.find(id);
            if (fit != idc.end()) {
                auto const idx = fit->second;
                orgs[idx].second = glm::vec3(b_xAcc->Get_f(pIdx), b_yAcc->Get_f(pIdx), b_zAcc->Get_f(pIdx));
            }
        }

        auto& ray_vec = rays[plIdx];
        ray_vec.reserve(origins.size());

        for (auto const& el : orgs) {
            auto const& origin = el.first;
            auto const& dest = el.second;

            if (origin.x <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                origin.y <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                origin.z <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                dest.x <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                dest.y <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                dest.z <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon())
                continue;

            /*if (std::isnan(origin.x) || std::isnan(origin.y) || std::isnan(origin.z) || std::isnan(dest.x) ||
                std::isnan(dest.y) || std::isnan(dest.z))
                continue;*/

            auto const& dir = dest - origin;
            //auto dir = glm::vec3(0.f, 0.f, 1.f);
            auto const tMax = glm::length(dir);
            if (tMax != 0.0f) {
                // ray_vec.emplace_back(origin, dir, 0.f, std::numeric_limits<float>::max());
                ray_vec.emplace_back(
                    origin, glm::normalize(dir), std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
            } else {
                ray_vec.emplace_back(
                    origin, glm::normalize(dir), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
            }
        }
    }

    return rays;
}

inline std::vector<std::vector<RayH>> rays_from_particles_rng(
    core::moldyn::MultiParticleDataCall& call, unsigned int base_frame_id) {
    std::vector<std::vector<RayH>> rays;
    std::vector<std::vector<std::pair<glm::vec3, glm::vec3>>> origins;
    std::vector<std::unordered_map<std::uint64_t, std::uint64_t>> idx_map;

    bool got_0 = false;
    do {
        call.SetFrameID(base_frame_id, true);
        got_0 = call(1);
        got_0 = got_0 && call(0);
    } while (call.FrameID() != base_frame_id && !got_0 && (call.Unlock(), true));

    auto const a_pl_count = call.GetParticleListCount();

    rays.resize(a_pl_count);
    origins.resize(a_pl_count);

    for (unsigned int plIdx = 0; plIdx < a_pl_count; ++plIdx) {
        auto const& part_a = call.AccessParticles(plIdx);
        auto const a_pCount = part_a.GetCount();

        auto& orgs = origins[plIdx];

        orgs.resize(a_pCount);
        std::fill(orgs.begin(), orgs.end(),
            std::make_pair<glm::vec3, glm::vec3>(
                glm::vec3(std::numeric_limits<float>::lowest()), glm::vec3(std::numeric_limits<float>::lowest())));

        auto a_xAcc = part_a.GetParticleStore().GetXAcc();
        auto a_yAcc = part_a.GetParticleStore().GetYAcc();
        auto a_zAcc = part_a.GetParticleStore().GetZAcc();
        auto a_idAcc = part_a.GetParticleStore().GetIDAcc();

        for (std::size_t pIdx = 0; pIdx < a_pCount; ++pIdx) {
            orgs[pIdx] = std::make_pair(glm::vec3(a_xAcc->Get_f(pIdx), a_yAcc->Get_f(pIdx), a_zAcc->Get_f(pIdx)),
                glm::vec3(std::numeric_limits<float>::lowest()));
        }
    }

    for (unsigned int plIdx = 0; plIdx < a_pl_count; ++plIdx) {
        auto const& part_b = call.AccessParticles(plIdx);

        auto& orgs = origins[plIdx];

        auto& ray_vec = rays[plIdx];
        ray_vec.reserve(origins.size());

        for (auto const& el : orgs) {
            auto const& origin = el.first;

            /*if (std::isnan(origin.x) || std::isnan(origin.y) || std::isnan(origin.z) || std::isnan(dest.x) ||
                std::isnan(dest.y) || std::isnan(dest.z))
                continue;*/

            // auto const& dir = dest - origin;
            auto dir = glm::vec3(0.f, 0.f, 1.f);
            auto const tMax = glm::length(dir);
            if (tMax != 0.0f) {
                // ray_vec.emplace_back(origin, dir, 0.f, std::numeric_limits<float>::max());
                ray_vec.emplace_back(
                    origin, glm::normalize(dir), std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
            } else {
                ray_vec.emplace_back(
                    origin, glm::normalize(dir), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
            }
        }
    }

    return rays;
}
} // namespace megamol::optix_hpg
