/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */
#include "protein_calls/RMSD.h"

#include <Eigen/Eigen>
#include <glm/gtc/type_ptr.hpp>

#include "mmcore/utility/log/Log.h"

megamol::protein_calls::RMSDReturn megamol::protein_calls::CalculateRMSD(std::vector<glm::vec3>& toFit,
    const std::vector<glm::vec3>& reference, const RMSDMode mode, const std::vector<float>& weights) {
    RMSDReturn result{};
    size_t const len = std::min(toFit.size(), reference.size());

    if (len < 2) {
        result.success = false;
        return result;
    }

    bool const useMasses = toFit.size() == weights.size() ? true : false;
    // calculate the Root Mean Square Deviation
    float massSum = 0.0f;
    result.rmsdValue = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        const float weight = useMasses ? weights[i] : 1.0f;
        massSum += weight;
        const auto diff = reference[i] - toFit[i];
        result.rmsdValue += weight * (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
    }
    result.rmsdValue = std::sqrt(result.rmsdValue / massSum);

    // if we just want to have the RMSD value, we can return here
    if (mode == RMSDMode::ONLY_RMSD) {
        result.success = true;
        return result;
    }

    /**
     * The following is a direct implementation of the Kabsch algorithm as described in
     * https://en.wikipedia.org/wiki/Kabsch_algorithm
     */

    // calculate (mass) center of both toFit and reference
    // as well as the translation vector
    glm::vec3 fitCenter = toFit.front();
    glm::vec3 refCenter = reference.front();
    for (size_t i = 1; i < len; ++i) {
        fitCenter += toFit[i];
        refCenter += reference[i];
    }
    fitCenter /= massSum;
    refCenter /= massSum;
    result.toFitCenter = fitCenter;
    result.referenceCenter = refCenter;
    result.translationVector = refCenter - fitCenter;
    result.translationValid = true;

    // calculate covariance matrix H
    Eigen::MatrixXf P_base(len, 3);
    Eigen::MatrixXf Q_ref(len, 3);
    for (size_t i = 0; i < len; ++i) {
        // move the mass centers to origin beforehand
        P_base(i, 0) = toFit[i].x - fitCenter.x;
        P_base(i, 1) = toFit[i].y - fitCenter.y;
        P_base(i, 2) = toFit[i].z - fitCenter.z;
        Q_ref(i, 0) = reference[i].x - refCenter.x;
        Q_ref(i, 1) = reference[i].y - refCenter.y;
        Q_ref(i, 2) = reference[i].z - refCenter.z;
    }
    P_base.transposeInPlace();
    Eigen::Matrix3f H = P_base * Q_ref; // H = P_trans * Q

    // calculate Singular Value Decomposition of H
    // H = U*S*V_trans
    // we are only interested in U and V
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto U = svd.matrixU();
    auto V = svd.matrixV();
    // d = sign(det(V*U_trans))
    auto d = (V * U.transpose()).determinant();
    d = d >= 0.0f ? 1.0f : -1.0f;

    // calculate the rotation matrix
    // R = V * M * U_trans
    // where M is a unit matrix with the lower right corner set to d
    Eigen::Matrix3f M;
    M << 1, 0, 0, 0, 1, 0, 0, 0, d;
    Eigen::Matrix3f R = V * M * U.transpose();
    // here, the Kabsch algorithm is finished

    // transfer result to glm (both should be column major at this point)
    result.rotationMatrix = glm::make_mat3(R.data());
    result.rotationValid = true;

    if (mode == RMSDMode::RMSD_CALC_MATRICES) {
        result.success = true;
        return result;
    }

    // apply the rotation if requested
    if (mode == RMSDMode::RMSD_ONLY_ROTATION || mode == RMSDMode::RMSD_FULL_ALIGNMENT) {
        for (auto& coord : toFit) {
            coord -= fitCenter;                    // move to origin
            coord = result.rotationMatrix * coord; // apply rotation
            coord += fitCenter;                    // move back
        }
    }

    // apply the translation if requested
    if (mode == RMSDMode::RMSD_ONLY_MOVEMENT || mode == RMSDMode::RMSD_FULL_ALIGNMENT) {
        for (auto& coord : toFit) {
            coord += result.translationVector;
        }
    }

    result.success = true;
    return result;
}

float megamol::protein_calls::CalculateRMSDValue(
    std::vector<glm::vec3>& toFit, const std::vector<glm::vec3>& reference, const std::vector<float>& weights) {
    const auto res = CalculateRMSD(toFit, reference, RMSDMode::ONLY_RMSD, weights);
    if (!res.success) {
        core::utility::log::Log::DefaultLog.WriteError(
            "The RMSD value calculation failed. This may be due to wrongly sized input arrays.");
        return 0.0;
    }
    return res.rmsdValue;
}
