#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace megamol::protein_calls {
enum class RMSDMode {
    ONLY_RMSD = 0,
    RMSD_CALC_MATRICES = 1,
    RMSD_ONLY_MOVEMENT = 2,
    RMSD_ONLY_ROTATION = 3,
    RMSD_FULL_ALIGNMENT = 4
};

/**
 * Struct for transporting the return value(s) of the RMSD calculation
 */
struct RMSDReturn {
    /** flag determining whether the rotation matrix in this struct is valid */
    bool rotationValid = false;
    /** flag determining whether the translation vector in this struct is valid */
    bool translationValid = false;
    /** center of the toFit coordinates. If weights are given, this is the mass center */
    glm::vec3 toFitCenter = glm::vec3(0.0f);
    /** center of the reference coordinates. If weights are given, this is the mass center */
    glm::vec3 referenceCenter = glm::vec3(0.0f);
    /** translation vector that moves the coordinates to the reference coordinates */
    glm::vec3 translationVector = glm::vec3(0.0f);
    /** rotation matrix that aligns the rotation of the toFit coordinate */
    glm::mat3 rotationMatrix = glm::mat3(1.0f);
    /** the calculated RMSD value */
    float rmsdValue = 0.0f;
    /** flag determining whether the calculation was successful */
    bool success = false;
};

/**
 * Calculates the Root Mean Square Deviation between two sets of coordinates.
 * If asked for, this method also calculates translation and rotation vectors to fit the given coordinates onto each other.
 * If the length of toFit and reference mismatch, only the first n coordinates are compared, where n is the minimum size of both vectors.
 *
 * The mode ONLY_RMSD just calculates the Root Means Square Deviation value but skips all other computations.
 * The mode RMSD_CALC_MATRICES additionally calculates the matrices and vectors necessary for fitting the coordinates onto each other.
 * The mode RMSD_ONLY_MOVEMENT applies the movement onto the reference coordinates but applies no rotation.
 * The mode RMSD_ONLY_ROTATION applies just the rotation onto the reference coordinates's rotation but applies no movement.
 * The mode RMSD_FULL_ALIGNMENT applies both rotation and translation so that the coordinates lie mostly onto each other.
 *
 * @param toFit Coordinates that are fitted to the given reference coordinates
 * @param reference Reference coordinates the toFit coordinates are aligned to
 * @param mode The used calculation mode. For details, see above.
 * @param weights Weight vector to steer the influence of each coordinate. If the size of this vector does not match the size of toFit, weights are set to 1.
 *
 * @return Struct containing all calculated results
 */
RMSDReturn CalculateRMSD(std::vector<glm::vec3>& toFit, const std::vector<glm::vec3>& reference,
    const RMSDMode mode = RMSDMode::ONLY_RMSD, const std::vector<float>& weights = {});

/**
 * Calculates the Root Mean Square Deviation between two sets of coordinates.
 * If the length of toFit and reference mismatch, only the first n coordinates are compared, where n is the minimum size of both vectors.
 *
 * @param toFit Coordinates that are fitted to the given reference coordinates
 * @param reference Reference coordinates the toFit coordinates are aligned to
 * @param weights Weight vector to steer the influence of each coordinate. If the size of this vector does not match the size of toFit, weights are set to 1.
 *
 * @return The RMSD value
 */
float CalculateRMSDValue(
    std::vector<glm::vec3>& toFit, const std::vector<glm::vec3>& reference, const std::vector<float>& weights = {});

} // namespace megamol::protein_calls
