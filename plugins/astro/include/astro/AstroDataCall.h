/*
 * AstroDataCall.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ASTRODATACALL_H_INCLUDED
#define MEGAMOLCORE_ASTRODATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/math/Cuboid.h"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace megamol::astro {

typedef std::shared_ptr<std::vector<glm::vec3>> vec3ArrayPtr;
typedef std::shared_ptr<std::vector<float>> floatArrayPtr;
typedef std::shared_ptr<std::vector<bool>> boolArrayPtr;
typedef std::shared_ptr<std::vector<int64_t>> idArrayPtr;

class AstroDataCall : public core::AbstractGetData3DCall {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "AstroDataCall";
    }


    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to get astronomical particle data.";
    }

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /** Index of the 'GetExtent' function */
    static const unsigned int CallForGetExtent;

    /** Ctor. */
    AstroDataCall();

    /** Dtor. */
    ~AstroDataCall() override;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 2;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "getData";
        case 1:
            return "getExtent";
        }
        return "";
    }

    /**
     * Sets the position vector
     *
     * @param positionVec Pointer to the new position vector to be set
     */
    inline void SetPositions(vec3ArrayPtr& positionVec) {
        this->positions = positionVec;
    }

    /**
     * Retrieve the pointer to the vector storing the positions
     *
     * @return Pointer to the position array
     */
    inline const vec3ArrayPtr GetPositions() const {
        return this->positions;
    }

    /**
     * Sets the velocity vector
     *
     * @param velocityVec Pointer to the new velocity vector to be set
     */
    inline void SetVelocities(vec3ArrayPtr& velocityVec) {
        this->velocities = velocityVec;
    }

    /**
     * Retrieve the pointer to the vector storing the velocities
     *
     * @return Pointer to the velocity array
     */
    inline const vec3ArrayPtr GetVelocities() const {
        return this->velocities;
    }

    /**
     * Sets the velocity derivative vector
     *
     * @param derivatives Pointer to the new velocity derivative vector to be set
     */
    inline void SetVelocityDerivatives(vec3ArrayPtr& derivatives) {
        this->velocityDerivatives = derivatives;
    }

    /**
     * Retrieve the pointer to the vector storing the velocity derivatives
     *
     * @return Pointer to the velocity derivative array
     */
    inline const vec3ArrayPtr GetVelocityDerivatives() const {
        return this->velocityDerivatives;
    }

    /**
     * Sets the temperature vector
     *
     * @param temparatureVec Pointer to the new temperature vector to be set
     */
    inline void SetTemperature(floatArrayPtr& temparatureVec) {
        this->temperatures = temparatureVec;
    }

    /**
     * Retrieve the pointer to the vector storing the temperature
     *
     * @return Pointer to the temperature array
     */
    inline const floatArrayPtr GetTemperature() const {
        return this->temperatures;
    }

    /**
     * Sets the temperature derivative vector
     *
     * @param derivatives Pointer to the new temperature derivative vector to be set
     */
    inline void SetTemperatureDerivatives(floatArrayPtr& derivatives) {
        this->temperatureDerivatives = derivatives;
    }

    /**
     * Retrieve the pointer to the vector storing the temperature derivatives
     *
     * @return Pointer to the temperature derivative array
     */
    inline const floatArrayPtr GetTemperatureDerivatives() const {
        return this->temperatureDerivatives;
    }

    /**
     * Sets the mass vector
     *
     * @param massVec Pointer to the new mass vector to be set
     */
    inline void SetMass(floatArrayPtr& massVec) {
        this->masses = massVec;
    }

    /**
     * Retrieve the pointer to the vector storing the mass
     *
     * @return Pointer to the mass array
     */
    inline const floatArrayPtr GetMass() const {
        return this->masses;
    }

    /**
     * Sets the internal energy vector
     *
     * @param internalEnergyVec Pointer to the new internal energy vector to be set
     */
    inline void SetInternalEnergy(floatArrayPtr& internalEnergyVec) {
        this->internalEnergies = internalEnergyVec;
    }

    /**
     * Retrieve the pointer to the vector storing the internal energy
     *
     * @return Pointer to the internal energy array
     */
    inline const floatArrayPtr GetInternalEnergy() const {
        return this->internalEnergies;
    }

    /**
     * Sets the internal energy derivative vector
     *
     * @param derivative Pointer to the new internal energy derivative vector to be set
     */
    inline void SetInternalEnergyDerivatives(floatArrayPtr& derivative) {
        this->internalEnergyDerivatives = derivative;
    }

    /**
     * Retrieve the pointer to the vector storing the internal energy derivatives
     *
     * @return Pointer to the internal energy derivative array
     */
    inline const floatArrayPtr GetInternalEnergyDerivatives() const {
        return this->internalEnergyDerivatives;
    }

    /**
     * Sets the smoothing length vector
     *
     * @param smoothingLengthVec Pointer to the new smoothing length vector to be set
     */
    inline void SetSmoothingLength(floatArrayPtr& smoothingLengthVec) {
        this->smoothingLengths = smoothingLengthVec;
    }

    /**
     * Retrieve the pointer to the vector storing the smoothing length
     *
     * @return Pointer to the smoothing length array
     */
    inline const floatArrayPtr GetSmoothingLength() const {
        return this->smoothingLengths;
    }

    /**
     * Sets the smoothing length derivative vector
     *
     * @param derivative Pointer to the new smoothing length derivative vector to be set
     */
    inline void SetSmoothingLengthDerivatives(floatArrayPtr& derivative) {
        this->smoothingLengthDerivatives = derivative;
    }

    /**
     * Retrieve the pointer to the vector storing the smoothing length
     *
     * @return Pointer to the smoothing length array
     */
    inline const floatArrayPtr GetSmoothingLengthDerivatives() const {
        return this->smoothingLengthDerivatives;
    }

    /**
     * Sets the molecular weight vector
     *
     * @param molecularWeightVec Pointer to the new molecular weight vector to be set
     */
    inline void SetMolecularWeights(floatArrayPtr& molecularWeightVec) {
        this->molecularWeights = molecularWeightVec;
    }

    /**
     * Retrieve the pointer to the vector storing the molecular weight
     *
     * @return Pointer to the molecular weight array
     */
    inline const floatArrayPtr GetMolecularWeights() const {
        return this->molecularWeights;
    }

    /**
     * Sets the molecular weight derivative vector
     *
     * @param derivative Pointer to the new molecular weight derivative vector to be set
     */
    inline void SetMolecularWeightDerivatives(floatArrayPtr& derivative) {
        this->molecularWeightDerivatives = derivative;
    }

    /**
     * Retrieve the pointer to the vector storing the molecular weight derivative
     *
     * @return Pointer to the molecular weight derivative array
     */
    inline const floatArrayPtr GetMolecularWeightDerivatives() const {
        return this->molecularWeightDerivatives;
    }

    /**
     * Sets the density vector
     *
     * @param densityVec Pointer to the new density vector to be set
     */
    inline void SetDensity(floatArrayPtr& densityVec) {
        this->densities = densityVec;
    }

    /**
     * Retrieve the pointer to the vector storing the density
     *
     * @return Pointer to the density array
     */
    inline const floatArrayPtr GetDensity() const {
        return this->densities;
    }

    /**
     * Sets the density derivative vector
     *
     * @param derivative Pointer to the new density derivative vector to be set
     */
    inline void SetDensityDerivative(floatArrayPtr& derivative) {
        this->densityDerivatives = derivative;
    }

    /**
     * Retrieve the pointer to the vector storing the density derivative
     *
     * @return Pointer to the density derivative array
     */
    inline const floatArrayPtr GetDensityDerivative() const {
        return this->densityDerivatives;
    }

    /**
     * Sets the gravitational potential vector
     *
     * @param gravitationalPotentialVec Pointer to the new gravitational potential vector to be set
     */
    inline void SetGravitationalPotential(floatArrayPtr& gravitationalPotentialVec) {
        this->gravitationalPotentials = gravitationalPotentialVec;
    }

    /**
     * Retrieve the pointer to the vector storing the gravitational potential
     *
     * @return Pointer to the gravitational potential array
     */
    inline const floatArrayPtr GetGravitationalPotential() const {
        return this->gravitationalPotentials;
    }

    /**
     * Sets the gravitational potential derivative vector
     *
     * @param derivative Pointer to the new gravitational potential derivative vector to be set
     */
    inline void SetGravitationalPotentialDerivatives(floatArrayPtr& derivative) {
        this->gravitationalPotentialDerivatives = derivative;
    }

    /**
     * Retrieve the pointer to the vector storing the gravitational potential derivatives
     *
     * @return Pointer to the gravitational potential derivative array
     */
    inline const floatArrayPtr GetGravitationalPotentialDerivatives() const {
        return this->gravitationalPotentialDerivatives;
    }

    /**
     * Sets the entropy vector
     *
     * @param entropyVec Pointer to the new entropy vector to be set
     */
    inline void SetEntropy(floatArrayPtr& entropyVec) {
        this->entropies = entropyVec;
    }

    /**
     * Retrieve the pointer to the vector storing the entropy
     *
     * @return Pointer to the entropy array
     */
    inline const floatArrayPtr GetEntropy() const {
        return this->entropies;
    }

    /**
     * Sets the entropy derivative vector
     *
     * @param derivative Pointer to the new entropy derivative vector to be set
     */
    inline void SetEntropyDerivatives(floatArrayPtr& derivative) {
        this->entropyDerivatives = derivative;
    }

    /**
     * Retrieve the pointer to the vector storing the entropy derivatives
     *
     * @return Pointer to the entropy derivative array
     */
    inline const floatArrayPtr GetEntropyDerivatives() const {
        return this->entropyDerivatives;
    }

    /**
     * Sets the baryon flag vector
     *
     * @param isBaryonVec Pointer to the new baryon flag vector to be set
     */
    inline void SetIsBaryonFlags(boolArrayPtr& isBaryonVec) {
        this->isBaryonFlags = isBaryonVec;
    }

    /**
     * Retrieve the pointer to the vector storing the baryon flags
     * The content is true if the respecting particle is a baryon particle. It will contain 'false' if the particle is
     * dark matter.
     *
     * @return Pointer to the baryon flag array
     */
    inline const boolArrayPtr GetIsBaryonFlags() const {
        return this->isBaryonFlags;
    }

    /**
     * Sets the star flag vector
     *
     * @param isStarVec Pointer to the new star flag vector to be set
     */
    inline void SetIsStarFlags(boolArrayPtr& isStarVec) {
        this->isStarFlags = isStarVec;
    }

    /**
     * Retrieve the pointer to the vector storing the star flags
     *
     * @return Pointer to the star flag array
     */
    inline const boolArrayPtr GetIsStarFlags() const {
        return this->isStarFlags;
    }

    /**
     * Sets the wind flag vector
     *
     * @param isWindVec Pointer to the new wind flag vector to be set
     */
    inline void SetIsWindFlags(boolArrayPtr& isWindVec) {
        this->isWindFlags = isWindVec;
    }

    /**
     * Retrieve the pointer to the vector storing the wind flags
     *
     * @return Pointer to the wind flag array
     */
    inline const boolArrayPtr GetIsWindFlags() const {
        return this->isWindFlags;
    }

    /**
     * Sets the star forming gas flag vector
     *
     * @param isWindVec Pointer to the new star forming gas flag vector to be set
     */
    inline void SetIsStarFormingGasFlags(boolArrayPtr& isStarFormingGasVec) {
        this->isStarFormingGasFlags = isStarFormingGasVec;
    }

    /**
     * Retrieve the pointer to the vector storing the star forming gas flags
     *
     * @return Pointer to the star forming gas flag array
     */
    inline const boolArrayPtr GetIsStarFormingGasFlags() const {
        return this->isStarFormingGasFlags;
    }

    /**
     * Sets the AGN flag vector
     *
     * @param isWindVec Pointer to the new AGN flag vector to be set
     */
    inline void SetIsAGNFlags(boolArrayPtr& isAGNVec) {
        this->isAGNFlags = isAGNVec;
    }

    /**
     * Retrieve the pointer to the vector storing the AGN flags
     *
     * @return Pointer to the AGN flag array
     */
    inline const boolArrayPtr GetIsAGNFlags() const {
        return this->isAGNFlags;
    }

    /**
     * Sets the particle ID vector
     *
     * @param particleIDVec Pointer to the new particle ID vector to be set
     */
    inline void SetParticleIDs(idArrayPtr& particleIDVec) {
        this->particleIDs = particleIDVec;
    }

    /**
     * Retrieve the pointer to the vector storing the particle IDs
     *
     * @return Pointer to the particle ID array
     */
    inline const idArrayPtr GetParticleIDs() const {
        return this->particleIDs;
    }

    /**
     * Sets the AGN distance vector
     *
     * @param agnDistances Pointer to the new agn distance vector to be set
     */
    inline void SetAGNDistances(floatArrayPtr& agnDistances) {
        this->agnDistances = agnDistances;
    }

    /**
     * Retrieve the pointer to the vector storing the AGN distances
     *
     * @return Pointer to the agn distance array
     */
    inline const floatArrayPtr GetAgnDistances() const {
        return this->agnDistances;
    }

    /**
     * Retrieve the number of particles stored in this call.
     * This will only result in a value greater 0 if the positions array is set.
     * All other arrays may be unset.
     *
     * @return The numbers of particles stored
     */
    inline size_t GetParticleCount() const {
        if (positions == nullptr)
            return 0;
        return positions->size();
    }

    /**
     * Clears all of the stored values for a clean start.
     */
    inline void ClearValues() {
        this->positions.reset();
        this->velocities.reset();
        this->temperatures.reset();
        this->masses.reset();
        this->internalEnergies.reset();
        this->smoothingLengths.reset();
        this->molecularWeights.reset();
        this->densities.reset();
        this->gravitationalPotentials.reset();
        this->entropies.reset();
        this->isBaryonFlags.reset();
        this->isStarFlags.reset();
        this->isWindFlags.reset();
        this->isStarFormingGasFlags.reset();
        this->isAGNFlags.reset();
        this->particleIDs.reset();
        this->agnDistances.reset();

        this->velocityDerivatives.reset();
        this->temperatureDerivatives.reset();
        this->internalEnergyDerivatives.reset();
        this->smoothingLengthDerivatives.reset();
        this->molecularWeightDerivatives.reset();
        this->densityDerivatives.reset();
        this->gravitationalPotentialDerivatives.reset();
        this->entropyDerivatives.reset();
    }

private:
    /** Pointer to the position array */
    vec3ArrayPtr positions;

    /** Pointer to the velocity array */
    vec3ArrayPtr velocities;

    /** Pointer to the array containing the velocity derivatives */
    vec3ArrayPtr velocityDerivatives;

    /** Pointer to the temperature array */
    floatArrayPtr temperatures;

    /** Pointer to the array containing the temperature derivatives */
    floatArrayPtr temperatureDerivatives;

    /** Pointer to the mass array */
    floatArrayPtr masses;

    /** Pointer to the interal energy array */
    floatArrayPtr internalEnergies;

    /** Pointer to the array containing the internal energy derivatives */
    floatArrayPtr internalEnergyDerivatives;

    /** Pointer to the smoothing length array */
    floatArrayPtr smoothingLengths;

    /** Pointer to the array containing the smoothing length derivatives */
    floatArrayPtr smoothingLengthDerivatives;

    /** Pointer to the molecular weight array */
    floatArrayPtr molecularWeights;

    /** Pointer to the array containing the molecular weight derivatives */
    floatArrayPtr molecularWeightDerivatives;

    /** Pointer to the density array */
    floatArrayPtr densities;

    /** Pointer to the array containing the density derivatives */
    floatArrayPtr densityDerivatives;

    /** Pointer to the gravitational potential array */
    floatArrayPtr gravitationalPotentials;

    /** Pointer to the array containing the gravitational potential derivatives */
    floatArrayPtr gravitationalPotentialDerivatives;

    /** Pointer to the entropy array */
    floatArrayPtr entropies;

    /** Pointer to the array containing the entropy derivatives */
    floatArrayPtr entropyDerivatives;

    /** Pointer to the baryon flag array */
    boolArrayPtr isBaryonFlags;

    /** Pointer to the star flag array */
    boolArrayPtr isStarFlags;

    /** Pointer to the wind flag array */
    boolArrayPtr isWindFlags;

    /** Pointer to the star forming gas flag array */
    boolArrayPtr isStarFormingGasFlags;

    /** Pointer to the AGN flag array */
    boolArrayPtr isAGNFlags;

    /** Pointer to the particle ID array */
    idArrayPtr particleIDs;

    /** Pointer to the array storing the distance to the AGNs */
    floatArrayPtr agnDistances;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<AstroDataCall> AstroDataCallDescription;

} // namespace megamol::astro

#endif
