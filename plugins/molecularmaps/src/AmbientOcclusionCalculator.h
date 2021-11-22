/*
 * AmbientOcclusionCalculator.h
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_AMBIENTOCCLUSIONCALCULATOR_H_INCLUDED
#define MMMOLMAPPLG_AMBIENTOCCLUSIONCALCULATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CoreInstance.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"
#include <cfloat>

namespace megamol {
namespace molecularmaps {

/**
 * Class computing ambient occlusion factors per vertex for a given mesh
 */
class AmbientOcclusionCalculator {
public:
    /**
     * Struct for settings for the ambient occlusion.
     */
    struct AOSettings {
        /* The angle factor for the direction generation */
        float angleFactor;
        /* Scaling parameter for the final ambient occlusion value */
        float evalFactor;
        /* The falloff parameter for the distance function */
        float falloffFactor;
        /* Influence factor of a single sphere on a voxel */
        float genFac;
        /* The maximal distance to the last ao sample */
        float maxDist;
        /* The minimal distance to the first ao sample */
        float minDist;
        /* Number of sample directions */
        int numSampleDirections;
        /* Scaling factor for particle radii */
        float scaling;
        /* Voxel count of the volume in x-direction */
        unsigned int volSizeX;
        /* Voxel count of the volume in x-direction */
        unsigned int volSizeY;
        /* Voxel count of the volume in x-direction */
        unsigned int volSizeZ;

        /**
         * Constructor
         */
        AOSettings(void)
                : angleFactor(1.0f)
                , evalFactor(1.0f)
                , falloffFactor(1.0f)
                , genFac(1.0f)
                , maxDist(0.5f)
                , minDist(0.5f)
                , numSampleDirections(8)
                , scaling(1.0f)
                , volSizeX(1)
                , volSizeY(1)
                , volSizeZ(1) {}

        /**
         * Copy constructor
         */
        AOSettings(const AOSettings& aos) {
            this->angleFactor = aos.angleFactor;
            this->evalFactor = aos.evalFactor;
            this->falloffFactor = aos.falloffFactor;
            this->genFac = aos.genFac;
            this->maxDist = aos.maxDist;
            this->minDist = aos.minDist;
            this->numSampleDirections = aos.numSampleDirections;
            this->scaling = aos.scaling;
            this->volSizeX = aos.volSizeX;
            this->volSizeY = aos.volSizeY;
            this->volSizeZ = aos.volSizeZ;
        }

        /**
         * Computes whether this struct or another given one are different from each other
         *
         * @param aos The other struct
         * @return True if the both structs are different from each other, false otherwise
         */
        bool isDirty(const AOSettings& aos) {
            if (std::abs(this->angleFactor - aos.angleFactor) > FLT_EPSILON)
                return true;
            if (std::abs(this->evalFactor - aos.evalFactor) > FLT_EPSILON)
                return true;
            if (std::abs(this->falloffFactor - aos.falloffFactor) > FLT_EPSILON)
                return true;
            if (std::abs(this->genFac - aos.genFac) > FLT_EPSILON)
                return true;
            if (std::abs(this->maxDist - aos.maxDist) > FLT_EPSILON)
                return true;
            if (std::abs(this->minDist - aos.minDist) > FLT_EPSILON)
                return true;
            if (std::abs(this->scaling - aos.scaling) > FLT_EPSILON)
                return true;
            if (this->numSampleDirections != aos.numSampleDirections)
                return true;
            if (this->volSizeX != aos.volSizeX)
                return true;
            if (this->volSizeY != aos.volSizeY)
                return true;
            if (this->volSizeZ != aos.volSizeZ)
                return true;
            return false;
        }
    };

    /**
     * Constructor
     */
    AmbientOcclusionCalculator(void);

    /**
     * Destructor
     */
    virtual ~AmbientOcclusionCalculator(void);

    /**
     * Calculates one brightness value per vertex and returns a pointer to the resulting array of values, where
     * each of the values is written three times in a row.
     *
     * @param settings struct containing all settings for the ambient occlusion
     * @return Pointer to the vector containing the resulting brightness values per vertex three times in a row. If
     * the method was not successful, it returns a nullptr.
     */
    const std::vector<float>* calculateVertexShadows(AOSettings settings);

    /**
     * Tells the module to erase the stored vertex shadow data.
     */
    void clearStoredShadowData(void);

    /**
     * Returns the stored vertex shadow array.
     *
     * @return Pointer to the vector containing the resulting brightness values per vertex three times in a row. If
     * the method was not successful, it returns a nullptr.
     */
    const std::vector<float>* getVertexShadows(void);

    /**
     * Initializes the class with a given triangle mesh.
     *
     * @param instance Pointer to the megamol core instance.
     * @param vertices Pointer to the vector containing the vertex positions.
     * @param vertex_normals Pointer to the vector containing the vertex normals.
     * @param mdc The call containing the necessary protein information.
     * @return True on success, false otherwise.
     */
    bool initilialize(core::CoreInstance* instance, const std::vector<float>* vertices,
        const std::vector<float>* vertex_normals, protein_calls::MolecularDataCall* mdc);

    /**
     * Reloads the shader programs of this ambient occlusion calculator.
     *
     * @param instance Pointer to the megamol core instance.
     * @return True on success, false otherwise.
     */
    bool loadShaders(core::CoreInstance* instance);

private:
    /**
     * Creates and uploads the direction texture
     */
    void calcDirections(void);

    /**
     * Creates and uploads the level texture.
     *
     * @param aoWidthX The size of a single voxel in x-direction
     * @param aoWidthY The size of a single voxel in y-direction
     * @param aoWidthZ The size of a single voxel in z-direction
     * @param diag The length of the clip box diagonal
     */
    void calcLevels(float aoWidthX, float aoWidthY, float aoWidthZ, float diag);

    /**
     * Creates an empty volume on the GPU
     *
     * @param settings The settings struct containing the necessary measurements
     */
    void createEmptyVolume(AOSettings settings);

    /**
     * Creates the Shadow volume on the CPU and uploads it on the GPU
     *
     * @param settings The settings struct containing the necessary measurements
     */
    void createVolumeCPU(AOSettings settings);

    /**
     * Reads the colour data from the SSBO and writes the result into the resultVector
     */
    void readColourData();

    /**
     * Resizes the GPU representation of the volume
     *
     * @param settings The settings struct containing the necessary measurements
     */
    void resizeVolume(AOSettings settings);

    /**
     * Uploads the vertex data to the GPU.
     */
    void uploadVertexData();

    /** The compute shader for the ambient occlusion computation */
    vislib::graphics::gl::GLSLComputeShader aoComputeShader;

    /** The maximum number of samples */
    int aoSampleMax;

    /** Handle for the SSBO containing the final vertex colors */
    GLuint colourSSBOHandle;

    /** Handle for the direction texture */
    GLuint dirTexture;

    /** Dirty flag for the internal data */
    bool isDirty;

    /** The hash of the most recently used protein data */
    SIZE_T lastProteinDataHash;

    /** Handle for the level texture */
    GLuint lvlTexture;

    /** Pointer to the call containing the protein data */
    protein_calls::MolecularDataCall* mdc;

    /** Handle for the SSBO containing the protein atom position data */
    GLuint normalSSBOHandle;

    /** Resulting vector storing one brightness float value per vertex */
    std::vector<float> resultVector;

    /** The settings struct */
    AOSettings settings;

    /** Flag indicating whether the shaders have changed */
    bool shaderChanged;

    /** Pointer to the vector containing the vertex positions */
    const std::vector<float>* vertex_normals;

    /** Handle for the SSBO containing the vertex data */
    GLuint vertexSSBOHandle;

    /** Pointer to the vector containing the vertex positions */
    const std::vector<float>* vertices;

    /** Handle of the used volume texture */
    GLuint volTexture;

    /** Flag determining whether the volume has been initialized */
    bool volumeInitialized;
};

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_AMBIENTOCCLUSIONCALCULATOR_H_INCLUDED */
