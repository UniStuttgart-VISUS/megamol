#pragma once

#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/PerAtomFloatCall.h"

#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>

namespace megamol::protein_calls {
class ProteinColor {
public:
    /**
     * Enum for all possible coloring modes. The modes should always be numbered incrementally starting at 0.
     * The last value 'MODE_COUNT' is special and should always reflect the nubmer of other modes.
     */
    enum class ColoringMode {
        ELEMENT = 0,
        SECONDARY_STRUCTURE = 1,
        RAINBOW = 2,
        BFACTOR = 3,
        CHARGE = 4,
        OCCUPANCY = 5,
        CHAIN = 6,
        MOLECULE = 7,
        RESIDUE = 8,
        AMINOACID = 9,
        RMSF = 10,
        HYDROPHOBICITY = 11,
        BINDINGSITE = 12,
        PER_ATOM_FLOAT = 13,
        HEIGHTMAP_COLOR = 14,
        HEIGHTMAP_VALUE = 15,
        MODE_COUNT = 16
    };

    /**
     * TODO is this function really necessary? Currently only used by the SolventVolumeRenderer.
     */
    // static void FillAminoAcidColorTable(std::vector<glm::vec3> aminoAcidColorTable);

    /**
     * Returns a human-readable name for a given coloring mode.
     *
     * @param mode The mode to return the name for
     * @return The human-readable name of the mode
     */
    static std::string GetName(ProteinColor::ColoringMode mode);

    /**
     * Read a color table from file. If the reading was not successful, the color table will be filled with predefined colors.
     *
     * @param filename The filename of the color table file
     * @param OUT_colorTable The resulting color lookup table
     * @return True on success, false otherwise
     */
    static bool ReadColorTableFromFile(std::string filename, std::vector<glm::vec3>& OUT_colorTable);

    /**
     * Read a color table from file. If the reading was not successful, the color table will be filled with predefined colors.
     *
     * @param filename The filename of the color table file
     * @param OUT_colorTable The resulting color lookup table
     * @return True on success, false otherwise
     */
    static bool ReadColorTableFromFile(std::filesystem::path filename, std::vector<glm::vec3>& OUT_colorTable);

    /**
     * Creates a rainbow color table of a given size. Any requested length below 16 will result in a rainbow table of length 16.
     *
     * @param size The length of the resulting rainbow color table
     * @param OUT_rainbowTable Will contain the table of rainbow colors
     */
    static void MakeRainbowColorTable(unsigned int size, std::vector<glm::vec3>& OUT_rainbowTable);

    /**
     * Makes a color table by weighting two differnt coloring modes.
     * The color table is only compute if it is empty or if the recomputation is forced.
     *
     * @param mdc The call containing the molecular data
     * @param colMode0 The first coloring mode
     * @param colMode1 The second coloring mode
     * @param weight0 The weighting factor for the first coloring mode
     * @param weight1 The weighting factor for the second coloring mode
     * @param OUT_colorTable The generated color table as output
     * @param colorLookupTable The color table used for interpolation
     * @param fileColorTable The color table that was read from a file
     * @param rainbowColorTable The rainbow color table used for the rainbow coloring mode
     * @param bsc Optional pointer to a binding site call for the binding site coloring mode
     * @param psc Optional pointer to a per atom float call for optional value visualization
     * @param forceRecompute When set to true, this forces the recalculation of the color table
     * @param useNeighbors If set to true, the neighbor coloring mode can be enabled
     * @param enzymeMode This can enable the special enzyme binding site coloring mode
     * @param gxtype Additional flag for the enzyme coloring mode telling it if the protein is a gx-type
     */
    static void MakeWeightedColorTable(const megamol::protein_calls::MolecularDataCall& mdc,
        const ColoringMode colMode0, const ColoringMode colMode1, float weight0, float weight1,
        std::vector<glm::vec3>& OUT_colorTable, const std::vector<glm::vec3>& colorLookupTable,
        const std::vector<glm::vec3>& fileColorTable, const std::vector<glm::vec3>& rainbowColorTable,
        const protein_calls::BindingSiteCall* bsc = nullptr, const protein_calls::PerAtomFloatCall* psc = nullptr,
        bool forceRecompute = false, bool useNeighbors = false, bool enzymeMode = false, bool gxtype = true);

    /**
     * Make color tables for all atoms according to the current coloring mode
     * The color table is only compute if it is empty or if the recomputation is forced
     *
     * @param mdc The call containing the molecular data
     * @param colMode The first coloring mode
     * @param OUT_colorTable The generated color table as output
     * @param colorLookupTable The color table used for interpolation
     * @param fileColorTable The color table that was read from a file
     * @param rainbowColorTable The rainbow color table used for the rainbow coloring mode
     * @param bsc Optional pointer to a binding site call for the binding site coloring mode
     * @param psc Optional pointer to a per atom float call for optional value visualization
     * @param forceRecompute When set to true, this forces the recalculation of the color table
     * @param useNeighbors If set to true, the neighbor coloring mode can be enabled
     * @param enzymeMode This can enable the special enzyme binding site coloring mode
     * @param gxtype Additional flag for the enzyme coloring mode telling it if the protein is a gx-type
     * @return A pair containing the min value of the visualized quantity as first and the max value as second value.
     * A correct result is only guaranteed if the recomputation is forced.
     */
    static std::pair<float, float> MakeColorTable(const megamol::protein_calls::MolecularDataCall& mdc,
        const ColoringMode colMode, std::vector<glm::vec3>& OUT_colorTable,
        const std::vector<glm::vec3>& colorLookupTable, const std::vector<glm::vec3>& fileColorTable,
        const std::vector<glm::vec3>& rainbowColorTable, const protein_calls::BindingSiteCall* bsc = nullptr,
        const protein_calls::PerAtomFloatCall* psc = nullptr, bool forceRecompute = false, bool useNeighbors = false,
        bool enzymeMode = false, bool gxtype = true);

    /**
     * Interpolates on a given color table linearly
     *
     * @param val The value to find the color for
     * @param colors The color table to interpolate on
     * @param minVal The minimum possible value for val
     * @param maxVal The maximum possible value for val
     * @return The interpolated color
     */
    static glm::vec3 InterpolateMultipleColors(
        float val, const std::vector<glm::vec3>& colors, float minVal = 0.0f, float maxVal = 0.0f);
};
} // namespace megamol::protein_calls
