#include "protein_calls/ProteinColor.h"

#include "mmcore/utility/ColourParser.h"
#include "mmcore/utility/log/Log.h"
#include "protein_calls/ProteinHelpers.h"
#include <fstream>
#include <limits>

using namespace megamol;
using namespace megamol::protein_calls;

std::string ProteinColor::GetName(ProteinColor::ColoringMode mode) {
    switch (mode) {
    case ProteinColor::ColoringMode::ELEMENT:
        return "Element";
    case ProteinColor::ColoringMode::SECONDARY_STRUCTURE:
        return "Secondary Structure";
    case ProteinColor::ColoringMode::RAINBOW:
        return "Rainbow";
    case ProteinColor::ColoringMode::BFACTOR:
        return "B-Factor";
    case ProteinColor::ColoringMode::CHARGE:
        return "Charge";
    case ProteinColor::ColoringMode::OCCUPANCY:
        return "Occupancy";
    case ProteinColor::ColoringMode::CHAIN:
        return "Chain";
    case ProteinColor::ColoringMode::MOLECULE:
        return "Molecule";
    case ProteinColor::ColoringMode::RESIDUE:
        return "Residue";
    case ProteinColor::ColoringMode::AMINOACID:
        return "Amino Acid";
    case ProteinColor::ColoringMode::RMSF:
        return "Root Mean Square Fluctuation";
    case ProteinColor::ColoringMode::HYDROPHOBICITY:
        return "Hydrophobicity";
    case ProteinColor::ColoringMode::BINDINGSITE:
        return "Binding Site";
    case ProteinColor::ColoringMode::PER_ATOM_FLOAT:
        return "Per-Atom Float Value";
    case ProteinColor::ColoringMode::HEIGHTMAP_COLOR:
        return "Heightmap Color";
    case ProteinColor::ColoringMode::HEIGHTMAP_VALUE:
        return "Heightmap Value";
    default:
        return "";
    }
}

bool ProteinColor::ReadColorTableFromFile(std::string filename, std::vector<glm::vec3>& OUT_colorTable) {
    OUT_colorTable.clear();
    std::ifstream file(filename);
    if (file.is_open()) {
        glm::vec3 res{};
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && core::utility::ColourParser::FromString(line.c_str(), res.r, res.g, res.b)) {
                OUT_colorTable.push_back(res);
            }
        }
        return true;
    }

    if (OUT_colorTable.size() == 0) {
        OUT_colorTable = {glm::vec3(0.7f, 0.8f, 0.4f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(0.5f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.0f),
            glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 0.5f, 0.5f), glm::vec3(0.0f, 0.0f, 0.5f),
            glm::vec3(0.5f, 0.0f, 0.5f), glm::vec3(1.0f, 0.50f, 0.0f), glm::vec3(0.0f, 0.50f, 1.0f),
            glm::vec3(1.0f, 0.50f, 1.0f), glm::vec3(0.5f, 0.25f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f),
            glm::vec3(0.5f, 1.0f, 0.5f), glm::vec3(0.75f, 1.0f, 0.0f), glm::vec3(0.5f, 0.0f, 0.75f),
            glm::vec3(1.0f, 0.5f, 0.5f), glm::vec3(0.75f, 1.0f, 0.75f), glm::vec3(0.75f, 0.75f, 0.5f),
            glm::vec3(1.0f, 0.75f, 0.5f)};
    }
    return false;
}

bool ProteinColor::ReadColorTableFromFile(std::filesystem::path filename, std::vector<glm::vec3>& OUT_colorTable) {
    return ReadColorTableFromFile(filename.string(), OUT_colorTable);
}

void ProteinColor::MakeRainbowColorTable(unsigned int size, std::vector<glm::vec3>& OUT_rainbowTable) {
    OUT_rainbowTable.clear();
    uint32_t n = size / 4;
    // enforce the minimum size of 16
    if (size < 16) {
        n = 4;
    }
    OUT_rainbowTable.reserve(size);
    float f = 1.0f / static_cast<float>(n);
    glm::vec3 color(1.0, 0.0, 0.0);
    for (uint32_t i = 0; i < n; ++i) {
        color.y = std::min(color.y + f, 1.0f);
        OUT_rainbowTable.push_back(color);
    }
    for (unsigned int i = 0; i < n; i++) {
        color.x = std::max(color.x - f, 0.0f);
        OUT_rainbowTable.push_back(color);
    }
    for (unsigned int i = 0; i < n; i++) {
        color.z = std::min(color.z + f, 1.0f);
        OUT_rainbowTable.push_back(color);
    }
    for (unsigned int i = 0; i < n; i++) {
        color.y = std::max(color.y - f, 0.0f);
        OUT_rainbowTable.push_back(color);
    }
}

void ProteinColor::MakeWeightedColorTable(const megamol::protein_calls::MolecularDataCall& mdc,
    const ProteinColor::ColoringMode colMode0, const ProteinColor::ColoringMode colMode1, float weight0, float weight1,
    std::vector<glm::vec3>& OUT_colorTable, const std::vector<glm::vec3>& colorLookupTable,
    const std::vector<glm::vec3>& fileColorTable, const std::vector<glm::vec3>& rainbowColorTable,
    const protein_calls::BindingSiteCall* bsc, const protein_calls::PerAtomFloatCall* psc, bool forceRecompute,
    bool useNeighbors, bool enzymeMode, bool gxtype) {

    if (forceRecompute) {
        OUT_colorTable.clear();
    }

    glm::clamp(weight0, 0.0f, 1.0f);
    glm::clamp(weight1, 0.0f, 1.0f);

    weight0 = weight0 / (weight0 + weight1);
    weight1 = weight1 / (weight0 + weight1);

    if (OUT_colorTable.empty()) {
        std::vector<glm::vec3> color0;
        std::vector<glm::vec3> color1;

        ProteinColor::MakeColorTable(mdc, colMode0, color0, colorLookupTable, fileColorTable, rainbowColorTable, bsc,
            psc, forceRecompute, useNeighbors, enzymeMode, gxtype);
        ProteinColor::MakeColorTable(mdc, colMode1, color1, colorLookupTable, fileColorTable, rainbowColorTable, bsc,
            psc, forceRecompute, useNeighbors, enzymeMode, gxtype);

        for (uint32_t i = 0; i < mdc.AtomCount(); ++i) {
            OUT_colorTable.push_back(color0[i] * weight0 + color1[i] * weight1);
        }
    }
}

void ProteinColor::MakeColorTable(const megamol::protein_calls::MolecularDataCall& mdc,
    const ProteinColor::ColoringMode colMode, std::vector<glm::vec3>& OUT_colorTable,
    const std::vector<glm::vec3>& colorLookupTable, const std::vector<glm::vec3>& fileColorTable,
    const std::vector<glm::vec3>& rainbowColorTable, const protein_calls::BindingSiteCall* bsc,
    const protein_calls::PerAtomFloatCall* psc, bool forceRecompute, bool useNeighbors, bool enzymeMode, bool gxtype) {

    if (forceRecompute) {
        OUT_colorTable.clear();
    }
    if (!OUT_colorTable.empty()) {
        return;
    }

    OUT_colorTable.reserve(mdc.AtomCount());

    if (colMode == ColoringMode::ELEMENT) {
        for (size_t i = 0; i < mdc.AtomCount(); ++i) {
            glm::vec3 col = glm::make_vec3(mdc.AtomTypes()[mdc.AtomTypeIndices()[i]].Colour());
            OUT_colorTable.push_back(col / 255.0f);
        }
    } else if (colMode == ColoringMode::SECONDARY_STRUCTURE) {
        glm::vec3 colNone(0.0, 1.0, 0.0);
        glm::vec3 colHelix(1.0, 0.0, 0.0);
        glm::vec3 colSheet(0.0, 0.0, 1.0);
        glm::vec3 colCoil(136.0 / 255.0);
        OUT_colorTable.resize(mdc.AtomCount(), colNone);
        for (size_t sec = 0; sec < mdc.SecondaryStructureCount(); ++sec) {
            uint32_t idx = mdc.SecondaryStructures()[sec].FirstAminoAcidIndex();
            uint32_t cnt = mdc.SecondaryStructures()[sec].AminoAcidCount();
            auto elemType = mdc.SecondaryStructures()[sec].Type();
            for (size_t res = idx; res < idx + cnt; ++res) {
                uint32_t atomIdx = mdc.Residues()[res]->FirstAtomIndex();
                uint32_t atomCnt = atomIdx + mdc.Residues()[res]->AtomCount();
                for (size_t atom = atomIdx; atom < atomCnt; ++atom) {
                    if (elemType == megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_HELIX) {
                        OUT_colorTable[atom] = colHelix;
                    } else if (elemType == megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_SHEET) {
                        OUT_colorTable[atom] = colSheet;
                    } else if (elemType == megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_COIL) {
                        OUT_colorTable[atom] = colCoil;
                    }
                }
            }
        }
    } else if (colMode == ColoringMode::RAINBOW) {
        for (size_t i = 0; i < mdc.AtomCount(); ++i) {
            uint32_t idx = static_cast<uint32_t>(i / static_cast<float>(mdc.AtomCount()) * rainbowColorTable.size());
            OUT_colorTable.push_back(rainbowColorTable[idx]);
        }
    } else if (colMode == ColoringMode::BFACTOR) {
        for (uint32_t i = 0; i < mdc.AtomCount(); ++i) {
            auto col = InterpolateMultipleColors(
                mdc.AtomBFactors()[i], colorLookupTable, mdc.MinimumBFactor(), mdc.MaximumBFactor());
            OUT_colorTable.push_back(col);
        }
        if (mdc.MaximumBFactor() - mdc.MinimumBFactor() < std::numeric_limits<float>::epsilon()) {
            OUT_colorTable.clear();
            OUT_colorTable.resize(mdc.AtomCount(), glm::vec3(1.0));
        }
    } else if (colMode == ColoringMode::CHARGE) {
        for (uint32_t i = 0; i < mdc.AtomCount(); ++i) {
            auto col = InterpolateMultipleColors(
                mdc.AtomCharges()[i], colorLookupTable, mdc.MinimumCharge(), mdc.MaximumCharge());
            OUT_colorTable.push_back(col);
        }
        if (mdc.MaximumCharge() - mdc.MinimumCharge() < std::numeric_limits<float>::epsilon()) {
            OUT_colorTable.clear();
            OUT_colorTable.resize(mdc.AtomCount(), glm::vec3(1.0));
        }
    } else if (colMode == ColoringMode::OCCUPANCY) {
        for (uint32_t i = 0; i < mdc.AtomCount(); ++i) {
            auto col = InterpolateMultipleColors(
                mdc.AtomOccupancies()[i], colorLookupTable, mdc.MinimumOccupancy(), mdc.MaximumOccupancy());
            OUT_colorTable.push_back(col);
        }
        if (mdc.MaximumOccupancy() - mdc.MinimumOccupancy() < std::numeric_limits<float>::epsilon()) {
            OUT_colorTable.clear();
            OUT_colorTable.resize(mdc.AtomCount(), glm::vec3(1.0));
        }
    } else if (colMode == ColoringMode::CHAIN) {
        // get the last atom of the last res of the last mol of the first chain
        uint32_t cntChain = 0;
        uint32_t cntMol = mdc.Chains()[cntChain].MoleculeCount() - 1;
        uint32_t cntRes = mdc.Molecules()[cntMol].FirstResidueIndex() + mdc.Molecules()[cntMol].ResidueCount() - 1;
        uint32_t cntAtom = mdc.Residues()[cntRes]->FirstAtomIndex() + mdc.Residues()[cntRes]->AtomCount() - 1;
        // get the first color
        uint32_t idx = 0;
        auto color = fileColorTable[idx % fileColorTable.size()];
        // loop over all atoms
        for (size_t atom = 0; atom < mdc.AtomCount(); ++atom) {
            // check, if the last atom of the current chain is reached
            if (atom > cntAtom) {
                // get the last atom of the last res of the last mol of the next chain
                cntChain++;
                cntMol = mdc.Chains()[cntChain].FirstMoleculeIndex() + mdc.Chains()[cntChain].MoleculeCount() - 1;
                cntRes = mdc.Molecules()[cntMol].FirstResidueIndex() + mdc.Molecules()[cntMol].ResidueCount() - 1;
                cntAtom = mdc.Residues()[cntRes]->FirstAtomIndex() + mdc.Residues()[cntRes]->AtomCount() - 1;
                // get the next color
                idx++;
                color = fileColorTable[idx % fileColorTable.size()];
            }
            OUT_colorTable.push_back(color);
        }
    } else if (colMode == ColoringMode::MOLECULE) {
        // get the last atom of the last res of the first mol
        uint32_t cntMol = 0;
        uint32_t cntRes = mdc.Molecules()[cntMol].FirstResidueIndex() + mdc.Molecules()[cntMol].ResidueCount() - 1;
        uint32_t cntAtom = mdc.Residues()[cntRes]->FirstAtomIndex() + mdc.Residues()[cntRes]->AtomCount() - 1;
        // get the first color
        uint32_t idx = 0;
        auto color = fileColorTable[idx % fileColorTable.size()];
        // loop over all atoms
        for (size_t atom = 0; atom < mdc.AtomCount(); ++atom) {
            // check, if the last atom of the current chain is reached
            if (atom > cntAtom) {
                // get the last atom of the last res of the next mol
                cntMol++;
                cntRes = mdc.Molecules()[cntMol].FirstResidueIndex() + mdc.Molecules()[cntMol].ResidueCount() - 1;
                cntAtom = mdc.Residues()[cntRes]->FirstAtomIndex() + mdc.Residues()[cntRes]->AtomCount() - 1;
                // get the next color
                idx++;
                color = fileColorTable[idx % fileColorTable.size()];
            }
            OUT_colorTable.push_back(color);
        }
    } else if (colMode == ColoringMode::RESIDUE) {
        // loop over all residues
        for (uint32_t res = 0; res < mdc.ResidueCount(); ++res) {
            // loop over all atoms of the current residue
            uint32_t idx = mdc.Residues()[res]->FirstAtomIndex();
            uint32_t cnt = mdc.Residues()[res]->AtomCount();
            // get residue type index
            uint32_t resTypeIdx = mdc.Residues()[res]->Type();
            for (uint32_t atom = idx; atom < idx + cnt; ++atom) {
                // Special cases for water/toluol/methanol
                if (mdc.ResidueTypeNames()[resTypeIdx].Equals("SOL")) {
                    // Water
                    OUT_colorTable.push_back(glm::vec3(0.3, 0.3, 1.0));
                } else if (mdc.ResidueTypeNames()[resTypeIdx].Equals("MeOH")) {
                    // methanol
                    OUT_colorTable.push_back(glm::vec3(1.0, 0.5, 0.0));
                } else if (mdc.ResidueTypeNames()[resTypeIdx].Equals("TOL")) {
                    // toluol
                    OUT_colorTable.push_back(glm::vec3(1.0, 1.0, 1.0));
                } else {
                    OUT_colorTable.push_back(fileColorTable[resTypeIdx % fileColorTable.size()]);
                }
            }
        }
    } else if (colMode == ColoringMode::AMINOACID) {
        // loop over all residues
        for (uint32_t res = 0; res < mdc.ResidueCount(); ++res) {
            // loop over all atoms of the current residue
            uint32_t idx = mdc.Residues()[res]->FirstAtomIndex();
            uint32_t cnt = mdc.Residues()[res]->AtomCount();
            // get residue type index
            uint32_t resTypeIdx = mdc.Residues()[res]->Type();
            uint32_t colour_idx = 0;
            if (mdc.ResidueTypeNames()[resTypeIdx].Equals("ALA") || mdc.ResidueTypeNames()[resTypeIdx].Equals("VAL") ||
                mdc.ResidueTypeNames()[resTypeIdx].Equals("MET") || mdc.ResidueTypeNames()[resTypeIdx].Equals("LEU") ||
                mdc.ResidueTypeNames()[resTypeIdx].Equals("ILE") || mdc.ResidueTypeNames()[resTypeIdx].Equals("PRO") ||
                mdc.ResidueTypeNames()[resTypeIdx].Equals("TRP") || mdc.ResidueTypeNames()[resTypeIdx].Equals("PHE")) {
                colour_idx = 0;
            } else if (mdc.ResidueTypeNames()[resTypeIdx].Equals("TYR") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("TYM") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("THR") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("GLN") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("GLY") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("SER") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("CYS") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("CYX") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("CYM") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("ASN") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("ASH")) {
                colour_idx = 1;
            } else if (mdc.ResidueTypeNames()[resTypeIdx].Equals("LYS") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("LYN") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("ARG") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("HIS") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("HID") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("HIE") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("HIP")) {
                colour_idx = 2;
            } else if (mdc.ResidueTypeNames()[resTypeIdx].Equals("GLU") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("GLH") ||
                       mdc.ResidueTypeNames()[resTypeIdx].Equals("ASP")) {
                colour_idx = 3;
            } else {
                colour_idx = 4;
            }
            for (uint32_t atom = idx; atom < idx + cnt; ++atom) {
                if (colour_idx == 0) {
                    OUT_colorTable.push_back(glm::vec3(255, 230, 128) / 255.0f);
                } else if (colour_idx == 1) {
                    OUT_colorTable.push_back(glm::vec3(170, 222, 135) / 255.0f);
                } else if (colour_idx == 2) {
                    OUT_colorTable.push_back(glm::vec3(128, 179, 255) / 255.0f);
                } else if (colour_idx == 3) {
                    OUT_colorTable.push_back(glm::vec3(255, 170, 170) / 255.0f);
                } else {
                    OUT_colorTable.push_back(glm::vec3(180) / 255.0f);
                }
            }
        }
    } else if (colMode == ColoringMode::RMSF) {
        OUT_colorTable.resize(mdc.AtomCount(), glm::vec3(1.0));
        // TODO
    } else if (colMode == ColoringMode::HYDROPHOBICITY) {
        auto bounds = protein_calls::GetHydrophobicityBounds();
        for (uint32_t res = 0; res < mdc.ResidueCount(); ++res) {
            uint32_t idx = mdc.Residues()[res]->FirstAtomIndex();
            uint32_t cnt = mdc.Residues()[res]->AtomCount();
            uint32_t resTypeIdx = mdc.Residues()[res]->Type();
            float val = protein_calls::GetHydrophibicityByResName(mdc.ResidueTypeNames()[resTypeIdx].PeekBuffer());
            auto col = InterpolateMultipleColors(val, colorLookupTable, bounds.x, bounds.y);
            for (uint32_t atom = idx; atom < idx + cnt; ++atom) {
                OUT_colorTable.push_back(col);
            }
        }
    } else if (colMode == ColoringMode::BINDINGSITE) {
        // initialize all colors as white
        OUT_colorTable.resize(mdc.AtomCount(), glm::vec3(1.0));
        if (bsc != nullptr) {
            for (uint32_t chain = 0; chain < mdc.ChainCount(); ++chain) {
                uint32_t firstMol = mdc.Chains()[chain].FirstMoleculeIndex();
                for (uint32_t mol = firstMol; mol < firstMol + mdc.Chains()[chain].MoleculeCount(); ++mol) {
                    uint32_t firstRes = mdc.Molecules()[mol].FirstResidueIndex();
                    for (unsigned int res = 0; res < mdc.Molecules()[res].ResidueCount(); ++res) {
                        // loop over all binding sites
                        for (uint32_t bs = 0; bs < bsc->GetBindingSiteCount(); ++bs) {
                            for (unsigned int bsResCnt = 0; bsResCnt < bsc->GetBindingSite(bs)->Count(); ++bsResCnt) {
                                vislib::Pair<char, unsigned int> bsRes = bsc->GetBindingSite(bs)->operator[](bsResCnt);
                                if (mdc.Chains()[chain].Name() == bsRes.First() &&
                                    mdc.Residues()[firstRes + res]->OriginalResIndex() == bsRes.Second() &&
                                    mdc.ResidueTypeNames()[mdc.Residues()[firstRes + res]->Type()] ==
                                        bsc->GetBindingSiteResNames(bs)->operator[](bsResCnt)) {
                                    uint32_t firstAtom = mdc.Residues()[firstRes + res]->FirstAtomIndex();
                                    for (unsigned int aCnt = 0; aCnt < mdc.Residues()[firstRes + res]->AtomCount();
                                         aCnt++) {
                                        uint32_t atomIdx = firstAtom + aCnt;
                                        OUT_colorTable[atomIdx] =
                                            glm::make_vec3(bsc->GetBindingSiteColor(bs).PeekComponents());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            core::utility::log::Log::DefaultLog.WriteWarn(
                "No Binding Site Call connected. Falling back to white color");
        }
    } else if (colMode == ColoringMode::PER_ATOM_FLOAT) {
        if (psc == nullptr) {
            OUT_colorTable.resize(mdc.AtomCount(), glm::vec3(1.0));
            core::utility::log::Log::DefaultLog.WriteWarn(
                "No Per-Atom Float Call connected. Falling back to white color");
        } else {
            for (uint32_t i = 0; i < mdc.AtomCount(); ++i) {
                auto col =
                    InterpolateMultipleColors(psc->GetFloat()[i], colorLookupTable, psc->MinValue(), psc->MaxValue());
                OUT_colorTable.push_back(col);
            }
            if (psc->MaxValue() - psc->MinValue() < std::numeric_limits<float>::epsilon()) {
                OUT_colorTable.clear();
                OUT_colorTable.resize(mdc.AtomCount(), glm::vec3(1.0));
            }
        }
    } else if (colMode == ColoringMode::HEIGHTMAP_COLOR || colMode == ColoringMode::HEIGHTMAP_VALUE) {
        // initialize all colors as white
        OUT_colorTable.resize(mdc.AtomCount(), glm::vec3(1.0));
        // loop over all protein atoms to calculate centroid
        glm::vec3 centroid(0.0f, 0.0f, 0.0f);
        for (uint32_t i = 0; i < mdc.AtomCount(); ++i) {
            centroid += glm::make_vec3(&mdc.AtomPositions()[3 * i]);
        }
        centroid /= static_cast<float>(mdc.AtomCount());
        // calculate maximum distance
        float max_dist = 0.0f;
        float dist, steps;
        unsigned int col_cnt = 15;
        for (unsigned int i = 0; i < mdc.AtomCount(); ++i) {
            dist = sqrt(pow(centroid.x - mdc.AtomPositions()[3 * i + 0], 2.0f) +
                        pow(centroid.y - mdc.AtomPositions()[3 * i + 1], 2.0f) +
                        pow(centroid.z - mdc.AtomPositions()[3 * i + 2], 2.0f));
            if (dist > max_dist)
                max_dist = dist;
        }
        steps = max_dist / static_cast<float>(col_cnt);
        // calculate the colors
        std::vector<glm::vec3> colours = {glm::vec3(160.0, 194.0, 222.0), glm::vec3(173.0, 203.0, 230.0),
            glm::vec3(185.0, 213.0, 237.0), glm::vec3(196.0, 223.0, 244.0), glm::vec3(206.0, 231.0, 249.0),
            glm::vec3(218.0, 240.0, 253.0), glm::vec3(172.0, 208.0, 165.0), glm::vec3(148.0, 191.0, 139.0),
            glm::vec3(168.0, 198.0, 143.0), glm::vec3(209.0, 215.0, 171.0), glm::vec3(239.0, 235.0, 192.0),
            glm::vec3(222.0, 214.0, 163.0), glm::vec3(211.0, 202.0, 157.0), glm::vec3(202.0, 185.0, 130.0),
            glm::vec3(195.0, 167.0, 107.0), glm::vec3(195.0, 167.0, 107.0), glm::vec3(195.0, 167.0, 107.0)};
        // normalize the colors
        for (auto& col : colours) {
            col /= 255.0f;
        }

        unsigned int bin;
        float alpha, upper_bound, lower_bound;
        for (uint32_t cnt = 0; cnt < mdc.AtomCount(); ++cnt) {
            dist = sqrt(pow(centroid[0] - mdc.AtomPositions()[3 * cnt + 0], 2.0f) +
                        pow(centroid[1] - mdc.AtomPositions()[3 * cnt + 1], 2.0f) +
                        pow(centroid[2] - mdc.AtomPositions()[3 * cnt + 2], 2.0f));
            if (colMode == ColoringMode::HEIGHTMAP_COLOR) {
                bin = static_cast<unsigned int>(std::truncf(dist / steps));
                lower_bound = float(bin) * steps;
                upper_bound = float(bin + 1) * steps;
                alpha = (dist - lower_bound) / (upper_bound - lower_bound);
                OUT_colorTable[cnt] = glm::mix(colours[bin], colours[bin + 1], alpha);
            } else if (colMode == ColoringMode::HEIGHTMAP_VALUE) {
                dist /= max_dist;
                OUT_colorTable[cnt] = glm::vec3(dist);
            }
        }
    }
}

glm::vec3 ProteinColor::InterpolateMultipleColors(
    float val, const std::vector<glm::vec3>& colors, float minVal, float maxVal) {
    float alpha = val;
    if (alpha < minVal) {
        alpha = minVal;
    }
    if (alpha > maxVal) {
        alpha = maxVal;
    }
    float stepSize = (maxVal - minVal) / static_cast<float>(colors.size());
    uint32_t bin = static_cast<uint32_t>(std::truncf((alpha - minVal) / stepSize));
    float lower_bound = minVal + static_cast<float>(bin) * stepSize;
    float upper_bound = minVal + static_cast<float>(bin + 1) * stepSize;
    alpha = (alpha - lower_bound) / (upper_bound - lower_bound);
    // special case for certain floating point errors
    if (bin >= colors.size() - 1) {
        return colors[colors.size() - 1];
    }
    return glm::mix(colors[bin], colors[bin + 1], alpha);
}
