#ifndef MMPROTEINPLUGIN_PROTEINHELPERS_H_INCLUDED
#define MMPROTEINPLUGIN_PROTEINHELPERS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "glm/glm.hpp"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib/math/Vector.h"
#include <algorithm>
#include <string>

namespace megamol::protein_calls {

/**
 * Computes and returns the approximate positions of the binding site of a molecule.
 * To do so, the atom positions of the following atoms are averaged:
 * 1. The second oxygen of the catalytic serine
 * 2. The second side chain nitrogen of the catalytic histidine
 * 3. The backbone nitrogen of the first oxyanion hole residue
 * 4. The backbone nitrogen of the second oxyanion hole residue or, in the case of non-gx-types, the first sidechain
 oxygen

 * This computation is only possible if the incoming BindingSiteCall has at least three binding sites of which the third
 * consists of at least two amino acids.
 * If there is too few data, this method just averages the atom positions of all available binding site atoms.
 *
 * It is assumed that the necessary data is already present in the call.
 *
 * @param mdc The call containing the molecular data.
 * @param bsc The call containing the data of the binding site.
 * @param isgx Determination whether the incoming protein is a gx- or a y-type.
 * @return The estimated position of the binding site.
 */
inline vislib::math::Vector<float, 3> EstimateBindingSitePosition(
    protein_calls::MolecularDataCall* mdc, protein_calls::BindingSiteCall* bsc) {
    vislib::math::Vector<float, 3> result(0.0f, 0.0f, 0.0f);

    bool isgx = bsc->isOfGxType();

    // check the binding sites
    bool fallbackMode = false;
    auto bscount = bsc->GetBindingSiteCount();
    if (bscount < 3 || !bsc->isEnzymeMode()) {
        fallbackMode = true;
    } else {
        if (bsc->GetBindingSite(0)->size() < 1 || bsc->GetBindingSite(1)->size() < 1 ||
            bsc->GetBindingSite(2)->size() < 2) {
            fallbackMode = true;
        }
    }
    if (!fallbackMode) {
        bscount = 3;
    }

    // average the positions
    uint32_t firstMol;
    uint32_t firstRes;
    uint32_t firstAtom;
    uint32_t sum = 0u;

    for (uint32_t cCnt = 0; cCnt < mdc->ChainCount(); cCnt++) {
        firstMol = mdc->Chains()[cCnt].FirstMoleculeIndex();
        for (uint32_t mCnt = firstMol; mCnt < firstMol + mdc->Chains()[cCnt].MoleculeCount(); mCnt++) {
            firstRes = mdc->Molecules()[mCnt].FirstResidueIndex();
            for (uint32_t rCnt = firstRes; rCnt < firstRes + mdc->Molecules()[mCnt].ResidueCount(); rCnt++) {
                firstAtom = mdc->Residues()[firstRes + rCnt]->FirstAtomIndex();
                auto bla = firstAtom;
                for (uint32_t bsidx = 0; bsidx < bscount; bsidx++) {
                    auto bsRes = bsc->GetBindingSite(bsidx);
                    for (uint32_t bsridx = 0; bsridx < bsRes->size(); bsridx++) {

                        auto cname = mdc->Chains()[cCnt].Name();
                        auto bname = bsRes->operator[](bsridx).first;

                        auto orgidx = mdc->Residues()[firstRes + rCnt]->OriginalResIndex();
                        auto bsorgidx = bsRes->operator[](bsridx).second;

                        auto typn = mdc->ResidueTypeNames()[mdc->Residues()[firstRes + rCnt]->Type()];
                        auto typb = bsc->GetBindingSiteResNames(bsidx)->operator[](bsridx);

                        if (mdc->Chains()[cCnt].Name() == bsRes->operator[](bsridx).first &&
                            mdc->Residues()[firstRes + rCnt]->OriginalResIndex() == bsRes->operator[](bsridx).second &&
                            mdc->ResidueTypeNames()[mdc->Residues()[firstRes + rCnt]->Type()].PeekBuffer() ==
                                bsc->GetBindingSiteResNames(bsidx)->operator[](bsridx)) {
                            for (uint32_t aCnt = firstAtom;
                                 aCnt < firstAtom + mdc->Residues()[firstRes + rCnt]->AtomCount(); aCnt++) {
                                auto atomName = mdc->AtomTypes()[mdc->AtomTypeIndices()[aCnt]].Name();
                                if (fallbackMode || // TODO further assumptions
                                    (atomName.Equals("OG") && bsidx == 0) || (atomName.Equals("ND2") && bsidx == 1) ||
                                    (atomName.Equals("N") && bsidx == 2 && bsridx == 0) ||
                                    (atomName.Equals("N") && bsidx == 2 && bsridx == 1 && isgx) ||
                                    (atomName.Equals("OH") && bsidx == 2 && bsridx == 1 && !isgx)) {

                                    vislib::math::Vector<float, 3> atpos(&mdc->AtomPositions()[3 * aCnt]);
                                    result += atpos;
                                    sum++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    result /= static_cast<float>(sum);
    return result;
}

/**
 * Gets the preset protein color as glm vec3
 *
 * @param atomName The element of the atom stored as string
 * @return The color representing the element as glm vec3
 */
inline glm::vec3 GetElementColorPreset(std::string const& atomName) {
    if (atomName == "H") {
        return glm::vec3(240.0f, 240.0f, 240.0f) / 255.0f;
    }
    if (atomName == "C") {
        return glm::vec3(90.0f, 90.0f, 90.0f) / 255.0f;
    }
    if (atomName == "N") {
        return glm::vec3(37.0f, 136.0f, 195.0f) / 255.0f;
    }
    if (atomName == "O") {
        return glm::vec3(206.0f, 34.0f, 34.0f) / 255.0f;
    }
    if (atomName == "S") {
        return glm::vec3(255.0f, 215.0f, 0.0f) / 255.0f;
    }
    if (atomName == "P") {
        return glm::vec3(255.0f, 128.0f, 64.0f) / 255.0f;
    }
    if (atomName == "M") {
        return glm::vec3(90.0f, 90.0f, 90.0f) / 255.0f;
    }

    // special values for a SFB Demo
    if (atomName == "Po") { // Pore
        return glm::vec3(149.0f, 149.0f, 149.0f) / 255.0f;
    }
    if (atomName == "P1") { // Pore (coarse)
        return glm::vec3(149.0f, 149.0f, 149.0f) / 255.0f;
    }
    if (atomName == "XX") { // CL
        return glm::vec3(154.0f, 205.0f, 50.0f) / 255.0f;
    }
    if (atomName == "YY") { // NA
        return glm::vec3(255.0f, 215.0f, 20.0f) / 255.0f;
    }
    if (atomName == "ZZ") { // DNA center
        return glm::vec3(240.0f, 240.0f, 240.0f) / 255.0f;
    }
    if (atomName == "QQ") { // DNA base
        return glm::vec3(240.0f, 80.0f, 50.0f) / 255.0f;
    }

    // default grey
    return glm::vec3(191.0f) / 255.0f;
}

/**
 * Gets the preset protein color as a unsigned char vislib vector
 *
 * @param atomName The element of the atom stored as string
 * @return The color representing the element as vislib unsigned char vector
 */
inline vislib::math::Vector<unsigned char, 3> GetElementColorPresetVislib(std::string const& atomName) {
    auto const v = GetElementColorPreset(atomName) * 255.0f;
    return vislib::math::Vector<unsigned char, 3>(
        static_cast<unsigned char>(v.x), static_cast<unsigned char>(v.y), static_cast<unsigned char>(v.z));
}

/**
 * Gets the one letter amino acid code for a given three letter code
 *
 * @param resName The 3 letter code of the amino acid
 * @return The one letter code of the amino acid
 */
inline char GetAminoAcidOneLetterCode(const std::string& resName) {
    std::string name = resName;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    if (name == "ALA")
        return 'A';
    else if (name == "ARG")
        return 'R';
    else if (name == "ASN")
        return 'N';
    else if (name == "ASP")
        return 'D';
    else if (name == "CYS")
        return 'C';
    else if (name == "GLN")
        return 'Q';
    else if (name == "GLU")
        return 'E';
    else if (name == "GLY")
        return 'G';
    else if (name == "HIS")
        return 'H';
    else if (name == "ILE")
        return 'I';
    else if (name == "LEU")
        return 'L';
    else if (name == "LYS")
        return 'K';
    else if (name == "MET")
        return 'M';
    else if (name == "PHE")
        return 'F';
    else if (name == "PRO")
        return 'P';
    else if (name == "SER")
        return 'S';
    else if (name == "THR")
        return 'T';
    else if (name == "TRP")
        return 'W';
    else if (name == "TYR")
        return 'Y';
    else if (name == "VAL")
        return 'V';
    else if (name == "ASH")
        return 'D';
    else if (name == "CYX")
        return 'C';
    else if (name == "CYM")
        return 'C';
    else if (name == "GLH")
        return 'E';
    else if (name == "HID")
        return 'H';
    else if (name == "HIE")
        return 'H';
    else if (name == "HIP")
        return 'H';
    else if (name == "MSE")
        return 'M';
    else if (name == "LYN")
        return 'K';
    else if (name == "TYM")
        return 'Y';
    else
        return '?';
}

/**
 * Get the hydrophobicity value for an amino acid according to the method of Monera et al.
 *
 * @param resName The 3 letter code of the amino acid
 * @return The hydrophicity value of the amino acid
 */
inline float GetHydrophobicityByResNameMonera(const std::string& resName) {
    std::string name = resName;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    // O.D.Monera, T.J.Sereda, N.E.Zhou, C.M.Kay, R.S.Hodges
    // "Relationship of sidechain hydrophobicity and alpha-helical propensity on the stability of the single-stranded
    // amphipathic alpha-helix." J.Pept.Sci. 1995 Sep - Oct; 1(5) : 319 - 29.
    if (name == "ALA")
        return 41.0f;
    else if (name == "ARG")
        return -14.0f;
    else if (name == "LEU")
        return 97.0f;
    else if (name == "LYS")
        return -23.0f;
    else if (name == "MET")
        return 74.0f;
    else if (name == "GLN")
        return -10.0f;
    else if (name == "ILE")
        return 99.0f;
    else if (name == "TRP")
        return 97.0f;
    else if (name == "PHE")
        return 100.0f;
    else if (name == "TYR")
        return 63.0f;
    else if (name == "CYS")
        return 49.0f;
    else if (name == "VAL")
        return 76.0f;
    else if (name == "ASN")
        return -28.0f;
    else if (name == "SER")
        return -5.0f;
    else if (name == "HIS")
        return 8.0f;
    else if (name == "GLU")
        return -31.0f;
    else if (name == "THR")
        return 13.0f;
    else if (name == "ASP")
        return -55.0f;
    else if (name == "GLY")
        return 0.0f;
    else if (name == "PRO")
        return -46.0f;
    return 0.0f;
}

/**
 * Get the hydrophobicity value for an amino acid according to the method of Kyte and Doolittle
 *
 * @param resName The 3 letter code of the amino acid
 * @return The hydrophicity value of the amino acid
 */
inline float GetHydrophobicityByResNameKyte(std::string resName) {
    std::string name = resName;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    // J. Kyte, R. F. Doolittle
    // "A simple method for displaying the hydropathic character of a protein."
    // J. Mol. Biol. 1982 May 5; 157(1) : 105 - 32.
    if (name == "ALA")
        return 1.8f;
    else if (name == "ARG")
        return -4.5f;
    else if (name == "ASN")
        return -3.5f;
    else if (name == "ASP")
        return -3.5f;
    else if (name == "CYS")
        return 2.5f;
    else if (name == "GLU")
        return -3.5f;
    else if (name == "GLN")
        return -3.5f;
    else if (name == "GLY")
        return -0.4f;
    else if (name == "HIS")
        return -3.2f;
    else if (name == "ILE")
        return 4.5f;
    else if (name == "LEU")
        return 3.8f;
    else if (name == "LYS")
        return -3.9f;
    else if (name == "MET")
        return 1.9f;
    else if (name == "PHE")
        return 2.8f;
    else if (name == "PRO")
        return -1.6f;
    else if (name == "SER")
        return -0.8f;
    else if (name == "THR")
        return -0.7f;
    else if (name == "TRP")
        return -0.9f;
    else if (name == "TYR")
        return -1.3f;
    else if (name == "VAL")
        return 4.2f;
    return 0.0f;
}

enum AminoAcidCategory { UNPOLAR = 0, POLAR = 1, BASIC = 2, ACIDIC = 3, NONE = 4 };

/*
 * Get the amino acid property value for an amino acid
 *
 * @param resName The 3 letter code of the amino acid
 * @return The property category of the amino acid
 */
inline AminoAcidCategory GetAminoAcidPropertiesByResName(std::string resName) {
    std::string name = resName;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    if (name == "ALA")
        return AminoAcidCategory::UNPOLAR;
    else if (name == "ARG")
        return AminoAcidCategory::BASIC;
    else if (name == "ASN")
        return AminoAcidCategory::POLAR;
    else if (name == "ASP")
        return AminoAcidCategory::ACIDIC;
    else if (name == "CYS")
        return AminoAcidCategory::POLAR;
    else if (name == "GLU")
        return AminoAcidCategory::ACIDIC;
    else if (name == "GLN")
        return AminoAcidCategory::POLAR;
    else if (name == "GLY")
        return AminoAcidCategory::POLAR;
    else if (name == "HIS")
        return AminoAcidCategory::BASIC;
    else if (name == "ILE")
        return AminoAcidCategory::UNPOLAR;
    else if (name == "LEU")
        return AminoAcidCategory::UNPOLAR;
    else if (name == "LYS")
        return AminoAcidCategory::BASIC;
    else if (name == "MET")
        return AminoAcidCategory::UNPOLAR;
    else if (name == "PHE")
        return AminoAcidCategory::UNPOLAR;
    else if (name == "PRO")
        return AminoAcidCategory::UNPOLAR;
    else if (name == "SER")
        return AminoAcidCategory::POLAR;
    else if (name == "THR")
        return AminoAcidCategory::POLAR;
    else if (name == "TRP")
        return AminoAcidCategory::UNPOLAR;
    else if (name == "TYR")
        return AminoAcidCategory::POLAR;
    else if (name == "VAL")
        return AminoAcidCategory::UNPOLAR;
    return AminoAcidCategory::NONE;
}

/**
 * Get the min and max value of the hydrophobicity table of Monera et al.
 *
 * @return vec2 where x stores the min value and y the max value
 */
inline glm::vec2 GetHydrophobicityBoundsMonera() {
    return glm::vec2(-100.0, 100.0);
}

/**
 * Get the min and max value of the hydrophobicity table of Kyte and Doolittle.
 *
 * @return vec2 where x stores the min value and y the max value
 */
inline glm::vec2 GetHydrophobicityBoundsKyte() {
    return glm::vec2(-4.5, 4.5);
}

/**
 * Get the min and max value of the hydrophobicity table. The current standard is the method by Kyte and Doolittle.
 *
 * @return vec2 where x stores the min value and y the max value
 */
inline glm::vec2 GetHydrophobicityBounds() {
    return GetHydrophobicityBoundsKyte();
}

/**
 * Get the hydrophobicity value for an amino acid. The current standard is the method by Kyte and Doolittle.
 *
 * @param resName The 3 letter code of the amino acid
 * @return The hydrophicity value of the amino acid
 */
inline float GetHydrophibicityByResName(std::string resName) {
    return GetHydrophobicityByResNameKyte(resName);
}

} // namespace megamol::protein_calls

#endif
