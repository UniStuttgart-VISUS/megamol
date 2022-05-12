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

namespace megamol {
namespace protein_calls {

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
                            mdc->Residues()[firstRes + rCnt]->OriginalResIndex() ==
                                bsRes->operator[](bsridx).second &&
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
 * Gets the one letter amino acid code for a given three letter code
 *
 * @param resName The 3 letter code of the amino acid
 * @return The one letter code of the amino acid
 */
inline char GetAminoAcidOneLetterCode(const std::string& resName) {
    std::string name = resName;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    if (name.compare("ALA") == 0)
        return 'A';
    else if (name.compare("ARG") == 0)
        return 'R';
    else if (name.compare("ASN") == 0)
        return 'N';
    else if (name.compare("ASP") == 0)
        return 'D';
    else if (name.compare("CYS") == 0)
        return 'C';
    else if (name.compare("GLN") == 0)
        return 'Q';
    else if (name.compare("GLU") == 0)
        return 'E';
    else if (name.compare("GLY") == 0)
        return 'G';
    else if (name.compare("HIS") == 0)
        return 'H';
    else if (name.compare("ILE") == 0)
        return 'I';
    else if (name.compare("LEU") == 0)
        return 'L';
    else if (name.compare("LYS") == 0)
        return 'K';
    else if (name.compare("MET") == 0)
        return 'M';
    else if (name.compare("PHE") == 0)
        return 'F';
    else if (name.compare("PRO") == 0)
        return 'P';
    else if (name.compare("SER") == 0)
        return 'S';
    else if (name.compare("THR") == 0)
        return 'T';
    else if (name.compare("TRP") == 0)
        return 'W';
    else if (name.compare("TYR") == 0)
        return 'Y';
    else if (name.compare("VAL") == 0)
        return 'V';
    else if (name.compare("ASH") == 0)
        return 'D';
    else if (name.compare("CYX") == 0)
        return 'C';
    else if (name.compare("CYM") == 0)
        return 'C';
    else if (name.compare("GLH") == 0)
        return 'E';
    else if (name.compare("HID") == 0)
        return 'H';
    else if (name.compare("HIE") == 0)
        return 'H';
    else if (name.compare("HIP") == 0)
        return 'H';
    else if (name.compare("MSE") == 0)
        return 'M';
    else if (name.compare("LYN") == 0)
        return 'K';
    else if (name.compare("TYM") == 0)
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
    if (name.compare("ALA") == 0)
        return 41.0f;
    else if (name.compare("ARG") == 0)
        return -14.0f;
    else if (name.compare("LEU") == 0)
        return 97.0f;
    else if (name.compare("LYS") == 0)
        return -23.0f;
    else if (name.compare("MET") == 0)
        return 74.0f;
    else if (name.compare("GLN") == 0)
        return -10.0f;
    else if (name.compare("ILE") == 0)
        return 99.0f;
    else if (name.compare("TRP") == 0)
        return 97.0f;
    else if (name.compare("PHE") == 0)
        return 100.0f;
    else if (name.compare("TYR") == 0)
        return 63.0f;
    else if (name.compare("CYS") == 0)
        return 49.0f;
    else if (name.compare("VAL") == 0)
        return 76.0f;
    else if (name.compare("ASN") == 0)
        return -28.0f;
    else if (name.compare("SER") == 0)
        return -5.0f;
    else if (name.compare("HIS") == 0)
        return 8.0f;
    else if (name.compare("GLU") == 0)
        return -31.0f;
    else if (name.compare("THR") == 0)
        return 13.0f;
    else if (name.compare("ASP") == 0)
        return -55.0f;
    else if (name.compare("GLY") == 0)
        return 0.0f;
    else if (name.compare("PRO") == 0)
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
    if (name.compare("ALA") == 0)
        return 1.8f;
    else if (name.compare("ARG") == 0)
        return -4.5f;
    else if (name.compare("ASN") == 0)
        return -3.5f;
    else if (name.compare("ASP") == 0)
        return -3.5f;
    else if (name.compare("CYS") == 0)
        return 2.5f;
    else if (name.compare("GLU") == 0)
        return -3.5f;
    else if (name.compare("GLN") == 0)
        return -3.5f;
    else if (name.compare("GLY") == 0)
        return -0.4f;
    else if (name.compare("HIS") == 0)
        return -3.2f;
    else if (name.compare("ILE") == 0)
        return 4.5f;
    else if (name.compare("LEU") == 0)
        return 3.8f;
    else if (name.compare("LYS") == 0)
        return -3.9f;
    else if (name.compare("MET") == 0)
        return 1.9f;
    else if (name.compare("PHE") == 0)
        return 2.8f;
    else if (name.compare("PRO") == 0)
        return -1.6f;
    else if (name.compare("SER") == 0)
        return -0.8f;
    else if (name.compare("THR") == 0)
        return -0.7f;
    else if (name.compare("TRP") == 0)
        return -0.9f;
    else if (name.compare("TYR") == 0)
        return -1.3f;
    else if (name.compare("VAL") == 0)
        return 4.2f;
    return 0.0f;
}

/**
 * Get the min and max value of the hydrophobicity table of Monera et al.
 *
 * @return vec2 where x stores the min value and y the max value
 */
inline glm::vec2 GetHydrophobicityBoundsMonera(void) {
    return glm::vec2(-100.0, 100.0);
}

/**
 * Get the min and max value of the hydrophobicity table of Kyte and Doolittle.
 *
 * @return vec2 where x stores the min value and y the max value
 */
inline glm::vec2 GetHydrophobicityBoundsKyte(void) {
    return glm::vec2(-4.5, 4.5);
}

/**
 * Get the min and max value of the hydrophobicity table. The current standard is the method by Kyte and Doolittle.
 *
 * @return vec2 where x stores the min value and y the max value
 */
inline glm::vec2 GetHydrophobicityBounds(void) {
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

} // namespace protein_calls
} /* end namespace megamol */

#endif
