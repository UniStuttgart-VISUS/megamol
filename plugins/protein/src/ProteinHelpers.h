#ifndef MMPROTEINPLUGIN_PROTEINHELPERS_H_INCLUDED
#define MMPROTEINPLUGIN_PROTEINHELPERS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib/math/Vector.h"

namespace megamol {
namespace protein {

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
        if (bsc->GetBindingSite(0)->Count() < 1 || bsc->GetBindingSite(1)->Count() < 1 ||
            bsc->GetBindingSite(2)->Count() < 2) {
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
                    for (uint32_t bsridx = 0; bsridx < bsRes->Count(); bsridx++) {

                        auto cname = mdc->Chains()[cCnt].Name();
                        auto bname = bsRes->operator[](bsridx).First();

                        auto orgidx = mdc->Residues()[firstRes + rCnt]->OriginalResIndex();
                        auto bsorgidx = bsRes->operator[](bsridx).Second();

                        auto typn = mdc->ResidueTypeNames()[mdc->Residues()[firstRes + rCnt]->Type()];
                        auto typb = bsc->GetBindingSiteResNames(bsidx)->operator[](bsridx);

                        if (mdc->Chains()[cCnt].Name() == bsRes->operator[](bsridx).First() &&
                            mdc->Residues()[firstRes + rCnt]->OriginalResIndex() ==
                                bsRes->operator[](bsridx).Second() &&
                            mdc->ResidueTypeNames()[mdc->Residues()[firstRes + rCnt]->Type()] ==
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

} /* end namespace protein */
} /* end namespace megamol */

#endif
