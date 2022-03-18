
#ifndef MMPROTEIN_RMSF_H_INCLUDED
#define MMPROTEIN_RMSF_H_INCLUDED

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/MolecularDataCall.h"

namespace megamol {
namespace protein_calls {

/**
 * Computes the RMSF for all atoms stored in mol.
 * Will store the RMSF values as B-factor in mol.
 *
 * @param mol The molecular data call
 * @return 'false' if no data is available or if the molecule just one time step; 'true' on success.
 */
bool computeRMSF(protein_calls::MolecularDataCall* mol);
} // namespace protein_calls
} // namespace megamol

#endif // MMPROTEIN_RMSF_H_INCLUDED
