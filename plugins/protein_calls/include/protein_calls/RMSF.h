
#pragma once

#include "protein_calls/MolecularDataCall.h"

namespace megamol::protein_calls {

/**
 * Computes the RMSF for all atoms stored in mol.
 * Will store the RMSF values as B-factor in mol.
 *
 * @param mol The molecular data call
 * @return 'false' if no data is available or if the molecule just one time step; 'true' on success.
 */
bool computeRMSF(protein_calls::MolecularDataCall* mol);
} // namespace megamol::protein_calls
