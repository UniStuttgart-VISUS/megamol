#include "FEMDataCall.h"

namespace megamol
{
	namespace archvis
	{
		void FEMDataCall::setFEMData(std::shared_ptr<FEMModel> const & fem_data)
		{
			m_fem_data = fem_data;
		}

		std::shared_ptr<FEMModel> FEMDataCall::getFEMData()
		{
			return m_fem_data;
		}
	}
}
