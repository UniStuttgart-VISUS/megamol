#include "FEMDataCall.h"

namespace megamol
{
	namespace archvis
	{
		void FEMDataCall::setFEMData(std::shared_ptr<FEMDataStorage> const & fem_data)
		{
			m_fem_data = fem_data;
		}

		std::shared_ptr<FEMDataStorage> FEMDataCall::getFEMData()
		{
			return m_fem_data;
		}

		void FEMDataCall::setUpdateFlag()
		{
			m_update_flag = true;
		}

		bool FEMDataCall::getUpdateFlag()
		{
			return m_update_flag;
		}

		void FEMDataCall::clearUpdateFlag()
		{
			m_update_flag = false;
		}
	}
}
