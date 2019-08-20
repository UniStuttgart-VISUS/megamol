/*
* MSMDataCall.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef MSM_DATA_CALL_H_INCLUDED
#define MSM_DATA_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetDataCall.h"

#include "ScaleModel.h"

namespace megamol{
namespace archvis{

class MSMDataCall : public megamol::core::AbstractGetDataCall 
{
public:
    MSMDataCall();
    ~MSMDataCall() = default;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "MSMDataCall"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Call that gives access to a loaded MSM dataset."; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return AbstractGetDataCall::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return AbstractGetDataCall::FunctionName(idx); }

    void setMSM(std::shared_ptr<ScaleModel> const& msm);

    std::shared_ptr<ScaleModel> getMSM();

private:
    std::shared_ptr<ScaleModel> m_MSM;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<MSMDataCall> MSMDataCallDescription;

}
}


#endif // !MSM_DATA_CALL_H_INCLUDED
