/*
 * 3DUIRenderTaskDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef THREE_DIMENSIONAL_UI_RENDER_TASK_DATA_SOURCE
#define THREE_DIMENSIONAL_UI_RENDER_TASK_DATA_SOURCE

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "AbstractGPURenderTaskDataSource.h"
#include "mesh/CallGltfData.h"

#include "3DInteractionCollection.h"

namespace megamol {
namespace mesh {

    class ThreeDimensionalUIRenderTaskDataSource : public AbstractGPURenderTaskDataSource
    {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName(void) { return "UIRenderTaskDataSource"; }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char* Description(void) {
            return "....TODO...";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) { return true; }

        ThreeDimensionalUIRenderTaskDataSource();
        ~ThreeDimensionalUIRenderTaskDataSource();
    
    protected:
        bool getDataCallback(core::Call& caller);

        bool getInteractionCallback(core::Call& caller);

    private:

        std::shared_ptr<ThreeDimensionalInteractionCollection> m_interaction_collection;

        megamol::core::CalleeSlot m_3DInteraction_calleeSlot;
        megamol::core::CallerSlot m_glTF_callerSlot;

    };

}
}

#endif // !THREE_DIMENSIONAL_UI_RENDER_TASK_DATA_SOURCE
