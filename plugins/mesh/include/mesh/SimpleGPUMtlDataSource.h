/*
 * SimpleGPUMtlDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef SIMPLE_GPU_MTL_DATA_SOURCE_H_INCLUDED
#define SIMPLE_GPU_MTL_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"

#include "mesh/AbstractGPUMaterialDataSource.h"

namespace megamol {
namespace mesh {

class SimpleGPUMtlDataSource : public AbstractGPUMaterialDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "SimpleGPUMtlDataSource"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "mesh data source for loading a BTF Shader file"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }


    SimpleGPUMtlDataSource();
    ~SimpleGPUMtlDataSource();

protected:
    virtual bool create();

    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller);

private:
    uint32_t m_version;

    /** The btf file name */
    core::param::ParamSlot m_btf_filename_slot;
};

} // namespace mesh
} // namespace megamol

#endif // !SIMPLE_GPU_MTL_DATA_SOURCE_H_INCLUDED
