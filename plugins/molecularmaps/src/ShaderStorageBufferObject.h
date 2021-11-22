/*
 * ShaderStorageBufferObject.h
 * Copyright (C) 2009-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_SHADERSTORAGEBUFFEROBJECT_H_INCLUDED
#define MMMOLMAPPLG_SHADERSTORAGEBUFFEROBJECT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <iostream>
#include <vislib/graphics/gl/IncludeAllGL.h>

namespace megamol {
namespace molecularmaps {

/**
 * Class encapsulating an OpenGL Shader Storage Buffer Object (SSBO)
 */
class ShaderStorageBufferObject {
public:
    /** Ctor */
    ShaderStorageBufferObject(void);

    /** Dtor */
    virtual ~ShaderStorageBufferObject(void);

    /**
     * Initializes the SSBO
     *
     * @param p_data_ptr Pointer to the data
     * @param p_data_size The size of the incoming data
     * @param p_usage The usage flag of the SSBO
     * @param p_binding The position the SSBO should be bound to
     */
    template<typename T>
    bool init(T* p_data_ptr, GLuint p_data_size, GLenum p_usage, GLuint p_binding);

    /**
     * Retrieves the data stored in the SSBO
     *
     * @param data_ptr The pointer the data is copied to
     * @param p_data_size The amount of data that is copied
     */
    template<typename T>
    void GetData(T* data_ptr, GLuint p_data_size);

    /**
     * Sets the data of the SSBO
     *
     * @param p_data_ptr Pointer to the data the SSBO is set to
     * @param p_data_size The size of the incoming data
     * @param p_usage The usage flag of the SSBO
     */
    template<typename T>
    void SetData(T* p_data_ptr, GLuint p_data_size, GLenum p_usage);

    /**
     * Unbinds the SSBO
     */
    void UnbindBuffer(void);

    /**
     * Initializes the atomic counter
     *
     * @param p_binding_ac The position the atomic counter should be bound to
     * @return True on success. False otherwise.
     */
    bool initAtomicCounter(GLuint p_binding_ac);

    /**
     * Binds the atomic counter so that it can be used
     */
    void BindAtomicCounter(void);

    /**
     * Retrieves the current value of the atomic counter
     *
     * @return The current value of the atomic counter
     */
    GLuint GetAtomicCounterVal(void);

    /**
     * Resets the atomic counter to the given value
     *
     * @param p_value The new value of the counter
     */
    void ResetAtomicCounter(GLuint p_value);

    /**
     * Unbinds the atomic counter
     */
    void UnbindAtomicCounter(void);

private:
    /**
     * Copy constructor made private to prevent unwanted copys
     */
    ShaderStorageBufferObject(const ShaderStorageBufferObject& copy);

    /** Handle of the SSBO */
    GLuint m_ssbo;
    /** Binding point of the SSBO */
    GLuint m_binding;
    /** Handle of the atomic counter buffer */
    GLuint m_atomic_counter_buffer;
    /** Binding point of the atomic counter buffer */
    GLuint m_binding_ac;
};

#include "ShaderStorageBufferObject.inl"

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_SHADERSTORAGEBUFFEROBJECT_H_INCLUDED */
