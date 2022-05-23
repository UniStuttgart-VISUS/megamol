/*
 * ShaderStorageBufferObject.cpp
 * Copyright (C) 2009-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ShaderStorageBufferObject.h"

using namespace megamol;
using namespace megamol::molecularmaps;

/*
 * ShaderStorageBufferObject::ShaderStorageBufferObject
 */
ShaderStorageBufferObject::ShaderStorageBufferObject(void) {}

/*
 * ShaderStorageBufferObject::~ShaderStorageBufferObject
 */
ShaderStorageBufferObject::~ShaderStorageBufferObject(void) {
    glDeleteBuffers(1, &m_ssbo);
    glDeleteBuffers(1, &m_atomic_counter_buffer);
}

/*
 * ShaderStorageBufferObject::UnbindBuffer
 */
void ShaderStorageBufferObject::UnbindBuffer() {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, m_binding, 0);
}

/*
 * ShaderStorageBufferObject::initAtomicCounter
 */
bool ShaderStorageBufferObject::initAtomicCounter(GLuint p_binding_ac) {
    GLint error;
    error = glGetError(); // catch error that comes out of nowhere...

    m_binding_ac = p_binding_ac;
    glGenBuffers(1, &m_atomic_counter_buffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_atomic_counter_buffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    error = glGetError();
    if (error != 0)
        return false;
    return true;
}

/*
 * ShaderStorageBufferObject::BindAtomicCounter
 */
void ShaderStorageBufferObject::BindAtomicCounter() {
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, m_binding_ac, m_atomic_counter_buffer);
}

/*
 * ShaderStorageBufferObject::GetAtomicCounterVal
 */
GLuint ShaderStorageBufferObject::GetAtomicCounterVal() {
    GLuint value;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_atomic_counter_buffer);
    GLuint* ptr = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), GL_MAP_READ_BIT);
    value = ptr[0];
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    return value;
}

/*
 * ShaderStorageBufferObject::ResetAtomicCounter
 */
void ShaderStorageBufferObject::ResetAtomicCounter(GLuint p_value) {
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_atomic_counter_buffer);
    GLuint* ptr = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint),
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    ptr[0] = p_value;
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
}

/*
 * ShaderStorageBufferObject::UnbindAtomicCounter
 */
void ShaderStorageBufferObject::UnbindAtomicCounter() {
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, m_binding_ac, 0);
}
