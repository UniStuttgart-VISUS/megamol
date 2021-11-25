/*
 * ShaderStorageBufferObject.inl
 * Copyright (C) 2009-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

/*
 * ShaderStorageBufferObject::init
 */
template<typename T>
bool ShaderStorageBufferObject::init(T* p_data_ptr, GLuint p_data_size, GLenum p_usage, GLuint p_binding) {
    m_binding = p_binding;

    // generate buffer
    glGenBuffers(1, &m_ssbo);

    // upload data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * p_data_size, NULL, p_usage);
    if (p_data_ptr != NULL) {
        T* ptr = (T*)glMapBufferRange(
            GL_SHADER_STORAGE_BUFFER, 0, sizeof(T) * p_data_size, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        for (GLuint i = 0; i < p_data_size; i++) {
            ptr[i] = p_data_ptr[i];
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        return true;
    }
    return false;
}

/*
 * ShaderStorageBufferObject::GetData
 */
template<typename T>
void ShaderStorageBufferObject::GetData(T* data_ptr, GLuint p_data_size) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo);
    T* data_ptr_gpu = (T*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    memcpy(data_ptr, data_ptr_gpu, p_data_size * sizeof(T));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

/*
 * ShaderStorageBufferObject::SetData
 */
template<typename T>
void ShaderStorageBufferObject::SetData(T* p_data_ptr, GLuint p_data_size, GLenum p_usage) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, m_binding, m_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * p_data_size, p_data_ptr, p_usage);
}
