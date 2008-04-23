/*
 * GLSLShader.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLSLSHADER_H_INCLUDED
#define VISLIB_GLSLSHADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractOpenGLShader.h"
#include "vislib/ExtensionsDependent.h"
#include "vislib/glverify.h"
#include "vislib/types.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * Class of GLSLShaders.
     *
     * Use a 'Create' Method to construct a shader programm. After this was 
     * successful you can activate the shader using 'Enable'. Although you can
     * access the shader programm handle directly it is recommended to use the
     * classes methods where possible (e.g.: better use 'SetParameter' then 
     * 'glUniform').
     */
    class GLSLShader : public AbstractOpenGLShader, 
            public ExtensionsDependent<GLSLShader> {

    public:

        /**
         * Answer whether 'hProg' is a valid program object handle, that also 
         * has not been scheduled for deletion.
         * 
         * @param hProg A program handle.
         *
         * @return true, if 'hProg' is a valid handle, false otherwise.
         */
        static bool IsValidHandle(GLhandleARB hProg);

        /**
         * Answer the extensions that are required for ARB shaders as
         * space-separated ANSI strings.
         *
         * @return The extensions that are requiered for ARB shaders.
         */
        static const char *RequiredExtensions(void);

        /** Vertex shader source for fixed function transformation. */
        static const char *FTRANSFORM_VERTEX_SHADER_SRC;

        /** Ctor. */
        GLSLShader(void);

        /** Dtor. */
        virtual ~GLSLShader(void);

        /**
         * Binds a generic vertex attribute to a named shader attribute, 
         * allowing to use more descriptive names in the shader sources.
         * The attribute binding is performed with the next call of 'Link'.
         *
         * @param index The index of the generic vertex attribute.
         * @param name  The name of the shader attribute.
         *
         * @return true if the binding was successful (however, you must still
         *         call 'Link' before anything happens), false on error.
         */
        virtual GLenum BindAttribute(GLint index, const char *name);

        /**
         * Compiles a new shader program object using a vertex shader and a
         * fragment shader. Both shader sources will be compiled into shader
         * objects and both will be attached to a program object. The program
         * object will not be linked. You must call 'Link' before you can use
         * the shader. Using 'vertexShaderSrc' as source code of the 
         * vertex shader and 'fragmentShaderSrc' as source code of the pixel
         * shader.
         *
         * @param vertexShaderSrc   The null terminated source string of the 
         *                          vertex shader.
         * @param fragmentShaderSrc The null terminated source string of the
         *                          pixel shader.
         *
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         */
        virtual bool Compile(const char *vertexShaderSrc, 
            const char *fragmentShaderSrc);

        /**
         * Compiles a new shader program object using a vertex shader and a
         * fragment shader. Both shader sources will be compiled into shader
         * objects and both will be attached to a program object. The program
         * object will not be linked. You must call 'Link' before you can use
         * the shader. Using the concatenation of the null terminated
         * strings in 'vertexShaderSrc' as source code of the vertex shader 
         * and the content of 'fragmentShaderSrc' as source code of the pixel
         * shader.
         *
         * @param vertexShaderSrc      An array of 'cntVertexShaderSrc' null
         *                             terminated strings forming the vertex
         *                             shader.
         * @param cntVertexShaderSrc   The number of elements in 
         *                             'vertexShaderSrc'.
         * @param fragmentShaderSrc    An array of 'cntFragmentShaderSrc' null
         *                             terminated strings forming the pixel
         *                             shader.
         * @param cntFragmentShaderSrc The number of elements in
         *                             'fragmentShaderSrc'.
         * @param insertLineDirective  Indicates whether the '#line' directive
         *                             should be inserted between each two
         *                             shader source strings.
         *
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         */
        virtual bool Compile(const char **vertexShaderSrc, 
            const SIZE_T cntVertexShaderSrc, const char **fragmentShaderSrc,
            const SIZE_T cntFragmentShaderSrc, 
            bool insertLineDirective = true);

        /**
         * Compiles a new shader program object using a vertex shader and a
         * fragment shader. Both shader sources will be compiled into shader
         * objects and both will be attached to a program object. The program
         * object will not be linked. You must call 'Link' before you can use
         * the shader. Loading the shader code from two files.
         *
         * @param vertexShaderFile   The name of the vertex shader source file.
         * @param fragmentShaderFile The name of the pixel shader source file.
         * 
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         * @throws IOException     If reading the shader code from an open
         *                         file failed.
         */
        virtual bool CompileFromFile(const char *vertexShaderFile,
            const char *fragmentShaderFile);

        /**
         * Compiles a new shader program object using a vertex shader and a
         * fragment shader. Both shader sources will be compiled into shader
         * objects and both will be attached to a program object. The program
         * object will not be linked. You must call 'Link' before you can use
         * the shader. Loading the shader code from several files.
         *
         * @param vertexShaderFiles      Array of names of the vertex shader 
         *                               source files.
         * @param cntVertexShaderFiles   Number of vertex shader source files
         * @param fragmentShaderFiles    Array of names of the fragment shader
         *                               source files.
         * @param cntFragmentShaderFiles Number of fragment shader source files
         * @param insertLineDirective    Indicates whether the '#line' 
         *                               directive should be inserted between 
         *                               each two shader source strings.
         * 
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         * @throws IOException     If reading the shader code from an open
         *                         file failed.
         */
        virtual bool CompileFromFile(const char **vertexShaderFiles,
            const SIZE_T cntVertexShaderFiles, const char **fragmentShaderFiles,
            const SIZE_T cntFragmentShaderFiles, 
            bool insertLineDirective = true);

        /**
         * Create a new shader using 'vertexShaderSrc' as source code of the 
         * vertex shader and 'fragmentShaderSrc' as source code of the pixel
         * shader. This is equal to the corresponding calls of 'Compile' and
         * 'Link'.
         *
         * @param vertexShaderSrc   The null terminated source string of the 
         *                          vertex shader.
         * @param fragmentShaderSrc The null terminated source string of the
         *                          pixel shader.
         *
         * @return true, if the shader was successfully created.
         *
         * @throws OpenGLException If an OpenGL call for creating the shader
         *                         fails.
         */
        virtual bool Create(const char *vertexShaderSrc, 
            const char *fragmentShaderSrc);

        /**
         * Create a new shader using the concatenation of the null terminated
         * strings in 'vertexShaderSrc' as source code of the vertex shader 
         * and the content of 'fragmentShaderSrc' as source code of the pixel
         * shader. This is equal to the corresponding calls of 'Compile' and
         * 'Link'.
         *
         * @param vertexShaderSrc      An array of 'cntVertexShaderSrc' null
         *                             terminated strings forming the vertex
         *                             shader.
         * @param cntVertexShaderSrc   The number of elements in 
         *                             'vertexShaderSrc'.
         * @param fragmentShaderSrc    An array of 'cntFragmentShaderSrc' null
         *                             terminated strings forming the pixel
         *                             shader.
         * @param cntFragmentShaderSrc The number of elements in
         *                             'fragmentShaderSrc'.
         * @param insertLineDirective  Indicates whether the '#line' directive
         *                             should be inserted between each two
         *                             shader source strings.
         *
         * @return true, if the shader was successfully created.
         *
         * @throws OpenGLException If an OpenGL call for creating the shader
         *                         fails.
         */
        virtual bool Create(const char **vertexShaderSrc, 
            const SIZE_T cntVertexShaderSrc, const char **fragmentShaderSrc,
            const SIZE_T cntFragmentShaderSrc, 
            bool insertLineDirective = true);

        /**
         * Crate a new shader loading the shader code from two files. This is 
         * equal to the corresponding calls of 'Compile' and 'Link'.
         *
         * @param vertexShaderFile   The name of the vertex shader source file.
         * @param fragmentShaderFile The name of the pixel shader source file.
         *
         * @return true, if the shader was successfully created, false, if one
         *         of the shader files could not be opened.
         * 
         * @throws OpenGLException If an OpenGL call for creating the shader
         *                         fails.
         * @throws IOException     If reading the shader code from an open
         *                         file failed.
         */
        virtual bool CreateFromFile(const char *vertexShaderFile,
            const char *fragmentShaderFile);

        /**
         * Create a new shader loading the shader code from several files. This
         * is equal to the corresponding calls of 'Compile' and 'Link'.
         *
         * @param vertexShaderFiles      Array of names of the vertex shader 
         *                               source files.
         * @param cntVertexShaderFiles   Number of vertex shader source files
         * @param fragmentShaderFiles    Array of names of the fragment shader
         *                               source files.
         * @param cntFragmentShaderFiles Number of fragment shader source files
         * @param insertLineDirective    Indicates whether the '#line' 
         *                               directive should be inserted between 
         *                               each two shader source strings.
         * 
         * @return true, if the shader was successfully created, false, if one
         *         of the shader files could not be opened.
         * 
         * @throws OpenGLException If an OpenGL call for creating the shader
         *                         fails.
         * @throws IOException     If reading the shader code from an open
         *                         file failed.
         */
        virtual bool CreateFromFile(const char **vertexShaderFiles,
            const SIZE_T cntVertexShaderFiles, const char **fragmentShaderFiles,
            const SIZE_T cntFragmentShaderFiles, 
            bool insertLineDirective = true);

        /**
         * Disable the shader. This method changes the GL to render using
         * the fixed function pipeline.
         *
         * It is safe to call this method, if the shader has not been
         * successfully created or enable.
         *
         * @return GL_NO_ERROR in case of success, an error code, if the
         *         shader was active but could not be disabled.
         */
        virtual GLenum Disable(void);
        
        /**
         * Enables the shader. The shader must have been successfully created
         * before.
         *
         * @return GL_NO_ERROR in case of success, an error code, if the
         *         shader could not be enabled.
         */
        virtual GLenum Enable(void);

        /**
         * Links the shader. The shader must already be successfully compiled.
         *
         * @return true If the shader linked successfully.
         *
         * @throws OpenGLException If an OpenGL call for linking the shader
         *                         fails.
         */
        virtual bool Link(void);

        /**
         * Answers the location of a uniform parameter. This location can be
         * used as parameter for 'glUniform' functions and 'SetParameter' 
         * methods.
         *
         * @param name The name of the uniform parameter. Must not be NULL.
         *
         * @return The location of the uniform parameter. A value less then
         *         Zero indicates an invalid position.
         */
        GLint ParameterLocation(const char *name) const;

        /**
         * Answer the OpenGL program handle. Use with care since manipulating
         * the handle may result in undefined behaviour. Use this handle for
         * calling OpenGL functions not wrapped by this class.
         *
         * @return The OpenGL program handle.
         */
        inline GLhandleARB ProgramHandle(void) const {
            return this->hProgObj;
        }

        /**
         * Releases all resources allocated by the shader. It is safe to
         * 
         * @return GL_NO_ERROR if the resource have been released or have
         *         never been allocated, an error code, if they have been
         *         allocated but could not be released.
         */
        virtual GLenum Release(void);

        /**
         * Sets the one component of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The name of the parameter.
         * @param v1   The value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const char *name, const float v1) {
            return this->SetParameter(this->ParameterLocation(name), v1);
        }

        /**
         * Sets the two components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The name of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const char *name, const float v1, 
                const float v2) {
            return this->SetParameter(this->ParameterLocation(name), v1, v2);
        }

        /**
         * Sets the three components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The name of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         * @param v3   The third value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const char *name, const float v1, 
                const float v2, const float v3) {
            return this->SetParameter(this->ParameterLocation(name), 
                v1, v2, v3);
        }

        /**
         * Sets the four components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The name of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         * @param v3   The third value for the parameter.
         * @param v4   The fourth value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const char *name, const float v1,
                const float v2, const float v3, const float v4) {
            return this->SetParameter(this->ParameterLocation(name), 
                v1, v2, v3, v4);
        }

        /**
         * Sets the one component of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The name of the parameter.
         * @param v1   The value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const char *name, const int v1) {
            return this->SetParameter(this->ParameterLocation(name), v1);
        }

        /**
         * Sets the two components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The name of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const char *name, const int v1, 
                const int v2) {
            return this->SetParameter(this->ParameterLocation(name), v1, v2);
        }

        /**
         * Sets the three components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The name of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         * @param v3   The third value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const char *name, const int v1, 
                const int v2, const int v3) {
            return this->SetParameter(this->ParameterLocation(name), 
                v1, v2, v3);
        }

        /**
         * Sets the four components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The name of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         * @param v3   The third value for the parameter.
         * @param v4   The fourth value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const char *name, const int v1,
                const int v2, const int v3, const int v4) {
            return this->SetParameter(this->ParameterLocation(name), 
                v1, v2, v3, v4);
        }

        /**
         * Sets the one component of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The location of the parameter.
         * @param v1   The value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLint name, const float v1);

        /**
         * Sets the two components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The location of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLint name, const float v1, 
            const float v2);

        /**
         * Sets the three components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The location of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         * @param v3   The third value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLint name, const float v1, 
            const float v2, const float v3);

        /**
         * Sets the four components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The location of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         * @param v3   The third value for the parameter.
         * @param v4   The fourth value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLint name, const float v1,
            const float v2, const float v3, const float v4);

        /**
         * Sets the one component of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The location of the parameter.
         * @param v1   The value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLint name, const int v1);

        /**
         * Sets the two components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The location of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLint name, const int v1, 
            const int v2);

        /**
         * Sets the three components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The location of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         * @param v3   The third value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLint name, const int v1, 
            const int v2, const int v3);

        /**
         * Sets the four components of the parameter 'name'.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name The location of the parameter.
         * @param v1   The first value for the parameter.
         * @param v2   The second value for the parameter.
         * @param v3   The third value for the parameter.
         * @param v4   The fourth value for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLint name, const int v1,
            const int v2, const int v3, const int v4);

        /**
         * Sets the one component of the parameter 'name'. 'value' points to 
         * an array with 'count' times one float values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The name of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray1(const char *name, const GLsizei count, 
                const float *value) {
            return this->SetParameterArray1(this->ParameterLocation(name), count,
                value);
        }

        /**
         * Sets the two components of the parameter 'name'. 'value' points to 
         * an array with 'count' times two float values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The name of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray2(const char *name, const GLsizei count, 
                const float *value) {
            return this->SetParameterArray2(this->ParameterLocation(name), count,
                value);
        }

        /**
         * Sets the three components of the parameter 'name'. 'value' points to
         * an array with 'count' times three float values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The name of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray3(const char *name, const GLsizei count, 
                const float *value) {
            return this->SetParameterArray3(this->ParameterLocation(name), count,
                value);
        }

        /**
         * Sets the four components of the parameter 'name'. 'value' points to 
         * an array with 'count' times four float values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The name of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray4(const char *name, const GLsizei count, 
                const float *value) {
            return this->SetParameterArray4(this->ParameterLocation(name), count,
                value);
        }

        /**
         * Sets the one component of the parameter 'name'. 'value' points to 
         * an array with 'count' times one integer values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The name of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray1(const char *name, const GLsizei count, 
                const int *value) {
            return this->SetParameterArray1(this->ParameterLocation(name), count,
                value);
        }

        /**
         * Sets the two components of the parameter 'name'. 'value' points to 
         * an array with 'count' times two integer values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The name of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray2(const char *name, const GLsizei count, 
                const int *value) {
            return this->SetParameterArray2(this->ParameterLocation(name), count,
                value);
        }

        /**
         * Sets the three components of the parameter 'name'. 'value' points to
         * an array with 'count' times three integer values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The name of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray3(const char *name, const GLsizei count, 
                const int *value) {
            return this->SetParameterArray3(this->ParameterLocation(name), count,
                value);
        }

        /**
         * Sets the four components of the parameter 'name'. 'value' points to 
         * an array with 'count' times four integer values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The name of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray4(const char *name, const GLsizei count, 
                const int *value) {
            return this->SetParameterArray4(this->ParameterLocation(name), count,
                value);
        }

        /**
         * Sets the one component of the parameter 'name'. 'value' points to 
         * an array with 'count' times one float values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The location of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray1(const GLint name, 
            const GLsizei count, const float *value);

        /**
         * Sets the two components of the parameter 'name'. 'value' points to 
         * an array with 'count' times two float values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The location of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray2(const GLint name, 
            const GLsizei count, const float *value);

        /**
         * Sets the three components of the parameter 'name'. 'value' points to
         * an array with 'count' times three float values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The location of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray3(const GLint name, 
            const GLsizei count, const float *value);

        /**
         * Sets the four components of the parameter 'name'. 'value' points to 
         * an array with 'count' times four float values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The location of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray4(const GLint name, 
            const GLsizei count, const float *value);

        /**
         * Sets the one component of the parameter 'name'. 'value' points to 
         * an array with 'count' times one integer values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The location of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray1(const GLint name, 
            const GLsizei count, const int *value);

        /**
         * Sets the two components of the parameter 'name'. 'value' points to 
         * an array with 'count' times two integer values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The location of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray2(const GLint name, 
            const GLsizei count, const int *value);

        /**
         * Sets the three components of the parameter 'name'. 'value' points to
         * an array with 'count' times three integer values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The location of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray3(const GLint name, 
            const GLsizei count, const int *value);

        /**
         * Sets the four components of the parameter 'name'. 'value' points to 
         * an array with 'count' times four integer values.
         *
         * The shader must be enabled before calling this method.
         *
         * @param name  The location of the parameter.
         * @param count The number of value groups in the array.
         * @param value The values for the parameter.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameterArray4(const GLint name, 
            const GLsizei count, const int *value);

        /**
         * Casts this object to the GLhandleABR value representing the OpenGL
         * program handle.
         *
         * @return The OpenGL program handle.
         */
        inline operator GLhandleARB(void) const {
            return this->hProgObj;
        }

    protected:

        /**
         * Creates and compiles a new shader.
         *
         * @param type Specifies the type of shader to be created.
         * @param src The shader source code elements.
         * @param cnt The number of source code elements.
         * @param insertLineDirective Indicates whether to insert '#line' 
         *                            directives between each two shader source
         *                            elements.
         *
         * @return the compiled shader object.
         *
         * @throw CompileException if there was a compilation error.
         */
        GLhandleARB compileNewShader(GLenum type, const char **src, 
            GLsizei cnt, bool insertLineDirective); 

        /**
         * Answer the shader error string for the specified program object.
         *
         * @param hProg A handle to a program object.
         *
         * @return A string holding the compiler error string.
         *
         * @throws OpenGLException If the compile errors could not be retrieved, 
         *                         i. e. because 'hProg' is not a valid shader
         *                         program.
         */
        static StringA getProgramInfoLog(GLhandleARB hProg);

        /** 
         * Answer the compile status of the program designated by 'hProg'.
         *
         * @param hProg A handle to a program object.
         *
         * @return true, if the program was successfully compiled, false 
         *         otherwise.
         *
         * @throws OpenGLException If the compile status could not be retrieved, 
         *                         i. e. because 'hProg' is not a valid shader
         *                         handle.
         */
        static bool isCompiled(GLhandleARB hProg);

        /** 
         * Answer the linker status of the program designated by 'hProg'.
         *
         * @param hProg A handle to a program object.
         *
         * @return true, if the program was successfully linked, false 
         *         otherwise.
         *
         * @throws OpenGLException If the linker status could not be retrieved, 
         *                         i. e. because 'hProg' is not a valid shader
         *                         handle.
         */
        static bool isLinked(GLhandleARB hProg);

        /** Handle of the program object. */
        GLhandleARB hProgObj;
    };

    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLSLSHADER_H_INCLUDED */
