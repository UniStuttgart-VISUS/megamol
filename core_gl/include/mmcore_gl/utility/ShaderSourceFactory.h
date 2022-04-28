/*
 * ShaderSourceFactory.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef BEZTUBE_SHADERSOURCEFACTORY_H_INCLUDED
#define BEZTUBE_SHADERSOURCEFACTORY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore_gl/utility/BTFParser.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"


namespace megamol {
namespace core_gl {
namespace utility {


/**
 * Factory class for shader sources.
 *
 * Shaders and shader snippets are adressed by names with namespaces,
 * similar to slots in the view graph. The first namespace identifies the
 * btf file name and must not be omitted in the function calls.
 */
class ShaderSourceFactory {
public:
    /** The btf parser has access to the internal data structure */
    friend class BTFParser;

    /** do not insert "#line" */
    static const UINT32 FLAGS_NO_LINE_PRAGMAS;

    /** insert "#line" in GLSL style */
    static const UINT32 FLAGS_GLSL_LINE_PRAGMAS;

    /** insern "#line" in HLSL style */
    static const UINT32 FLAGS_HLSL_LINE_PRAGMAS;

    /** The default flags */
    static const UINT32 FLAGS_DEFAULT_FLAGS;

    /**
     * Ctor.
     *
     * @param shader_dirs_ The shader dirs of the config.
     */
    ShaderSourceFactory(vislib::Array<vislib::StringW> shader_dirs_);

    /**
     * Dtor.
     */
    ~ShaderSourceFactory(void);

    /**
     * Loads a btf based on its root-namespace name.
     *
     * @param name The root-namespace name of the btf file to load.
     * @param forceReload Force the reloading of the btf file.
     *
     * @return 'true' on success, 'false' on failure (an error message
     *         will be logged).
     */
    bool LoadBTF(const vislib::StringA& name, bool forceReload = false);

    /**
     * Makes the requested shader snippet.
     *
     * @param name The fully qualified name of the shader snippet to
     *             create, including namespaces.
     * @param flags The construction flags
     *
     * @return The requested shader source snippet or 'NULL' in case of
     *         an error (an error messages will be logged).
     */
    vislib::SmartPtr<vislib_gl::graphics::gl::ShaderSource::Snippet> MakeShaderSnippet(
        const vislib::StringA& name, UINT32 flags = FLAGS_DEFAULT_FLAGS);

    /**
     * Makes the requested shader.
     *
     * @param name The fully qualified name of the shader to create,
     *             including namespaces.
     * @param outSrc The shader source object to receive the new shader
     *               code.
     * @param flags The construction flags
     *
     * @return 'true' on success, 'false' if the shader code could not
     *         be created (an error messages will be logged).
     */
    bool MakeShaderSource(const vislib::StringA& name, vislib_gl::graphics::gl::ShaderSource& outShaderSrc,
        UINT32 flags = FLAGS_DEFAULT_FLAGS);

private:
    /**
     * Utility class allowing to push the root namespace ontop of each
     * working stack.
     */
    class ShallowBTFNamespace : public BTFParser::BTFNamespace {
    public:
        /**
         * Ctor.
         */
        ShallowBTFNamespace(BTFParser::BTFNamespace& n) : BTFParser::BTFNamespace(), n(n) {
            this->SetName(n.Name());
        }

        /**
         * Dtor.
         */
        virtual ~ShallowBTFNamespace(void) {
            // intentionally empty
        }

        /**
         * Gets the list of the children.
         *
         * @return The list of the children.
         */
        virtual vislib::SingleLinkedList<vislib::SmartPtr<BTFElement>>& Children(void) {
            return this->n.Children();
        }

        /**
         * Gets the list of the children.
         *
         * @return The list of the children.
         */
        virtual const vislib::SingleLinkedList<vislib::SmartPtr<BTFElement>>& Children(void) const {
            return this->n.Children();
        }

        /**
         * Gets the child element with the given name.
         *
         * @param name The name to search for.
         *
         * @return The found child element or NULL.
         */
        virtual vislib::SmartPtr<BTFElement> FindChild(const vislib::StringA& name) const {
            return this->n.FindChild(name);
        }

    private:
        /** The real namespace object. */
        BTFParser::BTFNamespace& n;
    };

    /**
     * Answers the file number for the given top-level namespace.
     *
     * @param name The top-level namespace
     *
     * @return The file number
     */
    int fileNo(const vislib::StringA& name);

    /**
     * Requests a BTFElement
     *
     * @param name The name (including namespace) of the BTFElement to
     *             return.
     *
     * @return The requested BTFElement or NULL in case of an error
     *         (an error message will be logged).
     */
    vislib::SmartPtr<BTFParser::BTFElement> getBTFElement(const vislib::StringA& name);

    /**
     * Makes the requested shader
     *
     * @param s The source object
     * @param o The destination object
     *
     * @return 'true' on success, 'false' if the shader code could not
     *         be created (an error messages will be logged).
     */
    bool makeShaderSource(BTFParser::BTFShader* s, vislib_gl::graphics::gl::ShaderSource& o, UINT32 flags);

    /**
     * Makes the requested snippet.
     *
     * @param s The snippet source object.
     *
     * @return The snippet object.
     */
    vislib::SmartPtr<vislib_gl::graphics::gl::ShaderSource::Snippet> makeSnippet(
        BTFParser::BTFSnippet* s, UINT32 flags);

    /** The shader dirs */
    vislib::Array<vislib::StringW> shader_dirs;

    /** The (empty-named) root namespace */
    BTFParser::BTFNamespace root;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The stable ordered list of fileIds */
    vislib::Array<vislib::StringA> fileIds;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};

} /* end namespace utility */
} // namespace core_gl
} /* end namespace megamol */

#endif /* BEZTUBE_SHADERSOURCEFACTORY_H_INCLUDED */
