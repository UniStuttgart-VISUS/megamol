#include "ProbeBillboardGlyphMaterial.h"

#include "mesh/MeshCalls.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "probe/ProbeCalls.h"

megamol::probe_gl::ProbeBillboardGlyphMaterial::ProbeBillboardGlyphMaterial()
        : m_version(0)
        , m_glyph_images_slot("GetProbes", "Slot for accessing a probe collection") {

    this->m_glyph_images_slot.SetCompatibleCall<mesh::CallImageDescription>();
    this->MakeSlotAvailable(&this->m_glyph_images_slot);
}

megamol::probe_gl::ProbeBillboardGlyphMaterial::~ProbeBillboardGlyphMaterial() {}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::create() {


    auto create_progam = [this](vislib::StringA shader_base_name) -> std::shared_ptr<ShaderProgram> {
        vislib_gl::graphics::gl::ShaderSource vert_shader_src;
        vislib_gl::graphics::gl::ShaderSource frag_shader_src;

        vislib::StringA vertShaderName = shader_base_name + "::vertex";
        vislib::StringA fragShaderName = shader_base_name + "::fragment";

        auto ssf =
            std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());

        if (!ssf->MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src)) {
            throw;
        }
        if (!ssf->MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src)) {
            throw;
        }

        std::string vertex_src(vert_shader_src.WholeCode(), (vert_shader_src.WholeCode()).Length());
        std::string fragment_src(frag_shader_src.WholeCode(), (frag_shader_src.WholeCode()).Length());

        std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> shader_srcs;
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Vertex, vertex_src});
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Fragment, fragment_src});

        auto shader_prgm = std::make_unique<glowl::GLSLProgram>(shader_srcs);
        shader_prgm->setDebugLabel(std::string(shader_base_name)); // TODO debug label not set in time for catch...

        return shader_prgm;
    };

    try {
        this->m_textured_glyph_prgm = create_progam("TexturedProbeGlyph");
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n", "TexturedProbeGlyph", exc.what(),
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: Unknown exception\n");
        return false;
    }

    try {
        this->m_scalar_probe_glyph_prgm = create_progam("ScalarProbeGlyph");
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n", "ScalarProbeGlyph", exc.what(),
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: Unknown exception\n");
        return false;
    }

    try {
        this->m_scalar_distribution_probe_glyph_prgm = create_progam("ScalarDistributionProbeGlyph");
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n", "ScalarDistributionProbeGlyph",
            exc.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: Unknown exception\n");
        return false;
    }

    try {
        this->m_vector_probe_glyph_prgm = create_progam("VectorProbeGlyph");
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n", "VectorProbeGlyph", exc.what(),
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: Unknown exception\n");
        return false;
    }

    try {
        this->m_clusterID_glyph_prgm = create_progam("ClusterIDGlyph");
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n", "ClusterIDGlyph", exc.what(),
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader: Unknown exception\n");
        return false;
    }

    // Set initial state of module
    ++m_version;
    m_material_collection.first = std::make_shared<mesh_gl::GPUMaterialCollection>();
    m_material_collection.first->addMaterial("ProbeBillboard_Textured", m_textured_glyph_prgm);
    m_material_collection.second.push_back("ProbeBillboard_Textured");
    m_material_collection.first->addMaterial("ProbeBillboard_Scalar", m_scalar_probe_glyph_prgm);
    m_material_collection.second.push_back("ProbeBillboard_Scalar");
    m_material_collection.first->addMaterial(
        "ProbeBillboard_ScalarDistribution", m_scalar_distribution_probe_glyph_prgm);
    m_material_collection.second.push_back("ProbeBillboard_ScalarDistribution");
    m_material_collection.first->addMaterial("ProbeBillboard_Vector", m_vector_probe_glyph_prgm);
    m_material_collection.second.push_back("ProbeBillboard_Vector");
    m_material_collection.first->addMaterial("ProbeBillboard_ClusterID", m_clusterID_glyph_prgm);
    m_material_collection.second.push_back("ProbeBillboard_ClusterID");

    return true;
}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getDataCallback(core::Call& caller) {

    mesh_gl::CallGPUMaterialData* lhs_mtl_call = dynamic_cast<mesh_gl::CallGPUMaterialData*>(&caller);
    mesh_gl::CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<mesh_gl::CallGPUMaterialData>();

    if (lhs_mtl_call == NULL) {
        return false;
    }

    std::vector<std::shared_ptr<mesh_gl::GPUMaterialCollection>> gpu_mtl_collections;
    // if there is a material connection to the right, issue callback
    if (rhs_mtl_call != nullptr) {
        (*rhs_mtl_call)(0);
        if (rhs_mtl_call->hasUpdate()) {
            ++m_version;
        }
        gpu_mtl_collections = rhs_mtl_call->getData();
    }
    gpu_mtl_collections.push_back(m_material_collection.first);

    mesh::CallImage* ic = this->m_glyph_images_slot.CallAs<mesh::CallImage>();
    if (ic != NULL) {
        if (!(*ic)(0))
            return false;

        auto image_meta_data = ic->getMetaData();

        // something has changed in the neath...
        bool something_has_changed = ic->hasUpdate();

        if (something_has_changed) {
            ++m_version;

            auto img_data = ic->getData();

            // use first image to determine size -> assumes same size for all images
            auto img_height = img_data->accessImages().front().height;
            auto img_width = img_data->accessImages().front().width;
            auto img_format =
                mesh::ImageDataAccessCollection::convertToGLInternalFormat(img_data->accessImages().front().format);

            glowl::TextureLayout tex_layout;
            tex_layout.width = img_width;
            tex_layout.height = img_height;
            tex_layout.depth = 2048;
            tex_layout.levels = 1;
            // TODO
            tex_layout.format = GL_RGBA;
            tex_layout.type = GL_UNSIGNED_BYTE;
            // TODO
            tex_layout.internal_format = img_format;

            tex_layout.int_parameters = {
                {GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}};

            size_t img_cnt = img_data->accessImages().size();
            size_t required_tx_arrays = static_cast<size_t>(std::ceil(static_cast<double>(img_cnt) / 2048.0));

            std::vector<std::shared_ptr<glowl::Texture>> textures(required_tx_arrays, nullptr);
            for (auto& tx_array : textures) {
                auto new_tex_ptr = std::make_shared<glowl::Texture2DArray>("ProbeGlyph", tex_layout, nullptr);
                tx_array = std::static_pointer_cast<glowl::Texture>(new_tex_ptr);
            }

            auto images = img_data->accessImages();
            for (size_t i = 0; i < images.size(); ++i) {
                auto texture_idx = i / 2048;
                auto slice_idx = i % 2048;
                textures[texture_idx]->bindTexture();

                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, slice_idx, tex_layout.width, tex_layout.height, 1,
                    tex_layout.format, tex_layout.type, images[i].data);

                glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
            }

            glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

            for (auto& identifier : m_material_collection.second) {
                m_material_collection.first->deleteMaterial(identifier);
            }
            m_material_collection.second.clear();

            m_material_collection.first->addMaterial("ProbeBillboard_Textured", m_textured_glyph_prgm, textures);
            m_material_collection.second.push_back("ProbeBillboard_Textured");
            m_material_collection.first->addMaterial("ProbeBillboard_Scalar", m_scalar_probe_glyph_prgm);
            m_material_collection.second.push_back("ProbeBillboard_Scalar");
            m_material_collection.first->addMaterial("ProbeBillboard_Vector", m_vector_probe_glyph_prgm);
            m_material_collection.second.push_back("ProbeBillboard_Vector");
            m_material_collection.first->addMaterial("ProbeBillboard_ClusterID", m_clusterID_glyph_prgm);
            m_material_collection.second.push_back("ProbeBillboard_ClusterID");
        }
    }

    lhs_mtl_call->setData(gpu_mtl_collections, m_version);

    return true;
}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getMetaDataCallback(core::Call& caller) {

    // if (!mesh::AbstractGPUMaterialDataSource::getMetaDataCallback(caller)) return false;

    // auto lhs_mtl_call = dynamic_cast<mesh::CallGPUMaterialData*>(&caller);
    auto glyph_image_call = m_glyph_images_slot.CallAs<mesh::CallImage>();
    //
    // auto lhs_mtl_meta_data = lhs_mtl_call->getMetaData();
    // auto glyph_image_meta_data = glyph_image_call->getMetaData();

    if (glyph_image_call != NULL) {
        if (!(*glyph_image_call)(1))
            return false;
    }

    return true;
}
