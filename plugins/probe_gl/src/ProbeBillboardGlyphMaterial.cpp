#include "ProbeBillboardGlyphMaterial.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

megamol::probe_gl::ProbeBillboardGlyphMaterial::ProbeBillboardGlyphMaterial() 
    : m_version(0)
    , m_textured_glyph_mtl_idx(0)
    , m_vector_glpyh_mtl_idx(0)
    , m_scalar_glyph_mtl_idx(0)
    , m_glyph_images_slot("GetProbes", "Slot for accessing a probe collection") 
{

    this->m_glyph_images_slot.SetCompatibleCall<mesh::CallImageDescription>();
    this->MakeSlotAvailable(&this->m_glyph_images_slot);

}

megamol::probe_gl::ProbeBillboardGlyphMaterial::~ProbeBillboardGlyphMaterial() {}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::create() {


    auto create_progam = [this](vislib::StringA shader_base_name, std::shared_ptr<ShaderProgram> shader_prgm)
    {
        vislib::graphics::gl::ShaderSource vert_shader_src;
        vislib::graphics::gl::ShaderSource frag_shader_src;

        vislib::StringA vertShaderName = shader_base_name + "::vertex";
        vislib::StringA fragShaderName = shader_base_name + "::fragment";

        this->instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
        this->instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);

        std::string vertex_src(vert_shader_src.WholeCode(), (vert_shader_src.WholeCode()).Length());
        std::string fragment_src(frag_shader_src.WholeCode(), (frag_shader_src.WholeCode()).Length());

        bool prgm_error = false;

        if (!vertex_src.empty()) 
            prgm_error |= !shader_prgm->compileShaderFromString(&vertex_src, ShaderProgram::VertexShader);
        if (!fragment_src.empty())
            prgm_error |=
                !shader_prgm->compileShaderFromString(&fragment_src, ShaderProgram::FragmentShader);

        prgm_error |= !shader_prgm->link();

        if (prgm_error) {
            std::cerr << "Error during shader program creation of \"" << shader_prgm->getDebugLabel()
                      << "\"" << std::endl;
            std::cerr << shader_prgm->getLog();
        }
    };


    this->m_textured_glyph_prgm = std::make_shared<ShaderProgram>();
    create_progam("TexturedProbeGlyph", m_textured_glyph_prgm);

    this->m_scalar_probe_glyph_prgm = std::make_shared<ShaderProgram>();
    create_progam("ScalarProbeGlyph", m_scalar_probe_glyph_prgm);

    this->m_vector_probe_glyph_prgm = std::make_shared<ShaderProgram>();
    create_progam("VectorProbeGlyph", m_vector_probe_glyph_prgm);

    // Set intial state of module
    ++m_version;
    m_textured_glyph_mtl_idx = m_gpu_materials->addMaterial(m_textured_glyph_prgm);
    m_scalar_glyph_mtl_idx = m_gpu_materials->addMaterial(m_scalar_probe_glyph_prgm);
    m_vector_glpyh_mtl_idx = m_gpu_materials->addMaterial(m_vector_probe_glyph_prgm);

    return true;
}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getDataCallback(core::Call& caller) {

    mesh::CallGPUMaterialData* lhs_mtl_call = dynamic_cast<mesh::CallGPUMaterialData*>(&caller);
    if (lhs_mtl_call == NULL) return false;

    std::shared_ptr<mesh::GPUMaterialCollecton> mtl_collection;
    // no incoming material -> use your own material storage
    if (lhs_mtl_call->getData() == nullptr) 
        mtl_collection = this->m_gpu_materials;
    else
        mtl_collection = lhs_mtl_call->getData();

    // if there is a material connection to the right, pass on the material collection
    mesh::CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<mesh::CallGPUMaterialData>();
    if (rhs_mtl_call != NULL){
        auto rhs_version = rhs_mtl_call->version();
        rhs_mtl_call->setData(mtl_collection, rhs_version);
        (*rhs_mtl_call)(0);
    }

    mesh::CallImage* ic = this->m_glyph_images_slot.CallAs<mesh::CallImage>();
    if (ic != NULL){
        if (!(*ic)(0)) return false;

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

            // ToDo Clear only existing entry in collection?
            mtl_collection->clearMaterials();

            mtl_collection->addMaterial(this->m_textured_glyph_prgm, textures);
            m_gpu_materials->addMaterial(this->m_scalar_probe_glyph_prgm);
            m_gpu_materials->addMaterial(this->m_vector_probe_glyph_prgm);
        }

    }
    
    if (lhs_mtl_call->version() < m_version){
        lhs_mtl_call->setData(mtl_collection, m_version);
    }

    return true; 
}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getMetaDataCallback(core::Call& caller) {

    //if (!mesh::AbstractGPUMaterialDataSource::getMetaDataCallback(caller)) return false;

    //auto lhs_mtl_call = dynamic_cast<mesh::CallGPUMaterialData*>(&caller);
    auto glyph_image_call = m_glyph_images_slot.CallAs<mesh::CallImage>();
    //
    //auto lhs_mtl_meta_data = lhs_mtl_call->getMetaData();
    //auto glyph_image_meta_data = glyph_image_call->getMetaData();

    if (glyph_image_call != NULL) {
        if (!(*glyph_image_call)(1)) return false;
    }

    return true;
}
