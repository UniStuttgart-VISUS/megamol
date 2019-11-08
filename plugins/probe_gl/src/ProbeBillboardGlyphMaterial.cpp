#include "ProbeBillboardGlyphMaterial.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

megamol::probe_gl::ProbeBillboardGlyphMaterial::ProbeBillboardGlyphMaterial() 
    : m_glyph_images_slot("GetProbes", "Slot for accessing a probe collection"), m_glyph_images_slot_cached_hash(0) {

    this->m_glyph_images_slot.SetCompatibleCall<mesh::CallImageDescription>();
    this->MakeSlotAvailable(&this->m_glyph_images_slot);

}

megamol::probe_gl::ProbeBillboardGlyphMaterial::~ProbeBillboardGlyphMaterial() {}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::create() {
    // create shader program
    vislib::graphics::gl::ShaderSource vert_shader_src;
    vislib::graphics::gl::ShaderSource frag_shader_src;

    vislib::StringA shader_base_name("ProbeGlyph");
    vislib::StringA vertShaderName = shader_base_name + "::vertex";
    vislib::StringA fragShaderName = shader_base_name + "::fragment";

    this->instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
    this->instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);

    std::string vertex_src(vert_shader_src.WholeCode(), (vert_shader_src.WholeCode()).Length());
    std::string fragment_src(frag_shader_src.WholeCode(), (frag_shader_src.WholeCode()).Length());

    this->m_billboard_glyph_prgm = std::make_shared<ShaderProgram>();

    bool prgm_error = false;

    if (!vertex_src.empty()) 
        prgm_error |= !this->m_billboard_glyph_prgm->compileShaderFromString(&vertex_src, ShaderProgram::VertexShader);
    if (!fragment_src.empty())
        prgm_error |= !this->m_billboard_glyph_prgm->compileShaderFromString(&fragment_src, ShaderProgram::FragmentShader);

    prgm_error |= !this->m_billboard_glyph_prgm->link();

    if (prgm_error) {
        std::cout << "Error during shader program creation of \"" << this->m_billboard_glyph_prgm->getDebugLabel()
                  << "\"" << std::endl;
        std::cout << this->m_billboard_glyph_prgm->getLog();
    }


    return true;
}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getDataCallback(core::Call& caller) {

    mesh::CallGPUMaterialData* lhs_mtl_call = dynamic_cast<mesh::CallGPUMaterialData*>(&caller);
    if (lhs_mtl_call == NULL) return false;

    // no incoming material -> use your own material storage
    if (lhs_mtl_call->getData() == nullptr) lhs_mtl_call->setData(this->m_gpu_materials);
    std::shared_ptr<mesh::GPUMaterialCollecton> mtl_collection = lhs_mtl_call->getData();

    // if there is a material connection to the right, pass on the material collection
    mesh::CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<mesh::CallGPUMaterialData>();
    if (rhs_mtl_call != NULL) rhs_mtl_call->setData(mtl_collection);

    
    mesh::CallImage* ic = this->m_glyph_images_slot.CallAs<mesh::CallImage>();
    if (ic == NULL) return false;

    auto image_meta_data = ic->getMetaData();

    if (image_meta_data.m_data_hash > this->m_glyph_images_slot_cached_hash)
    {
        this->m_glyph_images_slot_cached_hash = image_meta_data.m_data_hash;

        if (!(*ic)(0)) return false;
        auto img_data = ic->getData();

        std::vector<std::shared_ptr<glowl::Texture>> textures;

        struct col_rgba {
            unsigned char r, g, b, a = 0;
        };
        std::array<col_rgba, 40000> dummy_data;
        for (auto& d : dummy_data) {
            d.r = 255;
            d.a = 255;
        }

        auto images = img_data->accessImages();
        for (auto& img : images)
        {
            glowl::TextureLayout tex_layout;
            tex_layout.width = img.width;
            tex_layout.height = img.height;
            tex_layout.depth = 1;
            tex_layout.levels = 1;
            //TODO
            tex_layout.format = GL_RGBA;
            tex_layout.type = GL_UNSIGNED_BYTE;
            //TODO
            tex_layout.internal_format = mesh::ImageDataAccessCollection::convertToGLInternalFormat(img.format);

            auto new_tex_ptr = std::make_shared<glowl::Texture2D>("ProbeGlyph",tex_layout,img.data);
            //auto new_tex_ptr = std::make_shared<glowl::Texture2D>("ProbeGlyph", tex_layout, dummy_data.data());

            textures.push_back(std::static_pointer_cast<glowl::Texture>(new_tex_ptr));
        }

        //ToDo Clear only existing entry in collection?
        mtl_collection->clearMaterials();

        mtl_collection->addMaterial(this->m_billboard_glyph_prgm,textures);
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

    if (!(*glyph_image_call)(1)) return false;

    return true;
}
