use std::fmt::Display;


pub struct Group(pub u32);
impl Display for Group{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
pub struct Binding(pub u32);
impl Display for Binding{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct InputTextureSlot{
    name: String,
    group: Group,
    binding: Binding,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
}

impl InputTextureSlot{
    pub fn new(
        name: String,
        group: Group,
        binding: Binding,
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self{
        return Self{name, group, binding, sample_type, view_dimension}
    }
    pub fn name(&self) -> &str{
        return &self.name
    }
    pub fn to_wgsl_declaration(&self) -> String{
        let name = &self.name;
        let sample_type = match self.sample_type {
            wgpu::TextureSampleType::Float { .. } => "f32",
            _ => panic!("can't handle sample types different than Float for now")
        };
        let texture_base_type = match self.view_dimension{
            wgpu::TextureViewDimension::D2 => "texture_2d",
            _ => panic!("can't handle any other view dimension for now")
        };
        let group = self.group;
        let binding = self.binding;
        format!(
            "@group({group}) @binding({binding}) var {name} : {texture_base_type}<{sample_type}>;"
        )
    }
    pub fn to_binding_type(&self) -> wgpu::BindingType{
        wgpu::BindingType::Texture {
            sample_type: self.sample_type,
            view_dimension: self.view_dimension,
            multisampled: false
        }
    }
    pub fn to_bind_group_layout_entry(&self) -> wgpu::BindGroupLayoutEntry{
        return wgpu::BindGroupLayoutEntry{
            binding: self.binding.0,
            count: None,
            ty: self.to_binding_type(),
            visibility: wgpu::ShaderStages::COMPUTE,
        }
    }
    pub fn allocate_texture(&self, device: &wgpu::Device, size: wgpu::Extent3d) -> wgpu::Texture{
        device.create_texture(&wgpu::TextureDescriptor{
            dimension: self.view_dimension.compatible_texture_dimension(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            label: None,
            mip_level_count: 1, //FIXME: double check it
            sample_count: 1,
            size,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }
}

impl Display for InputTextureSlot{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_wgsl_declaration())
    }
}

pub struct OutpuTextureSlot{
    name: String,
    group: Group,
    binding: Binding,
    texture_format: wgpu::TextureFormat,
    view_dimension: wgpu::TextureViewDimension,

}
impl OutpuTextureSlot{
    pub fn new(
        name: String,
        group: Group,
        binding: Binding,
        texture_format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self{
        Self{name, group, binding, texture_format, view_dimension}
    }
    pub fn name(&self) -> &str{
        return &self.name
    }
    pub fn to_wgsl_declaration(&self) -> String{
        let name = &self.name;
        let texture_format = match self.texture_format {
            wgpu::TextureFormat::Rgba8Unorm => "rgba8unorm",
            _ => panic!("can't handle sample types different than Float for now")
        };
        let texture_base_type = match self.view_dimension{
            wgpu::TextureViewDimension::D2 => "texture_storage_2d",
            _ => panic!("can't handle any other view dimension for now")
        };
        let group = self.group;
        let binding = self.binding;
        format!(
            "@group({group}) @binding({binding}) var {name} : {texture_base_type}<{texture_format}, write>;"
        )
    }
    pub fn to_binding_type(&self) -> wgpu::BindingType{
        wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: self.texture_format,
            view_dimension: self.view_dimension,
        }
    }
    pub fn to_bind_group_layout_entry(&self) -> wgpu::BindGroupLayoutEntry{
        return wgpu::BindGroupLayoutEntry{
            binding: self.binding.0,
            count: None,
            ty: self.to_binding_type(),
            visibility: wgpu::ShaderStages::COMPUTE,
        }
    }
}
impl Display for OutpuTextureSlot{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_wgsl_declaration())
    }
}