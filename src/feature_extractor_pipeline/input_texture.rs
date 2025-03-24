use std::fmt::Display;

use crate::util::{Binding, Group};

pub struct InputTextureSlot {
    name: String,
    group: Group,
    binding: Binding,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
}

impl InputTextureSlot {
    pub fn new(
        name: String,
        group: Group,
        binding: Binding,
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Self {
        return Self {
            name,
            group,
            binding,
            sample_type,
            view_dimension,
        };
    }
    pub fn create_texture(&self, device: &wgpu::Device, size: wgpu::Extent3d) -> InputTexture {
        let name = &self.name;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("input_texture__{name}")),
            dimension: self.view_dimension.compatible_texture_dimension(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            mip_level_count: 1, //FIXME: double check it
            sample_count: 1,
            size,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        return InputTexture {
            texture,
            texture_view,
            binding: self.binding,
        };
    }
    pub fn name(&self) -> &str {
        return &self.name;
    }
    pub fn view_dimension(&self) -> &wgpu::TextureViewDimension {
        return &self.view_dimension;
    }
    pub fn to_wgsl_declaration(&self) -> String {
        let name = &self.name;
        let sample_type = match self.sample_type {
            wgpu::TextureSampleType::Float { .. } => "f32",
            _ => panic!("can't handle sample types different than Float for now"),
        };
        let texture_base_type = match self.view_dimension {
            wgpu::TextureViewDimension::D2 => "texture_2d",
            wgpu::TextureViewDimension::D3 => "texture_3d",
            _ => panic!(
                "can't handle any other view dimension for now: {:?}",
                self.view_dimension
            ),
        };
        let group = &self.group;
        let binding = &self.binding;
        format!("@group({group}) @binding({binding}) var {name} : {texture_base_type}<{sample_type}>;")
    }
    pub fn to_binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            sample_type: self.sample_type,
            view_dimension: self.view_dimension,
            multisampled: false,
        }
    }
    pub fn to_bind_group_layout_entry(&self) -> wgpu::BindGroupLayoutEntry {
        return wgpu::BindGroupLayoutEntry {
            binding: self.binding.into(),
            count: None,
            ty: self.to_binding_type(),
            visibility: wgpu::ShaderStages::COMPUTE,
        };
    }
}

impl Display for InputTextureSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_wgsl_declaration())
    }
}

pub struct InputTexture {
    binding: Binding,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
}
impl InputTexture {
    pub fn to_bind_group_entry(&self) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding: self.binding.into(),
            resource: wgpu::BindingResource::TextureView(&self.texture_view),
        }
    }
    pub fn write_texture(&self, queue: &wgpu::Queue, image: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>) {
        queue.write_texture(
            self.texture.as_image_copy(),
            image.as_raw(),
            wgpu::TexelCopyBufferLayout {
                bytes_per_row: Some(4 * image.width()),
                rows_per_image: Some(image.height()),
                offset: 0,
            },
            wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
        )
    }
}
