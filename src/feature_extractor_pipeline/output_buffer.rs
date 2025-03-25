use std::fmt::Display;

use crate::util::{Binding, BufferSize, Group};

pub struct OutputBuffer {
    binding: Binding,
    buffer: wgpu::Buffer,
    size: BufferSize,
}
impl OutputBuffer {
    pub fn size(&self) -> BufferSize {
        return self.size;
    }
}

impl OutputBuffer {
    pub fn to_bind_group_entry(&self) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding: self.binding.into(),
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.buffer,
                offset: 0,
                size: Some(self.size.requested_size().into()), //FIXME?
            }),
        }
    }
    pub fn raw(&self) -> &wgpu::Buffer {
        return &self.buffer;
    }
}

pub struct OutputBufferSlot {
    pub name: String,
    pub group: Group,
    pub binding: Binding,
}
impl OutputBufferSlot {
    pub fn create_buffer(&self, device: &wgpu::Device, size: BufferSize) -> OutputBuffer {
        let name = &self.name;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("output_buffer__{name}")),
            mapped_at_creation: false,
            size: size.padded_size().into(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        return OutputBuffer {
            buffer,
            size,
            binding: self.binding,
        };
    }
    pub fn name(&self) -> &str {
        return &self.name;
    }
    pub fn to_binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None, //FIXME?
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
impl Display for OutputBufferSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = &self.name;
        let group = &self.group;
        let binding = &self.binding;
        write!(f, "@group({group}) @binding({binding}) var<storage, read_write> {name} : array<vec4<f32>>;")
    }
}
