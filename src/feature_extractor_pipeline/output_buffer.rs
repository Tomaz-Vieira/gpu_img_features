use std::{fmt::Display, marker::PhantomData};

use crate::{util::{Binding, Group}, wgsl::ShaderTypeExt};

pub struct OutputBufferSlot<T> {
    pub name: String,
    pub group: Group,
    pub binding: Binding,
    pub marker: PhantomData<T>,
}
impl<T: ShaderTypeExt> OutputBufferSlot<T> {
    pub fn create_buffer(&self, device: &wgpu::Device, size: wgpu::BufferSize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("output_buffer__{}", self.name)),
            mapped_at_creation: false,
            size: size.into(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
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
impl<T: ShaderTypeExt> Display for OutputBufferSlot<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = &self.name;
        let group = &self.group;
        let binding = &self.binding;
        let element_type_name = T::wgsl_type_name();
        write!(
            f,
            "@group({group}) @binding({binding}) var<storage, read_write> {name} : array<{element_type_name}>;",
        )
    }
}
