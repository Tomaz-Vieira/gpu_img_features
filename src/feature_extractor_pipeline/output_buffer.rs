use std::{fmt::Display, marker::PhantomData};

use encase::nalgebra::Vector3;

use crate::{util::{Binding, Group}, wgsl::ShaderTypeExt};

use super::kernel::gaussian_blur::GaussianBlur;

pub struct OutputBufferSlot<T> {
    pub name: String,
    pub group: Group,
    pub binding: Binding,
    pub marker: PhantomData<T>,
}
impl<T> OutputBufferSlot<T> {
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

pub struct KernelBufferSlot<T> {
    name: String,
    group: Group,
    binding: Binding,
    marker: PhantomData<T>,
    kernel: GaussianBlur,
    buffer: wgpu::Buffer,
}
impl<T> KernelBufferSlot<T> {
    pub fn new(
        device: &wgpu::Device,
        name: String,
        group: Group,
        binding: Binding,
        kernel: GaussianBlur,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("uniform_buffer__{name}")),
            mapped_at_creation: true,
            size: kernel.required_size_in_bytes() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        {
            let mut bytes_slice = buffer.slice(..).get_mapped_range_mut();
            let kernel_values = bytemuck::cast_slice_mut::<_, f32>(&mut bytes_slice);
            kernel.fill_slice_yx(kernel_values);
        }
        buffer.unmap();
        Self{
            name, group, binding, buffer, kernel, marker: PhantomData,
        }
    }
    pub fn kernel(&self) -> &GaussianBlur{
        &self.kernel
    }
    pub fn wgsl_kernel_value_at_center_offset(&self, offset_variable: &str) -> String{
        let slot_name = &self.name;
        let kernel_side_len = self.kernel.kernel_side_len();
        format!("{slot_name}[{offset_variable}.y * {kernel_side_len} + {offset_variable}.x]")
    }
    pub fn to_binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
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
impl<T: ShaderTypeExt> Display for KernelBufferSlot<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = &self.name;
        let group = &self.group;
        let binding = &self.binding;
        let element_type_name = T::wgsl_type_name();
        write!(
            f,
            "@group({group}) @binding({binding}) var<uniform> {name} : array<{element_type_name}>;",
        )
    }
}
