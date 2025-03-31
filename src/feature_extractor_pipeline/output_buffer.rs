use std::{fmt::Display, marker::PhantomData};

use crate::{util::{Binding, Extent3dExt, Group}, wgsl::ShaderTypeExt};

use super::kernel::gaussian_blur::GaussianBlur;

pub struct OutputBufferSlot<T, const KSIDE: usize> {
    pub name: String,
    pub group: Group,
    pub binding: Binding,
    pub img_extent: wgpu::Extent3d,
    pub kernels: Vec<GaussianBlur<KSIDE>>,
    pub marker: PhantomData<T>,
}

impl<T: ShaderTypeExt, const KSIDE: usize> OutputBufferSlot<T, KSIDE> {
    #[allow(non_snake_case)]
    pub fn wgsl_indexing_from_kernIdx_xyzOffset(&self, kern_idx_expr: &str, xyz_offset_expr: &str) -> String{
        format!("[{kern_idx_expr}][{xyz_offset_expr}.z][{xyz_offset_expr}.y][{xyz_offset_expr}.x]")
    }
    pub fn output_buffer_size<ElmntTy: ShaderTypeExt>(&self) -> u64{
        self.kernels.len() as u64 * self.img_extent.to_buffer_size::<ElmntTy>()
    }
    pub fn create_output_buffer<ElmntTy: ShaderTypeExt>(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("output_buffer__{}", self.name)),
            mapped_at_creation: false,
            size: self.output_buffer_size::<ElmntTy>(),
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

impl<T: ShaderTypeExt, const KSIDE: usize> Display for OutputBufferSlot<T, KSIDE>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = &self.name;
        let group = &self.group;
        let binding = &self.binding;
        let img_array_type = self.img_extent.to_wgsl_array_type::<T>();
        let num_kernels = self.kernels.len();
        write!(
            f,
            "@group({group}) @binding({binding}) var<storage, read_write> {name} : array<{img_array_type}, {num_kernels}>;",
        )
    }
}

pub struct KernelBufferSlot<const KSIDE: usize> {
    name: String,
    group: Group,
    binding: Binding,
    kernel: GaussianBlur<KSIDE>,
    buffer: wgpu::Buffer,
}
impl<const KSIDE: usize> KernelBufferSlot<KSIDE> {
    pub fn new(
        device: &wgpu::Device,
        name: String,
        group: Group,
        binding: Binding,
        kernel: GaussianBlur<KSIDE>,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("kernel_buffer__{name}")),
            mapped_at_creation: true,
            size: kernel.required_size_in_bytes() as u64,
            usage: wgpu::BufferUsages::STORAGE,
        });

        {
            let mut bytes_slice = buffer.slice(..).get_mapped_range_mut();
            let kernel_values: &mut [[f32; KSIDE]] = bytemuck::cast_slice_mut(&mut bytes_slice);
            let sized: &mut [[f32; KSIDE]; KSIDE] = kernel_values.try_into().unwrap();
            kernel.fill_slice_yx(sized);
        }
        buffer.unmap();
        Self{
            name, group, binding, buffer, kernel
        }
    }
    pub fn kernel(&self) -> &GaussianBlur<KSIDE> {
        &self.kernel
    }
    pub fn wgsl_kernel_value_at_center_offset(&self, offset_variable: &str) -> String{
        let slot_name = &self.name;
        let idx_expr = self.kernel.wgsl_indexing_from_yx_offset(offset_variable);
        format!("{slot_name}{idx_expr}")
    }
    pub fn to_binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
    pub fn to_bind_group_entry(&self) -> wgpu::BindGroupEntry{
        wgpu::BindGroupEntry{
            binding: self.binding.into(),
            resource: self.buffer.as_entire_binding(),
        }
    }
}
impl<const KSIDE: usize> Display for KernelBufferSlot< KSIDE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = &self.name;
        let group = &self.group;
        let binding = &self.binding;
        write!(
            f,
            "@group({group}) @binding({binding}) var<storage, read> {name} : array<array<f32, {KSIDE}>, {KSIDE}>;",
        )
    }
}
