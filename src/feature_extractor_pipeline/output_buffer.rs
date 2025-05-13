use std::{fmt::Display, marker::PhantomData, time::Instant};

use nalgebra::Vector2;

use crate::{util::{Binding, Extent3dExt, Group, MegsPerMs}, wgsl::ShaderTypeExt};

use super::kernel::gaussian_blur::GaussianBlur;

pub struct OutputBufferSlot<T, const KSIDE: usize> {
    pub name: String,
    pub group: Group,
    pub binding: Binding,
    pub img_extent: wgpu::Extent3d,
    pub marker: PhantomData<T>,
}

impl<T: ShaderTypeExt, const KSIDE: usize> OutputBufferSlot<T, KSIDE> {
    #[allow(non_snake_case)]
    pub fn wgsl_indexing_from_kernIdx_xyzOffset(&self, xyz_offset_expr: &str) -> String{
        format!("[{xyz_offset_expr}.z][{xyz_offset_expr}.y][{xyz_offset_expr}.x]")
    }
    pub fn output_buffer_size(&self) -> u64{
        self.img_extent.to_buffer_size::<T>() //FIXME: assumes oiutput has same simensions as img? maybe this is ok
    }
    pub fn create_output_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let size = self.output_buffer_size();
        eprintln!("Gonna create an output buffer of size {size}");
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("output_buffer__{}", self.name)),
            mapped_at_creation: false,
            size,
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
        write!(
            f,
            "@group({group}) @binding({binding}) var<storage, read_write> {name} : {img_array_type};",
            // "@group({group}) @binding({binding}) var<storage, read_write> {name} : array<{img_array_type}, {num_kernels}>;",
        )
    }
}

pub struct KernelsInBuffSlot<const KSIDE: usize> {
    name: String,
    group: Group,
    binding: Binding,
    kernels: Vec<GaussianBlur<KSIDE>>,
    buffer: wgpu::Buffer,
}
impl<const KSIDE: usize> KernelsInBuffSlot<KSIDE> {
    pub fn new(
        device: &wgpu::Device,
        name: String,
        group: Group,
        binding: Binding,
        kernels: Vec<GaussianBlur<KSIDE>>,
    ) -> Self {
        let buffer_byte_length = kernels[0].required_size_in_bytes() * kernels.len(); //FIXME: [0]

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("kernel_buffer__{name}")),
            mapped_at_creation: true,
            size: buffer_byte_length as u64,
            usage: wgpu::BufferUsages::STORAGE,
        });

        {
            let mut bytes_slice = buffer.slice(..).get_mapped_range_mut();
            let kernel_values: &mut [f32] = bytemuck::cast_slice_mut(&mut bytes_slice);

            let iradius = i64::try_from((KSIDE - 1) / 2).unwrap();
            let mut offset: usize = 0;

            let start = Instant::now();
            for y in -iradius..=iradius{
                for x in -iradius..=iradius{
                    for kern in &kernels{
                        kernel_values[offset] = kern.kernel_at(Vector2::new(x, y));
                        offset += 1;
                    }
                }
            }
            let duration = Instant::now() - start;
            let megabytes_per_s = MegsPerMs::from_num_bytes_duration(&*bytes_slice, duration);
            eprintln!("Copied {buffer_byte_length} bytes form cpu to GPU in {duration:?} at {megabytes_per_s}");
        }

        buffer.unmap();
        Self{
            name, group, binding, buffer, kernels
        }
    }
    pub fn kernels(&self) -> &[GaussianBlur<KSIDE>] {
        &self.kernels
    }
    pub fn radius(&self) -> usize{
        (KSIDE - 1) / 2
    }
    pub fn write_wgsl_feature_calcs(&self, mut out: &mut impl std::fmt::Write) -> Result<(), std::fmt::Error> {
        let radius = self.radius();
        let slot_name = &self.name;
        let num_kernels = self.kernels().len();

        for (k_idx, _kernel) in self.kernels.iter().enumerate(){
            //FIXME: assumes input image has 3 channels
            write!(&mut out, "
                var feature_{k_idx}: vec3<f32> = vec3(0.0, 0.0, 0.0);"
            )?;
        }
        write!(&mut out, "
                var in_buf_kernels_offset: i32 = 0;
                for (var y=-{radius}; y<={radius}; y++){{
                    for (var x=-{radius}; x<={radius}; x++){{
                        let offset = vec2<i32>(x, y);
                        let sample_coords: vec2<i32> = vec2<i32>(
                            clamp(current_coords.x + offset.x, 0, texture_upper_limit.x),
                            clamp(current_coords.y + offset.y, 0, texture_upper_limit.y),
                        );
                        let sample = textureLoad(input_image, sample_coords, 0).xyz;
                        {}
                        in_buf_kernels_offset += {num_kernels};
                    }}
                }}
            ",
            self.kernels.iter().enumerate()
                .map(|(k_idx, _kern)| format!("
                        //FIXME: ilastik features don't go from 0 to 1.0, but from 0.0 to 255.0, i think
                        feature_{k_idx} += sample * {slot_name}[in_buf_kernels_offset + {k_idx}] * 255.0;"))
                .collect::<Vec<_>>()
                .join("")
        )
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
impl<const KSIDE: usize> Display for KernelsInBuffSlot< KSIDE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = &self.name;
        let group = &self.group;
        let binding = &self.binding;
        write!(
            f,
            "@group({group}) @binding({binding}) var<storage, read> {name} : array<f32>;",
        )
    }
}
