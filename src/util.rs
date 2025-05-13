use std::ops::Deref;
use std::num::NonZeroU8;
use std::fmt::Display;
use std::time::{Duration, Instant};
use colored::Colorize;

use crate::wgsl::ShaderTypeExt;

pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
impl Display for WorkgroupSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { x, y, z } = self;
        write!(f, "@workgroup_size({x}, {y}, {z})")
    }
}

pub trait Extent3dExt {
    fn num_dispatch_work_groups(&self, size: &WorkgroupSize) -> (u32, u32, u32);
    fn to_padded_buffer_size(&self, format: wgpu::TextureFormat) -> u32;
    fn to_buffer_size<ElmntTy: ShaderTypeExt>(&self) -> u64;
    fn to_wgsl_array_type<ElmntTy: ShaderTypeExt>(&self) -> String;
}
impl Extent3dExt for wgpu::Extent3d {
    fn num_dispatch_work_groups(&self, size: &WorkgroupSize) -> (u32, u32, u32) {
        return (
            (self.width + size.x - 1) / size.x,
            (self.height + size.y - 1) / size.y,
            (self.depth_or_array_layers + size.z - 1) / size.z,
        );
    }
    fn to_padded_buffer_size(&self, format: wgpu::TextureFormat) -> u32 {
        //FIXME: assuming 2D
        let bytes_per_texel: u32 = match format {
            wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => todo!("Add other texture formats"),
        };
        let actual_width = self.width * bytes_per_texel;
        let missing_padding = 256 - (actual_width % 256);
        let padded_width = actual_width + missing_padding;

        return padded_width * self.height;
    }
    fn to_buffer_size<ElmntTy: ShaderTypeExt>(&self) -> u64 {
        let bytes_per_element = std::mem::size_of::<ElmntTy>() as u32;
        // eprintln!("Bytes per element of {}: {bytes_per_element}", std::any::type_name::<ElmntTy>());
        u64::from(self.width * self.height * self.depth_or_array_layers * bytes_per_element)
    }
    fn to_wgsl_array_type<ElmntTy: ShaderTypeExt>(&self) -> String{
        let element_type_name = ElmntTy::wgsl_type_name();
        let Self{width, height, depth_or_array_layers: depth} = self;

        let x_array = format!("array<{element_type_name}, {width}>");
        let y_array = format!("array<{x_array}, {height}>");
        let z_array = format!("array<{y_array}, {depth}>");
        z_array
    }
}

pub trait ImageBufferExt {
    fn extent(&self) -> wgpu::Extent3d;
}
impl<Pix, Container> ImageBufferExt for image::ImageBuffer<Pix, Container>
where
    Pix: image::Pixel,
    Container: Deref<Target = [Pix::Subpixel]>,
{
    fn extent(&self) -> wgpu::Extent3d {
        let (width, height) = self.dimensions();
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Group(pub u32);
impl Into<u32> for Group {
    fn into(self) -> u32 {
        return self.0;
    }
}
impl Display for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Copy, Clone)]
pub struct Binding(pub u32);
impl Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Into<u32> for Binding {
    fn into(self) -> u32 {
        return self.0;
    }
}

pub struct NumChannels(pub NonZeroU8);
impl From<u8> for NumChannels {
    fn from(value: u8) -> Self {
        return Self(value.try_into().unwrap());
    }
}
impl Into<u32> for NumChannels {
    fn into(self) -> u32 {
        return u32::from(u8::from(self.0));
    }
}

pub fn timeit<F, OUT>(task_name: &str, f: F) -> OUT
where
    F: FnOnce() -> OUT
{
    let start = std::time::Instant::now();
    let out = f();
    eprintln!("{task_name} took {:?}", std::time::Instant::now() - start);
    out
}

pub struct MegsPerMs(f64);

impl MegsPerMs{
    pub fn from_num_bytes_duration(num_bytes: impl Into<NumBytes>, duration: Duration) -> Self{
        let bytes_per_ms = num_bytes.into().0 as f64 / (duration.as_secs_f64() * 1000.0);
        return Self(bytes_per_ms / (1024.0 * 1024.0))
    }
}

impl std::fmt::Display for MegsPerMs{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.5} MB/ms", self.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NumBytes(usize);

impl std::fmt::Display for NumBytes{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out: f64 = self.0 as f64;
        let mut num_divisions = 0;

        while out > 1024.0 {
            out /= 1024.0;
            num_divisions += 1;
        }

        match num_divisions{
            0 => write!(f, "{out:.1} B"),
            1 => write!(f, "{out:.1} K"),
            2 => write!(f, "{out:.1} M"),
            3 => write!(f, "{out:.1} G"),
            _ => panic!("this looks like too big of  a number: {}", self.0)
        }
    }
}

impl<T: Sized> From<&[T]> for NumBytes{
    fn from(value: &[T]) -> Self {
        return Self(value.len() * size_of::<T>())
    }
}

pub fn copy_bytes<T: bytemuck::AnyBitPattern>(source: &[u8], descr: &str) -> Vec<T>{
    let start = Instant::now();
    let out: Vec<T> = bytemuck::cast_slice::<_, _>(source).to_owned();
    let num_copied_bytes = NumBytes::from(out.as_slice());
    let duration = Instant::now() - start;
    let megs_per_ms = MegsPerMs::from_num_bytes_duration(num_copied_bytes, duration);
    eprintln!(
        "Copied {} {} in {} at {}",
        num_copied_bytes.to_string().blue(),
        descr.red(),
        format!("{:.1}ms", (duration.as_secs_f64() * 1000.0)).blue(),
        megs_per_ms.to_string().yellow()
    );

    out
}


