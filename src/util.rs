use std::{
    fmt::Display,
    num::{NonZeroU64, NonZeroU8},
    ops::Deref,
};

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
    fn to_buffer_size(&self, num_channels: NumChannels) -> BufferSize;
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
        let bytes_per_texel: u32 = match format {
            wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => todo!("Add other texture formats"),
        };
        let actual_width = self.width * bytes_per_texel;
        let missing_padding = actual_width % 256;
        let padded_width = actual_width + missing_padding;

        return padded_width * self.height;
    }
    fn to_buffer_size(&self, num_channels: NumChannels) -> BufferSize {
        let num_chanels_u32: u32 = num_channels.into();
        let bytes_per_channel = 4;

        BufferSize::from(self.width * self.height * self.depth_or_array_layers * num_chanels_u32 * bytes_per_channel)
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

#[derive(Copy, Clone, Debug)]
pub struct BufferSize {
    requested_size: NonZeroU64,
    padded_size: NonZeroU64,
}

impl BufferSize {
    pub fn new(requested_size: NonZeroU64) -> Self {
        let missing_padding = u64::from(requested_size) % 256; //FIXME: use BIND_BUFFER_ALIGNMENT
        let padded_size = NonZeroU64::new(u64::from(requested_size) + missing_padding).unwrap();
        return Self {
            requested_size,
            padded_size,
        };
    }
    pub fn requested_size(&self) -> NonZeroU64 {
        self.requested_size
    }
    pub fn padded_size(&self) -> NonZeroU64 {
        self.padded_size
    }
}
impl From<u32> for BufferSize {
    fn from(value: u32) -> Self {
        let v = u64::from(value);
        return Self::new(NonZeroU64::try_from(v).unwrap());
    }
}
