#![feature(adt_const_params)]

pub mod feature_extractor_pipeline;
pub mod util;
pub mod wgsl;

use wgpu::{
    Instance, InstanceDescriptor, Backends, RequestAdapterOptions,
};
use pollster::FutureExt;

use crate::{feature_extractor_pipeline::pipeline::FeatureExtractorPipeline, util::{ImageBufferExt, WorkgroupSize, Arg}};


fn main() {
    let instance = Instance::new(InstanceDescriptor{
        backends: Backends::VULKAN,
        dx12_shader_compiler: wgpu::Dx12Compiler::default(),
    });

    let adapter = instance.request_adapter(&RequestAdapterOptions{
        compatible_surface: None,
        force_fallback_adapter: false,
        power_preference: wgpu::PowerPreference::HighPerformance,
    }).block_on().unwrap();

    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .block_on().unwrap();


    let img = image::io::Reader::open("./c_cells_1.png").unwrap().decode().unwrap();
    let img_rgba8 = img.to_rgba8();

    let dims = img_rgba8.dimensions();
    println!("Image has these dimensions:{:?} ", dims);

    let pipeline = FeatureExtractorPipeline::new(
        &device,
        Arg::<"tile_size", _>(img_rgba8.extent()),
        WorkgroupSize{
            x: 16,
            y: 16,
            z: 1,
        }
    );


    pipeline.process(&device, &queue, &img_rgba8);
}
