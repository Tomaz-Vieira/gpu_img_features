#![feature(adt_const_params)]

pub mod feature_extractor_pipeline;
pub mod util;

use wgpu::{
    Instance, InstanceDescriptor, Backends, RequestAdapterOptions, PipelineLayoutDescriptor, BindGroupLayout,
};
use pollster::FutureExt;


fn main() {
    println!("Hello, world!");

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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label: Some("my_encoder")});
    let compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label: Some("my_compute_pass")});
    // compute_pass.set

}
