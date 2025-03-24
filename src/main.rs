pub mod feature_extractor_pipeline;
pub mod util;
pub mod wgsl;

use feature_extractor_pipeline::pipeline::FeatureExtractorPipeline;
use pollster::FutureExt;
use util::{ImageBufferExt, WorkgroupSize};


fn main() {
    // wgpu uses `log` for all of our logging, so we initialize a logger with the `env_logger` crate.
    //
    // To change the log level, set the `RUST_LOG` environment variable. See the `env_logger`
    // documentation for more information.
    env_logger::init();

    // We first initialize an wgpu `Instance`, which contains any "global" state wgpu needs.
    //
    // This is what loads the vulkan/dx12/metal/opengl libraries.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor{
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    // We then create an `Adapter` which represents a physical gpu in the system. It allows
    // us to query information about it and create a `Device` from it.
    //
    // This function is asynchronous in WebGPU, so request_adapter returns a future. On native/webgl
    // the future resolves immediately, so we can block on it without harm.
    let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions{
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }
        )
        .block_on()
        .expect("Failed to create adapter");

    // Print out some basic information about the adapter.
    println!("Running on Adapter: {:#?}", adapter.get_info());

    // Check to see if the adapter supports compute shaders. While WebGPU guarantees support for
    // compute shaders, wgpu supports a wider range of devices through the use of "downlevel" devices.
    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("Adapter does not support compute shaders");
    }

    // We then create a `Device` and a `Queue` from the `Adapter`.
    //
    // The `Device` is used to create and manage GPU resources.
    // The `Queue` is a queue used to submit work for the GPU to process.
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    )
    .block_on()
    .expect("Failed to create device");

    let img = image::io::Reader::open("./c_cells_1.png").unwrap().decode().unwrap();
    let img_rgba8 = img.to_rgba8();

    let dims = img_rgba8.dimensions();
    println!("Image has these dimensions:{:?} ", dims);

    let pipeline = FeatureExtractorPipeline::new(
        &device,
        img_rgba8.extent(),
        WorkgroupSize{
            x: 16,
            y: 16,
            z: 1,
        }
    );

    pipeline.process(&device, &queue, &img_rgba8);
}
