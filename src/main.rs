pub mod feature_extractor_pipeline;
pub mod util;
pub mod wgsl;

use feature_extractor_pipeline::{kernel::gaussian_blur::GaussianBlur, pipeline::FeatureExtractorPipeline};
use pollster::FutureExt;
use rand::RngCore;
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
        // flags: wgpu::InstanceFlags::debugging(),
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

    let mut rng = rand::rng();

    const WIDTH: usize = 1024;
    const HEIGHT: usize = 1024;
    const NUM_CHANNELS: usize = 4; //FIXME
    const NUM_IMAGES: usize = 1;
    const KERNEL_SIDE: usize = 41;

    let images: Vec<image::ImageBuffer<image::Rgba<u8>, Vec<u8>>> = (0..NUM_IMAGES)
        .map(|_| {
            let mut bytes: Vec<u8> = vec![0; WIDTH * HEIGHT * NUM_CHANNELS];
            rng.fill_bytes(&mut bytes);
            image::ImageBuffer::from_raw(WIDTH as u32, HEIGHT as u32, bytes).unwrap()
        })
        .collect();

    // let img = image::io::Reader::open("./big.png").unwrap().decode().unwrap();
    // let img1_rgba8 = img.to_rgba8();
    
    // let img = image::io::Reader::open("./big2.png").unwrap().decode().unwrap();
    // let img2_rgba8 = img.to_rgba8();

    // let img = image::io::Reader::open("./big3.png").unwrap().decode().unwrap();
    // let img3_rgba8 = img.to_rgba8();

    // let images = [
    //     img1_rgba8,
    //     img2_rgba8,
    //     img3_rgba8,
    // ];

    let dims = images[0].dimensions();
    println!("Image has these dimensions:{:?} ", dims);

    let kernels = vec![
        GaussianBlur::<KERNEL_SIDE>{ sigma: 1.0 },
        GaussianBlur::<KERNEL_SIDE>{ sigma: 3.0 },
        GaussianBlur::<KERNEL_SIDE>{ sigma: 5.0 },
        GaussianBlur::<KERNEL_SIDE>{ sigma: 10.0 },
        // GaussianBlur::<KERNEL_SIDE>{ sigma: 4.84089642 },
    ];

    let pipeline = FeatureExtractorPipeline::new(
        &device,
        WorkgroupSize{
            x: 16,
            y: 16,
            z: 1,
        },
        kernels.clone(),
        images[0].extent(),
    );


    let start = std::time::Instant::now();
    for (img_idx, input_img) in images.iter().enumerate(){
        std::thread::scope(|s|{
            s.spawn(||{
                // let input_img = &images[t];
                let features: Vec<[f32; 4]> = {
                    let start = std::time::Instant::now();
                    let features = pipeline.process(&device, &queue, input_img).unwrap();
                    let x = input_img.width();
                    let y = input_img.height();
                    let num_kernels = kernels.len();
                    let time_diff = std::time::Instant::now() - start;
                    println!(
                        "Convolved a {x}x{y} (3 channels) image with {num_kernels} kernel(s) of {KERNEL_SIDE}x{KERNEL_SIDE} in {time_diff:?}"
                    );
                    features
                };

                for (kern_idx, _kernel) in kernels.iter().enumerate() {
                    let width = input_img.width();
                    let height = input_img.height();
                    let num_pixels = (width * height) as usize;
                    let img_slice = &features[(kern_idx * num_pixels)..(kern_idx + 1) * num_pixels];
                    let img_slice_f32: &[f32] = bytemuck::cast_slice(img_slice);
                    assert!(img_slice_f32.len() == num_pixels * 4);

                    
                    let rgba_u8: Vec<u8> = {
                        let start = std::time::Instant::now();
                        let converted = img_slice_f32.iter()
                            .map(|channel| (*channel * 255.0) as u8)
                            .collect::<Vec<_>>();
                        eprintln!("Converted img to u8 in {:?}", std::time::Instant::now() - start);
                        converted
                    };
                    let parsed = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, rgba_u8)
                        .expect("Could not parse rgba u8 image!!!");
                    // parsed.save(format!("blurred_t{img_idx:?}_{kern_idx}.png")).unwrap();
                    eprintln!("blurred_t{img_idx:?}_{kern_idx}.png {}", parsed.width());
                }
            });
        });
    }
    eprintln!("Took {:?} to run all {} images",std::time::Instant::now() - start, images.len());

}
