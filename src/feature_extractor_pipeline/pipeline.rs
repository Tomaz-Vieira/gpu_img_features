use std::num::NonZeroU8;
use std::fmt::Write;

use encase::nalgebra::Vector3;
use wgpu::{BindGroupDescriptor, BindGroupLayoutDescriptor, ShaderModuleDescriptor};

use crate::{
    feature_extractor_pipeline::reader_buffer::ReaderBuffer,
    util::{Binding, Extent3dExt, Group, ImageBufferExt, NumChannels, WorkgroupSize},
};

use super::{input_texture::InputTextureSlot, kernel::gaussian_blur::GaussianBlur, output_buffer::OutputBufferSlot};

pub struct FeatureExtractorPipeline {
    input_texture_slot: InputTextureSlot,
    output_buffer_slot: OutputBufferSlot,
    workgroup_size: WorkgroupSize,
    slots_bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
impl FeatureExtractorPipeline {
    pub fn new(
        device: &wgpu::Device,
        tile_size: wgpu::Extent3d,
        workgroup_size: WorkgroupSize,
        kernels: &[GaussianBlur],
        radius: NonZeroU8,
    ) -> Self {
        let radius: i32 = u8::from(radius) as i32;
        let input_texture_view_dimension = match tile_size.depth_or_array_layers {
            1 => wgpu::TextureViewDimension::D2,
            _ => wgpu::TextureViewDimension::D3,
        };

        let input_texture_slot = InputTextureSlot::new(
            "input_image".into(),
            Group(0),
            Binding(0),
            wgpu::TextureSampleType::Float { filterable: false },
            input_texture_view_dimension,
        );
        let output_buffer_slot = OutputBufferSlot{
            name: "output_features".into(),
            group: Group(0),
            binding: Binding(1),
        };

        let input_name = input_texture_slot.name();
        let output_name = output_buffer_slot.name();
        let mut code = String::with_capacity(1024 * 1024);
        write!(&mut code, "
            {input_texture_slot}
            {output_buffer_slot}

            @compute {workgroup_size}
            fn extract_features(
                @builtin(global_invocation_id) global_id : vec3<u32>,
            ) {{
                let dimensions = textureDimensions(input_image);
                let texture_upper_limit = vec2<i32>(dimensions.xy) - vec2<i32>(1, 1);
                let current_coords = vec2<i32>(global_id.xy);
        ").unwrap();

        for (idx, _kernel) in kernels.iter().enumerate(){
            write!(&mut code, "
                var acc_{idx}: vec3<f32> = vec3(0.0, 0.0, 0.0);
            ").unwrap();
        }
    
        write!(&mut code, "
                if(global_id.x >= dimensions.x || global_id.y >= dimensions.y) {{
                    return;
                }}
        ").unwrap();

        let mut kernel_sum: f32 = 0.0;

        for y in -radius..=radius{
            for x in -radius..=radius{
                write!(&mut code, "
                    {{
                    let offset = vec2<i32>({x}, {y});
                       let sample_coords: vec2<i32> = vec2<i32>(
                           clamp(current_coords.x + offset.x, 0, texture_upper_limit.x),
                           clamp(current_coords.y + offset.y, 0, texture_upper_limit.y),
                       );
                       let sample = textureLoad(input_image, sample_coords, 0).xyz;
                ").unwrap();
                for (k_idx, kernel) in kernels.iter().enumerate(){
                    let kernel_value = kernel.kernel_at(Vector3::new(x, y, 0));
                    kernel_sum += kernel_value;
                    write!(&mut code, "
                        acc_{k_idx} += sample * {kernel_value};
                    ").unwrap();
                } 
                write!(&mut code, "
                    }}
                ").unwrap();
            }
        }

        write!(&mut code, "
                {output_name}[global_id.y * dimensions.x + global_id.x] = vec4<f32>(acc_0, 1.0); //FIXME!!!!!!! acc_<IDX>!!
            }} //closes extract_features fn
        ").unwrap();

        println!("{}", code);
        println!("With this sum: {kernel_sum}");

        let inst = std::time::Instant::now();
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("feature_extractor_comp_shader"),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let dur = std::time::Instant::now() - inst;
        println!("How much time just to compile? {dur:?}");

        let slots_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("feature_Extractor_slots_group_layout"),
            entries: &[
                input_texture_slot.to_bind_group_layout_entry(),
                output_buffer_slot.to_bind_group_layout_entry(),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("feature_extractor_pipeline_layout"),
            push_constant_ranges: &[],
            bind_group_layouts: &[&slots_bind_group_layout],
        });

        Self {
            input_texture_slot,
            output_buffer_slot,
            workgroup_size,
            slots_bind_group_layout,
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("my_pipeline"),
                entry_point: Some("extract_features"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        }
    }
    pub fn process(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ) /* -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> */
    {
        let input_texture = self.input_texture_slot.create_texture(device, image.extent());
        input_texture.write_texture(queue, image);

        //FIXME: right now we are hardcoding that the output has 4 channels
        let num_channels = NumChannels(NonZeroU8::new(4).unwrap());
        let output_buffer_size = image.extent().to_buffer_size(num_channels);
        println!("Output buffer will have {output_buffer_size:?} bytes");
        let output_buffer = self.output_buffer_slot.create_buffer(device, output_buffer_size);
        let read_buffer = ReaderBuffer::new("my_read_buffer", &output_buffer, device);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("binding_for_filter_pipeline"),
            layout: &self.slots_bind_group_layout,
            entries: &[input_texture.to_bind_group_entry(), output_buffer.to_bind_group_entry()],
        });

        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("my_encoder_for_filtering"),
        });

        {
            let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("my_compute_pass"),
                timestamp_writes: None, 
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let (x, y, z) = image.extent().num_dispatch_work_groups(&self.workgroup_size);
            compute_pass.dispatch_workgroups(x, y, z);
            // drop(compute_pass); //FIXME?: forcing pass to end here, I hope
        }

        read_buffer.encode_copy(&mut command_encoder);

        queue.submit(Some(command_encoder.finish()));

        read_buffer.map_async(wgpu::MapMode::Read, move |_| {
            //FIXME: check result and, if successul, set a condvar or something
            println!("buffer is mapped!");
        });

        device.poll(wgpu::Maintain::Wait);

        // Gets contents of buffer
        let data = read_buffer.get_mapped_range();
        println!("Copying data from buffer into CPU...........");
        let data_cpy: Vec<u8> = bytemuck::cast_slice::<_, f32>(&data)
            .iter()
            .map(|channel| (channel * 255.0) as u8)
            .collect();

        if let Some(out_img) =
            image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(image.width(), image.height(), data_cpy)
        {
            out_img.save("blurred.png").unwrap();
        } else {
            println!("Could not make image from result!")
        };

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        read_buffer.unmap(); // Unmaps buffer from memory
                             // If you are familiar with C++ these 2 lines can be thought of similarly to:
                             //   delete myPointer;
                             //   myPointer = NULL;
                             // It effectively frees the memory

        // let mapped_range = buffer_slice.get_mapped_range();

        // {
        //     if let Some(out_img) = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(image.width(), image.height(), mapped_range){
        //         out_img.save("blurred.png").unwrap();
        //     }else{
        //         println!("Could not make image from result!")
        //     };
        // }
    }
}
