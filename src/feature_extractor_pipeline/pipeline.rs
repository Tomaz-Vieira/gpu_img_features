use std::num::NonZeroU8;
use std::fmt::Write;

use encase::nalgebra::Vector4;
use wgpu::{BindGroupLayoutDescriptor, ShaderModuleDescriptor};

use crate::util::{Binding, Extent3dExt, Group, ImageBufferExt, NumChannels, WorkgroupSize};

use super::{input_texture::InputTextureSlot, kernel::gaussian_blur::GaussianBlur, output_buffer::{KernelBufferSlot, OutputBufferSlot}};

pub struct FeatureExtractorPipeline {
    input_texture_slot: InputTextureSlot,
    kernels_bind_group: wgpu::BindGroup,
    output_buffer_slot: OutputBufferSlot<Vector4<u32>>,
    workgroup_size: WorkgroupSize,
    pipeline: wgpu::ComputePipeline,
}
impl FeatureExtractorPipeline {
    pub const INOUT_GROUP: Group = Group(0);
    pub const KERNELS_GROUP: Group = Group(1);

    pub fn new(
        device: &wgpu::Device,
        tile_size: wgpu::Extent3d,
        workgroup_size: WorkgroupSize,
        kernels: Vec<GaussianBlur>,
    ) -> Self {
        let input_texture_view_dimension = match tile_size.depth_or_array_layers {
            1 => wgpu::TextureViewDimension::D2,
            _ => wgpu::TextureViewDimension::D3,
        };

        let input_texture_slot = InputTextureSlot::new(
            "input_image".into(),
            Self::INOUT_GROUP,
            Binding(0),
            wgpu::TextureSampleType::Float { filterable: false },
            input_texture_view_dimension,
        );
        let output_buffer_slot = OutputBufferSlot::<Vector4<u32>>{
            name: "output_features".into(),
            group: Self::INOUT_GROUP,
            binding: Binding(1),
            marker: std::marker::PhantomData,
        };
        let kernel_buffer_slots: Vec<KernelBufferSlot<f32>> = kernels.into_iter().enumerate()
            .map(|(k_idx, kernel)| KernelBufferSlot::new(
                device,
                format!("kernel_{k_idx}"),
                Self::KERNELS_GROUP,
                Binding(k_idx as u32),
                kernel,
            ))
            .collect();
        let kernel_buffer_slots_decls = kernel_buffer_slots.iter()
            .map(|kbs| kbs.to_string())
            .collect::<Vec<String>>()
            .join("\n            ");

        // let input_name = input_texture_slot.name();
        let output_name = &output_buffer_slot.name;
        let mut code = String::with_capacity(1024 * 1024);
        write!(&mut code, "
            {input_texture_slot}
            {output_buffer_slot}
            {kernel_buffer_slots_decls}

            @compute {workgroup_size}
            fn extract_features(
                @builtin(global_invocation_id) global_id : vec3<u32>,
            ) {{
                let dimensions = textureDimensions(input_image);
                let texture_upper_limit = vec2<i32>(dimensions.xy) - vec2<i32>(1, 1);
                let current_coords = vec2<i32>(global_id.xy);

                if(global_id.x >= dimensions.x || global_id.y >= dimensions.y) {{
                    return;
                }}
        ").unwrap();

        for (k_idx, _kernel_slot) in kernel_buffer_slots.iter().enumerate(){
            write!(&mut code, "
                var acc_{k_idx}: vec3<f32> = vec3(0.0, 0.0, 0.0);
            ").unwrap();
        }

        let radius = kernel_buffer_slots[0].kernel().radius; //FIXME! assumes all kernels same size
        write!(&mut code, "
                for (var y=-{radius}; y<={radius}; y++){{
                    for (var x=-{radius}; x<={radius}; x++){{
                        let offset = vec2<i32>(x, y);
                        let sample_coords: vec2<i32> = vec2<i32>(
                            clamp(current_coords.x + offset.x, 0, texture_upper_limit.x),
                            clamp(current_coords.y + offset.y, 0, texture_upper_limit.y),
                        );
                        let sample = textureLoad(input_image, sample_coords, 0).xyz;
        ").unwrap();

        for (k_idx, kernel_slot) in kernel_buffer_slots.iter().enumerate(){
            let kernel_value_expr = kernel_slot.wgsl_kernel_value_at_center_offset("offset");
            write!(&mut code, "
                        acc_{k_idx} += sample * {kernel_value_expr};
            ").unwrap();
        }

        write!(&mut code, "
                    }} // close for x
                }} // close for y

                {output_name}[global_id.y * dimensions.x + global_id.x] = vec4<u32>(vec3<u32>(acc_0 * 255.0), 255); //FIXME!!!!!!! acc_<IDX>!!
            }} //closes extract_features fn
        ").unwrap();

        println!("Shader code:\n{code}");
        // panic!("check code");

        let shader_module = {
            let inst = std::time::Instant::now();
            let shader_module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("feature_extractor_comp_shader"),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
            println!("How much time just to compile? {:?}", std::time::Instant::now() - inst);
            shader_module
        };

        // ------------------ Layout --------------------
        let inout_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("inout_group_layout"),
            entries: &[
                input_texture_slot.to_bind_group_layout_entry(),
                output_buffer_slot.to_bind_group_layout_entry(),
            ],
        });

        let kernels_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("kernels_group_layout"),
            entries: &kernel_buffer_slots.iter()
                .map(|kbs| kbs.to_bind_group_layout_entry())
                .collect::<Vec<_>>()
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("feature_extractor_pipeline_layout"),
            push_constant_ranges: &[],
            bind_group_layouts: &[
                &inout_bind_group_layout,
                &kernels_bind_group_layout,
            ],
        });
        // ------------------ END Layout --------------------

        Self {
            input_texture_slot,
            output_buffer_slot,
            workgroup_size,
            kernels_bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor{
                label: Some("kernels_unifors_group"),
                layout: &kernels_bind_group_layout,
                entries: &kernel_buffer_slots.iter()
                    .map(|kbs| kbs.to_bind_group_entry())
                    .collect::<Vec<_>>()
            }),
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
    ) -> Result<image::ImageBuffer<image::Rgba<u8>, Vec<u8>>, String> {
        let input_texture = self.input_texture_slot.create_texture(device, image.extent());
        {
            let start = std::time::Instant::now();
            input_texture.write_texture(queue, image);
            println!("Uploaded texture in {:?}", std::time::Instant::now() - start);
        }

        //FIXME: right now we are hardcoding that the output has 4 channels
        let num_channels = NumChannels(NonZeroU8::new(4).unwrap());
        let output_buffer_size = image.extent().to_buffer_size(num_channels);
        println!("Output buffer will have {output_buffer_size:?} bytes");
        let output_buffer = self.output_buffer_slot.create_buffer(device, output_buffer_size);
        let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_buffer"),
            mapped_at_creation: false,
            size: output_buffer_size.into(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });

        let inout_binding_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binding_for_filter_pipeline"),
            layout: &self.pipeline.get_bind_group_layout(Self::INOUT_GROUP.into()),
            entries: &[
                input_texture.to_bind_group_entry(),
                wgpu::BindGroupEntry{
                    binding: self.output_buffer_slot.binding.into(),
                    resource: output_buffer.as_entire_binding(),
                },
            ],
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
            compute_pass.set_bind_group(Self::INOUT_GROUP.into(), &inout_binding_group, &[]);
            compute_pass.set_bind_group(Self::KERNELS_GROUP.into(), &self.kernels_bind_group, &[]);
            let (x, y, z) = image.extent().num_dispatch_work_groups(&self.workgroup_size);
            println!("Dispatch workgrounps: x: {x} y: {y} z: {z}");
            compute_pass.dispatch_workgroups(x, y, z);
            // drop(compute_pass); //FIXME?: forcing pass to end here, I hope
        }

        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &read_buffer,
            0,
            output_buffer_size.into(),
        );

        queue.submit(Some(command_encoder.finish()));

        let read_buffer_slice = read_buffer.slice(..);
        read_buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
            //FIXME: check result and, if successul, set a condvar or something
            println!("buffer is mapped!");
        });

        device.poll(wgpu::Maintain::Wait);

        // Gets contents of buffer

        let out = {
            let read_buffer_view = read_buffer_slice.get_mapped_range();
            let data_cpy: Vec<u8> = bytemuck::cast_slice::<_, u32>(&read_buffer_view)
                .iter()
                .map(|channel| *channel as u8)
                .collect();

            match image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(image.width(), image.height(), data_cpy) {
                Some(img) => Ok(img),
                None =>Err(format!("Copuld not make image form result!"))
            }
        };

        read_buffer.unmap();
        out
    }
}
