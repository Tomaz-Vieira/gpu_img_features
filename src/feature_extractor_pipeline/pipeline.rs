use std::fmt::Write;
use std::time::Instant;

use nalgebra::Vector4;
use wgpu::{BindGroupLayoutDescriptor, ShaderModuleDescriptor};

use crate::decision_tree::RandomForest;
use crate::util::{timeit, Binding, Extent3dExt, Group, ImageBufferExt, WorkgroupSize};

use super::input_texture::InputTextureSlot;
use super::output_buffer::{KernelsInBuffSlot, OutputBufferSlot};
use super::kernel::gaussian_blur::GaussianBlur;

pub struct FeatureExtractorPipeline<const KSIDE: usize> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    input_texture_slot: InputTextureSlot,
    kernels_bind_group: wgpu::BindGroup,
    output_buffer_slot: OutputBufferSlot<Vector4<f32>, KSIDE>,
    workgroup_size: WorkgroupSize,
    pipeline: wgpu::ComputePipeline,
}
impl<const KSIDE: usize> FeatureExtractorPipeline<KSIDE> {
    pub const INOUT_GROUP: Group = Group(0);
    pub const KERNELS_GROUP: Group = Group(1);

    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        workgroup_size: WorkgroupSize,
        kernels: Vec<GaussianBlur<KSIDE>>,
        forest: &RandomForest,
        img_extent: wgpu::Extent3d,
    ) -> Self {
        assert!(forest.highest_feature_idx() + 1 == kernels.len() * 3);
        let input_texture_view_dimension = match img_extent.depth_or_array_layers {
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
        let output_buffer_slot = OutputBufferSlot::<Vector4<f32>, KSIDE>{
            name: "output_features_buf".into(),
            group: Self::INOUT_GROUP,
            binding: Binding(1),
            img_extent,
            marker: std::marker::PhantomData,
        };
        let kernel_buffer_slot: KernelsInBuffSlot<KSIDE> = KernelsInBuffSlot::new(
            &device,
            "in_buf_kernels".to_owned(),
            Self::KERNELS_GROUP,
            Binding(0),
            kernels,
        );
        let output_name = &output_buffer_slot.name;
        let mut code = String::with_capacity(1024 * 1024);
        write!(&mut code, "
            {input_texture_slot}
            {output_buffer_slot}
            {kernel_buffer_slot}

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

        kernel_buffer_slot.write_wgsl_feature_calcs(&mut code).unwrap();

        forest.write_wgsl(&mut code).unwrap();

        let output_indexing = output_buffer_slot.wgsl_indexing_from_kernIdx_xyzOffset("global_id");
        write!(&mut code, "
            if class_0_score > class_1_score {{
                {output_name}{output_indexing} = vec4(255.0, 0.0, 0.0, 1.0); //FIXME! hardcoded alpha channel!!
            }} else {{
                {output_name}{output_indexing} = vec4(0.0, 255.0, 0.0, 1.0); //FIXME! hardcoded alpha channel!!
            }}"
        ).unwrap();

        write!(&mut code, "
            }} //closes extract_features fn
        ").unwrap();

        // eprintln!("Shader code:");
        for (line_idx, line) in code.lines().enumerate(){
            eprintln!("{:03} {line}", line_idx + 1);
        }

        let shader_module = timeit("compiling compute shader", ||{
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("feature_extractor_comp_shader"),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            })
        });

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
            entries: &[kernel_buffer_slot.to_bind_group_layout_entry()],
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
                entries: &[kernel_buffer_slot.to_bind_group_entry()],
            }),
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("my_pipeline"),
                entry_point: Some("extract_features"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
            device,
            queue,
        }
    }
    pub fn process(
        &self,
        img: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ) -> Result<Vec<[f32; 4]>, String> {
        {
            let expected_extent = self.output_buffer_slot.img_extent;
            let found_extent = img.extent();
            if found_extent != expected_extent {
                return Err(format!(
                    "Expected image with extent {expected_extent:?}, found {found_extent:?}",
                ))
            }
        }
        let input_texture = self.input_texture_slot.create_texture(&self.device, img.extent());
        input_texture.write_texture(&self.queue, img);

        //FIXME: hardcoding vec4, expecting it to always be a rgba image
        let output_buffer = self.output_buffer_slot.create_output_buffer(&self.device);
        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_buffer"),
            mapped_at_creation: false,
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });

        let inout_binding_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        let mut command_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
            let (x, y, z) = img.extent().num_dispatch_work_groups(&self.workgroup_size);
            println!("Dispatch workgrounps: x: {x} y: {y} z: {z}");
            compute_pass.dispatch_workgroups(x, y, z);
            // drop(compute_pass); //FIXME?: forcing pass to end here, I hope
        }

        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &read_buffer,
            0,
            output_buffer.size(),
        );

        self.queue.submit(Some(command_encoder.finish()));

        let read_buffer_slice = read_buffer.slice(..);

        let start_of_map_async = Instant::now();
        read_buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
            let time_until_mapping = Instant::now() - start_of_map_async;
            eprintln!("Mapping the output buffer to CPU memory space (so, waiting for compute to finish?) took {time_until_mapping:?}");
            //FIXME: check result and, if successul, set a condvar or something
            // println!("buffer is mapped!");
        });

        self.device.poll(wgpu::PollType::Wait).unwrap();

        // Gets contents of buffer

        let out: Vec<[f32; 4]> = timeit("copying data back from GPU", ||{
            let read_buffer_view = read_buffer_slice.get_mapped_range();
            let data_cpy: Vec<[f32; 4]> = bytemuck::cast_slice::<_, _>(&read_buffer_view).to_owned();
            data_cpy
        });

        read_buffer.unmap();
        Ok(out)
    }
}
