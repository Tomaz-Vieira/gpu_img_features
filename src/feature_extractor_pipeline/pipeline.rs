use wgpu::{ShaderModuleDescriptor, BindGroupLayoutDescriptor};

use super::texture_slots::{InputTextureSlot, OutpuTextureSlot, Group, Binding};

pub struct FeatureExtractorPipelineLayout{
    input_texture_slot: InputTextureSlot,
    output_texture_slot: OutpuTextureSlot,
    shader_module: wgpu::ShaderModule,
    slots_bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}
impl FeatureExtractorPipelineLayout{
    pub fn new(device: &wgpu::Device) -> Self{
        let input_texture_slot = InputTextureSlot::new(
            "input_image".into(),
            Group(0),
            Binding(0),
            wgpu::TextureSampleType::Float { filterable: false },
            wgpu::TextureViewDimension::D2,
        );
        let input_name = input_texture_slot.name();

        let output_texture_slot = OutpuTextureSlot::new(
            "output_features".into(),
            Group(0),
            Binding(1),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureViewDimension::D2,
        );
        let output_name = output_texture_slot.name();

        let shader_module = device.create_shader_module(ShaderModuleDescriptor{
            label: Some("feature_extractor_comp_shader"),
            source: wgpu::ShaderSource::Wgsl(format!("
                {input_texture_slot}
                {output_texture_slot}

                @compute @workgroup_size(16, 16)
                fn grayscale_main(
                    @builtin(global_invocation_id) global_id : vec3<u32>,
                ) {{
                    let dimensions = textureDimensions({input_name});
                    let coords = vec2<i32>(global_id.xy);

                    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {{
                        return;
                    }}

                    let color = textureLoad({input_name}, coords.xy, 0);
                    let gray = dot(vec3<f32>(0.299, 0.587, 0.114), color.rgb);

                    textureStore({output_name}, coords.xy, vec4<f32>(gray, gray, gray, color.a));
                }}
            ").into())
        });

        let slots_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("feature_Extractor_slots_group_layout"),
            entries: &[
                input_texture_slot.to_bind_group_layout_entry(), output_texture_slot.to_bind_group_layout_entry(),
            ]
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("feature_extractor_pipeline_layout"),
            push_constant_ranges: &[],
            bind_group_layouts: &[
                &slots_bind_group_layout,
            ],
        });

        Self{
            input_texture_slot, output_texture_slot, shader_module, slots_bind_group_layout, pipeline_layout
        }
    }
}