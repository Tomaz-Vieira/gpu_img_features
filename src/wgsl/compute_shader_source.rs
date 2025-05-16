use std::marker::PhantomData;

use nalgebra::Vector3;

use crate::util::WorkgroupSize;
use crate::wgsl::buffer::OutputBuffer;
use crate::wgsl::expression::Expression;
use crate::wgsl::statement::Statement;
use crate::wgsl::texture::Texture2dDecl;
use crate::wgsl::Wgsl;

pub struct ComputeShaderSource {
    pub workgroup_size: WorkgroupSize,
    pub main_fn_name: String,
    pub input_textures: Vec<Texture2dDecl>,
    pub output_buffers: Vec<Box<dyn OutputBuffer>>,
    pub statements: Vec<Box<dyn Statement>>,
}

impl ComputeShaderSource {
    pub fn wgsl(&self) -> String {
        let Self {
            workgroup_size,
            main_fn_name,
            ..
        } = &self;
        let input_textures_wgsl = self
            .input_textures
            .iter()
            .map(|tex| tex.wgsl())
            .collect::<Vec<_>>()
            .join("\n");
        let output_buffers_wgsl = self
            .output_buffers
            .iter()
            .map(|tex| tex.wgsl())
            .collect::<Vec<_>>()
            .join("\n");
        let global_invocation_id = Self::global_invocation_id().wgsl();
        let statements_wgsl = self
            .statements
            .iter()
            .map(|st| st.wgsl())
            .collect::<Vec<_>>()
            .join("\n                ");
        format!(
            "
            {input_textures_wgsl}

            {output_buffers_wgsl}

            @compute {workgroup_size}
            fn {main_fn_name}(
                @builtin(global_invocation_id) {global_invocation_id} : vec3<u32>,
            ){{
                {statements_wgsl}
            }}
        "
        )
    }
    pub fn global_invocation_id() -> Expression<Vector3<u32>> {
        return Expression("global_invocation_id".into(), PhantomData);
    }
}
