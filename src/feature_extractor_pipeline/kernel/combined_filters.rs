use crate::feature_extractor_pipeline::kernel::CenterOffset;
use crate::wgsl::buffer::OutputBufferDecl;
use crate::wgsl::compute_shader_source::ComputeShaderSource;
use crate::wgsl::declaration::FunctionVarDecl;
use crate::wgsl::expression::Expression;
use crate::wgsl::statement::{Assignment, BufferWrite, Statement};
use crate::wgsl::texture::Texture2dDecl;
use crate::wgsl::{FVec4, IVec2};

use super::gaussian_blur::GaussianBlur;

// pub struct CombinedFilterValue<const GAUSS_BLUR_COUNT: u8>{
//     gaussian_blur: [glam::Vec4; GAUSS_BLUR_COUNT]
// }
// impl<const GAUSS_BLUR_COUNT: u8> NamableType for CombinedFilterValue<GAUSS_BLUR_COUNT>{
//     fn wgsl() -> String { "CombinedFilterValue".into() }
// }

pub struct CombinedFilters {
    pub kernels: Vec<GaussianBlur>,
    pub input_texture: Texture2dDecl,
    pub output_buffer: OutputBufferDecl<FVec4>,
}
impl CombinedFilters {
    pub fn produce_shader(&self) -> ComputeShaderSource {
        let Self {
            kernels,
            input_texture,
            output_buffer,
        } = self;
        let mut statements = Vec::<Box<dyn Statement>>::new();

        let accumulator_vars: Vec<FunctionVarDecl<FVec4>> = kernels
            .iter()
            .map(|kernel| {
                let acc_var = kernel.accumulator_var();
                statements.push(Box::new(acc_var.clone()));
                acc_var
            })
            .collect();

        let sample_var = FunctionVarDecl::<FVec4> {
            name: "sample".into(),
            initializer: None,
        };
        statements.push(Box::new(sample_var.clone()));

        let max_radius: i32 = self
            .kernels
            .iter()
            .map(|k| k.max_offset)
            .max()
            .unwrap()
            .try_into()
            .unwrap();

        //FIXME: add to statements
        let texture_dimension_decl = FunctionVarDecl {
            name: "dimensions".into(),
            initializer: Some(self.input_texture.textureDimensions()),
        };
        statements.push(Box::new(texture_dimension_decl.clone()));

        let texture_dims_max_x = FunctionVarDecl {
            name: "texture_dims_max_x".into(),
            initializer: Some(texture_dimension_decl.x() - 1),
        };
        statements.push(Box::new(texture_dims_max_x.clone()));

        let texture_dims_max_y = FunctionVarDecl {
            name: "texture_dims_max_y".into(),
            initializer: Some(texture_dimension_decl.y() - 1),
        };
        statements.push(Box::new(texture_dims_max_y.clone()));

        for y in -max_radius..=max_radius {
            for x in -max_radius..=max_radius {
                let center_offset = CenterOffset { x, y, z: 0 };
                let x = Expression::from(x);
                let y = Expression::from(y);
                //FIXME: use sampler to clamp instead of clmaping in code

                statements.push(Box::new(Assignment {
                    assignee: sample_var.clone(),
                    value: self.input_texture.textureLoad(
                        Expression::<IVec2>::construct(
                            x.clamped(0, &Expression::from(&texture_dims_max_x)),
                            y.clamped(0, &Expression::from(&texture_dims_max_y)),
                        ),
                        0,
                    ),
                }));

                for (kernel_index, kernel) in self.kernels.iter().enumerate() {
                    if let Some(shader_line) = kernel.accumulate(&accumulator_vars[kernel_index], &center_offset) {
                        statements.push(Box::new(shader_line));
                    }
                }
            }
        }

        let global_invocation_id = ComputeShaderSource::global_invocation_id();
        let buffer_output_index = (global_invocation_id.y() * texture_dimension_decl.x()) + global_invocation_id.x();
        accumulator_vars.iter().enumerate().for_each(|(output_index, acc_var)| {
            statements.push(Box::new(BufferWrite {
                buffer: output_buffer.clone(),
                index: buffer_output_index.clone(),
                value: acc_var.into(),
            }));
        });

        return ComputeShaderSource {
            workgroup_size: 16,
            main_fn_name: "main".into(),
            input_textures: vec![input_texture.clone()],
            output_buffers: vec![Box::new(output_buffer.clone())],
            statements,
        };
    }
}

#[test]
fn produce_shader() {
    use crate::util::{Binding, Group};
    use std::marker::PhantomData;

    let shader_code = CombinedFilters {
        kernels: vec![
            GaussianBlur {
                sigma: 0.3,
                max_offset: 3,
            },
            GaussianBlur {
                sigma: 0.7,
                max_offset: 3,
            },
        ],
        input_texture: Texture2dDecl {
            group: Group(0),
            binding: Binding(0),
            name: "my_texture".into(),
        },
        output_buffer: OutputBufferDecl::<FVec4> {
            group: Group(0),
            binding: Binding(1),
            name: "my_output".into(),
            marker: PhantomData,
        },
    }
    .produce_shader()
    .wgsl();
    println!("Shader code:\n{shader_code}");
}
