use encase::nalgebra::{Vector2, Vector3, Vector4};
use paste::paste;

pub mod buffer;
pub mod compute_shader_source;
pub mod declaration;
pub mod expression;
pub mod statement;
pub mod texture;

pub trait Wgsl {
    fn wgsl(&self) -> String;
}
pub trait ShaderTypeExt: encase::ShaderType {
    fn wgsl_type_name() -> String;
}

pub type FVec4 = Vector4<f32>;
pub type FVec3 = Vector3<f32>;
pub type UVec4 = Vector4<u32>;
pub type UVec3 = Vector3<u32>;
pub type UVec2 = Vector2<u32>;
pub type IVec2 = Vector2<i32>;

macro_rules! impl_ShaderTypeExt_for_primitive {($primitive:ty) => {
    impl ShaderTypeExt for $primitive {
        fn wgsl_type_name() -> String {
            stringify!($primitive).into()
        }
    }
};}
impl_ShaderTypeExt_for_primitive!(f32);
impl_ShaderTypeExt_for_primitive!(u32);
impl_ShaderTypeExt_for_primitive!(i32);

macro_rules! impl_ShaderTypeExt_for_vec {($length:literal, $item_type:ty) => { paste! {
    impl ShaderTypeExt for [<Vector $length>]<$item_type>{
        fn wgsl_type_name() -> String {
            format!("vec{}<{}>", stringify!($length), $item_type::wgsl_type_name()).into()
        }
    }
}};}
impl_ShaderTypeExt_for_vec!(4, f32);
impl_ShaderTypeExt_for_vec!(3, f32);
impl_ShaderTypeExt_for_vec!(2, f32);

impl_ShaderTypeExt_for_vec!(4, i32);
impl_ShaderTypeExt_for_vec!(3, i32);
impl_ShaderTypeExt_for_vec!(2, i32);

impl_ShaderTypeExt_for_vec!(4, u32);
impl_ShaderTypeExt_for_vec!(3, u32);
impl_ShaderTypeExt_for_vec!(2, u32);
