use std::marker::PhantomData;

use nalgebra::{Vector2, Vector4};

use crate::util::{Group, Binding};
use super::{expression::Expression, IVec2};

#[derive(Clone)]
pub struct Texture2dDecl{
    pub group: Group,
    pub binding: Binding,
    pub name: String,
}
impl Texture2dDecl{
    pub fn wgsl(&self) -> String{
        let Self{group, binding, name, sample_type, ..} = &self;
        format!(
            "@group({group}) @binding({binding}) var {name} : texture_2d<f32>;"
        )
    }
    #[allow(non_snake_case)]
    pub fn textureLoad(
        &self,
        coords: Expression<IVec2>,
        level: Expression<u32>
    ) -> Expression<Vector4<f32>>{
        let texture_name = &self.name;
        return Expression(format!("textureLoad({texture_name}, {coords}, {level})"), PhantomData)
    }
    #[allow(non_snake_case)]
    pub fn textureDimensions(&self) -> Expression<Vector2<u32>>{
        let texture_name = &self.name;
        Expression(format!("textureDimensions({texture_name})"), PhantomData)
    }
}
