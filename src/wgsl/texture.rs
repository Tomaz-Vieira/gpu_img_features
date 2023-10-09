use std::marker::PhantomData;

use encase::nalgebra::{Vector2, Vector4};

use super::{expression::Expression, IVec2};
use crate::util::{Binding, Group};

#[derive(Clone)]
pub struct Texture2dDecl {
    pub group: Group,
    pub binding: Binding,
    pub name: String,
}
impl Texture2dDecl {
    pub fn wgsl(&self) -> String {
        let Self {
            group, binding, name, ..
        } = &self;
        format!("@group({group}) @binding({binding}) var {name} : texture_2d<f32>;")
    }
    #[allow(non_snake_case)]
    pub fn textureLoad(
        &self,
        coords: impl Into<Expression<IVec2>>,
        level: impl Into<Expression<u32>>,
    ) -> Expression<Vector4<f32>> {
        let texture_name = &self.name;
        let coords_expr = coords.into();
        let level_expr = level.into();
        return Expression(
            format!("textureLoad({texture_name}, {coords_expr}, {level_expr})"),
            PhantomData,
        );
    }
    #[allow(non_snake_case)]
    pub fn textureDimensions(&self) -> Expression<Vector2<u32>> {
        let texture_name = &self.name;
        Expression(format!("textureDimensions({texture_name})"), PhantomData)
    }
}
