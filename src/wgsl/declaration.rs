use std::marker::PhantomData;

use super::{expression::Expression, Wgsl, statement::Statement, ShaderTypeExt};
use encase::nalgebra::{Vector3, Vector2};

pub enum AddressSpace{
    Storage,
    Function,
}
impl AddressSpace{
    pub fn wgsl(&self) -> String{
        match self{
            Self::Storage => "storage".into(),
            Self::Function => "function".into(),
        }
    }
}

pub enum AccessMode{
    Read,
    Write,
    ReadWrite,
}
impl AccessMode{
    pub fn wgsl(&self) -> String{
        match  self{
            Self::Read => "read".into(),
            Self::Write => "write".into(),
            Self::ReadWrite => "read_write".into(),
        }
    }
}

#[derive(Clone)]
pub struct FunctionVarDecl<T: ShaderTypeExt>{
    pub name: String,
    pub initializer: Option<Expression<T>>,
}

impl<T: ShaderTypeExt> From<&FunctionVarDecl<T>> for Expression<T>{
    fn from(value: &FunctionVarDecl<T>) -> Self {
        Expression(value.name.clone(), PhantomData)
    }
}

impl<T: ShaderTypeExt> Wgsl for FunctionVarDecl<T>{
    fn wgsl(&self) -> String{
        let name = &self.name;
        let type_name = T::wgsl_type_name();
        let initializer_str = match &self.initializer{
            Some(init) => format!(" = {}", init.wgsl()),
            None => "".into(),
        };
        format!("var {name}: {type_name}{initializer_str};")
    }
}

impl<T: ShaderTypeExt> Statement for FunctionVarDecl<T>{}

macro_rules! impl_FunctionVarDecl_xy {
    ($item_type:ty) => {
        impl FunctionVarDecl<Vector2<$item_type>>{
            pub fn x(&self) ->Expression<$item_type>{
                Expression(format!("{}.x", Expression::from(self)), PhantomData)
            }
            pub fn y(&self) ->Expression<$item_type>{
                Expression(format!("{}.y", Expression::from(self)), PhantomData)
            }
        }
    };
}

macro_rules! impl_FunctionVarDecl_xyz {
    ($item_type:ty) => {
        impl FunctionVarDecl<Vector3<$item_type>>{
            pub fn x(&self) ->Expression<$item_type>{
                Expression(format!("{}.x", Expression::from(self)), PhantomData)
            }
            pub fn y(&self) ->Expression<$item_type>{
                Expression(format!("{}.y", Expression::from(self)), PhantomData)
            }
            pub fn z(&self) ->Expression<$item_type>{
                Expression(format!("{}.z", Expression::from(self)), PhantomData)
            }
        }
    };
}
impl_FunctionVarDecl_xyz!(f32);
impl_FunctionVarDecl_xyz!(u32);

impl_FunctionVarDecl_xy!(f32);
impl_FunctionVarDecl_xy!(u32);