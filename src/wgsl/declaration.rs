use std::marker::PhantomData;

use super::{expression::Expression, statement::Statement, ShaderTypeExt, Wgsl};
use nalgebra::{Vector2, Vector3};

pub enum AddressSpace {
    Storage,
    Function,
}
impl AddressSpace {
    pub fn wgsl(&self) -> String {
        match self {
            Self::Storage => "storage".into(),
            Self::Function => "function".into(),
        }
    }
}

pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}
impl AccessMode {
    pub fn wgsl(&self) -> String {
        match self {
            Self::Read => "read".into(),
            Self::Write => "write".into(),
            Self::ReadWrite => "read_write".into(),
        }
    }
}

pub struct LocalVarDecl<T: ShaderTypeExt> {
    pub name: String,
    pub initializer: Option<Expression<T>>,
}

impl<T: ShaderTypeExt> LocalVarDecl<T>{
    pub fn as_expr(&self) -> Expression<T>{
        Expression::from(self)
    }
}

impl<T: ShaderTypeExt> Clone for LocalVarDecl<T>{
    fn clone(&self) -> Self {
        Self{name: self.name.clone(), initializer: self.initializer.clone()}
    }
}

impl<T: ShaderTypeExt> From<&LocalVarDecl<T>> for Expression<T> {
    fn from(value: &LocalVarDecl<T>) -> Self {
        Expression(value.name.clone(), PhantomData)
    }
}

impl<T: ShaderTypeExt> Wgsl for LocalVarDecl<T> {
    fn wgsl(&self) -> String {
        let name = &self.name;
        let type_name = T::wgsl_type_name();
        let initializer_str = match &self.initializer {
            Some(init) => format!(" = {}", init.wgsl()),
            None => "".into(),
        };
        format!("var {name}: {type_name}{initializer_str};")
    }
}

impl<T: ShaderTypeExt> Statement for LocalVarDecl<T> {}

macro_rules! impl_LocalVarDecl_xy {
    ($item_type:ty) => {
        impl LocalVarDecl<Vector2<$item_type>> {
            pub fn x(&self) -> Expression<$item_type> {
                Expression(format!("{}.x", Expression::from(self)), PhantomData)
            }
            pub fn y(&self) -> Expression<$item_type> {
                Expression(format!("{}.y", Expression::from(self)), PhantomData)
            }
        }
    };
}

macro_rules! impl_LocalVarDecl_xyz {
    ($item_type:ty) => {
        impl LocalVarDecl<Vector3<$item_type>> {
            pub fn x(&self) -> Expression<$item_type> {
                Expression(format!("{}.x", Expression::from(self)), PhantomData)
            }
            pub fn y(&self) -> Expression<$item_type> {
                Expression(format!("{}.y", Expression::from(self)), PhantomData)
            }
            pub fn z(&self) -> Expression<$item_type> {
                Expression(format!("{}.z", Expression::from(self)), PhantomData)
            }
        }
    };
}
impl_LocalVarDecl_xyz!(f32);
impl_LocalVarDecl_xyz!(u32);

impl_LocalVarDecl_xy!(f32);
impl_LocalVarDecl_xy!(u32);
